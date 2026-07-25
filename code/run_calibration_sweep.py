"""Run a controlled Kilosort threshold sweep on one denoised hybrid clip."""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.comparison as sc
import spikeinterface.curation as sic
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss


DATA = Path("../data")
RESULTS = Path("../results")
SCRATCH = Path("../scratch")
CONFIG_PATH = Path(__file__).with_name("calibration_sweep.json")
SORTER_PARAMS_PATH = Path(__file__).with_name("params.json")

PREPROCESSING_PIPELINE = {
    "phase_shift": {"margin_ms": 100.0},
    "highpass_filter": {"freq_min": 300.0, "margin_ms": 5.0},
    "detect_and_remove_bad_channels": {
        "method": "coherence+psd",
        "dead_channel_threshold": -0.5,
        "noisy_channel_threshold": 1.0,
        "outside_channel_threshold": -0.3,
        "outside_channels_location": "top",
        "n_neighbors": 11,
        "channel_filters": {"noise", "dead", "out"},
        "seed": 0,
    },
    "common_reference": {"reference": "global", "operator": "median"},
}


def _single_path(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(
            f"expected exactly one {description}, found {len(paths)}: {paths}"
        )
    return paths[0]


def discover_inputs(data_folder: Path) -> tuple[Path, Path]:
    recording_candidates = sorted(
        path.parent for path in data_folder.rglob("si_folder.json")
    )
    gt_candidates = sorted(data_folder.rglob("sorting.zarr"))
    return (
        _single_path(recording_candidates, "denoised binary recording"),
        _single_path(gt_candidates, "ground-truth sorting.zarr"),
    )


def preprocessing_pipeline(recording: si.BaseRecording) -> dict:
    pipeline = copy.deepcopy(PREPROCESSING_PIPELINE)
    if "inter_sample_shift" not in recording.get_property_keys():
        pipeline.pop("phase_shift")
    return pipeline


def load_sweep_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text())
    thresholds = [float(value) for value in config["thresholds"]]
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError(f"invalid thresholds: {thresholds}")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError(f"thresholds must be unique: {thresholds}")
    config["thresholds"] = thresholds
    duration_keys = [
        key
        for key in ("expected_duration_s", "minimum_duration_s")
        if key in config
    ]
    if len(duration_keys) != 1:
        raise ValueError(
            "configure exactly one of expected_duration_s or minimum_duration_s"
        )
    duration_key = duration_keys[0]
    config[duration_key] = float(config[duration_key])
    if config[duration_key] <= 0:
        raise ValueError(f"{duration_key} must be positive")
    return config


def evaluate_threshold(
    threshold: float,
    gt_sorting: si.BaseSorting,
    tested_sorting: si.BaseSorting,
    match_score: float,
    well_detected_score: float,
) -> tuple[pd.DataFrame, dict]:
    comparison = sc.compare_sorter_to_ground_truth(
        gt_sorting,
        tested_sorting,
        exhaustive_gt=False,
        match_score=match_score,
        well_detected_score=well_detected_score,
    )
    performance = comparison.get_performance(method="by_unit", output="pandas")
    counts = comparison.get_performance(method="raw_count", output="pandas")
    per_unit = performance[["accuracy", "precision", "recall"]].join(
        counts[["tp", "fn", "fp", "num_gt"]]
    )
    per_unit.index.name = "gt_unit_id"
    per_unit.insert(0, "Th_learned", threshold)
    per_unit = per_unit.reset_index()

    total_spikes = int(tested_sorting.to_spike_vector().size)
    summary = {
        "Th_learned": threshold,
        "mean_accuracy": float(per_unit["accuracy"].mean()),
        "mean_precision": float(per_unit["precision"].mean()),
        "mean_recall": float(per_unit["recall"].mean()),
        "gt_units_detected": int((per_unit["accuracy"] > 0).sum()),
        "gt_units_above_0_8_accuracy": int(
            (per_unit["accuracy"] > well_detected_score).sum()
        ),
        "sorter_units": int(tested_sorting.get_num_units()),
        "sorted_spike_events": total_spikes,
        "true_positive_injected_spikes": int(per_unit["tp"].sum()),
        "false_positive_spikes_in_gt_matched_clusters": int(per_unit["fp"].sum()),
    }
    return per_unit, summary


def main() -> None:
    config = load_sweep_config()
    RESULTS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    recording_folder, gt_folder = discover_inputs(DATA)

    recording = si.load(recording_folder)
    if recording.get_num_segments() != 1:
        raise ValueError("calibration recording must have one segment")
    duration_s = recording.get_total_duration()
    if "expected_duration_s" in config:
        expected_duration_s = config["expected_duration_s"]
        if not np.isclose(
            duration_s,
            expected_duration_s,
            atol=1 / recording.sampling_frequency,
        ):
            raise ValueError(
                f"unexpected recording duration: {duration_s} != "
                f"{expected_duration_s}"
            )
    elif duration_s < config["minimum_duration_s"]:
        raise ValueError(
            f"recording is too short: {duration_s} < "
            f"{config['minimum_duration_s']}"
        )

    gt_sorting = si.read_zarr(gt_folder)
    if not np.isclose(
        gt_sorting.get_sampling_frequency(), recording.get_sampling_frequency()
    ):
        raise ValueError("recording/GT sampling-frequency mismatch")
    gt_sorting = gt_sorting.frame_slice(
        start_frame=0, end_frame=recording.get_num_samples(segment_index=0)
    )

    si.set_global_job_kwargs(
        n_jobs=int(os.getenv("CO_CPUS", "-1")),
        chunk_duration="1s",
        progress_bar=False,
        mp_context="spawn",
    )
    pipeline = preprocessing_pipeline(recording)
    preprocessed = spre.apply_preprocessing_pipeline(recording, pipeline)
    if preprocessed.get_num_channels() < 0.5 * recording.get_num_channels():
        raise RuntimeError("preprocessing removed more than half of the channels")
    preprocessed_folder = SCRATCH / "preprocessed_calibration"
    if preprocessed_folder.exists():
        shutil.rmtree(preprocessed_folder)
    preprocessed = preprocessed.save(folder=preprocessed_folder)

    base_params = json.loads(SORTER_PARAMS_PATH.read_text())["sorter"]
    per_unit_frames = []
    summaries = []
    started = time.perf_counter()
    for threshold in config["thresholds"]:
        label = f"th{threshold:g}".replace(".", "p")
        sorter_folder = SCRATCH / f"ks4_{label}"
        output_folder = RESULTS / f"spikesorted_{label}"
        for folder in (sorter_folder, output_folder):
            if folder.exists():
                shutil.rmtree(folder)

        sorter_params = copy.deepcopy(base_params)
        sorter_params["Th_learned"] = threshold
        threshold_started = time.perf_counter()
        sorting = ss.run_sorter(
            "kilosort4",
            preprocessed,
            folder=sorter_folder,
            verbose=False,
            delete_output_folder=False,
            remove_existing_folder=True,
            **sorter_params,
        )
        sorting = sorting.remove_empty_units()
        sorting = sic.remove_excess_spikes(sorting=sorting, recording=preprocessed)
        sorting = sorting.save(folder=output_folder)
        log_source = sorter_folder / "spikeinterface_log.json"
        if log_source.is_file():
            shutil.copy(log_source, output_folder / log_source.name)

        per_unit, summary = evaluate_threshold(
            threshold,
            gt_sorting,
            sorting,
            match_score=float(config["match_score"]),
            well_detected_score=float(config["well_detected_score"]),
        )
        summary["sorter_run_time_s"] = time.perf_counter() - threshold_started
        summary["sorted_spike_events_per_s"] = (
            summary["sorted_spike_events"] / duration_s
        )
        per_unit_frames.append(per_unit)
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    per_unit_table = pd.concat(per_unit_frames, ignore_index=True)
    summary_table = pd.DataFrame(summaries)
    per_unit_table.to_csv(RESULTS / "threshold_sweep_per_unit.csv", index=False)
    summary_table.to_csv(RESULTS / "threshold_sweep_summary.csv", index=False)
    manifest = {
        "recording_folder": str(recording_folder),
        "gt_folder": str(gt_folder),
        "duration_s": duration_s,
        "sampling_frequency": recording.get_sampling_frequency(),
        "input_channels": recording.get_num_channels(),
        "preprocessed_channels": preprocessed.get_num_channels(),
        "thresholds": config["thresholds"],
        "preprocessing_pipeline": pipeline,
        "elapsed_s": time.perf_counter() - started,
    }
    (RESULTS / "threshold_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=list) + "\n"
    )


if __name__ == "__main__":
    main()