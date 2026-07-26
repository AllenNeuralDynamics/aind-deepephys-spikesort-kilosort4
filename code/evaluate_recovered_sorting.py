"""Evaluate one recovered full-recording sorting against the exact hybrid GT."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import spikeinterface as si
import spikeinterface.comparison as sc


DATA = Path("../data")
RESULTS = Path("../results")
EXPECTED_DURATION_S = 7144.262
EXPECTED_GT_EVENTS = 1_070_127
EXPECTED_THRESHOLDS = {"Th_universal": 9, "Th_learned": 10}


def single_path(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(
            f"expected exactly one {description}, found {len(paths)}: {paths}"
        )
    return paths[0]


def sorter_params(sorting_folder: Path) -> dict:
    provenance = json.loads((sorting_folder / "provenance.json").read_text())
    return provenance["annotations"]["__sorting_info__"]["params"][
        "sorter_params"
    ]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sorting_folder = single_path(
        sorted(
            path.parent
            for path in DATA.rglob("si_folder.json")
            if path.parent.name.startswith("spikesorted_")
        ),
        "recovered spikesorted folder",
    )
    gt_folder = single_path(sorted(DATA.rglob("sorting.zarr")), "GT sorting.zarr")

    tested_sorting = si.load(sorting_folder)
    gt_sorting = si.read_zarr(gt_folder)
    if tested_sorting.get_sampling_frequency() != gt_sorting.get_sampling_frequency():
        raise ValueError("sorting/GT sampling-frequency mismatch")
    gt_events = int(gt_sorting.to_spike_vector().size)
    if gt_events != EXPECTED_GT_EVENTS:
        raise ValueError(f"unexpected GT event count: {gt_events}")

    params = sorter_params(sorting_folder)
    actual_thresholds = {key: params[key] for key in EXPECTED_THRESHOLDS}
    if actual_thresholds != EXPECTED_THRESHOLDS:
        raise ValueError(
            f"unexpected recovered sorter thresholds: {actual_thresholds}"
        )

    comparison = sc.compare_sorter_to_ground_truth(
        gt_sorting,
        tested_sorting,
        exhaustive_gt=False,
        match_score=0.2,
        well_detected_score=0.8,
    )
    performance = comparison.get_performance(method="by_unit", output="pandas")
    counts = comparison.get_performance(method="raw_count", output="pandas")
    per_unit = performance[["accuracy", "precision", "recall"]].join(
        counts[["tp", "fn", "fp", "num_gt"]]
    )
    per_unit.index.name = "gt_unit_id"
    per_unit = per_unit.reset_index()

    spike_events = int(tested_sorting.to_spike_vector().size)
    log = json.loads((sorting_folder / "spikeinterface_log.json").read_text())
    summary = {
        **EXPECTED_THRESHOLDS,
        "mean_accuracy": float(per_unit["accuracy"].mean()),
        "mean_precision": float(per_unit["precision"].mean()),
        "mean_recall": float(per_unit["recall"].mean()),
        "gt_units_detected": int((per_unit["accuracy"] > 0).sum()),
        "gt_units_above_0_8_accuracy": int((per_unit["accuracy"] > 0.8).sum()),
        "sorter_units": int(tested_sorting.get_num_units()),
        "sorted_spike_events": spike_events,
        "sorted_spike_events_per_s": spike_events / EXPECTED_DURATION_S,
        "true_positive_injected_spikes": int(per_unit["tp"].sum()),
        "false_positive_spikes_in_gt_matched_clusters": int(per_unit["fp"].sum()),
        "sorter_run_time_s": float(log["run_time"]),
    }
    per_unit.to_csv(RESULTS / "recovered_threshold10_per_unit.csv", index=False)
    pd.DataFrame([summary]).to_csv(
        RESULTS / "recovered_threshold10_summary.csv", index=False
    )
    manifest = {
        "sorting_folder": str(sorting_folder),
        "gt_folder": str(gt_folder),
        "duration_s": EXPECTED_DURATION_S,
        "sampling_frequency": tested_sorting.get_sampling_frequency(),
        "gt_events": gt_events,
        "sorter_parameters": params,
    }
    (RESULTS / "recovered_threshold10_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()