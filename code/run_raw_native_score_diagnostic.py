"""Learn raw-native templates and trace events at thresholds 8 and 10.75."""
from __future__ import annotations

import json
import os
import shutil
import time
from importlib.metadata import version

import numpy as np
import torch

import spikeinterface as si

import run_score_diagnostic as diagnostic


def main() -> None:
    from kilosort import io as ks_io

    started = time.perf_counter()
    versions = {
        "kilosort": version("kilosort"),
        "spikeinterface": version("spikeinterface"),
    }
    expected_versions = {"kilosort": "4.1.7", "spikeinterface": "0.104.7"}
    if versions != expected_versions:
        raise RuntimeError(
            f"unexpected diagnostic environment: {versions} != {expected_versions}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("raw-native score diagnostic requires a CUDA device")

    diagnostic.RESULTS.mkdir(parents=True, exist_ok=True)
    diagnostic.SCRATCH.mkdir(parents=True, exist_ok=True)
    raw_path, gt_path = diagnostic.discover_raw_inputs()
    raw = si.read_zarr(raw_path)
    if raw.get_num_segments() != 1:
        raise ValueError("raw-native diagnostic input must have one segment")
    sampling_frequency = raw.get_sampling_frequency()
    end_frame = int(round(diagnostic.EXPECTED_DURATION_S * sampling_frequency))
    if raw.get_num_samples(segment_index=0) < end_frame:
        raise ValueError("raw recording is shorter than 1200 seconds")
    raw = raw.frame_slice(0, end_frame)
    gt = si.read_zarr(gt_path).frame_slice(0, end_frame)
    if not np.isclose(gt.get_sampling_frequency(), sampling_frequency):
        raise ValueError("raw and GT sampling frequencies differ")

    raw_preprocessed, kept_ids, removed_ids = diagnostic.raw_native_preprocessing(raw)
    si.set_global_job_kwargs(
        n_jobs=int(os.getenv("CO_CPUS", "-1")),
        chunk_duration="1s",
        progress_bar=False,
        mp_context="spawn",
    )
    raw_folder = diagnostic.SCRATCH / "diagnostic_raw_native"
    raw_saved = diagnostic.save_recording(raw_preprocessed, raw_folder)
    template_learning_folder = diagnostic.SCRATCH / "raw_native_template_learning"
    ops_path, templates, final_reference = diagnostic.learn_templates(
        raw_saved, template_learning_folder
    )

    device = torch.device("cuda")
    ops = ks_io.load_ops(ops_path, device=device)
    learned_templates = torch.from_numpy(templates).to(device)
    shutil.copy2(ops_path, diagnostic.RESULTS / "raw_native_ops.npy")
    np.save(diagnostic.RESULTS / "raw_native_learned_templates.npy", templates)
    np.save(diagnostic.RESULTS / "raw_native_whitening_matrix.npy", ops["Wrot"].cpu())
    np.savez_compressed(
        diagnostic.RESULTS / "raw_native_template_learning_final_sort.npz",
        **final_reference,
    )
    shutil.rmtree(template_learning_folder)

    scoreable_start, scoreable_stop = diagnostic.replay_frame_bounds(ops, end_frame)
    gt_all, background, gt_sampled, gt_excluded_by_unit = (
        diagnostic.sample_gt_and_background(
            gt, end_frame, ops["nt"], scoreable_start, scoreable_stop
        )
    )
    output = diagnostic.replay_domain(
        "raw_native",
        diagnostic.binary_path(raw_saved),
        ops,
        learned_templates,
        gt_all,
        gt_sampled,
        background,
        gt,
        final_reference,
        device,
        lineage_thresholds=(diagnostic.TEMPLATE_LEARNING_THRESHOLD,),
        archive_mode="compact",
    )
    diagnostic.write_tabular_outputs([output])
    del raw_saved
    shutil.rmtree(raw_folder)

    manifest = {
        "diagnostic": "raw-native templates and preprocessing replayed on raw voltage",
        "kilosort_version": versions["kilosort"],
        "spikeinterface_version": versions["spikeinterface"],
        "template_source": "raw",
        "template_learning_Th_learned": diagnostic.TEMPLATE_LEARNING_THRESHOLD,
        "replay_thresholds": diagnostic.THRESHOLDS,
        "lineage_thresholds": [diagnostic.TEMPLATE_LEARNING_THRESHOLD],
        "event_archive_mode": "compact",
        "lineage_stages": [
            "detection_template",
            "final_clustering",
            "cluster_merging",
            "duplicate_removal",
        ],
        "lineage_delta_time_ms": diagnostic.LINEAGE_DELTA_TIME_MS,
        "lineage_match_score": diagnostic.LINEAGE_MATCH_SCORE,
        "lineage_status_codes": diagnostic.STATUS_NAMES,
        "lineage_scope": "raw-native first-1200-second diagnostic; templates and transform learned from raw voltage",
        "raw_Th8_lineage_reference": "exact template-learning sorting times and cluster partition, allowing cluster ID permutation",
        "lineage_npz_time_coordinates": "recording sample indices after applying bfile.imin",
        "lineage_peel_source": "captured inline during the same exact extraction pass that produced each event",
        "independent_replay_role": "score calibration and non-authoritative extraction-difference summary only",
        "duration_s": diagnostic.EXPECTED_DURATION_S,
        "sampling_frequency": sampling_frequency,
        "gt_units": [int(unit_id) for unit_id in gt.unit_ids],
        "gt_events": int(sum(frames.size for frames in gt_all.values())),
        "gt_events_before_edge_exclusion": int(
            sum(frames.size for frames in gt_all.values())
            + sum(gt_excluded_by_unit.values())
        ),
        "gt_events_excluded_unscoreable_edges": int(
            sum(gt_excluded_by_unit.values())
        ),
        "gt_events_excluded_unscoreable_edges_by_unit": gt_excluded_by_unit,
        "scoreable_frame_start_inclusive": scoreable_start,
        "scoreable_frame_stop_exclusive": scoreable_stop,
        "input_channels": raw.get_num_channels(),
        "matched_channels": len(kept_ids),
        "kept_channel_ids": kept_ids,
        "removed_channel_ids": removed_ids,
        "fixed_transform": "raw-derived channel mask, Kilosort whitening, drift, and learned templates",
        "elapsed_s": time.perf_counter() - started,
    }
    manifest_path = diagnostic.RESULTS / "raw_native_score_diagnostic_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=diagnostic.json_default) + "\n"
    )
    print(json.dumps(manifest, indent=2, default=diagnostic.json_default), flush=True)


if __name__ == "__main__":
    main()
