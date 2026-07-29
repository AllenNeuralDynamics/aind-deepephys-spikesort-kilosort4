"""Prepare deterministic TP/FP event selections for waveform inspection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from event_waveform import stratified_sample_indices


DEFAULT_UNITS = (337, 664, 793, 1300)
STATUS_CODES = {"fp": 1, "tp": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--ops", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events-per-class", type=int, default=128)
    parser.add_argument(
        "--units",
        type=int,
        nargs="+",
        default=list(DEFAULT_UNITS),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def main() -> None:
    args = parse_args()
    if args.events_per_class <= 1:
        raise ValueError("--events-per-class must exceed one")
    archive = np.load(args.archive)
    if "final_status" in archive.files:
        status = archive["final_status"]
        assigned_gt = archive["final_assigned_gt"]
    else:
        status = archive["duplicate_removal_status"]
        assigned_gt = archive["duplicate_removal_assigned_gt"]
    kept = archive["kept_after_dedup"]
    ops = np.load(args.ops, allow_pickle=True).item()
    coefficients = np.load(args.templates)
    w_pca = as_numpy(ops["wPCA"])
    nearest_channels = as_numpy(ops["iCC"]).astype(np.int64)

    units = []
    events = []
    for unit_id in args.units:
        unit_mask = kept & (assigned_gt == unit_id)
        if not np.any(unit_mask & (status == STATUS_CODES["tp"])):
            raise ValueError(f"GT unit {unit_id} has no final TP events")
        source_templates, source_counts = np.unique(
            archive["detection_template"][unit_mask], return_counts=True
        )
        dominant_template = int(source_templates[np.argmax(source_counts)])
        waveform = np.einsum("jk,jl->kl", coefficients[dominant_template], w_pca)
        peak_channel = int(np.argmax(np.ptp(waveform, axis=1)))
        channel_indices = nearest_channels[:, peak_channel].tolist()
        units.append(
            {
                "gt_unit_id": int(unit_id),
                "dominant_detection_template": dominant_template,
                "peak_channel_index": peak_channel,
                "channel_indices": [int(value) for value in channel_indices],
                "final_cluster": int(np.unique(archive["postmerge_cluster"][unit_mask])[0]),
                "final_tp_events": int((unit_mask & (status == STATUS_CODES["tp"])).sum()),
                "final_fp_events": int((unit_mask & (status == STATUS_CODES["fp"])).sum()),
            }
        )
        for class_name, status_code in STATUS_CODES.items():
            event_ids = np.flatnonzero(unit_mask & (status == status_code))
            if not event_ids.size:
                continue
            local_indices = stratified_sample_indices(
                archive["detection_score"][event_ids],
                archive["detection_peel"][event_ids],
                count=args.events_per_class,
                seed=unit_id * 10 + status_code,
            )
            selected = event_ids[local_indices]
            reference_ids: set[int] = set()
            if class_name == "tp":
                shuffled = selected.copy()
                np.random.default_rng(unit_id).shuffle(shuffled)
                reference_ids = set(
                    int(value) for value in shuffled[: shuffled.size // 2]
                )
            for event_id in selected:
                events.append(
                    {
                        "event_id": int(event_id),
                        "gt_unit_id": int(unit_id),
                        "event_class": class_name,
                        "reference_tp": int(event_id) in reference_ids,
                        "time_sample": int(archive["postmerge_time"][event_id]),
                        "detection_template": int(
                            archive["detection_template"][event_id]
                        ),
                        "detection_score": float(
                            archive["detection_score"][event_id]
                        ),
                        "detection_peel": int(
                            archive["detection_peel"][event_id]
                        ),
                    }
                )

    output = {
        "analysis": "posthoc native-denoised TP/FP waveform phenotype",
        "uses_ground_truth_for_analysis_labels": True,
        "uses_ground_truth_for_sorting_decisions": False,
        "source_computation": "d0efc11d-4141-443b-b381-98c9dbeecd6a",
        "source_archive": args.archive.name,
        "source_archive_sha256": sha256(args.archive),
        "source_ops_sha256": sha256(args.ops),
        "source_templates_sha256": sha256(args.templates),
        "events_per_available_class": args.events_per_class,
        "sampling": "four-by-four joint score-rank and peel-rank stratification",
        "units": units,
        "events": events,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "units": units,
                "selected_events": len(events),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()