"""Test GT-blind marginal-yield stopping during Kilosort matching pursuit."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
import torch

import peel_stopping
import run_score_diagnostic as diagnostic


THRESHOLD = 8.0
PATIENCE = 3
EXPECTED_DURATION_S = 1200.0
DOMAIN_LABELS = (
    "raw_native_peel_stopping",
    "denoised_native_peel_stopping",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=EXPECTED_DURATION_S,
        help="Recording duration; the registered full experiment uses 1200 seconds.",
    )
    parser.add_argument(
        "--domain",
        choices=("both", "raw", "denoised"),
        default="both",
        help="Domain subset; the registered full experiment uses both.",
    )
    parser.add_argument(
        "--skip-baseline-control",
        action="store_true",
        help="Skip the unchanged matcher arm that shares the learned native transform.",
    )
    args = parser.parse_args()
    if args.duration_s <= 0 or args.duration_s > EXPECTED_DURATION_S:
        parser.error(f"--duration-s must be in (0, {EXPECTED_DURATION_S:g}]")
    return args


def summarize_peel_comparison(
    baseline_peels: np.ndarray,
    stopped_peels: np.ndarray,
    max_peels: int,
    domain: str,
) -> list[dict]:
    """Summarize same-template baseline and stopped event counts by peel."""
    for label, peels in (("baseline", baseline_peels), ("stopped", stopped_peels)):
        if np.any((peels < 0) | (peels >= max_peels)):
            raise ValueError(f"{label} contains a peel outside [0, {max_peels})")
    baseline_counts = np.bincount(baseline_peels, minlength=max_peels)
    stopped_counts = np.bincount(stopped_peels, minlength=max_peels)
    baseline_late = np.cumsum(baseline_counts[::-1])[::-1]
    stopped_late = np.cumsum(stopped_counts[::-1])[::-1]
    rows = []
    for peel in range(max_peels):
        baseline_count = int(baseline_counts[peel])
        stopped_count = int(stopped_counts[peel])
        baseline_remaining = int(baseline_late[peel])
        stopped_remaining = int(stopped_late[peel])
        rows.append(
            {
                "domain": domain,
                "peel": peel,
                "baseline_events": baseline_count,
                "stopped_events": stopped_count,
                "event_delta": stopped_count - baseline_count,
                "event_ratio": (
                    stopped_count / baseline_count if baseline_count else np.nan
                ),
                "baseline_events_at_or_after": baseline_remaining,
                "stopped_events_at_or_after": stopped_remaining,
                "late_event_delta": stopped_remaining - baseline_remaining,
                "late_event_ratio": (
                    stopped_remaining / baseline_remaining
                    if baseline_remaining
                    else np.nan
                ),
            }
        )
    return rows


def run_peel_stopped_matching(
    ops: dict,
    X: torch.Tensor,
    U: torch.Tensor,
    ctc: torch.Tensor,
    patience: int = PATIENCE,
) -> dict[str, Any]:
    """Run exact matching until the marginal event yield converges."""
    matched = diagnostic.simulate_matching(
        ops,
        diagnostic.initial_projection(ops, X, U),
        U,
        ctc,
        THRESHOLD,
        X=X,
        stop_before_peel=lambda counts: peel_stopping.should_stop_for_low_yield(
            counts, patience=patience
        ),
    )
    counts = matched["peel_counts"].cpu().tolist()
    stop_triggered = bool(
        counts
        and counts[-1] > 0
        and peel_stopping.should_stop_for_low_yield(counts, patience=patience)
    )
    first_count = int(counts[0]) if counts else 0
    summary = {
        "stop_triggered": stop_triggered,
        "stop_peel": len(counts) - 1 if stop_triggered else -1,
        "first_peel_events": first_count,
        "marginal_yield_floor": peel_stopping.square_root_yield_floor(first_count),
        "triggering_peel_events": int(counts[-1]) if stop_triggered else 0,
        "evaluated_peels": len(counts),
        "accepted_peels": (
            int(matched["peel"].max()) + 1 if matched["peel"].numel() else 0
        ),
        "accepted_events": int(matched["time"].numel()),
    }
    return {**matched, "stopping_summary": summary}


def extract_peel_stopped(
    ops: dict,
    bfile,
    U: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, dict, np.ndarray, list[dict]]:
    """Run stopped extraction while preserving event and peel identity."""
    from kilosort import template_matching

    captured_rows = []
    captured_peels = []
    batch_summaries = []
    original_run_matching = template_matching.run_matching
    batch_index = 0

    def local_matching(local_ops, X, local_U, ctc, device=device):
        nonlocal batch_index
        matched = run_peel_stopped_matching(local_ops, X, local_U, ctc)
        local_st = torch.column_stack((matched["time"], matched["template"]))
        keep = np.ones(local_st.shape[0], dtype=bool)
        if batch_index == 0:
            local_times = local_st[:, 0].cpu().numpy()
            keep = (
                local_times
                - local_ops["nt"]
                - local_ops["nt"] // 2
                + local_ops["nt0min"]
            ) >= 0
        shift = batch_index * bfile.batch_downsampling * local_ops["batch_size"]
        rows = np.column_stack(
            (
                local_st[:, 0].cpu().numpy()
                - local_ops["nt"]
                + shift
                - local_ops["nt"] // 2
                + local_ops["nt0min"],
                local_st[:, 1].cpu().numpy(),
                matched["score"].cpu().numpy(),
            )
        )
        captured_rows.append(rows[keep])
        captured_peels.append(matched["peel"].cpu().numpy()[keep])
        batch_summaries.append(
            {"batch": batch_index, **matched["stopping_summary"]}
        )
        batch_index += 1
        return (
            local_st,
            matched["amplitude"].unsqueeze(1),
            matched["score"].unsqueeze(1),
            matched["Xres"],
        )

    template_matching.run_matching = local_matching
    try:
        st, tF, ops = template_matching.extract(ops, bfile, U, device=device)
    finally:
        template_matching.run_matching = original_run_matching
    if batch_index != int(bfile.n_batches):
        raise RuntimeError(
            f"peel-stopping capture covered {batch_index}/{bfile.n_batches} batches"
        )
    captured_st = np.concatenate(captured_rows)
    peels = np.concatenate(captured_peels)
    order = np.argsort(captured_st[:, 0])
    captured_st = captured_st[order]
    peels = peels[order]
    if not np.array_equal(
        captured_st[:, :2].astype(np.int64), st[:, :2].astype(np.int64)
    ):
        raise RuntimeError("peel-stopping capture changed extraction event identity")
    if not np.allclose(captured_st[:, 2], st[:, 2], rtol=0, atol=0):
        raise RuntimeError("peel-stopping capture changed extraction scores")
    return st, tF, ops, peels, batch_summaries


def process_domain(
    domain: str,
    recording: Any,
    gt: Any,
    device: torch.device,
    include_baseline_control: bool,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Learn a native transform, stop low-yield peels, and evaluate lineage."""
    from kilosort import io as ks_io

    preprocessed, kept_ids, removed_ids = diagnostic.raw_native_preprocessing(
        recording
    )
    saved_folder = diagnostic.SCRATCH / f"peel_stopping_{domain}"
    saved = diagnostic.save_recording(preprocessed, saved_folder)
    learning_folder = diagnostic.SCRATCH / f"peel_stopping_learning_{domain}"
    ops_path, templates, _ = diagnostic.learn_templates(saved, learning_folder)
    ops = ks_io.load_ops(ops_path, device=device)
    U = torch.from_numpy(templates).to(device)
    np.save(diagnostic.RESULTS / f"{domain}_native_templates.npy", templates)
    shutil.copy2(ops_path, diagnostic.RESULTS / f"{domain}_native_ops.npy")
    shutil.rmtree(learning_folder)

    bfile = ks_io.bfile_from_ops(
        ops=ops,
        filename=str(diagnostic.binary_path(saved)),
        device=device,
    )
    lineages = []
    baseline_peels = None
    baseline_events = None
    if include_baseline_control:
        baseline_domain = domain.replace(
            "_peel_stopping", "_baseline_control"
        )
        baseline_ops = ops.copy()
        baseline_ops["settings"] = ops["settings"].copy()
        baseline_st, baseline_tF, baseline_ops, baseline_peels = (
            diagnostic.extract_with_peels(baseline_ops, bfile, U, device)
        )
        baseline_events = int(baseline_st.shape[0])
        baseline_lineage = diagnostic.run_event_lineage(
            baseline_domain,
            THRESHOLD,
            baseline_ops,
            baseline_st,
            baseline_tF,
            baseline_peels,
            gt,
            float(ops["fs"]),
            int(bfile.imin),
            final_reference=None,
            archive_mode="none",
        )
        diagnostic.write_lineage_checkpoint(
            baseline_domain,
            THRESHOLD,
            baseline_lineage,
            {
                "domain": baseline_domain,
                "Th_learned": THRESHOLD,
                "algorithm": "unchanged_kilosort_4_1_7_same_template_control",
            },
        )
        lineages.append(baseline_lineage)
        del baseline_st, baseline_tF
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stopped_ops = ops.copy()
    stopped_ops["settings"] = ops["settings"].copy()
    st, tF, stopped_ops, peels, batch_summaries = extract_peel_stopped(
        stopped_ops, bfile, U, device
    )
    lineage = diagnostic.run_event_lineage(
        domain,
        THRESHOLD,
        stopped_ops,
        st,
        tF,
        peels,
        gt,
        float(ops["fs"]),
        int(bfile.imin),
        final_reference=None,
        archive_mode="none",
    )
    diagnostic.write_lineage_checkpoint(
        domain,
        THRESHOLD,
        lineage,
        {
            "domain": domain,
            "Th_learned": THRESHOLD,
            "algorithm": "square_root_marginal_yield_stopping",
            "patience": PATIENCE,
        },
    )
    lineages.append(lineage)
    peel_comparison_rows = (
        summarize_peel_comparison(
            baseline_peels,
            peels,
            int(ops["max_peels"]),
            domain,
        )
        if baseline_peels is not None
        else []
    )
    for row in batch_summaries:
        row["domain"] = domain
    metadata = {
        "domain": domain,
        "input_channels": recording.get_num_channels(),
        "native_channels": len(kept_ids),
        "kept_channel_ids": kept_ids,
        "removed_channel_ids": removed_ids,
        "templates": int(U.shape[0]),
        "baseline_control_events": baseline_events,
        "peel_stopped_events": int(st.shape[0]),
        "stopped_batches": int(
            sum(row["stop_triggered"] for row in batch_summaries)
        ),
        "total_batches": len(batch_summaries),
    }
    del st, tF, saved
    shutil.rmtree(saved_folder)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lineages, batch_summaries, peel_comparison_rows, metadata


def main() -> None:
    import spikeinterface as si

    args = parse_args()
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
        raise RuntimeError("peel-stopping diagnostic requires a CUDA device")

    diagnostic.RESULTS.mkdir(parents=True, exist_ok=True)
    diagnostic.SCRATCH.mkdir(parents=True, exist_ok=True)
    raw_path, denoised_path, gt_path = diagnostic.discover_inputs()
    raw = si.read_zarr(raw_path)
    denoised = si.load(denoised_path)
    sampling_frequency = raw.get_sampling_frequency()
    if not np.isclose(sampling_frequency, denoised.get_sampling_frequency()):
        raise ValueError("raw and denoised sampling frequencies differ")
    end_frame = int(round(args.duration_s * sampling_frequency))
    if raw.get_num_samples(segment_index=0) < end_frame:
        raise ValueError("raw recording is shorter than requested duration")
    raw = raw.frame_slice(0, end_frame)
    if denoised.get_num_samples(segment_index=0) < end_frame:
        raise ValueError("denoised recording is shorter than requested duration")
    denoised = denoised.frame_slice(0, end_frame)
    gt = si.read_zarr(gt_path).frame_slice(0, end_frame)
    si.set_global_job_kwargs(
        n_jobs=int(os.getenv("CO_CPUS", "-1")),
        chunk_duration="1s",
        progress_bar=False,
        mp_context="spawn",
    )
    device = torch.device("cuda")

    outputs = []
    all_batch_summaries = []
    all_peel_comparisons = []
    domains = []
    recordings = {
        "raw_native_peel_stopping": raw,
        "denoised_native_peel_stopping": denoised,
    }
    selected_domains = {
        "both": DOMAIN_LABELS,
        "raw": ("raw_native_peel_stopping",),
        "denoised": ("denoised_native_peel_stopping",),
    }[args.domain]
    for domain in selected_domains:
        lineages, batch_summaries, peel_comparisons, metadata = process_domain(
            domain,
            recordings[domain],
            gt,
            device,
            include_baseline_control=not args.skip_baseline_control,
        )
        outputs.extend(lineages)
        all_batch_summaries.extend(batch_summaries)
        all_peel_comparisons.extend(peel_comparisons)
        domains.append(metadata)

    for key, filename in (
        ("lineage_stages", "peel_stopping_stage_summary.csv"),
        ("lineage_units", "peel_stopping_unit_summary.csv"),
        ("lineage_clusters", "peel_stopping_cluster_summary.csv"),
        ("lineage_transitions", "peel_stopping_transition_summary.csv"),
        (
            "lineage_cluster_transitions",
            "peel_stopping_cluster_transition_summary.csv",
        ),
        ("lineage_stage_deltas", "peel_stopping_stage_deltas.csv"),
        ("lineage_scores", "peel_stopping_score_summary.csv"),
    ):
        pd.DataFrame([row for output in outputs for row in output[key]]).to_csv(
            diagnostic.RESULTS / filename, index=False
        )
    pd.DataFrame(all_batch_summaries).to_csv(
        diagnostic.RESULTS / "peel_stopping_batch_summary.csv", index=False
    )
    pd.DataFrame(all_peel_comparisons).to_csv(
        diagnostic.RESULTS / "peel_stopping_peel_comparison.csv", index=False
    )
    manifest = {
        "algorithm": "square-root marginal-yield peel stopping",
        "uses_ground_truth_for_decisions": False,
        "Th_learned": THRESHOLD,
        "decision": (
            "stop before the third consecutive peel with candidate count at or "
            "below ceil(sqrt(first-peel candidate count))"
        ),
        "decision_scope": "independent per batch",
        "patience": PATIENCE,
        "triggering_peel_is_accepted": False,
        "baseline_control": (
            "unchanged Kilosort 4.1.7 extraction using identical native "
            "preprocessing, ops, templates, binary, and batches"
        ),
        "duration_s": args.duration_s,
        "requested_domain": args.domain,
        "registered_full_policy": {
            "duration_s": EXPECTED_DURATION_S,
            "domain": "both",
            "Th_learned": THRESHOLD,
            "patience": PATIENCE,
            "include_baseline_control": True,
        },
        "sampling_frequency": sampling_frequency,
        "kilosort_version": versions["kilosort"],
        "spikeinterface_version": versions["spikeinterface"],
        "domains": domains,
        "elapsed_s": time.perf_counter() - started,
    }
    path = diagnostic.RESULTS / "peel_stopping_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, default=diagnostic.json_default) + "\n"
    )
    print(json.dumps(manifest, indent=2, default=diagnostic.json_default), flush=True)


if __name__ == "__main__":
    main()