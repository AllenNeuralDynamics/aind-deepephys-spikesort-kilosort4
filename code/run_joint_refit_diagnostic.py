"""Test GT-blind local joint amplitude refitting during Kilosort matching."""
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
from torch.nn.functional import conv1d, max_pool1d

import joint_refit


THRESHOLD = 8.0
EXPECTED_DURATION_S = 1200.0
DOMAIN_LABELS = ("raw_native_joint_refit", "denoised_native_joint_refit")


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
    args = parser.parse_args()
    if args.duration_s <= 0 or args.duration_s > EXPECTED_DURATION_S:
        parser.error(
            f"--duration-s must be in (0, {EXPECTED_DURATION_S:g}]"
        )
    return args


def _apply_deltas(
    B: torch.Tensor,
    Xres: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    ctc: torch.Tensor,
    times: torch.Tensor,
    templates: torch.Tensor,
    deltas: torch.Tensor,
    nt: int,
) -> None:
    changed = deltas != 0
    if not torch.any(changed):
        return
    times = times[changed]
    templates = templates[changed]
    deltas = deltas[changed]

    tiwave = torch.arange(-(nt // 2), nt // 2 + 1, device=B.device)
    waveform_indices = (times[:, None] + tiwave).reshape(-1)
    waveforms = torch.einsum("ijk,jl->kil", U[templates], W)
    waveform_deltas = (deltas[None, :, None] * waveforms).reshape(
        Xres.shape[0], -1
    )
    Xres.index_add_(1, waveform_indices, -waveform_deltas)

    trange = torch.arange(-nt, nt + 1, device=B.device)
    projection_indices = (times[:, None] + trange).reshape(-1)
    projection_deltas = (
        deltas[None, :, None] * ctc[:, templates, :]
    ).reshape(B.shape[0], -1)
    B.index_add_(1, projection_indices, -projection_deltas)


def _quantile(values: list[float], quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values else np.nan


def run_joint_refit_matching(
    ops: dict,
    X: torch.Tensor,
    U: torch.Tensor,
    ctc: torch.Tensor,
) -> dict[str, torch.Tensor | list[dict] | dict]:
    """Run greedy matching with a local NNLS block update after every peel."""
    started = time.perf_counter()
    nt = ops["nt"]
    max_events = 100_000
    W = ops["wPCA"].contiguous()
    nm = (U**2).sum(-1).sum(-1)
    projection = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt // 2)
    B = torch.einsum("ijk,kjl->il", U, projection)
    Xres = X.clone()
    trange = torch.arange(-nt, nt + 1, device=B.device)
    tiwave = torch.arange(-(nt // 2), nt // 2 + 1, device=B.device)

    event_times = torch.empty(max_events, dtype=torch.int64, device=B.device)
    event_templates = torch.empty(max_events, dtype=torch.int64, device=B.device)
    event_scores = torch.empty(max_events, dtype=B.dtype, device=B.device)
    event_amplitudes = torch.zeros(max_events, dtype=B.dtype, device=B.device)
    event_peels = torch.empty(max_events, dtype=torch.int64, device=B.device)
    event_lookup: dict[tuple[int, int], int] = {}
    event_count = 0
    candidate_count = 0
    redetection_count = 0
    telemetry_rows = []
    residual_energy = float(torch.sum(Xres**2))
    initial_residual_energy = residual_energy

    for peel in range(ops["max_peels"]):
        peel_started = time.perf_counter()
        field = torch.relu(B) ** 2 / nm.unsqueeze(-1)
        field[:, :nt] = 0
        field[:, -nt:] = 0
        best, templates_by_time = torch.max(field, 0)
        pooled = max_pool1d(
            best.unsqueeze(0).unsqueeze(0),
            2 * nt + 1,
            stride=1,
            padding=nt,
        )[0, 0]
        accepted = (pooled > THRESHOLD**2) & (torch.abs(pooled - best) < 1e-9)
        times = torch.nonzero(accepted)[:, 0]
        if times.numel() == 0:
            break

        templates = templates_by_time[times]
        scores = pooled[times].sqrt()
        greedy_projection = B[templates, times]
        greedy_amplitudes = greedy_projection / nm[templates]
        if times.numel() > 1 and torch.any(torch.diff(times) <= nt):
            greedy_gram = joint_refit.event_gram(ctc, times, templates, nt)
            greedy_energy_reduction = float(
                2 * torch.dot(greedy_amplitudes, greedy_projection)
                - torch.dot(greedy_amplitudes, greedy_gram @ greedy_amplitudes)
            )
        else:
            greedy_norms = ctc[templates, templates, nt]
            greedy_energy_reduction = float(
                torch.sum(
                    2 * greedy_amplitudes * greedy_projection
                    - greedy_amplitudes**2 * greedy_norms
                )
            )
        candidates_this_peel = int(times.numel())
        candidate_count += candidates_this_peel

        time_values = times.detach().cpu().tolist()
        template_values = templates.detach().cpu().tolist()
        focal_indices = []
        new_positions = []
        revived = []
        for position, key in enumerate(zip(time_values, template_values)):
            focal = event_lookup.get(key)
            if focal is None:
                focal = event_count + len(new_positions)
                if focal >= max_events:
                    raise RuntimeError(
                        f"joint refit exceeded {max_events} unique events in one batch"
                    )
                event_lookup[key] = focal
                new_positions.append(position)
            else:
                redetection_count += 1
                if event_amplitudes[focal] == 0:
                    revived.append((focal, position))
            focal_indices.append(focal)

        new_count = len(new_positions)
        if new_count:
            new_slice = slice(event_count, event_count + new_count)
            new_positions_tensor = torch.as_tensor(
                new_positions, dtype=torch.int64, device=B.device
            )
            event_times[new_slice] = times[new_positions_tensor]
            event_templates[new_slice] = templates[new_positions_tensor]
            event_scores[new_slice] = scores[new_positions_tensor]
            event_peels[new_slice] = peel
            event_count += new_count
        for focal, position in revived:
            event_scores[focal] = scores[position]
            event_peels[focal] = peel

        focal_indices_tensor = torch.as_tensor(
            focal_indices, dtype=torch.int64, device=B.device
        )
        event_amplitudes.index_add_(
            0, focal_indices_tensor, greedy_amplitudes
        )

        for parity in range(2):
            parity_times = times[parity::2]
            parity_templates = templates[parity::2]
            parity_amplitudes = greedy_amplitudes[parity::2]
            Xres[:, parity_times[:, None] + tiwave] -= parity_amplitudes[
                None, :, None
            ] * torch.einsum("ijk,jl->kil", U[parity_templates], W)
            B[:, parity_times[:, None] + trange] -= parity_amplitudes[
                None, :, None
            ] * ctc[:, parity_templates, :]

        residual_before_greedy = residual_energy
        residual_after_greedy = residual_before_greedy - greedy_energy_reduction
        residual_energy = residual_after_greedy

        active_indices = torch.nonzero(event_amplitudes[:event_count] > 0)[:, 0]
        sorted_times, order = torch.sort(event_times[active_indices])
        sorted_indices = active_indices[order]
        left = torch.searchsorted(sorted_times, times - nt, side="left")
        right = torch.searchsorted(sorted_times, times + nt, side="right")

        block_sizes = []
        conditions = []
        solver_iterations = []
        overlap_blocks = 0
        changed_amplitudes = 0
        zeroed_amplitudes = 0
        rank_deficient_blocks = 0
        nonconverged_blocks = 0
        reverted_blocks = 0
        refit_energy_reduction = 0.0
        amplitude_l1_change = 0.0
        amplitude_max_change = 0.0

        for position, focal in enumerate(focal_indices):
            block = sorted_indices[left[position] : right[position]]
            block = block[event_amplitudes[block] > 0]
            block_size = int(block.numel())
            block_sizes.append(block_size)
            if block_size <= 1:
                continue
            overlap_blocks += 1
            before = event_amplitudes[block].clone()
            refitted, deltas, stats = joint_refit.refit_block(
                B,
                ctc,
                event_times[block],
                event_templates[block],
                before,
                nt,
            )
            conditions.append(stats["condition"])
            solver_iterations.append(stats["iterations"])
            if stats["rank"] < block_size:
                rank_deficient_blocks += 1
            if not stats["converged"]:
                nonconverged_blocks += 1
            energy_reduction = stats["energy_reduction"]
            if not stats["converged"] or not np.isfinite(energy_reduction) or energy_reduction < 0:
                reverted_blocks += 1
                continue

            _apply_deltas(
                B,
                Xres,
                U,
                W,
                ctc,
                event_times[block],
                event_templates[block],
                deltas,
                nt,
            )
            event_amplitudes[block] = refitted
            absolute_change = torch.abs(deltas)
            changed_amplitudes += int(torch.count_nonzero(absolute_change).item())
            zeroed_amplitudes += int(torch.count_nonzero((before > 0) & (refitted == 0)).item())
            amplitude_l1_change += float(torch.sum(absolute_change))
            amplitude_max_change = max(
                amplitude_max_change, float(torch.max(absolute_change))
            )
            refit_energy_reduction += energy_reduction

        residual_energy -= refit_energy_reduction
        finite_conditions = [value for value in conditions if np.isfinite(value)]
        telemetry_rows.append(
            {
                "peel": peel,
                "detected_candidates": candidates_this_peel,
                "new_unique_events": new_count,
                "redetected_candidates": candidates_this_peel - new_count,
                "active_events_after_refit": int(
                    torch.count_nonzero(event_amplitudes[:event_count] > 0).item()
                ),
                "residual_energy_before_greedy": residual_before_greedy,
                "greedy_energy_reduction": greedy_energy_reduction,
                "residual_energy_after_greedy": residual_after_greedy,
                "refit_energy_reduction": refit_energy_reduction,
                "residual_energy_after_refit": residual_energy,
                "refit_to_greedy_energy_ratio": (
                    refit_energy_reduction / greedy_energy_reduction
                    if greedy_energy_reduction
                    else np.nan
                ),
                "refit_blocks": candidates_this_peel,
                "overlap_blocks": overlap_blocks,
                "block_size_mean": float(np.mean(block_sizes)),
                "block_size_p50": _quantile(block_sizes, 0.5),
                "block_size_p90": _quantile(block_sizes, 0.9),
                "block_size_p99": _quantile(block_sizes, 0.99),
                "block_size_max": max(block_sizes),
                "gram_condition_p50": _quantile(finite_conditions, 0.5),
                "gram_condition_p90": _quantile(finite_conditions, 0.9),
                "gram_condition_max": (
                    max(finite_conditions) if finite_conditions else np.nan
                ),
                "rank_deficient_blocks": rank_deficient_blocks,
                "nonconverged_blocks": nonconverged_blocks,
                "reverted_blocks": reverted_blocks,
                "solver_iterations_mean": (
                    float(np.mean(solver_iterations)) if solver_iterations else 0.0
                ),
                "solver_iterations_max": (
                    max(solver_iterations) if solver_iterations else 0
                ),
                "changed_amplitudes": changed_amplitudes,
                "zeroed_amplitudes": zeroed_amplitudes,
                "amplitude_l1_change": amplitude_l1_change,
                "amplitude_max_change": amplitude_max_change,
                "elapsed_s": time.perf_counter() - peel_started,
            }
        )

    positive = event_amplitudes[:event_count] > 0
    direct_final_energy = float(torch.sum(Xres**2))
    batch_summary = {
        "candidate_detections": candidate_count,
        "unique_event_variables": event_count,
        "redetected_candidates": redetection_count,
        "returned_positive_events": int(torch.count_nonzero(positive).item()),
        "zero_amplitude_events": int(torch.count_nonzero(~positive).item()),
        "initial_residual_energy": initial_residual_energy,
        "predicted_final_residual_energy": residual_energy,
        "direct_final_residual_energy": direct_final_energy,
        "residual_energy_audit_error": direct_final_energy - residual_energy,
        "elapsed_s": time.perf_counter() - started,
    }
    return {
        "time": event_times[:event_count][positive],
        "template": event_templates[:event_count][positive],
        "score": event_scores[:event_count][positive],
        "amplitude": event_amplitudes[:event_count][positive],
        "peel": event_peels[:event_count][positive],
        "Xres": Xres,
        "telemetry_rows": telemetry_rows,
        "batch_summary": batch_summary,
    }


def extract_joint_refit(
    ops: dict,
    bfile,
    U: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, dict, np.ndarray, list[dict], list[dict]]:
    """Run one joint-refit extraction while preserving event/peel identity."""
    from kilosort import template_matching

    captured_rows = []
    captured_peels = []
    telemetry_rows = []
    batch_summaries = []
    original_run_matching = template_matching.run_matching
    batch_index = 0

    def local_matching(local_ops, X, local_U, ctc, device=device):
        nonlocal batch_index
        matched = run_joint_refit_matching(local_ops, X, local_U, ctc)
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
        for row in matched["telemetry_rows"]:
            telemetry_rows.append({"batch": batch_index, **row})
        batch_summaries.append({"batch": batch_index, **matched["batch_summary"]})
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
            f"joint-refit capture covered {batch_index}/{bfile.n_batches} batches"
        )
    captured_st = np.concatenate(captured_rows)
    peels = np.concatenate(captured_peels)
    order = np.argsort(captured_st[:, 0])
    captured_st = captured_st[order]
    peels = peels[order]
    if not np.array_equal(
        captured_st[:, :2].astype(np.int64), st[:, :2].astype(np.int64)
    ):
        raise RuntimeError("joint-refit capture changed extraction event identity")
    if not np.allclose(captured_st[:, 2], st[:, 2], rtol=0, atol=0):
        raise RuntimeError("joint-refit capture changed extraction scores")
    return st, tF, ops, peels, telemetry_rows, batch_summaries


def process_domain(
    domain: str,
    recording: Any,
    gt: Any,
    device: torch.device,
) -> tuple[dict, list[dict], list[dict], dict]:
    """Learn a native transform, run joint refitting, and evaluate lineage."""
    from kilosort import io as ks_io
    import run_score_diagnostic as diagnostic

    preprocessed, kept_ids, removed_ids = diagnostic.raw_native_preprocessing(
        recording
    )
    saved_folder = diagnostic.SCRATCH / f"joint_refit_{domain}"
    saved = diagnostic.save_recording(preprocessed, saved_folder)
    learning_folder = diagnostic.SCRATCH / f"joint_refit_learning_{domain}"
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
    refit_ops = ops.copy()
    refit_ops["settings"] = ops["settings"].copy()
    st, tF, refit_ops, peels, telemetry_rows, batch_summaries = (
        extract_joint_refit(refit_ops, bfile, U, device)
    )
    lineage = diagnostic.run_event_lineage(
        domain,
        THRESHOLD,
        refit_ops,
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
            "algorithm": "one_hop_local_joint_nnls_refit",
        },
    )
    for row in telemetry_rows:
        row["domain"] = domain
    for row in batch_summaries:
        row["domain"] = domain
    metadata = {
        "domain": domain,
        "input_channels": recording.get_num_channels(),
        "native_channels": len(kept_ids),
        "kept_channel_ids": kept_ids,
        "removed_channel_ids": removed_ids,
        "templates": int(U.shape[0]),
        "joint_refit_events": int(st.shape[0]),
        "candidate_detections": int(
            sum(row["candidate_detections"] for row in batch_summaries)
        ),
        "redetected_candidates": int(
            sum(row["redetected_candidates"] for row in batch_summaries)
        ),
        "refit_energy_reduction": float(
            sum(row["refit_energy_reduction"] for row in telemetry_rows)
        ),
    }
    del st, tF, saved
    shutil.rmtree(saved_folder)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lineage, telemetry_rows, batch_summaries, metadata


def main() -> None:
    import spikeinterface as si
    import run_score_diagnostic as diagnostic

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
        raise RuntimeError("joint-refit diagnostic requires a CUDA device")

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
    all_telemetry_rows = []
    all_batch_summaries = []
    domains = []
    recordings = {
        "raw_native_joint_refit": raw,
        "denoised_native_joint_refit": denoised,
    }
    selected_domains = {
        "both": DOMAIN_LABELS,
        "raw": ("raw_native_joint_refit",),
        "denoised": ("denoised_native_joint_refit",),
    }[args.domain]
    for domain in selected_domains:
        lineage, telemetry_rows, batch_summaries, metadata = process_domain(
            domain, recordings[domain], gt, device
        )
        outputs.append(lineage)
        all_telemetry_rows.extend(telemetry_rows)
        all_batch_summaries.extend(batch_summaries)
        domains.append(metadata)

    for key, filename in (
        ("lineage_stages", "joint_refit_stage_summary.csv"),
        ("lineage_units", "joint_refit_unit_summary.csv"),
        ("lineage_clusters", "joint_refit_cluster_summary.csv"),
        ("lineage_transitions", "joint_refit_transition_summary.csv"),
        (
            "lineage_cluster_transitions",
            "joint_refit_cluster_transition_summary.csv",
        ),
        ("lineage_stage_deltas", "joint_refit_stage_deltas.csv"),
        ("lineage_scores", "joint_refit_score_summary.csv"),
    ):
        pd.DataFrame([row for output in outputs for row in output[key]]).to_csv(
            diagnostic.RESULTS / filename, index=False
        )
    pd.DataFrame(all_telemetry_rows).to_csv(
        diagnostic.RESULTS / "joint_refit_by_batch_peel.csv", index=False
    )
    pd.DataFrame(all_batch_summaries).to_csv(
        diagnostic.RESULTS / "joint_refit_batch_summary.csv", index=False
    )
    manifest = {
        "algorithm": "one-hop local joint nonnegative amplitude refit",
        "uses_ground_truth_for_decisions": False,
        "Th_learned": THRESHOLD,
        "refit_timing": "after original greedy subtraction on every peel",
        "refit_scope": "each newly detected event plus all positive-amplitude events with center lag within [-nt, nt]",
        "solver": "warm-start active-set NNLS with machine-precision eigenspace rank cutoff and no ridge",
        "block_order": "new events in ascending sample order; overlapping blocks are sequential block-coordinate updates",
        "duplicate_policy": "identical time-template atoms share one amplitude variable and can be revived after reaching zero",
        "duration_s": args.duration_s,
        "requested_domain": args.domain,
        "registered_full_policy": {
            "duration_s": EXPECTED_DURATION_S,
            "domain": "both",
            "Th_learned": THRESHOLD,
            "refit_scope": "one-hop full ctc support",
        },
        "sampling_frequency": sampling_frequency,
        "kilosort_version": versions["kilosort"],
        "spikeinterface_version": versions["spikeinterface"],
        "domains": domains,
        "elapsed_s": time.perf_counter() - started,
    }
    path = diagnostic.RESULTS / "joint_refit_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, default=diagnostic.json_default) + "\n"
    )
    print(json.dumps(manifest, indent=2, default=diagnostic.json_default), flush=True)


if __name__ == "__main__":
    main()