"""Test a GT-blind target-decoy FDR gate during Kilosort matching pursuit."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import max_pool1d

import spikeinterface as si

import run_score_diagnostic as diagnostic


TARGET_FDR = 0.05
MIN_THRESHOLD = 8.0
DOMAIN_LABELS = ("raw_native_fdr", "denoised_native_fdr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=diagnostic.EXPECTED_DURATION_S,
        help="Recording duration; the registered full experiment uses 1200 seconds.",
    )
    parser.add_argument(
        "--domain",
        choices=("both", "raw", "denoised"),
        default="both",
        help="Domain subset; the registered full experiment uses both.",
    )
    args = parser.parse_args()
    if args.duration_s <= 0 or args.duration_s > diagnostic.EXPECTED_DURATION_S:
        parser.error(
            f"--duration-s must be in (0, {diagnostic.EXPECTED_DURATION_S:g}]"
        )
    return args


def local_peak_scores(
    B: torch.Tensor,
    nm: torch.Tensor,
    nt: int,
    sign: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return template-winning temporal local maxima for one score sign."""
    field = torch.relu(sign * B) ** 2 / nm.unsqueeze(-1)
    field[:, :nt] = 0
    field[:, -nt:] = 0
    best, templates = torch.max(field, 0)
    pooled = max_pool1d(
        best.unsqueeze(0).unsqueeze(0),
        2 * nt + 1,
        stride=1,
        padding=nt,
    )[0, 0]
    mask = (best > 0) & (torch.abs(pooled - best) < 1e-9)
    times = torch.nonzero(mask)[:, 0]
    return best[mask].sqrt(), templates[mask], times


def target_decoy_threshold(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    floor: float,
    target_fdr: float,
) -> tuple[float, float, int, int, int, int]:
    """Choose the minimum knockoff-plus threshold satisfying target FDR."""
    if not 0 < target_fdr < 1:
        raise ValueError(f"target FDR must be in (0, 1): {target_fdr}")
    positive = positive_scores[positive_scores > floor]
    negative = negative_scores[negative_scores > floor]
    positive_count = int(positive.numel())
    negative_count = int(negative.numel())
    if positive_count == 0:
        return np.inf, np.inf, positive_count, negative_count, 0, 0

    # Kilosort's original rule is strict (`score > Th_learned`), so the floor
    # candidate excludes scores equal to 8 even though elevated candidates use
    # the usual target-decoy `>= candidate` counting rule.
    positive = positive.sort().values
    negative = negative.sort().values
    candidates = torch.unique(
        torch.cat(
            (
                torch.as_tensor(
                    [floor], dtype=positive.dtype, device=positive.device
                ),
                positive,
            )
        )
    ).sort().values
    targets = positive_count - torch.searchsorted(positive, candidates, right=False)
    decoys = negative_count - torch.searchsorted(negative, candidates, right=False)
    estimated = (1 + decoys) / targets.clamp_min(1)
    valid = (targets > 0) & (estimated <= target_fdr)
    accepted_indices = torch.nonzero(valid)
    if accepted_indices.numel() == 0:
        return np.inf, np.inf, positive_count, negative_count, 0, 0
    index = accepted_indices[0, 0]
    return (
        float(candidates[index]),
        float(estimated[index]),
        positive_count,
        negative_count,
        int(targets[index]),
        int(decoys[index]),
    )


def run_target_decoy_matching(
    ops: dict,
    X: torch.Tensor,
    U: torch.Tensor,
    ctc: torch.Tensor,
    target_fdr: float,
    minimum_threshold: float,
) -> dict[str, torch.Tensor | list[dict]]:
    """Run Kilosort matching pursuit with a sign-decoy FDR acceptance gate."""
    nt = ops["nt"]
    W = ops["wPCA"].contiguous()
    nm = (U**2).sum(-1).sum(-1)
    B = diagnostic.initial_projection(ops, X, U)
    Xres = X.clone()
    trange = torch.arange(-nt, nt + 1, device=B.device)
    tiwave = torch.arange(-(nt // 2), nt // 2 + 1, device=B.device)
    accepted_time = []
    accepted_template = []
    accepted_score = []
    accepted_amplitude = []
    accepted_peel = []
    gate_rows = []

    for peel in range(ops["max_peels"]):
        positive_scores, positive_templates, positive_times = local_peak_scores(
            B, nm, nt, sign=1
        )
        negative_scores, _, _ = local_peak_scores(B, nm, nt, sign=-1)
        (
            threshold,
            estimated_fdr,
            positive_floor_count,
            negative_floor_count,
            selected_target_count,
            selected_decoy_count,
        ) = target_decoy_threshold(
            positive_scores,
            negative_scores,
            minimum_threshold,
            target_fdr,
        )
        if np.isfinite(threshold):
            if threshold == minimum_threshold:
                accepted = positive_scores > minimum_threshold
            else:
                accepted = positive_scores >= threshold
        else:
            accepted = torch.zeros_like(positive_scores, dtype=torch.bool)
        scores = positive_scores[accepted]
        times = positive_times[accepted]
        templates = positive_templates[accepted]
        gate_rows.append(
            {
                "peel": peel,
                "minimum_threshold": minimum_threshold,
                "selected_threshold": threshold,
                "target_fdr": target_fdr,
                "estimated_fdr": estimated_fdr,
                "positive_local_maxima": int(positive_scores.numel()),
                "negative_local_maxima": int(negative_scores.numel()),
                "positive_count_above_floor": positive_floor_count,
                "negative_count_above_floor": negative_floor_count,
                "negative_to_positive_floor_ratio": (
                    negative_floor_count / positive_floor_count
                    if positive_floor_count
                    else np.nan
                ),
                "selected_target_count": selected_target_count,
                "selected_decoy_count": selected_decoy_count,
                "accepted_events": int(scores.numel()),
                "positive_score_max": (
                    float(positive_scores.max()) if positive_scores.numel() else np.nan
                ),
                "negative_score_max": (
                    float(negative_scores.max()) if negative_scores.numel() else np.nan
                ),
            }
        )
        if scores.numel() == 0:
            break

        iX = times.unsqueeze(1)
        iY = templates.unsqueeze(1)
        amplitude = B[iY, iX] / nm[iY]
        accepted_time.append(times)
        accepted_template.append(templates)
        accepted_score.append(scores)
        accepted_amplitude.append(amplitude[:, 0])
        accepted_peel.append(
            torch.full(
                (times.numel(),), peel, dtype=torch.int64, device=B.device
            )
        )
        for parity in range(2):
            Xres[:, iX[parity::2] + tiwave] -= amplitude[parity::2] * torch.einsum(
                "ijk, jl -> kil", U[iY[parity::2, 0]], W
            )
            B[:, iX[parity::2] + trange] -= amplitude[parity::2] * ctc[
                :, iY[parity::2, 0], :
            ]

    empty_long = torch.empty(0, dtype=torch.int64, device=B.device)
    empty_float = torch.empty(0, dtype=torch.float32, device=B.device)
    return {
        "time": torch.cat(accepted_time) if accepted_time else empty_long,
        "template": (
            torch.cat(accepted_template) if accepted_template else empty_long
        ),
        "score": torch.cat(accepted_score) if accepted_score else empty_float,
        "amplitude": (
            torch.cat(accepted_amplitude) if accepted_amplitude else empty_float
        ),
        "peel": torch.cat(accepted_peel) if accepted_peel else empty_long,
        "Xres": Xres,
        "gate_rows": gate_rows,
    }


def extract_target_decoy(
    ops: dict,
    bfile,
    U: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, dict, np.ndarray, list[dict]]:
    """Run one target-decoy extraction while preserving event/peel identity."""
    from kilosort import template_matching

    captured_rows = []
    captured_peels = []
    gate_rows = []
    original_run_matching = template_matching.run_matching
    batch_index = 0

    def adaptive_matching(local_ops, X, local_U, ctc, device=device):
        nonlocal batch_index
        matched = run_target_decoy_matching(
            local_ops,
            X,
            local_U,
            ctc,
            target_fdr=TARGET_FDR,
            minimum_threshold=MIN_THRESHOLD,
        )
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
        for row in matched["gate_rows"]:
            gate_rows.append({"batch": batch_index, **row})
        batch_index += 1
        return (
            local_st,
            matched["amplitude"].unsqueeze(1),
            matched["score"].unsqueeze(1),
            matched["Xres"],
        )

    template_matching.run_matching = adaptive_matching
    try:
        st, tF, ops = template_matching.extract(ops, bfile, U, device=device)
    finally:
        template_matching.run_matching = original_run_matching
    if batch_index != int(bfile.n_batches):
        raise RuntimeError(
            f"target-decoy capture covered {batch_index}/{bfile.n_batches} batches"
        )
    captured_st = np.concatenate(captured_rows)
    peels = np.concatenate(captured_peels)
    order = np.argsort(captured_st[:, 0])
    captured_st = captured_st[order]
    peels = peels[order]
    if not np.array_equal(
        captured_st[:, :2].astype(np.int64), st[:, :2].astype(np.int64)
    ):
        raise RuntimeError("target-decoy capture changed extraction event identity")
    if not np.allclose(captured_st[:, 2], st[:, 2], rtol=0, atol=0):
        raise RuntimeError("target-decoy capture changed extraction scores")
    return st, tF, ops, peels, gate_rows


def process_domain(
    domain: str,
    recording: si.BaseRecording,
    gt: si.BaseSorting,
    device: torch.device,
) -> tuple[dict, list[dict], dict]:
    """Learn a native transform, run the gate, and evaluate downstream lineage."""
    from kilosort import io as ks_io

    preprocessed, kept_ids, removed_ids = diagnostic.raw_native_preprocessing(
        recording
    )
    saved_folder = diagnostic.SCRATCH / f"target_decoy_{domain}"
    saved = diagnostic.save_recording(preprocessed, saved_folder)
    learning_folder = diagnostic.SCRATCH / f"target_decoy_learning_{domain}"
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
    adaptive_ops = ops.copy()
    adaptive_ops["settings"] = ops["settings"].copy()
    st, tF, adaptive_ops, peels, gate_rows = extract_target_decoy(
        adaptive_ops, bfile, U, device
    )
    lineage = diagnostic.run_event_lineage(
        domain,
        MIN_THRESHOLD,
        adaptive_ops,
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
        MIN_THRESHOLD,
        lineage,
        {
            "domain": domain,
            "Th_learned": MIN_THRESHOLD,
            "algorithm": "target_decoy_fdr",
            "target_fdr": TARGET_FDR,
        },
    )
    for row in gate_rows:
        row["domain"] = domain
    metadata = {
        "domain": domain,
        "input_channels": recording.get_num_channels(),
        "native_channels": len(kept_ids),
        "kept_channel_ids": kept_ids,
        "removed_channel_ids": removed_ids,
        "templates": int(U.shape[0]),
        "adaptive_events": int(st.shape[0]),
    }
    del st, tF, saved
    shutil.rmtree(saved_folder)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lineage, gate_rows, metadata


def main() -> None:
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
        raise RuntimeError("target-decoy diagnostic requires a CUDA device")

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
    all_gate_rows = []
    domains = []
    recordings = {
        "raw_native_fdr": raw,
        "denoised_native_fdr": denoised,
    }
    selected_domains = {
        "both": DOMAIN_LABELS,
        "raw": ("raw_native_fdr",),
        "denoised": ("denoised_native_fdr",),
    }[args.domain]
    for domain in selected_domains:
        recording = recordings[domain]
        lineage, gate_rows, metadata = process_domain(domain, recording, gt, device)
        outputs.append(lineage)
        all_gate_rows.extend(gate_rows)
        domains.append(metadata)

    for key, filename in (
        ("lineage_stages", "target_decoy_stage_summary.csv"),
        ("lineage_units", "target_decoy_unit_summary.csv"),
        ("lineage_clusters", "target_decoy_cluster_summary.csv"),
        ("lineage_transitions", "target_decoy_transition_summary.csv"),
        (
            "lineage_cluster_transitions",
            "target_decoy_cluster_transition_summary.csv",
        ),
        ("lineage_stage_deltas", "target_decoy_stage_deltas.csv"),
        ("lineage_scores", "target_decoy_score_summary.csv"),
    ):
        pd.DataFrame([row for output in outputs for row in output[key]]).to_csv(
            diagnostic.RESULTS / filename, index=False
        )
    pd.DataFrame(all_gate_rows).to_csv(
        diagnostic.RESULTS / "target_decoy_gate_by_batch_peel.csv", index=False
    )
    manifest = {
        "algorithm": "sign-reversed target-decoy knockoff-plus gate",
        "uses_ground_truth_for_decisions": False,
        "target_fdr": TARGET_FDR,
        "minimum_Th_learned": MIN_THRESHOLD,
        "threshold_scope": "independent per batch and peel",
        "decoy": "negative signed learned-template local maxima with identical competition and temporal suppression",
        "duration_s": args.duration_s,
        "requested_domain": args.domain,
        "registered_full_policy": {
            "duration_s": diagnostic.EXPECTED_DURATION_S,
            "domain": "both",
            "target_fdr": TARGET_FDR,
            "minimum_Th_learned": MIN_THRESHOLD,
        },
        "sampling_frequency": sampling_frequency,
        "kilosort_version": versions["kilosort"],
        "spikeinterface_version": versions["spikeinterface"],
        "domains": domains,
        "elapsed_s": time.perf_counter() - started,
    }
    path = diagnostic.RESULTS / "target_decoy_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, default=diagnostic.json_default) + "\n"
    )
    print(json.dumps(manifest, indent=2, default=diagnostic.json_default), flush=True)


if __name__ == "__main__":
    main()
