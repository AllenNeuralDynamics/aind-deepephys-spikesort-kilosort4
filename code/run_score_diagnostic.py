"""Replay fixed learned templates on matched raw and denoised 20-minute data."""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import conv1d, max_pool1d

import spikeinterface as si
import spikeinterface.preprocessing as spre
from spikeinterface.sorters.external.kilosort4 import Kilosort4Sorter


DATA = Path("../data")
RESULTS = Path("../results")
SCRATCH = Path("../scratch")
PARAMS_PATH = Path(__file__).with_name("params.json")
RAW_MOUNT = "probec_recording1_3"
DENOISED_MOUNT = "full96_om1_probec_1200s"
EXPECTED_DURATION_S = 1200.0
TEMPLATE_LEARNING_THRESHOLD = 8.0
THRESHOLDS = (8.0, 9.0, 10.0, 10.75)
GT_SAMPLES_PER_UNIT = 256
BACKGROUND_SAMPLES = 4096
MATCH_TOLERANCE_SAMPLES = 6
GT_SCORE_WINDOW_SAMPLES = 6
SEED = 0
SCORE_EDGES = np.concatenate(
    [np.arange(0, 20.1, 0.1), np.array([25.0, 30.0, 40.0, np.inf])]
)
BAD_CHANNEL_KWARGS = {
    "method": "coherence+psd",
    "dead_channel_threshold": -0.5,
    "noisy_channel_threshold": 1.0,
    "outside_channel_threshold": -0.3,
    "outside_channels_location": "top",
    "n_neighbors": 11,
    "channel_filters": {"noise", "dead", "out"},
    "seed": 0,
}


def single_path(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(
            f"expected exactly one {description}, found {len(paths)}: {paths}"
        )
    return paths[0]


def discover_inputs() -> tuple[Path, Path, Path]:
    denoised_root = DATA / DENOISED_MOUNT
    raw_root = DATA / RAW_MOUNT
    denoised = single_path(
        sorted(path.parent for path in denoised_root.rglob("si_folder.json")),
        "denoised BinaryFolder recording",
    )
    raw = single_path(sorted(raw_root.rglob("recording.zarr")), "raw recording.zarr")
    gt = single_path(sorted(raw_root.rglob("sorting.zarr")), "GT sorting.zarr")
    return raw, denoised, gt


def phase_highpass(recording: si.BaseRecording) -> si.BaseRecording:
    processed = recording
    if "inter_sample_shift" in processed.get_property_keys():
        processed = spre.phase_shift(
            processed, margin_ms=100.0, dtype="float32"
        )
    return spre.highpass_filter(
        processed, freq_min=300.0, margin_ms=5.0, dtype="float32"
    )


def matched_preprocessing(
    raw: si.BaseRecording, denoised: si.BaseRecording
) -> tuple[si.BaseRecording, si.BaseRecording, list, list]:
    if not np.array_equal(raw.channel_ids, denoised.channel_ids):
        raise ValueError("raw and denoised channel IDs or ordering differ")
    raw_filtered = phase_highpass(raw)
    denoised_filtered = phase_highpass(denoised)
    denoised_clean = spre.detect_and_remove_bad_channels(
        denoised_filtered, **BAD_CHANNEL_KWARGS
    )
    kept_ids = denoised_clean.channel_ids.tolist()
    removed_ids = [item for item in denoised.channel_ids.tolist() if item not in kept_ids]
    raw_clean = raw_filtered.channel_slice(kept_ids)
    raw_preprocessed = spre.common_reference(
        raw_clean, reference="global", operator="median"
    )
    denoised_preprocessed = spre.common_reference(
        denoised_clean, reference="global", operator="median"
    )
    for domain, processed in (
        ("raw", raw_preprocessed),
        ("denoised", denoised_preprocessed),
    ):
        if np.dtype(processed.get_dtype()) != np.dtype("float32"):
            raise ValueError(f"{domain} preprocessing did not produce float32")
        if processed.get_num_samples(segment_index=0) != raw.get_num_samples(
            segment_index=0
        ):
            raise ValueError(f"{domain} preprocessing changed sample count")
    if raw_preprocessed.get_num_channels() != denoised_preprocessed.get_num_channels():
        raise ValueError("matched preprocessing produced different channel counts")
    return raw_preprocessed, denoised_preprocessed, kept_ids, removed_ids


def save_recording(recording: si.BaseRecording, folder: Path) -> si.BaseRecording:
    if folder.exists():
        shutil.rmtree(folder)
    return recording.save(folder=folder)


def binary_path(recording: si.BaseRecording) -> Path:
    if not recording.binary_compatible_with(time_axis=0, file_paths_length=1):
        raise ValueError("saved diagnostic recording is not binary compatible")
    description = recording.get_binary_description()
    paths = description["file_paths"]
    return Path(single_path([Path(path) for path in paths], "binary file"))


def sample_gt_and_background(
    gt_sorting: si.BaseSorting,
    n_samples: int,
    nt: int,
    scoreable_start: int,
    scoreable_stop: int,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    rng = np.random.default_rng(SEED)
    input_by_unit = {
        int(unit_id): np.asarray(
            gt_sorting.get_unit_spike_train(unit_id, segment_index=0), dtype=np.int64
        )
        for unit_id in gt_sorting.unit_ids
    }
    all_by_unit = {
        unit_id: frames[(frames >= scoreable_start) & (frames < scoreable_stop)]
        for unit_id, frames in input_by_unit.items()
    }
    excluded_by_unit = {
        unit_id: int(input_by_unit[unit_id].size - frames.size)
        for unit_id, frames in all_by_unit.items()
    }
    empty_units = [unit_id for unit_id, frames in all_by_unit.items() if not frames.size]
    if empty_units:
        raise RuntimeError(f"GT units have no scoreable events: {empty_units}")
    sampled = {}
    for unit_id, frames in all_by_unit.items():
        count = min(GT_SAMPLES_PER_UNIT, frames.size)
        indices = np.sort(rng.choice(frames.size, size=count, replace=False))
        sampled[unit_id] = frames[indices]
    all_frames = np.sort(np.concatenate(list(input_by_unit.values())))
    background = []
    candidate_start = max(nt, scoreable_start)
    candidate_stop = min(n_samples - nt, scoreable_stop)
    if candidate_start >= candidate_stop:
        raise RuntimeError("no scoreable interval available for background sampling")
    while len(background) < BACKGROUND_SAMPLES:
        candidates = rng.integers(
            candidate_start, candidate_stop, size=BACKGROUND_SAMPLES * 2
        )
        indices = np.searchsorted(all_frames, candidates)
        left = np.maximum(indices - 1, 0)
        right = np.minimum(indices, all_frames.size - 1)
        distance = np.minimum(
            np.abs(candidates - all_frames[left]),
            np.abs(candidates - all_frames[right]),
        )
        background.extend(candidates[distance > 2 * nt].tolist())
    return (
        all_by_unit,
        np.sort(np.asarray(background[:BACKGROUND_SAMPLES])),
        sampled,
        excluded_by_unit,
    )


def initial_projection(ops: dict, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    W = ops["wPCA"].contiguous()
    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=ops["nt"] // 2)
    return torch.einsum("ijk, kjl -> il", U, B)


def simulate_matching(
    ops: dict,
    B_initial: torch.Tensor,
    U: torch.Tensor,
    ctc: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    """Run the exact 4.1.7 matching-pursuit decisions without feature export."""
    nt = ops["nt"]
    W = ops["wPCA"].contiguous()
    nm = (U**2).sum(-1).sum(-1)
    B = B_initial.clone()
    trange = torch.arange(-nt, nt + 1, device=B.device)
    accepted_time = []
    accepted_template = []
    accepted_score = []
    accepted_amplitude = []
    accepted_peel = []
    peel_counts = []
    for peel in range(ops["max_peels"]):
        Cf = torch.relu(B) ** 2 / nm.unsqueeze(-1)
        Cf[:, :nt] = 0
        Cf[:, -nt:] = 0
        Cfmax, imax = torch.max(Cf, 0)
        Cmax = max_pool1d(
            Cfmax.unsqueeze(0).unsqueeze(0), 2 * nt + 1, stride=1, padding=nt
        )
        mask = (Cmax[0, 0] > threshold**2) & (
            torch.abs(Cmax[0, 0] - Cfmax) < 1e-9
        )
        iX = torch.nonzero(mask)[:, :1]
        peel_counts.append(int(iX.shape[0]))
        if iX.numel() == 0:
            break
        iY = imax[iX]
        amplitude = B[iY, iX] / nm[iY]
        score = Cmax[0, 0, iX[:, 0]].sqrt()
        accepted_time.append(iX[:, 0])
        accepted_template.append(iY[:, 0])
        accepted_score.append(score)
        accepted_amplitude.append(amplitude[:, 0])
        accepted_peel.append(
            torch.full((iX.shape[0],), peel, dtype=torch.int64, device=B.device)
        )
        for parity in range(2):
            B[:, iX[parity::2] + trange] -= amplitude[parity::2] * ctc[
                :, iY[parity::2, 0], :
            ]
    empty_long = torch.empty(0, dtype=torch.int64, device=B.device)
    empty_float = torch.empty(0, dtype=torch.float32, device=B.device)
    return {
        "time": torch.cat(accepted_time) if accepted_time else empty_long,
        "template": torch.cat(accepted_template) if accepted_template else empty_long,
        "score": torch.cat(accepted_score) if accepted_score else empty_float,
        "amplitude": (
            torch.cat(accepted_amplitude) if accepted_amplitude else empty_float
        ),
        "peel": torch.cat(accepted_peel) if accepted_peel else empty_long,
        "peel_counts": torch.as_tensor(peel_counts, dtype=torch.int64),
    }


def score_histogram(values: np.ndarray) -> np.ndarray:
    return np.histogram(values, bins=SCORE_EDGES)[0]


def top_scores(
    B: torch.Tensor,
    nm: torch.Tensor,
    local_indices: np.ndarray,
    half_window: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if local_indices.size == 0:
        empty = np.empty(0)
        return empty, empty, empty.astype(np.int64), empty
    index = torch.as_tensor(local_indices, dtype=torch.int64, device=B.device)
    offsets = torch.arange(-half_window, half_window + 1, device=B.device)
    window = index.unsqueeze(1) + offsets.unsqueeze(0)
    normalized = torch.relu(B[:, window]) / nm.sqrt().reshape(-1, 1, 1)
    normalized = normalized.max(2).values
    values, templates = torch.topk(normalized, k=2, dim=0)
    top1 = values[0].cpu().numpy()
    top2 = values[1].cpu().numpy()
    return top1, top2, templates[0].cpu().numpy(), top1 - top2


def match_events_one_to_one(
    events: np.ndarray, gt_frames: np.ndarray, tolerance: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic maximum-cardinality match for two sorted trains."""
    event_order = np.argsort(events, kind="stable")
    gt_order = np.argsort(gt_frames, kind="stable")
    sorted_events = events[event_order]
    sorted_gt = gt_frames[gt_order]
    event_matches = []
    gt_matches = []
    event_index = 0
    gt_index = 0
    while event_index < sorted_events.size and gt_index < sorted_gt.size:
        event = sorted_events[event_index]
        gt = sorted_gt[gt_index]
        if abs(event - gt) <= tolerance:
            event_matches.append(event_order[event_index])
            gt_matches.append(gt_order[gt_index])
            event_index += 1
            gt_index += 1
        elif event < gt - tolerance:
            event_index += 1
        else:
            gt_index += 1
    return np.asarray(event_matches, dtype=np.int64), np.asarray(
        gt_matches, dtype=np.int64
    )


def batch_local_indices(
    frames: np.ndarray, ibatch: int, ops: dict, batch_length: int
) -> tuple[np.ndarray, np.ndarray]:
    shift = ibatch * ops["batch_size"] * ops.get("batch_downsampling", 1)
    local = frames - shift + ops["nt"] + ops["nt"] // 2 - ops["nt0min"]
    valid = (local >= ops["nt"]) & (local < batch_length - ops["nt"])
    return frames[valid], local[valid].astype(np.int64)


def replay_frame_bounds(ops: dict, n_samples: int) -> tuple[int, int]:
    if ops.get("batch_downsampling", 1) != 1:
        raise ValueError("score diagnostic requires batch_downsampling=1")
    offset = ops["nt0min"] - ops["nt"] // 2
    start = max(0, offset)
    stop = min(n_samples, ops["Nbatches"] * ops["batch_size"] + offset)
    if start >= stop:
        raise RuntimeError(f"invalid scoreable replay interval [{start}, {stop})")
    return start, stop


def replay_domain(
    domain: str,
    binary_file: Path,
    ops: dict,
    U: torch.Tensor,
    gt_all: dict[int, np.ndarray],
    gt_sampled: dict[int, np.ndarray],
    background: np.ndarray,
    device: torch.device,
) -> dict[str, list[dict]]:
    from kilosort import io as ks_io
    from kilosort import template_matching

    bfile = ks_io.bfile_from_ops(
        ops=ops, filename=str(binary_file), device=device
    )
    ctc = template_matching.prepare_matching(ops, U)
    nm = (U**2).sum(-1).sum(-1)
    if torch.any(nm <= 0):
        raise ValueError("learned template with nonpositive norm")

    hist_counts = np.zeros(SCORE_EDGES.size - 1, dtype=np.int64)
    local_threshold_counts = {threshold: 0 for threshold in THRESHOLDS}
    template_win_counts = np.zeros(U.shape[0], dtype=np.int64)
    template_bg_count = np.zeros(U.shape[0], dtype=np.int64)
    template_bg_sum = np.zeros(U.shape[0], dtype=np.float64)
    template_bg_sumsq = np.zeros(U.shape[0], dtype=np.float64)
    template_bg_max = np.full(U.shape[0], -np.inf)
    template_bg_exceed = {
        threshold: np.zeros(U.shape[0], dtype=np.int64) for threshold in THRESHOLDS
    }
    gt_score_values = {unit_id: [] for unit_id in gt_all}
    gt_margin_values = {unit_id: [] for unit_id in gt_all}
    sample_rows = []
    threshold_events = {threshold: [] for threshold in THRESHOLDS}
    threshold_event_scores = {threshold: [] for threshold in THRESHOLDS}
    threshold_event_peels = {threshold: [] for threshold in THRESHOLDS}

    for ibatch in range(ops["Nbatches"]):
        if ibatch % 50 == 0:
            print(f"[{domain}] batch {ibatch}/{ops['Nbatches']}", flush=True)
        X = bfile.padded_batch_to_torch(ibatch, ops)
        B0 = initial_projection(ops, X, U)
        Cf = torch.relu(B0) ** 2 / nm.unsqueeze(-1)
        Cf[:, : ops["nt"]] = 0
        Cf[:, -ops["nt"] :] = 0
        Cfmax, imax = torch.max(Cf, 0)
        Cmax = max_pool1d(
            Cfmax.unsqueeze(0).unsqueeze(0),
            2 * ops["nt"] + 1,
            stride=1,
            padding=ops["nt"],
        )
        shift = ibatch * ops["batch_size"] * ops.get("batch_downsampling", 1)
        local_grid = torch.arange(B0.shape[1], device=device)
        global_grid = (
            local_grid
            - ops["nt"]
            + shift
            - ops["nt"] // 2
            + ops["nt0min"]
        )
        valid_time = (global_grid >= 0) & (global_grid < bfile.n_samples)
        local_mask = (
            (Cfmax > 0)
            & valid_time
            & (torch.abs(Cmax[0, 0] - Cfmax) < 1e-9)
        )
        local_scores = Cfmax[local_mask].sqrt()
        local_templates = imax[local_mask]
        scores_np = local_scores.cpu().numpy()
        hist_counts += score_histogram(scores_np)
        template_win_counts += np.bincount(
            local_templates.cpu().numpy(), minlength=U.shape[0]
        )
        for threshold in THRESHOLDS:
            local_threshold_counts[threshold] += int((local_scores > threshold).sum())

        bg_frames, bg_local = batch_local_indices(
            background, ibatch, ops, B0.shape[1]
        )
        if bg_local.size:
            index = torch.as_tensor(bg_local, dtype=torch.int64, device=device)
            signed = B0[:, index] / nm.sqrt().unsqueeze(-1)
            signed_np = signed.cpu().numpy()
            template_bg_count += signed_np.shape[1]
            template_bg_sum += signed_np.sum(1)
            template_bg_sumsq += (signed_np**2).sum(1)
            template_bg_max = np.maximum(template_bg_max, signed_np.max(1))
            for threshold in THRESHOLDS:
                template_bg_exceed[threshold] += (signed_np > threshold).sum(1)
            top1, top2, winner, margin = top_scores(B0, nm, bg_local)
            for frame, score1, score2, template_id, gap in zip(
                bg_frames, top1, top2, winner, margin
            ):
                sample_rows.append(
                    {
                        "domain": domain,
                        "sample_type": "background",
                        "gt_unit_id": np.nan,
                        "frame": int(frame),
                        "top1_score": score1,
                        "top2_score": score2,
                        "winner_template": int(template_id),
                        "winner_margin": gap,
                    }
                )

        for unit_id, frames in gt_all.items():
            selected, local = batch_local_indices(frames, ibatch, ops, B0.shape[1])
            if local.size:
                top1, _, _, margin = top_scores(
                    B0, nm, local, half_window=GT_SCORE_WINDOW_SAMPLES
                )
                gt_score_values[unit_id].append(top1)
                gt_margin_values[unit_id].append(margin)
        for unit_id, frames in gt_sampled.items():
            selected, local = batch_local_indices(frames, ibatch, ops, B0.shape[1])
            top1, top2, winner, margin = top_scores(
                B0, nm, local, half_window=GT_SCORE_WINDOW_SAMPLES
            )
            for frame, score1, score2, template_id, gap in zip(
                selected, top1, top2, winner, margin
            ):
                sample_rows.append(
                    {
                        "domain": domain,
                        "sample_type": "gt",
                        "gt_unit_id": unit_id,
                        "frame": int(frame),
                        "top1_score": score1,
                        "top2_score": score2,
                        "winner_template": int(template_id),
                        "winner_margin": gap,
                    }
                )

        for threshold in THRESHOLDS:
            matched = simulate_matching(ops, B0, U, ctc, threshold)
            accepted_local = matched["time"].cpu().numpy()
            accepted_global = (
                accepted_local
                - ops["nt"]
                + shift
                - ops["nt"] // 2
                + ops["nt0min"]
            )
            accepted_score = matched["score"].cpu().numpy()
            accepted_peel = matched["peel"].cpu().numpy()
            valid = (accepted_global >= 0) & (accepted_global < bfile.n_samples)
            threshold_events[threshold].append(accepted_global[valid])
            threshold_event_scores[threshold].append(accepted_score[valid])
            threshold_event_peels[threshold].append(accepted_peel[valid])

        del B0, Cf, Cfmax, Cmax, local_scores, local_templates, X
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    hist_rows = []
    for index, count in enumerate(hist_counts):
        hist_rows.append(
            {
                "domain": domain,
                "score_bin_left": SCORE_EDGES[index],
                "score_bin_right": SCORE_EDGES[index + 1],
                "local_max_count": int(count),
            }
        )
    template_rows = []
    template_norms = nm.sqrt().cpu().numpy()
    if not np.all(template_bg_count == BACKGROUND_SAMPLES):
        raise RuntimeError(
            f"background score coverage differs by template: {template_bg_count}"
        )
    for template_id in range(U.shape[0]):
        count = template_bg_count[template_id]
        mean = template_bg_sum[template_id] / count
        variance = template_bg_sumsq[template_id] / count - mean**2
        row = {
            "domain": domain,
            "template_id": template_id,
            "template_norm": template_norms[template_id],
            "background_count": int(count),
            "signed_score_mean": mean,
            "signed_score_std": np.sqrt(max(variance, 0)),
            "signed_score_max": template_bg_max[template_id],
            "first_peel_win_count": int(template_win_counts[template_id]),
        }
        for threshold in THRESHOLDS:
            row[f"background_count_gt_{threshold:g}"] = int(
                template_bg_exceed[threshold][template_id]
            )
        template_rows.append(row)
    unit_order = list(gt_all)
    gt_frames_flat = np.concatenate([gt_all[unit_id] for unit_id in unit_order])
    gt_units_flat = np.concatenate(
        [np.full(gt_all[unit_id].size, unit_id, dtype=np.int64) for unit_id in unit_order]
    )
    gt_scores_flat = np.concatenate(
        [np.concatenate(gt_score_values[unit_id]) for unit_id in unit_order]
    )
    gt_margins_flat = np.concatenate(
        [np.concatenate(gt_margin_values[unit_id]) for unit_id in unit_order]
    )
    if not (
        gt_frames_flat.size == gt_scores_flat.size == gt_margins_flat.size
    ):
        raise RuntimeError("GT score replay did not cover every GT event exactly once")
    threshold_matches = {}
    threshold_peel_counts = {}
    for threshold in THRESHOLDS:
        events = np.concatenate(threshold_events[threshold])
        scores = np.concatenate(threshold_event_scores[threshold])
        peels = np.concatenate(threshold_event_peels[threshold])
        event_match, gt_match = match_events_one_to_one(
            events, gt_frames_flat, MATCH_TOLERANCE_SAMPLES
        )
        recovered = np.zeros(gt_frames_flat.size, dtype=bool)
        recovered[gt_match] = True
        recovered_peel = np.full(gt_frames_flat.size, -1, dtype=np.int64)
        recovered_peel[gt_match] = peels[event_match]
        threshold_matches[threshold] = {
            "events": events,
            "scores": scores,
            "peels": peels,
            "recovered": recovered,
            "recovered_peel": recovered_peel,
        }
        threshold_peel_counts[threshold] = np.bincount(
            peels, minlength=ops["max_peels"]
        )

    gt_rows = []
    for unit_id in unit_order:
        unit_mask = gt_units_flat == unit_id
        scores = gt_scores_flat[unit_mask]
        margins = gt_margins_flat[unit_mask]
        row = {
            "domain": domain,
            "gt_unit_id": unit_id,
            "gt_events": int(scores.size),
            "first_peel_score_mean": scores.mean(),
            "first_peel_score_std": scores.std(),
            "first_peel_score_median": np.median(scores),
            "first_peel_score_q10": np.quantile(scores, 0.1),
            "first_peel_score_q90": np.quantile(scores, 0.9),
            "winner_margin_mean": margins.mean(),
        }
        for threshold in THRESHOLDS:
            recovered = threshold_matches[threshold]["recovered"][unit_mask]
            recovered_peel = threshold_matches[threshold]["recovered_peel"][unit_mask]
            above = scores > threshold
            row[f"first_peel_count_gt_{threshold:g}"] = int(
                above.sum()
            )
            row[f"matching_pursuit_recovered_{threshold:g}"] = int(
                recovered.sum()
            )
            row[f"above_threshold_not_recovered_{threshold:g}"] = int(
                (above & ~recovered).sum()
            )
            row[f"below_threshold_but_recovered_{threshold:g}"] = int(
                (~above & recovered).sum()
            )
            row[f"recovered_on_later_peel_{threshold:g}"] = int(
                (recovered_peel > 0).sum()
            )
        gt_rows.append(row)
    threshold_rows = []
    peel_rows = []
    for threshold in THRESHOLDS:
        result = threshold_matches[threshold]
        count = result["events"].size
        recovered = int(result["recovered"].sum())
        total_gt = gt_frames_flat.size
        threshold_rows.append(
            {
                "domain": domain,
                "Th_learned": threshold,
                "accepted_events": int(count),
                "accepted_events_per_s": count / EXPECTED_DURATION_S,
                "accepted_score_mean": (
                    result["scores"].mean() if count else np.nan
                ),
                "gt_events_recovered_one_to_one_any_template": recovered,
                "gt_events_total": int(total_gt),
                "gt_event_recovery_fraction": recovered / total_gt,
                "first_peel_local_max_count_gt_threshold": int(
                    local_threshold_counts[threshold]
                ),
            }
        )
        for peel, peel_count in enumerate(threshold_peel_counts[threshold]):
            if peel_count == 0 and peel > 0:
                break
            peel_rows.append(
                {
                    "domain": domain,
                    "Th_learned": threshold,
                    "peel": peel,
                    "accepted_events": int(peel_count),
                }
            )
    return {
        "histogram": hist_rows,
        "templates": template_rows,
        "gt_units": gt_rows,
        "thresholds": threshold_rows,
        "peels": peel_rows,
        "samples": sample_rows,
    }


def learn_denoised_templates(
    recording: si.BaseRecording, output_folder: Path
) -> tuple[Path, np.ndarray]:
    from kilosort import template_matching

    sorter_params = json.loads(PARAMS_PATH.read_text())["sorter"]
    sorter_params = copy.deepcopy(sorter_params)
    sorter_params["Th_learned"] = TEMPLATE_LEARNING_THRESHOLD
    sorter_params["save_extra_vars"] = False
    captured = {}
    original_extract = template_matching.extract

    def capture_extract(ops, bfile, U, device=torch.device("cuda"), progress_bar=None):
        if "U" in captured:
            raise RuntimeError("learned-template extraction called more than once")
        captured["U"] = U.detach().cpu().numpy()
        return original_extract(
            ops, bfile, U, device=device, progress_bar=progress_bar
        )

    template_matching.extract = capture_extract
    try:
        Kilosort4Sorter.initialize_folder(
            recording, output_folder, verbose=True, remove_existing_folder=True
        )
        Kilosort4Sorter.set_params_to_folder(
            recording, output_folder, sorter_params, verbose=True
        )
        Kilosort4Sorter.setup_recording(recording, output_folder, verbose=True)
        Kilosort4Sorter.run_from_folder(
            output_folder, raise_error=True, verbose=True
        )
    finally:
        template_matching.extract = original_extract
    if "U" not in captured:
        raise RuntimeError("failed to capture learned templates")
    return output_folder / "sorter_output" / "ops.npy", captured["U"]


def main() -> None:
    from importlib.metadata import version
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
        raise RuntimeError("score diagnostic requires a CUDA device")
    RESULTS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    raw_path, denoised_path, gt_path = discover_inputs()
    raw = si.read_zarr(raw_path)
    denoised = si.load(denoised_path)
    if raw.get_num_segments() != 1 or denoised.get_num_segments() != 1:
        raise ValueError("diagnostic inputs must each have one segment")
    sampling_frequency = raw.get_sampling_frequency()
    if not np.isclose(sampling_frequency, denoised.get_sampling_frequency()):
        raise ValueError("raw and denoised sampling frequencies differ")
    end_frame = int(round(EXPECTED_DURATION_S * sampling_frequency))
    raw = raw.frame_slice(0, end_frame)
    if denoised.get_num_samples(segment_index=0) != end_frame:
        raise ValueError("denoised recording is not exactly 1200 seconds")
    gt = si.read_zarr(gt_path).frame_slice(0, end_frame)
    raw_preprocessed, denoised_preprocessed, kept_ids, removed_ids = (
        matched_preprocessing(raw, denoised)
    )
    si.set_global_job_kwargs(
        n_jobs=int(os.getenv("CO_CPUS", "-1")),
        chunk_duration="1s",
        progress_bar=False,
        mp_context="spawn",
    )
    denoised_folder = SCRATCH / "diagnostic_denoised"
    denoised_saved = save_recording(
        denoised_preprocessed, denoised_folder
    )
    template_learning_folder = SCRATCH / "diagnostic_template_learning"
    ops_path, templates = learn_denoised_templates(
        denoised_saved, template_learning_folder
    )
    device = torch.device("cuda")
    ops = ks_io.load_ops(ops_path, device=device)
    U = torch.from_numpy(templates).to(device)
    shutil.copy2(ops_path, RESULTS / "fixed_denoised_ops.npy")
    shutil.rmtree(template_learning_folder)
    scoreable_start, scoreable_stop = replay_frame_bounds(ops, end_frame)
    gt_all, background, gt_sampled, gt_excluded_by_unit = sample_gt_and_background(
        gt, end_frame, ops["nt"], scoreable_start, scoreable_stop
    )
    denoised_output = replay_domain(
        "denoised",
        binary_path(denoised_saved),
        ops,
        U,
        gt_all,
        gt_sampled,
        background,
        device,
    )
    del denoised_saved
    shutil.rmtree(denoised_folder)

    raw_folder = SCRATCH / "diagnostic_raw"
    raw_saved = save_recording(raw_preprocessed, raw_folder)
    raw_output = replay_domain(
        "raw",
        binary_path(raw_saved),
        ops,
        U,
        gt_all,
        gt_sampled,
        background,
        device,
    )
    del raw_saved
    shutil.rmtree(raw_folder)
    outputs = [raw_output, denoised_output]

    for key, filename in (
        ("histogram", "first_peel_local_max_histogram.csv"),
        ("templates", "template_background_scores.csv"),
        ("gt_units", "gt_unit_score_summary.csv"),
        ("thresholds", "threshold_replay_summary.csv"),
        ("peels", "peel_summary.csv"),
        ("samples", "score_samples.csv"),
    ):
        pd.DataFrame([row for output in outputs for row in output[key]]).to_csv(
            RESULTS / filename, index=False
        )
    np.save(RESULTS / "fixed_denoised_learned_templates.npy", templates)
    np.save(RESULTS / "fixed_denoised_whitening_matrix.npy", ops["Wrot"].cpu())
    manifest = {
        "diagnostic": "fixed denoised templates and preprocessing replayed on raw and denoised",
        "kilosort_version": versions["kilosort"],
        "spikeinterface_version": versions["spikeinterface"],
        "template_source": "denoised",
        "template_learning_Th_learned": TEMPLATE_LEARNING_THRESHOLD,
        "replay_thresholds": THRESHOLDS,
        "duration_s": EXPECTED_DURATION_S,
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
        "gt_detailed_samples_per_unit": GT_SAMPLES_PER_UNIT,
        "background_samples": BACKGROUND_SAMPLES,
        "background_definition": "random times more than 2*nt from injected GT; native spikes are not excluded",
        "match_tolerance_samples": MATCH_TOLERANCE_SAMPLES,
        "gt_score_window_samples": GT_SCORE_WINDOW_SAMPLES,
        "gt_recovery_definition": "global one-to-one temporal match to any accepted learned-template event",
        "seed": SEED,
        "input_channels": raw.get_num_channels(),
        "matched_channels": len(kept_ids),
        "kept_channel_ids": kept_ids,
        "removed_channel_ids": removed_ids,
        "fixed_transform": "denoised-derived channel mask, Kilosort whitening, drift, and learned templates",
        "first_peel_scores": "relu(<X,T>)/||T|| before threshold and subtraction",
        "elapsed_s": time.perf_counter() - started,
    }
    (RESULTS / "score_diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=list) + "\n"
    )
    print(json.dumps(manifest, indent=2, default=list), flush=True)


if __name__ == "__main__":
    main()