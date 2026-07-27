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
LINEAGE_THRESHOLDS = (8.0, 10.75)
LINEAGE_DELTA_TIME_MS = 0.4
LINEAGE_MATCH_SCORE = 0.2
GT_SAMPLES_PER_UNIT = 256
BACKGROUND_SAMPLES = 4096
MATCH_TOLERANCE_SAMPLES = 6
GT_SCORE_WINDOW_SAMPLES = 6
SEED = 0
SCORE_EDGES = np.concatenate(
    [np.arange(0, 20.1, 0.1), np.array([25.0, 30.0, 40.0, np.inf])]
)
STATUS_NAMES = {
    0: "unmatched_cluster",
    1: "fp_matched_cluster",
    2: "tp",
    3: "removed_by_dedup",
}
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
    raw_clean = raw_filtered.select_channels(kept_ids)
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


def is_matched_unit(unit_id) -> bool:
    return unit_id != -1 and unit_id != ""


def canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters by first occurrence for partition comparison."""
    canonical = np.empty(labels.size, dtype=np.int64)
    mapping = {}
    for index, label in enumerate(labels):
        if int(label) not in mapping:
            mapping[int(label)] = len(mapping)
        canonical[index] = mapping[int(label)]
    return canonical


def evaluate_lineage_stage(
    domain: str,
    threshold: float,
    stage: str,
    times: np.ndarray,
    labels: np.ndarray,
    event_ids: np.ndarray,
    total_events: int,
    gt_sorting: si.BaseSorting,
    sampling_frequency: float,
    all_gt_frames: np.ndarray,
    acg_threshold: float,
    ccg_threshold: float,
    detection_templates: np.ndarray,
) -> dict:
    """Apply benchmark unit matching and assign every event a stage status."""
    import spikeinterface.comparison as sc
    from kilosort import CCG

    if not (times.size == labels.size == event_ids.size):
        raise ValueError(f"{stage} lineage arrays differ in length")
    if np.unique(event_ids).size != event_ids.size:
        raise ValueError(f"{stage} lineage event IDs are not unique")
    tested_sorting = si.NumpySorting.from_samples_and_labels(
        times.astype(np.int64),
        labels.astype(np.int32),
        sampling_frequency=sampling_frequency,
    )
    comparison = sc.compare_sorter_to_ground_truth(
        gt_sorting,
        tested_sorting,
        exhaustive_gt=False,
        delta_time=LINEAGE_DELTA_TIME_MS,
        match_score=LINEAGE_MATCH_SCORE,
        compute_labels=True,
    )
    expected_delta_frames = int(
        LINEAGE_DELTA_TIME_MS / 1000 * sampling_frequency
    )
    if comparison.delta_frames != expected_delta_frames:
        raise RuntimeError(
            "SpikeInterface comparison tolerance differs from benchmark "
            f"expectation: {comparison.delta_frames} != {expected_delta_frames}"
        )
    counts = comparison.get_performance(method="raw_count", output="pandas")
    performance = comparison.get_performance(method="by_unit", output="pandas")

    status = np.full(total_events, 3, dtype=np.int8)
    status[event_ids] = 0
    assigned_gt = np.full(total_events, -1, dtype=np.int64)
    reverse_match = comparison.hungarian_match_21
    sorted_event_indices = np.argsort(times)
    for cluster_id in tested_sorting.unit_ids:
        cluster_indices = sorted_event_indices[labels[sorted_event_indices] == cluster_id]
        ordered_event_ids = event_ids[cluster_indices]
        gt_unit_id = reverse_match[cluster_id]
        if not is_matched_unit(gt_unit_id):
            continue
        assigned_gt[ordered_event_ids] = int(gt_unit_id)
        event_labels = comparison.get_labels2(cluster_id)[0]
        if event_labels.size != ordered_event_ids.size:
            raise RuntimeError(f"{stage} SpikeInterface labels lost event identity")
        status[ordered_event_ids] = 1
        status[ordered_event_ids[event_labels == "TP"]] = 2

    spike_vector = tested_sorting.to_spike_vector()
    is_refractory, contamination = CCG.refract(
        spike_vector["unit_index"],
        spike_vector["sample_index"] / sampling_frequency,
        acg_threshold=acg_threshold,
        ccg_threshold=ccg_threshold,
    )
    cluster_to_index = {
        int(cluster_id): index
        for index, cluster_id in enumerate(tested_sorting.unit_ids)
    }
    cluster_rows = []
    for cluster_id in tested_sorting.unit_ids:
        cluster_mask = labels == cluster_id
        cluster_event_ids = event_ids[cluster_mask]
        cluster_times = times[cluster_mask]
        cluster_index = cluster_to_index[int(cluster_id)]
        assigned = np.unique(assigned_gt[cluster_event_ids])
        if assigned.size != 1:
            raise RuntimeError(f"{stage} cluster has inconsistent GT assignment")
        source_templates, source_counts = np.unique(
            detection_templates[cluster_event_ids], return_counts=True
        )
        dominant_index = int(np.argmax(source_counts))
        proximal, _ = match_events_one_to_one(
            cluster_times.astype(np.int64), all_gt_frames, comparison.delta_frames
        )
        cluster_rows.append(
            {
                "domain": domain,
                "Th_learned": threshold,
                "stage": stage,
                "cluster_id": int(cluster_id),
                "assigned_gt_unit_id": int(assigned[0]),
                "events": int(cluster_event_ids.size),
                "detection_templates": int(source_templates.size),
                "dominant_detection_template": int(
                    source_templates[dominant_index]
                ),
                "dominant_detection_template_fraction": float(
                    source_counts[dominant_index] / cluster_event_ids.size
                ),
                "tp": int((status[cluster_event_ids] == 2).sum()),
                "fp_if_matched": int((status[cluster_event_ids] == 1).sum()),
                "gt_proximal_events_any_unit": int(proximal.size),
                "gt_negative_events_any_unit": int(
                    cluster_event_ids.size - proximal.size
                ),
                "is_refractory": int(is_refractory[cluster_index]),
                "contamination": float(contamination[cluster_index]),
            }
        )
    unit_rows = []
    for gt_unit_id in gt_sorting.unit_ids:
        cluster_id = counts.at[gt_unit_id, "tested_id"]
        matched = is_matched_unit(cluster_id)
        cluster_index = cluster_to_index[int(cluster_id)] if matched else None
        agreement = (
            comparison.agreement_scores.at[gt_unit_id, cluster_id]
            if matched
            else 0.0
        )
        unit_rows.append(
            {
                "domain": domain,
                "Th_learned": threshold,
                "stage": stage,
                "gt_unit_id": int(gt_unit_id),
                "cluster_id": int(cluster_id) if matched else -1,
                "agreement": float(agreement),
                "tp": int(counts.at[gt_unit_id, "tp"]),
                "fn": int(counts.at[gt_unit_id, "fn"]),
                "fp": int(counts.at[gt_unit_id, "fp"]),
                "num_gt": int(counts.at[gt_unit_id, "num_gt"]),
                "num_tested": int(counts.at[gt_unit_id, "num_tested"]),
                "accuracy": float(performance.at[gt_unit_id, "accuracy"]),
                "precision": float(performance.at[gt_unit_id, "precision"]),
                "recall": float(performance.at[gt_unit_id, "recall"]),
                "cluster_is_refractory": (
                    int(is_refractory[cluster_index]) if matched else np.nan
                ),
                "cluster_contamination": (
                    float(contamination[cluster_index]) if matched else np.nan
                ),
            }
        )

    event_match, _ = match_events_one_to_one(
        times.astype(np.int64), all_gt_frames, comparison.delta_frames
    )
    gt_proximal = np.zeros(total_events, dtype=bool)
    gt_proximal[event_ids[event_match]] = True
    tp = int((status == 2).sum())
    fp = int((status == 1).sum())
    fn = int(counts["fn"].astype(np.int64).sum())
    if tp != int(counts["tp"].astype(np.int64).sum()):
        raise RuntimeError(f"{stage} event TP labels disagree with count table")
    if fp != int(counts["fp"].astype(np.int64).sum()):
        raise RuntimeError(f"{stage} event FP labels disagree with count table")
    if int((status != 3).sum()) != times.size:
        raise RuntimeError(f"{stage} present-event status count is inconsistent")
    summary = {
        "domain": domain,
        "Th_learned": threshold,
        "stage": stage,
        "events_present": int(times.size),
        "events_removed": int(total_events - times.size),
        "clusters": int(tested_sorting.get_num_units()),
        "matched_gt_units": int(
            sum(is_matched_unit(value) for value in comparison.hungarian_match_12)
        ),
        "tp": tp,
        "fn": fn,
        "fp_in_matched_clusters": fp,
        "events_in_unmatched_clusters": int((status == 0).sum()),
        "pooled_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pooled_recall": tp / (tp + fn) if tp + fn else 0.0,
        "gt_proximal_events_any_unit": int(event_match.size),
        "gt_negative_events_any_unit": int(times.size - event_match.size),
    }
    return {
        "summary": summary,
        "units": unit_rows,
        "clusters": cluster_rows,
        "status": status,
        "assigned_gt": assigned_gt,
        "gt_proximal": gt_proximal,
    }


def lineage_score_rows(
    domain: str,
    threshold: float,
    stage: str,
    status: np.ndarray,
    scores: np.ndarray,
    peels: np.ndarray,
) -> list[dict]:
    """Summarize original extraction scores by downstream event status."""
    rows = []
    for status_code, status_name in STATUS_NAMES.items():
        for peel in range(int(peels.max()) + 1):
            selected = (status == status_code) & (peels == peel)
            values = scores[selected]
            if not values.size:
                continue
            rows.append(
                {
                    "domain": domain,
                    "Th_learned": threshold,
                    "stage": stage,
                    "status": status_name,
                    "peel": peel,
                    "events": int(values.size),
                    "score_mean": float(values.mean()),
                    "score_q10": float(np.quantile(values, 0.1)),
                    "score_median": float(np.median(values)),
                    "score_q90": float(np.quantile(values, 0.9)),
                }
            )
    return rows


def run_event_lineage(
    domain: str,
    threshold: float,
    ops: dict,
    st: np.ndarray,
    tF: torch.Tensor,
    peels: np.ndarray,
    gt_sorting: si.BaseSorting,
    sampling_frequency: float,
    imin: int,
    final_reference: dict | None = None,
) -> dict[str, list[dict]]:
    """Trace extracted events through clustering, merging, and deduplication."""
    from kilosort import clustering_qr, postprocessing, template_matching

    event_count = st.shape[0]
    if not (event_count == tF.shape[0] == peels.size):
        raise ValueError("extracted event lineage arrays differ in length")
    event_ids = np.arange(event_count, dtype=np.int64)
    detection_times = st[:, 0].astype(np.int64) + int(imin)
    detection_templates = st[:, 1].astype(np.int32)
    detection_scores = st[:, 2].astype(np.float32)
    all_gt_frames = np.sort(
        np.concatenate(
            [
                gt_sorting.get_unit_spike_train(unit_id, segment_index=0)
                for unit_id in gt_sorting.unit_ids
            ]
        ).astype(np.int64)
    )

    stages = []
    stages.append(
        (
            "detection_template",
            evaluate_lineage_stage(
                domain,
                threshold,
                "detection_template",
                detection_times,
                detection_templates,
                event_ids,
                event_count,
                gt_sorting,
                sampling_frequency,
                all_gt_frames,
                float(ops["settings"]["acg_threshold"]),
                float(ops["settings"]["ccg_threshold"]),
                detection_templates,
            ),
        )
    )

    premerge_clusters, Wall = clustering_qr.run(
        ops, st, tF, mode="template", device=torch.device("cuda")
    )
    stages.append(
        (
            "final_clustering",
            evaluate_lineage_stage(
                domain,
                threshold,
                "final_clustering",
                detection_times,
                premerge_clusters,
                event_ids,
                event_count,
                gt_sorting,
                sampling_frequency,
                all_gt_frames,
                float(ops["settings"]["acg_threshold"]),
                float(ops["settings"]["ccg_threshold"]),
                detection_templates,
            ),
        )
    )

    st_with_ids = np.column_stack((st, event_ids))
    _, merged_clusters, _, merged_st, _ = template_matching.merging_function(
        ops,
        Wall,
        premerge_clusters,
        st_with_ids,
        tF,
        device=torch.device("cuda"),
    )
    merged_event_ids = merged_st[:, 3].astype(np.int64)
    if not np.array_equal(np.sort(merged_event_ids), event_ids):
        raise RuntimeError("merging changed the event identity set")
    merged_times = np.empty(event_count, dtype=np.int64)
    merged_labels = np.empty(event_count, dtype=np.int32)
    merged_times[merged_event_ids] = (
        merged_st[:, 0].astype(np.int64) + int(imin)
    )
    merged_labels[merged_event_ids] = merged_clusters.astype(np.int32)
    stages.append(
        (
            "cluster_merging",
            evaluate_lineage_stage(
                domain,
                threshold,
                "cluster_merging",
                merged_times,
                merged_labels,
                event_ids,
                event_count,
                gt_sorting,
                sampling_frequency,
                all_gt_frames,
                float(ops["settings"]["acg_threshold"]),
                float(ops["settings"]["ccg_threshold"]),
                detection_templates,
            ),
        )
    )

    dedup_times, dedup_labels, keep_sorted = postprocessing.remove_duplicates(
        merged_st[:, 0].astype(np.int64) + int(imin),
        merged_clusters.astype(np.int32),
        dt=np.int32(ops["duplicate_spike_bins"]),
    )
    kept_event_ids = merged_event_ids[keep_sorted]
    kept = np.zeros(event_count, dtype=bool)
    kept[kept_event_ids] = True
    stages.append(
        (
            "duplicate_removal",
            evaluate_lineage_stage(
                domain,
                threshold,
                "duplicate_removal",
                dedup_times,
                dedup_labels,
                kept_event_ids,
                event_count,
                gt_sorting,
                sampling_frequency,
                all_gt_frames,
                float(ops["settings"]["acg_threshold"]),
                float(ops["settings"]["ccg_threshold"]),
                detection_templates,
            ),
        )
    )
    if final_reference is not None:
        reference_times = final_reference["times"].astype(np.int64)
        reference_labels = final_reference["labels"].astype(np.int64)
        if not np.array_equal(dedup_times, reference_times):
            raise RuntimeError("lineage final times differ from template-learning sort")
        if not np.array_equal(
            canonicalize_labels(dedup_labels), canonicalize_labels(reference_labels)
        ):
            raise RuntimeError(
                "lineage final cluster partition differs from template-learning sort"
            )
        stages[-1][1]["summary"]["template_learning_sort_identity"] = "exact"

    transition_rows = []
    cluster_transition_rows = []
    stage_delta_rows = []
    score_rows = []
    for stage_name, result in stages:
        score_rows.extend(
            lineage_score_rows(
                domain,
                threshold,
                stage_name,
                result["status"],
                detection_scores,
                peels,
            )
        )
    for (from_name, from_result), (to_name, to_result) in zip(stages, stages[1:]):
        from_summary = from_result["summary"]
        to_summary = to_result["summary"]
        stage_delta_rows.append(
            {
                "domain": domain,
                "Th_learned": threshold,
                "from_stage": from_name,
                "to_stage": to_name,
                "event_delta": (
                    to_summary["events_present"] - from_summary["events_present"]
                ),
                "cluster_delta": to_summary["clusters"] - from_summary["clusters"],
                "matched_gt_unit_delta": (
                    to_summary["matched_gt_units"]
                    - from_summary["matched_gt_units"]
                ),
                "tp_delta": to_summary["tp"] - from_summary["tp"],
                "fn_delta": to_summary["fn"] - from_summary["fn"],
                "fp_in_matched_clusters_delta": (
                    to_summary["fp_in_matched_clusters"]
                    - from_summary["fp_in_matched_clusters"]
                ),
                "events_in_unmatched_clusters_delta": (
                    to_summary["events_in_unmatched_clusters"]
                    - from_summary["events_in_unmatched_clusters"]
                ),
                "gt_negative_events_any_unit_delta": (
                    to_summary["gt_negative_events_any_unit"]
                    - from_summary["gt_negative_events_any_unit"]
                ),
            }
        )
        for from_code, from_status in STATUS_NAMES.items():
            for to_code, to_status in STATUS_NAMES.items():
                count = int(
                    (
                        (from_result["status"] == from_code)
                        & (to_result["status"] == to_code)
                    ).sum()
                )
                if count:
                    transition_rows.append(
                        {
                            "domain": domain,
                            "Th_learned": threshold,
                            "from_stage": from_name,
                            "to_stage": to_name,
                            "from_status": from_status,
                            "to_status": to_status,
                            "events": count,
                        }
                    )

    dedup_labels_by_event = np.full(event_count, -1, dtype=np.int32)
    dedup_labels_by_event[kept_event_ids] = dedup_labels
    stage_cluster_labels = [
        ("detection_template", detection_templates),
        ("final_clustering", premerge_clusters.astype(np.int32)),
        ("cluster_merging", merged_labels),
        ("duplicate_removal", dedup_labels_by_event),
    ]
    for (
        (from_name, from_labels),
        (to_name, to_labels),
        (_, from_result),
        (_, to_result),
    ) in zip(
        stage_cluster_labels,
        stage_cluster_labels[1:],
        stages,
        stages[1:],
    ):
        transitions = np.column_stack(
            (
                from_labels,
                to_labels,
                from_result["status"],
                to_result["status"],
            )
        )
        unique_transitions, transition_counts = np.unique(
            transitions, axis=0, return_counts=True
        )
        for values, count in zip(unique_transitions, transition_counts):
            from_cluster, to_cluster, from_status, to_status = values
            cluster_transition_rows.append(
                {
                    "domain": domain,
                    "Th_learned": threshold,
                    "from_stage": from_name,
                    "to_stage": to_name,
                    "from_cluster": int(from_cluster),
                    "to_cluster": int(to_cluster),
                    "from_status": STATUS_NAMES[int(from_status)],
                    "to_status": STATUS_NAMES[int(to_status)],
                    "events": int(count),
                }
            )

    archive = {
        "event_id": event_ids,
        "detection_time": detection_times,
        "postmerge_time": merged_times,
        "detection_template": detection_templates,
        "detection_score": detection_scores,
        "detection_peel": peels.astype(np.int16),
        "premerge_cluster": premerge_clusters.astype(np.int32),
        "postmerge_cluster": merged_labels,
        "kept_after_dedup": kept,
    }
    for stage_name, result in stages:
        archive[f"{stage_name}_status"] = result["status"]
        archive[f"{stage_name}_assigned_gt"] = result["assigned_gt"]
        archive[f"{stage_name}_gt_proximal"] = result["gt_proximal"]
    threshold_name = str(threshold).replace(".", "p")
    np.savez_compressed(
        RESULTS / f"event_lineage_{domain}_Th_{threshold_name}.npz", **archive
    )
    return {
        "lineage_stages": [result["summary"] for _, result in stages],
        "lineage_units": [
            row for _, result in stages for row in result["units"]
        ],
        "lineage_clusters": [
            row for _, result in stages for row in result["clusters"]
        ],
        "lineage_transitions": transition_rows,
        "lineage_cluster_transitions": cluster_transition_rows,
        "lineage_stage_deltas": stage_delta_rows,
        "lineage_scores": score_rows,
    }


def align_replay_peels(st: np.ndarray, replay: dict) -> np.ndarray:
    """Attach independently replayed peel IDs to Kilosort extraction events."""
    replay_scores = replay["scores"].astype(np.float32)
    replay_keys = list(
        zip(
            replay["events"].astype(np.int64),
            replay["templates"].astype(np.int64),
        )
    )
    peel_by_key = {}
    for key, score, peel in zip(replay_keys, replay_scores, replay["peels"]):
        peel_by_key.setdefault(key, []).append((float(score), int(peel)))
    extracted_scores = st[:, 2].astype(np.float32)
    extracted_keys = zip(
        st[:, 0].astype(np.int64),
        st[:, 1].astype(np.int64),
    )
    peels = np.empty(st.shape[0], dtype=np.int16)
    for index, (key, score) in enumerate(zip(extracted_keys, extracted_scores)):
        candidates = peel_by_key.get(key, [])
        if not candidates:
            raise RuntimeError(f"extracted event absent from replay: {key}")
        differences = np.abs(
            np.asarray([candidate[0] for candidate in candidates]) - float(score)
        )
        match_index = int(np.argmin(differences))
        if not np.isclose(
            candidates[match_index][0], float(score), rtol=1e-6, atol=1e-6
        ):
            raise RuntimeError(
                f"extracted event score differs from replay: {key}, {score}"
            )
        _, peels[index] = candidates.pop(match_index)
    if any(candidates for candidates in peel_by_key.values()):
        raise RuntimeError("replay contains events absent from exact extraction")
    return peels


def replay_domain(
    domain: str,
    binary_file: Path,
    ops: dict,
    U: torch.Tensor,
    gt_all: dict[int, np.ndarray],
    gt_sampled: dict[int, np.ndarray],
    background: np.ndarray,
    gt_sorting: si.BaseSorting,
    final_reference: dict | None,
    device: torch.device,
) -> dict[str, list[dict]]:
    from kilosort import io as ks_io
    from kilosort import template_matching

    bfile = ks_io.bfile_from_ops(
        ops=ops, filename=str(binary_file), device=device
    )
    if int(ops["Nbatches"]) != int(bfile.n_batches):
        raise RuntimeError(
            f"ops/binary batch mismatch: {ops['Nbatches']} != {bfile.n_batches}"
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
    threshold_event_templates = {threshold: [] for threshold in THRESHOLDS}

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
            accepted_template = matched["template"].cpu().numpy()
            valid = (accepted_global >= 0) & (accepted_global < bfile.n_samples)
            threshold_events[threshold].append(accepted_global[valid])
            threshold_event_scores[threshold].append(accepted_score[valid])
            threshold_event_peels[threshold].append(accepted_peel[valid])
            threshold_event_templates[threshold].append(accepted_template[valid])

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
        templates = np.concatenate(threshold_event_templates[threshold])
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
            "templates": templates,
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
    lineage_outputs = []
    for threshold in LINEAGE_THRESHOLDS:
        print(f"[{domain}] exact lineage at Th_learned={threshold:g}", flush=True)
        lineage_ops = ops.copy()
        lineage_ops["settings"] = ops["settings"].copy()
        lineage_ops["Th_learned"] = threshold
        st, tF, lineage_ops = template_matching.extract(
            lineage_ops, bfile, U, device=device
        )
        peels = align_replay_peels(st, threshold_matches[threshold])
        lineage_outputs.append(
            run_event_lineage(
                domain,
                threshold,
                lineage_ops,
                st,
                tF,
                peels,
                gt_sorting,
                float(ops["fs"]),
                int(bfile.imin),
                final_reference if threshold == TEMPLATE_LEARNING_THRESHOLD else None,
            )
        )
        del st, tF
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "histogram": hist_rows,
        "templates": template_rows,
        "gt_units": gt_rows,
        "thresholds": threshold_rows,
        "peels": peel_rows,
        "samples": sample_rows,
        "lineage_stages": [
            row for output in lineage_outputs for row in output["lineage_stages"]
        ],
        "lineage_units": [
            row for output in lineage_outputs for row in output["lineage_units"]
        ],
        "lineage_clusters": [
            row
            for output in lineage_outputs
            for row in output["lineage_clusters"]
        ],
        "lineage_transitions": [
            row
            for output in lineage_outputs
            for row in output["lineage_transitions"]
        ],
        "lineage_cluster_transitions": [
            row
            for output in lineage_outputs
            for row in output["lineage_cluster_transitions"]
        ],
        "lineage_stage_deltas": [
            row
            for output in lineage_outputs
            for row in output["lineage_stage_deltas"]
        ],
        "lineage_scores": [
            row for output in lineage_outputs for row in output["lineage_scores"]
        ],
    }


def learn_denoised_templates(
    recording: si.BaseRecording, output_folder: Path
) -> tuple[Path, np.ndarray, dict]:
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
    sorter_output = output_folder / "sorter_output"
    final_reference = {
        "times": np.load(sorter_output / "spike_times.npy"),
        "labels": np.load(sorter_output / "spike_clusters.npy"),
    }
    return sorter_output / "ops.npy", captured["U"], final_reference


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
    ops_path, templates, denoised_final_reference = learn_denoised_templates(
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
        gt,
        denoised_final_reference,
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
        gt,
        None,
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
        ("lineage_stages", "event_lineage_stage_summary.csv"),
        ("lineage_units", "event_lineage_unit_summary.csv"),
        ("lineage_clusters", "event_lineage_cluster_summary.csv"),
        ("lineage_transitions", "event_lineage_transition_summary.csv"),
        (
            "lineage_cluster_transitions",
            "event_lineage_cluster_transition_summary.csv",
        ),
        ("lineage_stage_deltas", "event_lineage_stage_deltas.csv"),
        ("lineage_scores", "event_lineage_score_summary.csv"),
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
        "lineage_thresholds": LINEAGE_THRESHOLDS,
        "lineage_stages": [
            "detection_template",
            "final_clustering",
            "cluster_merging",
            "duplicate_removal",
        ],
        "lineage_delta_time_ms": LINEAGE_DELTA_TIME_MS,
        "lineage_match_score": LINEAGE_MATCH_SCORE,
        "lineage_status_codes": STATUS_NAMES,
        "lineage_scope": "controlled fixed-denoised-template chain; not independently relearned production templates per domain",
        "denoised_Th8_lineage_reference": "exact template-learning sorting times and cluster partition, allowing cluster ID permutation",
        "lineage_reference_order": "exact Kilosort save order; not independently sorted",
        "lineage_npz_time_coordinates": "recording sample indices after applying bfile.imin",
        "lineage_evaluation": "SpikeInterface compare_sorter_to_ground_truth with full benchmark settings",
        "benchmark_evaluator_spikeinterface_version": "0.104.2",
        "lineage_spikeinterface_version": versions["spikeinterface"],
        "evaluator_matching_source_equivalence": "SpikeInterface 0.104.2 and 0.104.7 comparisontools.py and paircomparisons.py are byte-identical",
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