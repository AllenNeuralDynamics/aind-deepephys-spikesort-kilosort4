"""Pure helpers for posthoc TP/FP waveform comparisons."""
from __future__ import annotations

import numpy as np


def _rank_bins(values: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0:
        raise ValueError("bins must be positive")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.int64)
    ranks[order] = np.arange(values.size)
    return np.minimum(bins - 1, ranks * bins // max(values.size, 1))


def stratified_sample_indices(
    scores: np.ndarray,
    peels: np.ndarray,
    count: int,
    seed: int,
    bins: int = 4,
) -> np.ndarray:
    """Sample events across the joint score/peel rank distribution."""
    scores = np.asarray(scores)
    peels = np.asarray(peels)
    if scores.ndim != 1 or peels.ndim != 1 or scores.size != peels.size:
        raise ValueError("scores and peels must be equal-length vectors")
    if count < 0:
        raise ValueError("count must be nonnegative")
    if scores.size <= count:
        return np.arange(scores.size, dtype=np.int64)
    if count == 0:
        return np.empty(0, dtype=np.int64)

    groups = _rank_bins(scores, bins) * bins + _rank_bins(peels, bins)
    group_count = bins**2
    quota = count // group_count
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for group in range(group_count):
        members = np.flatnonzero(groups == group)
        rng.shuffle(members)
        selected.extend(members[:quota].tolist())

    selected_array = np.asarray(selected, dtype=np.int64)
    remaining = np.setdiff1d(
        np.arange(scores.size, dtype=np.int64),
        selected_array,
        assume_unique=False,
    )
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)].tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def quantile_sample_indices(
    values: np.ndarray,
    count: int,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> np.ndarray:
    """Select deterministic examples spanning an ordered value distribution."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be a vector")
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0 <= lower_quantile <= upper_quantile <= 1:
        raise ValueError("quantile bounds must satisfy 0 <= lower <= upper <= 1")
    order = np.argsort(values, kind="stable")
    sample_count = min(count, values.size)
    if not sample_count:
        return np.empty(0, dtype=np.int64)
    targets = np.linspace(lower_quantile, upper_quantile, sample_count) * (
        values.size - 1
    )
    available = set(range(values.size))
    positions = []
    for target in targets:
        position = min(available, key=lambda item: (abs(item - target), item))
        positions.append(position)
        available.remove(position)
    return order[np.sort(positions)]


def waveform_shape_metrics(
    waveforms: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cosine, optimal scale, and relative residual for each waveform."""
    waveforms = np.asarray(waveforms, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if waveforms.ndim < 2 or waveforms.shape[1:] != reference.shape:
        raise ValueError("waveforms must have shape (events, *reference.shape)")
    centered = waveforms - waveforms.mean(axis=tuple(range(1, waveforms.ndim)), keepdims=True)
    centered_reference = reference - reference.mean()
    flat = centered.reshape(centered.shape[0], -1)
    flat_reference = centered_reference.reshape(-1)
    reference_energy = float(np.dot(flat_reference, flat_reference))
    if reference_energy <= 0:
        raise ValueError("reference waveform has zero energy")
    waveform_energy = np.einsum("ij,ij->i", flat, flat)
    projection = flat @ flat_reference
    cosine = projection / np.sqrt(waveform_energy * reference_energy).clip(1e-12)
    scale = projection / reference_energy
    residual = flat - scale[:, None] * flat_reference[None, :]
    residual_fraction = np.sqrt(
        np.einsum("ij,ij->i", residual, residual) / waveform_energy.clip(1e-12)
    )
    return cosine, scale, residual_fraction


def paired_waveform_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return row-wise centered cosine similarity for paired waveforms."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim < 2:
        raise ValueError("paired waveforms must have equal event-first shapes")
    axes = tuple(range(1, first.ndim))
    first = first - first.mean(axis=axes, keepdims=True)
    second = second - second.mean(axis=axes, keepdims=True)
    first = first.reshape(first.shape[0], -1)
    second = second.reshape(second.shape[0], -1)
    numerator = np.einsum("ij,ij->i", first, second)
    denominator = np.sqrt(
        np.einsum("ij,ij->i", first, first)
        * np.einsum("ij,ij->i", second, second)
    )
    return numerator / denominator.clip(1e-12)


def rank_auc(labels: np.ndarray, values: np.ndarray) -> float:
    """Return the probability that a positive value exceeds a negative value."""
    labels = np.asarray(labels, dtype=bool)
    values = np.asarray(values, dtype=np.float64)
    if labels.ndim != 1 or values.ndim != 1 or labels.size != values.size:
        raise ValueError("labels and values must be equal-length vectors")
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        return np.nan
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1
        start = stop
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)