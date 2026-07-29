from __future__ import annotations

import numpy as np
import pytest

from event_waveform import (
    paired_waveform_cosine,
    quantile_sample_indices,
    rank_auc,
    stratified_sample_indices,
    waveform_shape_metrics,
)


def test_stratified_sample_is_deterministic_and_spans_distribution() -> None:
    scores = np.arange(100, dtype=np.float64)
    peels = np.tile(np.arange(10), 10)

    first = stratified_sample_indices(scores, peels, count=32, seed=7)
    second = stratified_sample_indices(scores, peels, count=32, seed=7)

    np.testing.assert_array_equal(first, second)
    assert first.size == 32
    assert np.unique(first).size == first.size
    assert scores[first].min() < 25
    assert scores[first].max() >= 75
    assert peels[first].min() <= 2
    assert peels[first].max() >= 7


def test_waveform_shape_metrics_separate_scale_from_shape() -> None:
    reference = np.array([[0.0, -1.0, 0.0], [0.0, -0.5, 0.0]])
    waveforms = np.stack((reference, 2 * reference, -reference))

    cosine, scale, residual = waveform_shape_metrics(waveforms, reference)

    np.testing.assert_allclose(cosine, [1.0, 1.0, -1.0])
    np.testing.assert_allclose(scale, [1.0, 2.0, -1.0])
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_waveform_shape_metrics_reject_zero_reference() -> None:
    with pytest.raises(ValueError, match="zero energy"):
        waveform_shape_metrics(np.ones((2, 3)), np.ones(3))


def test_rank_auc_handles_separation_and_ties() -> None:
    labels = np.array([False, False, True, True])

    assert rank_auc(labels, np.array([0.0, 1.0, 2.0, 3.0])) == 1.0
    assert rank_auc(labels, np.ones(4)) == 0.5


def test_paired_waveform_cosine_is_scale_invariant() -> None:
    first = np.array([[[0.0, -1.0, 0.0]], [[0.0, -2.0, 0.0]]])
    second = np.array([[[1.0, -1.0, 1.0]], [[0.0, 1.0, 0.0]]])

    np.testing.assert_allclose(paired_waveform_cosine(first, second), [1.0, -1.0])


def test_quantile_sample_indices_span_ordered_values() -> None:
    values = np.arange(100, dtype=np.float64)[::-1]

    selected = quantile_sample_indices(values, count=5)

    np.testing.assert_array_equal(values[selected], [10.0, 30.0, 49.0, 69.0, 89.0])
    assert np.unique(selected).size == 5


def test_quantile_sample_indices_return_all_small_inputs() -> None:
    values = np.array([3.0, 1.0, 2.0])

    selected = quantile_sample_indices(values, count=5)

    np.testing.assert_array_equal(values[selected], [1.0, 2.0, 3.0])