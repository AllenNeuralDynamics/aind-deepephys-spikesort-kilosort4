from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import nnls
from torch.nn.functional import conv1d

from joint_refit import event_gram, refit_block, solve_nonnegative_quadratic
from peel_stopping import square_root_yield_floor, should_stop_for_low_yield
from run_joint_refit_diagnostic import (
    run_joint_refit_matching,
    summarize_peel_comparison,
)
from run_peel_stopping_diagnostic import run_peel_stopped_matching
from run_score_diagnostic import simulate_matching


def prepare_ctc(U: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    nt = W.shape[1]
    WtW = conv1d(
        W.reshape(-1, 1, nt),
        W.reshape(-1, 1, nt),
        padding=nt,
    )
    WtW = torch.flip(WtW, [2])
    UtU = torch.einsum("ikl,jml->ijkm", U, U)
    return torch.einsum("ijkm,kml->ijl", UtU, WtW)


def make_problem() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(7)
    dtype = torch.float64
    nt = 9
    W = torch.randn(3, nt, dtype=dtype)
    U = torch.randn(4, 3, 5, dtype=dtype)
    ctc = prepare_ctc(U, W)
    times = torch.tensor([20, 24, 31, 45], dtype=torch.int64)
    templates = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
    tiwave = torch.arange(-(nt // 2), nt // 2 + 1)
    atoms = []
    for event_time, template in zip(times, templates):
        atom = torch.zeros((5, 64), dtype=dtype)
        atom[:, event_time + tiwave] = torch.einsum(
            "jk,jl->kl", U[template], W
        )
        atoms.append(atom)
    design = torch.stack([atom.reshape(-1) for atom in atoms], dim=1)
    return U, W, ctc, times, templates, design


def project(
    signal: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    projection = conv1d(
        signal.unsqueeze(1), W.unsqueeze(1), padding=W.shape[1] // 2
    )
    return torch.einsum("ijk,kjl->il", U, projection)


def test_event_gram_matches_explicit_shifted_waveforms() -> None:
    _, W, ctc, times, templates, design = make_problem()

    actual = event_gram(ctc, times, templates, W.shape[1])

    torch.testing.assert_close(actual, design.T @ design, rtol=1e-12, atol=1e-10)


def test_optimal_amplitudes_leave_residual_unchanged() -> None:
    U, W, ctc, times, templates, design = make_problem()
    amplitudes = torch.tensor([1.2, 0.6, 1.8, 0.9], dtype=W.dtype)
    signal = (design @ amplitudes).reshape(5, 64)
    residual = signal - (design @ amplitudes).reshape(5, 64)

    refitted, delta, diagnostics = refit_block(
        project(residual, U, W),
        ctc,
        times,
        templates,
        amplitudes,
        W.shape[1],
    )

    torch.testing.assert_close(refitted, amplitudes, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0, atol=1e-12)
    assert diagnostics["converged"]
    assert abs(diagnostics["energy_reduction"]) < 1e-20


def test_joint_refit_improves_overlapping_residual() -> None:
    U, W, ctc, times, templates, design = make_problem()
    true_amplitudes = torch.tensor([1.2, 0.6, 1.8, 0.9], dtype=W.dtype)
    signal = (design @ true_amplitudes).reshape(5, 64)
    greedy_amplitudes = torch.tensor([1.0, 0.8, 1.3, 0.7], dtype=W.dtype)
    residual = signal - (design @ greedy_amplitudes).reshape(5, 64)

    refitted, delta, diagnostics = refit_block(
        project(residual, U, W),
        ctc,
        times,
        templates,
        greedy_amplitudes,
        W.shape[1],
    )
    refitted_residual = residual - (design @ delta).reshape(5, 64)
    reference, _ = nnls(design.numpy(), signal.reshape(-1).numpy())

    np.testing.assert_allclose(refitted.numpy(), reference, rtol=1e-8, atol=1e-9)
    assert torch.sum(refitted_residual**2) < torch.sum(residual**2)
    np.testing.assert_allclose(
        diagnostics["energy_reduction"],
        float(torch.sum(residual**2) - torch.sum(refitted_residual**2)),
        rtol=1e-10,
        atol=1e-10,
    )


def test_negative_unconstrained_amplitude_is_zeroed() -> None:
    gram = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    rhs = torch.tensor([2.0, -1.0], dtype=torch.float64)

    amplitude, diagnostics = solve_nonnegative_quadratic(
        gram, rhs, torch.tensor([0.5, 0.5], dtype=torch.float64)
    )

    torch.testing.assert_close(
        amplitude, torch.tensor([1.0, 0.0], dtype=torch.float64)
    )
    assert diagnostics["converged"]
    assert diagnostics["solver"] == "active_set_eigh"
    assert torch.all(amplitude >= 0)


def test_positive_full_rank_solution_uses_cholesky_fast_path() -> None:
    gram = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    expected = torch.tensor([1.0, 2.0], dtype=torch.float64)

    amplitude, diagnostics = solve_nonnegative_quadratic(
        gram, gram @ expected, torch.tensor([0.5, 0.5], dtype=torch.float64)
    )

    torch.testing.assert_close(amplitude, expected)
    assert diagnostics["converged"]
    assert diagnostics["solver"] == "cholesky"
    assert diagnostics["rank"] == 2
    assert diagnostics["condition_is_proxy"]


def test_rank_deficient_duplicate_atoms_are_safe() -> None:
    gram = torch.tensor([[4.0, 4.0], [4.0, 4.0]], dtype=torch.float64)
    rhs = torch.tensor([6.0, 6.0], dtype=torch.float64)
    initial = torch.tensor([1.0, 0.0], dtype=torch.float64)
    objective_before = 0.5 * initial @ gram @ initial - rhs @ initial

    amplitude, diagnostics = solve_nonnegative_quadratic(gram, rhs, initial)
    objective_after = 0.5 * amplitude @ gram @ amplitude - rhs @ amplitude

    assert diagnostics["converged"]
    assert diagnostics["solver"] == "active_set_eigh"
    assert diagnostics["rank"] == 1
    assert torch.all(amplitude >= 0)
    torch.testing.assert_close(amplitude.sum(), torch.tensor(1.5, dtype=gram.dtype))
    assert objective_after <= objective_before


def test_matching_loop_preserves_exact_residual_energy_ledger() -> None:
    torch.manual_seed(11)
    dtype = torch.float64
    nt = 9
    W = torch.randn(2, nt, dtype=dtype)
    U = torch.randn(3, 2, 4, dtype=dtype)
    ctc = prepare_ctc(U, W)
    signal = torch.zeros(4, 256, dtype=dtype)
    tiwave = torch.arange(-(nt // 2), nt // 2 + 1)
    for event_time, template, amplitude in (
        (90, 0, 2.0),
        (94, 1, 1.7),
        (180, 2, 1.5),
    ):
        signal[:, event_time + tiwave] += amplitude * torch.einsum(
            "jk,jl->kl", U[template], W
        )

    result = run_joint_refit_matching(
        {"nt": nt, "wPCA": W, "max_peels": 10}, signal, U, ctc
    )
    summary = result["batch_summary"]
    relative_audit_error = abs(summary["residual_energy_audit_error"]) / summary[
        "initial_residual_energy"
    ]

    assert result["time"].numel() == 3
    assert torch.all(result["amplitude"] > 0)
    assert sum(row["overlap_blocks"] for row in result["telemetry_rows"]) > 0
    assert all(
        row["refit_energy_reduction"] >= 0
        for row in result["telemetry_rows"]
    )
    assert relative_audit_error < 1e-10


def test_peel_comparison_reports_exact_and_late_event_deltas() -> None:
    rows = summarize_peel_comparison(
        np.array([0, 0, 1, 2, 2, 3]),
        np.array([0, 0, 1, 2]),
        max_peels=5,
        domain="raw_native_joint_refit",
    )

    assert rows[0]["baseline_events"] == 2
    assert rows[0]["joint_refit_events"] == 2
    assert rows[0]["baseline_events_at_or_after"] == 6
    assert rows[0]["joint_refit_events_at_or_after"] == 4
    assert rows[0]["late_event_delta"] == -2
    assert rows[2]["event_delta"] == -1
    assert rows[3]["late_event_ratio"] == 0
    assert np.isnan(rows[4]["late_event_ratio"])


def test_square_root_yield_stopping_requires_consecutive_low_yield() -> None:
    assert square_root_yield_floor(484) == 22
    assert not should_stop_for_low_yield([484, 21, 30, 20, 19])
    assert should_stop_for_low_yield([484, 30, 22, 20, 19])


def test_square_root_yield_stopping_validates_inputs() -> None:
    with pytest.raises(ValueError, match="first-peel"):
        square_root_yield_floor(-1)
    with pytest.raises(ValueError, match="patience"):
        should_stop_for_low_yield([10], patience=0)
    with pytest.raises(ValueError, match="peel event"):
        should_stop_for_low_yield([10, -1, 0])


def test_matching_stopping_policy_runs_before_triggering_peel() -> None:
    U, W, ctc, _, _, design = make_problem()
    amplitudes = torch.tensor([1.2, 0.6, 1.8, 0.9], dtype=W.dtype)
    signal = (design @ amplitudes).reshape(5, 64)
    ops = {"nt": W.shape[1], "wPCA": W, "max_peels": 10}
    initial = project(signal, U, W)

    baseline = simulate_matching(ops, initial, U, ctc, 0.0, X=signal)
    no_stop = simulate_matching(
        ops,
        initial,
        U,
        ctc,
        0.0,
        X=signal,
        stop_before_peel=lambda _: False,
    )
    stopped = simulate_matching(
        ops,
        initial,
        U,
        ctc,
        0.0,
        X=signal,
        stop_before_peel=lambda _: True,
    )

    for key in ("time", "template", "score", "amplitude", "peel", "Xres"):
        torch.testing.assert_close(no_stop[key], baseline[key], rtol=0, atol=0)
    assert stopped["peel_counts"][0] > 0
    assert stopped["time"].numel() == 0
    torch.testing.assert_close(stopped["Xres"], signal, rtol=0, atol=0)


def test_peel_stopping_wrapper_reports_excluded_triggering_peel() -> None:
    U, W, ctc, _, _, design = make_problem()
    signal = (design[:, 0] * 100).reshape(5, 64)

    stopped = run_peel_stopped_matching(
        {"nt": W.shape[1], "wPCA": W, "max_peels": 10},
        signal,
        U,
        ctc,
        patience=1,
    )
    summary = stopped["stopping_summary"]

    assert summary["stop_triggered"]
    assert summary["stop_peel"] == 0
    assert summary["triggering_peel_events"] > 0
    assert summary["accepted_events"] == 0
    torch.testing.assert_close(stopped["Xres"], signal, rtol=0, atol=0)
