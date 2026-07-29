"""Local nonnegative amplitude refitting for Kilosort matching pursuit."""
from __future__ import annotations

from typing import Any

import torch


def event_gram(
    ctc: torch.Tensor,
    times: torch.Tensor,
    templates: torch.Tensor,
    nt: int,
) -> torch.Tensor:
    """Build the Gram matrix for time-shifted Kilosort template events."""
    if times.ndim != 1 or templates.ndim != 1 or times.shape != templates.shape:
        raise ValueError("times and templates must be one-dimensional and equal length")
    if ctc.ndim != 3 or ctc.shape[-1] != 2 * nt + 1:
        raise ValueError("ctc must have shape (templates, templates, 2 * nt + 1)")
    if times.numel() == 0:
        return ctc.new_empty((0, 0))

    lags = times[:, None] - times[None, :] + nt
    valid = (lags >= 0) & (lags < ctc.shape[-1])
    values = ctc[
        templates[:, None],
        templates[None, :],
        lags.clamp(0, ctc.shape[-1] - 1),
    ]
    gram = torch.where(valid, values, 0)
    return (gram + gram.T) / 2


def _passive_solution(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    passive: torch.Tensor,
) -> torch.Tensor:
    solution = torch.zeros_like(rhs)
    indices = torch.nonzero(passive)[:, 0]
    if indices.numel() == 0:
        return solution

    subgram = gram[indices[:, None], indices[None, :]]
    subrhs = rhs[indices]
    eigenvalues, eigenvectors = torch.linalg.eigh(subgram)
    cutoff = (
        torch.finfo(subgram.dtype).eps
        * max(1, subgram.shape[0])
        * eigenvalues[-1].abs().clamp_min(1)
    )
    inverse = torch.where(eigenvalues > cutoff, eigenvalues.reciprocal(), 0)
    solution[indices] = eigenvectors @ (inverse * (eigenvectors.T @ subrhs))
    return solution


def solve_nonnegative_quadratic(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve min 0.5*x.T*gram*x - rhs.T*x subject to x >= 0."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    if rhs.shape != (gram.shape[0],) or initial.shape != rhs.shape:
        raise ValueError("rhs and initial must match the Gram dimension")
    if gram.numel() == 0:
        return initial.clone(), {"iterations": 0, "converged": True, "rank": 0}

    gram = (gram + gram.T) / 2
    amplitude = initial.clamp_min(0).clone()
    scale = torch.stack(
        (
            gram.diagonal().abs().max(),
            rhs.abs().max(),
            torch.ones((), dtype=gram.dtype, device=gram.device),
        )
    ).max()
    tolerance = torch.finfo(gram.dtype).eps * max(1, gram.shape[0]) * scale
    passive = amplitude > tolerance
    max_iterations = max(1, 3 * gram.shape[0])
    iterations = 0

    while iterations < max_iterations:
        candidate = _passive_solution(gram, rhs, passive)
        while torch.any(passive & (candidate <= tolerance)):
            decreasing = passive & (candidate <= tolerance)
            step = torch.min(
                amplitude[decreasing]
                / (amplitude[decreasing] - candidate[decreasing]).clamp_min(tolerance)
            )
            amplitude += step * (candidate - amplitude)
            remove = passive & (amplitude <= tolerance)
            passive[remove] = False
            amplitude[remove] = 0
            candidate = _passive_solution(gram, rhs, passive)
            iterations += 1
            if iterations >= max_iterations:
                break
        amplitude = candidate.clamp_min(0)
        gradient = rhs - gram @ amplitude
        inactive_gradient = torch.where(passive, -torch.inf, gradient)
        entering = torch.argmax(inactive_gradient)
        if inactive_gradient[entering] <= tolerance:
            break
        passive[entering] = True
        iterations += 1

    eigenvalues = torch.linalg.eigvalsh(gram)
    rank_cutoff = (
        torch.finfo(gram.dtype).eps
        * max(1, gram.shape[0])
        * eigenvalues[-1].abs().clamp_min(1)
    )
    positive_eigenvalues = eigenvalues[eigenvalues > rank_cutoff]
    rank = int(positive_eigenvalues.numel())
    condition = (
        float(positive_eigenvalues[-1] / positive_eigenvalues[0])
        if positive_eigenvalues.numel()
        else float("inf")
    )
    gradient = rhs - gram @ amplitude
    converged = bool(
        torch.all(gradient[~passive] <= tolerance)
        and torch.all(torch.abs(gradient[passive]) <= 10 * tolerance)
    )
    return amplitude, {
        "iterations": iterations,
        "converged": converged,
        "rank": rank,
        "condition": condition,
        "tolerance": float(tolerance),
    }


def refit_block(
    B: torch.Tensor,
    ctc: torch.Tensor,
    times: torch.Tensor,
    templates: torch.Tensor,
    amplitudes: torch.Tensor,
    nt: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Refit one event block against the current residual projections."""
    gram = event_gram(ctc, times, templates, nt)
    residual_projection = B[templates, times]
    rhs = residual_projection + gram @ amplitudes
    refitted, diagnostics = solve_nonnegative_quadratic(gram, rhs, amplitudes)
    delta = refitted - amplitudes
    diagnostics.update(
        {
            "size": int(times.numel()),
            "energy_reduction": float(
                2 * torch.dot(delta, residual_projection)
                - torch.dot(delta, gram @ delta)
            ),
            "changed_amplitudes": int(torch.count_nonzero(delta).item()),
        }
    )
    return refitted, delta, diagnostics