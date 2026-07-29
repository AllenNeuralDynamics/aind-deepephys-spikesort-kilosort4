"""Ground-truth-independent stopping rules for Kilosort matching peels."""
from __future__ import annotations

import math
from collections.abc import Sequence


def square_root_yield_floor(first_peel_events: int) -> int:
    """Return the Poisson-scale marginal-yield floor for one batch."""
    if first_peel_events < 0:
        raise ValueError("first-peel event count must be nonnegative")
    return math.ceil(math.sqrt(first_peel_events))


def should_stop_for_low_yield(
    peel_counts: Sequence[int],
    patience: int = 3,
) -> bool:
    """Stop after consecutive peels fall below the first-peel yield scale."""
    if patience <= 0:
        raise ValueError("patience must be positive")
    if any(count < 0 for count in peel_counts):
        raise ValueError("peel event counts must be nonnegative")
    if len(peel_counts) < patience:
        return False
    floor = square_root_yield_floor(peel_counts[0])
    return all(count <= floor for count in peel_counts[-patience:])