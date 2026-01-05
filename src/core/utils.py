"""
Utilities: RNG, numeric helpers, sanity checks.

Intended to be lightweight and dependency-free (NumPy only).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


# ===== Randomness / Seeds =====================================================

def make_rng(seed: int | None = None) -> np.random.Generator:
    """
    Create a NumPy Generator with an optional seed.
    Use ONE rng per run and pass it explicitly to all simulators.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


# ===== Numerics ===============================================================

def trapz_increment(y_prev: float, y_curr: float, dt: float) -> float:
    """
    Trapezoidal increment for ∫ y dt over a single step [t, t+dt].
    Returns 0.5 * (y_prev + y_curr) * dt
    """
    return 0.5 * (y_prev + y_curr) * dt


def trapz(x: np.ndarray, y: np.ndarray) -> float:
    """
    Trapezoidal rule for ∫ y(x) dx over the full grid.
    Thin wrapper over numpy.trapz with shape/type assertions.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("trapz expects 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("trapz: x and y must have same length")
    return float(np.trapz(y, x))


# ===== Sanity checks / assertions ============================================

def assert_finite(name: str, arr: np.ndarray) -> None:
    """Raise if any NaN/Inf in arr."""
    arr = np.asarray(arr)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: contains NaN/Inf")


def assert_shape(name: str, arr: np.ndarray, expected: Tuple[int, ...]) -> None:
    """
    Assert exact shape. Use -1 as wildcard for any length on that axis.
    Example: assert_shape("rates", rates, (N, -1)) where rates.shape==(N,K)
    """
    arr = np.asarray(arr)
    if arr.ndim != len(expected):
        raise ValueError(f"{name}: expected {len(expected)}D, got {arr.ndim}D")
    for i, (got, exp) in enumerate(zip(arr.shape, expected)):
        if exp != -1 and got != exp:
            raise ValueError(f"{name}: axis {i} expected {exp}, got {got}")


def is_monotone_decreasing(x: Iterable[float]) -> bool:
    """True if strictly non-increasing (allows equal)."""
    prev = None
    for v in x:
        if prev is not None and v > prev + 1e-14:
            return False
        prev = v
    return True


# ===== Small helpers ==========================================================

@dataclass(frozen=True)
class Stats1D:
    mean: float
    std: float
    min: float
    max: float

def stats1d(a: np.ndarray) -> Stats1D:
    """Quick descriptive stats for logging/debug."""
    a = np.asarray(a, dtype=float).ravel()
    return Stats1D(
        mean=float(np.mean(a)),
        std=float(np.std(a, ddof=0)),
        min=float(np.min(a)),
        max=float(np.max(a)),
    )
