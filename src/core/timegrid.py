"""
Time grid utilities.

We represent time in *years* (ACT/365 or 30/360 choice comes earlier when you
build T and dt). The grid is uniform for simplicity: t_k = k * dt, k=0..K.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TimeGrid:
    """
    Uniform time grid on [0, T] with step dt (years).

    Attributes
    ----------
    T : float
        Horizon in years (e.g., 5.0 for 5Y).
    dt : float
        Time step in years (e.g., 1/12 for monthly).
    times : np.ndarray
        Array of shape (K+1,) with t_k = k*dt, last point == T (within tol).
    """
    T: float
    dt: float

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("TimeGrid: T must be > 0")
        if self.dt <= 0:
            raise ValueError("TimeGrid: dt must be > 0")
        # Compute number of steps with safe rounding
        K_float = self.T / self.dt
        K = int(round(K_float))
        if abs(K_float - K) > 1e-10:
            # Allow tiny floating mismatch, but warn users by being explicit.
            # You can make this a warning if you prefer.
            raise ValueError(
                f"TimeGrid: T/dt must be (almost) integer. Got T/dt={K_float:.12f}"
            )
        object.__setattr__(self, "_K", K)
        times = np.linspace(0.0, self.T, K + 1, dtype=float)
        object.__setattr__(self, "times", times)

        # Safety: ensure last time equals T to numerical tolerance
        if abs(self.times[-1] - self.T) > 1e-12:
            raise AssertionError("TimeGrid: last time is not equal to T (num issue)")

    @property
    def K(self) -> int:
        """Number of steps (so there are K+1 time points including t0=0)."""
        return self._K

    def index_of_time(self, t: float) -> int:
        """
        Return k such that times[k] == t within tolerance, else raise.
        Useful when aligning product cashflows to grid points.
        """
        idx = int(round(t / self.dt))
        if idx < 0 or idx > self.K:
            raise IndexError(f"time {t} out of grid [0,{self.T}]")
        if abs(self.times[idx] - t) > 1e-10:
            raise ValueError(f"time {t} is not on the grid (dt={self.dt})")
        return idx

    def nearest_index(self, t: float) -> int:
        """Return argmin |times - t| (no requirement that t lies on grid)."""
        return int(np.argmin(np.abs(self.times - t)))

    def contains(self, t: float) -> bool:
        """True if t in [0, T] (inclusive)."""
        return (t >= -1e-14) and (t <= self.T + 1e-14)
