"""
Cashflow schedule for vanilla IRS.

We keep it simple (year-fraction grid in ACT/365-equivalent years).
If you need business-day calendars, day-counts, etc., you can extend later.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Schedule:
    """
    Payment schedule between t_start and t_end (years), with fixed frequency.

    Parameters
    ----------
    t_start : float
        Start time in years (e.g., 0.0).
    t_end : float
        End time in years (e.g., 5.0).
    freq_per_year : int
        Payments per year (e.g., 2 for semi-annual, 4 for quarterly).
    """
    t_start: float
    t_end: float
    freq_per_year: int

    def __post_init__(self) -> None:
        if self.t_end <= self.t_start:
            raise ValueError("Schedule: t_end must be > t_start")
        if self.freq_per_year <= 0:
            raise ValueError("Schedule: freq_per_year must be > 0")
        object.__setattr__(self, "_dt", 1.0 / float(self.freq_per_year))
        # Build payment times excluding start, including end
        # e.g. start=0, end=5, semi-annual -> [0.5, 1.0, ..., 5.0]
        n = int(round((self.t_end - self.t_start) / self._dt))
        if abs(self.t_start + n * self._dt - self.t_end) > 1e-10:
            raise ValueError("Schedule: (t_end - t_start) must be multiple of 1/freq")
        pays = self.t_start + self._dt * np.arange(1, n + 1, dtype=float)
        object.__setattr__(self, "_pay_times", pays)
        object.__setattr__(self, "_accruals", np.full_like(pays, self._dt))

    @property
    def pay_times(self) -> np.ndarray:
        """Cashflow payment times T_j (shape (m,))."""
        return self._pay_times

    @property
    def accruals(self) -> np.ndarray:
        """Accrual factors α_j (shape (m,)); here constant dt = 1/freq."""
        return self._accruals

    @property
    def dt(self) -> float:
        """Accrual length (years)."""
        return self._dt

    def remaining_after(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Times & accruals of remaining payments strictly after time t.
        """
        mask = self._pay_times > (t + 1e-14)
        return self._pay_times[mask], self._accruals[mask]

# ---- ADD: explicit schedule (supports stubs) -------------------------------

@dataclass(frozen=True)
class ExplicitSchedule:
    """
    Schedule défini par pay_times explicites (supporte un 1er stub).
    pay_times: maturités (en années) depuis l'as-of, strictement croissantes.
    accruals : α_j = pay_times[j] - pay_times[j-1], avec pay_times[-1] après 0.
    """
    pay_times: np.ndarray
    accruals: np.ndarray

    @classmethod
    def from_pay_times(cls, pay_times) -> "ExplicitSchedule":
        pt = np.asarray(pay_times, dtype=float).ravel()
        if pt.size == 0:
            return cls(pt, pt.copy())
        if pt[0] <= 0.0 or np.any(np.diff(pt) <= 1e-14):
            raise ValueError("ExplicitSchedule: pay_times must be strictly increasing and > 0")
        acc = np.empty_like(pt)
        acc[0] = pt[0]
        acc[1:] = np.diff(pt)
        return cls(pt, acc)

    def remaining_after(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        mask = self.pay_times > (t + 1e-14)
        return self.pay_times[mask], self.accruals[mask]
