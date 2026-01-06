"""
Calendrier de cashflows pour un swap de taux vanilla (IRS).
Les temps sont exprimés en fractions d’années (équivalent ACT/365).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Schedule:
    """
    Calendrier de paiements entre t_start et t_end (en années), à fréquence fixe.

    Paramètres
    ----------
    t_start : float
        Date de départ en années (ex : 0.0).
    t_end : float
        Date de fin en années (ex : 5.0).
    freq_per_year : int
        Nombre de paiements par an (ex : 2 semestriel, 4 trimestriel).
    """
    t_start: float
    t_end: float
    freq_per_year: int

    def __post_init__(self) -> None:
        if self.t_end <= self.t_start:
            raise ValueError("Schedule: t_end doit être > t_start")
        if self.freq_per_year <= 0:
            raise ValueError("Schedule: freq_per_year doit être > 0")

        object.__setattr__(self, "_dt", 1.0 / float(self.freq_per_year))

        # Construit les dates de paiement en excluant le start mais en incluant le end.
        # Exemple : start=0, end=5, semestriel -> [0.5, 1.0, ..., 5.0]
        n = int(round((self.t_end - self.t_start) / self._dt))
        if abs(self.t_start + n * self._dt - self.t_end) > 1e-10:
            raise ValueError("Schedule: (t_end - t_start) must be multiple of 1/freq")

        pays = self.t_start + self._dt * np.arange(1, n + 1, dtype=float)
        object.__setattr__(self, "_pay_times", pays)

        # Accruals constants (ici = dt), pratique pour calculer l’annuity
        object.__setattr__(self, "_accruals", np.full_like(pays, self._dt))

    @property
    def pay_times(self) -> np.ndarray:
        """Dates de paiement des cashflows T_j (shape (m,))."""
        return self._pay_times

    @property
    def accruals(self) -> np.ndarray:
        """Facteurs d’accrual α_j (shape (m,)) ; ici constant dt = 1/freq."""
        return self._accruals

    @property
    def dt(self) -> float:
        """Longueur d’une période d’accrual (en années)."""
        return self._dt

    def remaining_after(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Renvoie les dates (et accruals) des paiements restant strictement après t.
        Utile pour calculer la MTM à une date intermédiaire t_k.
        """
        mask = self._pay_times > (t + 1e-14)
        return self._pay_times[mask], self._accruals[mask]


# ---- AJOUT : calendrier explicite (supporte les stubs) ----------------------

@dataclass(frozen=True)
class ExplicitSchedule:
    """
    Calendrier défini par des pay_times explicites.

    pay_times : maturités (en années) depuis l'as-of, strictement croissantes.
    accruals  : α_j = pay_times[j] - pay_times[j-1], avec un premier accrual α_0 = pay_times[0].
    """
    pay_times: np.ndarray
    accruals: np.ndarray

    @classmethod
    def from_pay_times(cls, pay_times) -> "ExplicitSchedule":
        pt = np.asarray(pay_times, dtype=float).ravel()

        if pt.size == 0:
            return cls(pt, pt.copy())

        if pt[0] <= 0.0 or np.any(np.diff(pt) <= 1e-14):
            raise ValueError("ExplicitSchedule: pay_times doit être strictement croissant et > 0")

        # Calcule les accruals : premier accrual = pt[0] 
        acc = np.empty_like(pt)
        acc[0] = pt[0]
        acc[1:] = np.diff(pt)
        return cls(pt, acc)

    def remaining_after(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Même API que Schedule : ne garde que les dates strictement après t."""
        mask = self.pay_times > (t + 1e-14)
        return self.pay_times[mask], self.accruals[mask]
