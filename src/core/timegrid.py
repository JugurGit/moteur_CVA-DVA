from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TimeGrid:
    """
    Grille de temps uniforme sur [0, T] avec un pas dt (en années).
    """
    T: float  # horizon (années)
    dt: float  # pas de temps (années)

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("TimeGrid: T doit être > 0")
        if self.dt <= 0:
            raise ValueError("TimeGrid: dt doit être > 0")

        # --- Détermination du nombre de pas K ---
        K_float = self.T / self.dt
        K = int(round(K_float))

        if abs(K_float - K) > 1e-10:
            raise ValueError(
                f"TimeGrid: T/dt doit être entier. Or, on a T/dt={K_float:.12f}"
            )

        object.__setattr__(self, "_K", K)

        # Construction de la grille 
        times = np.linspace(0.0, self.T, K + 1, dtype=float)
        object.__setattr__(self, "times", times)

        if abs(self.times[-1] - self.T) > 1e-12:
            raise AssertionError("TimeGrid: last time is not equal to T (num issue)")

    @property
    def K(self) -> int:
        """Nombre de pas (donc K+1 points en incluant t0=0)."""
        return self._K

    def index_of_time(self, t: float) -> int:
        """
        Renvoie k tel que times[k] == t à tolérance près, sinon lève une erreur.
        """
        idx = int(round(t / self.dt))

        if idx < 0 or idx > self.K:
            raise IndexError(f"time {t} out of grid [0,{self.T}]")

        if abs(self.times[idx] - t) > 1e-10:
            raise ValueError(f"time {t} is not on the grid (dt={self.dt})")
        return idx

    def nearest_index(self, t: float) -> int:
        """
        Renvoie l'indice du point de grille le plus proche de t.
        """
        return int(np.argmin(np.abs(self.times - t)))

    def contains(self, t: float) -> bool:
        """True si t ∈ [0, T] (inclus)"""
        return (t >= -1e-14) and (t <= self.T + 1e-14)
