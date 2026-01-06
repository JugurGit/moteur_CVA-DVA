"""
Construire les facteurs d’actualisation DF(0, t_k) sur une TimeGrid à partir d’une TermStructure.

On utilise la *même* structure de taux initiale que pour HW++ (ex : Nelson–Siegel),
afin de garantir la cohérence entre la courbe de départ et les simulations/pricers.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .termstructure.base_curve import TermStructure
from ..core.timegrid import TimeGrid


@dataclass
class DFCurveOnGrid:
    ts: TermStructure
    grid: TimeGrid

    def values(self) -> np.ndarray:
        """
        Renvoie un tableau de DF de taille (K+1,) :
            DF[k] = DF(0, t_k)
        où t_k sont les points de la grille (en années).
        """
        # Récupère les temps de la grille
        t = self.grid.times

        # Évalue la courbe de DF sur chaque point t_k
        df = np.array([self.ts.discount_factor(float(tt)) for tt in t], dtype=float)

        df[0] = 1.0

        for k in range(1, df.size):
            if df[k] > df[k - 1] + 1e-14:
                df[k] = df[k - 1]

        return df
