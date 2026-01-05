"""
Build discount factors DF(0, t_k) on a TimeGrid from a TermStructure.

We use the *same* initial term structure as for HW++ (e.g., Nelson–Siegel).
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
        Return DF array of shape (K+1,), DF[k] = DF(0, t_k).
        """
        t = self.grid.times
        df = np.array([self.ts.discount_factor(float(tt)) for tt in t], dtype=float)

        # Safety guards: DF(0)=1, DF decreasing (up to tiny numerical noise)
        df[0] = 1.0
        # enforce weak monotonicity by clipping tiny up-ticks due to floating noise
        for k in range(1, df.size):
            if df[k] > df[k-1] + 1e-14:
                df[k] = df[k-1]

        return df
