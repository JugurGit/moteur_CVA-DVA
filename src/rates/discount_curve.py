# src/rates/discount_curve.py
from __future__ import annotations
import math
from typing import Iterable, Sequence, Tuple, Literal, Optional
import numpy as np

class DiscountCurve:
    """
    Courbe d'actualisation initiale DF(0,t).
    Modes:
      - type='flat' : zero rate constant z -> DF(0,t)=exp(-z t)
      - type='points' : liste (t_i, DF_i), interpolation log-linéaire sur DF
    """
    def __init__(
        self,
        mode: Literal["flat", "points"],
        flat_zero_rate: Optional[float] = None,
        points: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        self.mode = mode
        if mode == "flat":
            if flat_zero_rate is None:
                raise ValueError("flat_zero_rate required for flat curve")
            if flat_zero_rate < -0.005:  # tolérance petits taux négatifs
                raise ValueError("flat_zero_rate seems too negative")
            self.z = float(flat_zero_rate)
            self._t_knots = np.array([0.0])
            self._df_knots = np.array([1.0])
        elif mode == "points":
            if not points:
                raise ValueError("points required for points curve")
            pts = np.array(points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValueError("points must be sequence of (t, df)")
            # tri par maturité
            pts = pts[np.argsort(pts[:, 0])]
            if pts[0, 0] < 0:
                raise ValueError("negative maturities not allowed")
            # ajout (0,1) si absent
            if pts[0, 0] > 0.0:
                pts = np.vstack([[0.0, 1.0], pts])
            # contrôles DF
            if not np.all(pts[:, 1] > 0):
                raise ValueError("all discount factors must be > 0")
            self._t_knots = pts[:, 0]
            self._df_knots = pts[:, 1]
        else:
            raise ValueError("mode must be 'flat' or 'points'")

    def df0(self, t: float) -> float:
        """DF(0,t)."""
        if t < 0:
            raise ValueError("t must be >= 0")
        if t == 0:
            return 1.0
        if self.mode == "flat":
            return math.exp(-self.z * t)
        # points: interpolation log-linéaire
        t_knots = self._t_knots
        df_knots = self._df_knots
        if t >= t_knots[-1]:
            # extrapolation log-linéaire à partir du dernier segment
            t0, t1 = t_knots[-2], t_knots[-1]
            ln0, ln1 = math.log(df_knots[-2]), math.log(df_knots[-1])
            slope = (ln1 - ln0) / (t1 - t0) if t1 > t0 else 0.0
            ln_df = ln1 + slope * (t - t1)
            return math.exp(ln_df)
        # interpolation dans l'intervalle
        idx = np.searchsorted(t_knots, t)  # position d'insertion
        i0 = idx - 1
        i1 = idx
        t0, t1 = t_knots[i0], t_knots[i1]
        ln0, ln1 = math.log(df_knots[i0]), math.log(df_knots[i1])
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        ln_df = (1 - w) * ln0 + w * ln1
        return math.exp(ln_df)

    def zero_rate(self, t: float) -> float:
        """
        z(0,t) implicite tel que DF(0,t) = exp(-z t), pour t>0.
        Pour t=0, renvoie z plat si mode flat, sinon 0.
        """
        if t <= 0:
            return self.z if self.mode == "flat" else 0.0
        df = self.df0(t)
        # évite division par zéro si df≈1 et t très petit
        return -math.log(df) / t
