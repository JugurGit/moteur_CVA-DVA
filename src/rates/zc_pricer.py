"""
Zero-coupon pricer under HW1F++:

P(t,T) = A(t,T) * exp( -B(t,T) * r_t ),
with
  B(t,T) = (1 - e^{-kappa (T - t)}) / kappa,
  ln A(t,T) = - ∫_t^T theta(s) * (1 - e^{-kappa (T - s)}) ds  +  0.5 * Var(∫_t^T r_u du),

and for an OU short rate,
  Var(∫_t^T r_u du) = (sigma^2 / (2 kappa^3)) * ( 2 kappa Δ + 4 e^{-kappa Δ} - e^{-2 kappa Δ} - 3 ), Δ = T - t.

We compute the integral with a small adaptive trapezoidal rule (accurate and robust).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Callable

import numpy as np

from .hw1f import HW1FModel
from .termstructure.base_curve import TermStructure


@dataclass
class ZCAnalyticHW:
    hw: HW1FModel
    ts: TermStructure   # kept in case you want quick DF(0,T), not strictly required

    # ---- Core affine functions ----------------------------------------------

    def B(self, t: float, T: float) -> float:
        a = self.hw.kappa
        if T < t:
            raise ValueError("B(t,T): requires T >= t")
        Δ = T - t
        if a == 0:
            return Δ  # limit case
        return (1.0 - np.exp(-a * Δ)) / a

    def var_integrated_ou(self, delta: float) -> float:
        """Var( ∫_0^Δ r_u du ) for an OU short-rate's *noise* part; depends only on Δ."""
        a, sig = self.hw.kappa, self.hw.sigma
        if delta <= 0.0:
            return 0.0
        e1 = np.exp(-a * delta)
        e2 = np.exp(-2.0 * a * delta)
        return (sig * sig) / (2.0 * a**3) * (2.0 * a * delta + 4.0 * e1 - e2 - 3.0)

    def _integral_theta_weighted(self, t: float, T: float) -> float:
        """
        I(t,T) = ∫_t^T theta(s) * (1 - e^{-kappa (T - s)}) ds
        Composite trapèze avec pas adaptatif en fonction de Δ = T - t.
        """
        if self.hw.theta_fn is None:
            raise RuntimeError("ZCAnalyticHW: theta_fn not set on HW model.")
        if T <= t:
            return 0.0

        a = self.hw.kappa
        Δ = T - t
        # plus Δ est grand, plus on raffine :  ~80 sous-pas par année (ajuste si besoin)
        n = int(max(64, np.ceil(80.0 * Δ)))
        grid = np.linspace(t, T, n + 1)
        weights = 1.0 - np.exp(-a * (T - grid))
        theta_vals = np.array([self.hw.theta_fn(s) for s in grid], dtype=float)
        return float(np.trapz(theta_vals * weights, grid))


    def A(self, t: float, T: float) -> float:
        if T < t:
            raise ValueError("A(t,T): requires T >= t")
        Δ = T - t
        I = self._integral_theta_weighted(t, T)
        var_I = self.var_integrated_ou(Δ)
        lnA = - I + 0.5 * var_I
        return float(np.exp(lnA))

    # ---- Bond price ----------------------------------------------------------

    def P(self, t: float, r_t: float, T: float) -> float:
        """Zero-coupon price P(t,T) under HW1F++."""
        # SPECIAL CASE: exact recollage at t=0
        if abs(t) < 1e-14:
            return float(self.ts.discount_factor(T))
        return self.A(t, T) * np.exp(-self.B(t, T) * r_t)


    def P_vector(self, t: float, r_t: float, T_list: Iterable[float]) -> np.ndarray:
        T_arr = np.asarray(list(T_list), dtype=float)
        A_vals = np.array([self.A(t, T) for T in T_arr], dtype=float)
        B_vals = np.array([self.B(t, T) for T in T_arr], dtype=float)
        return A_vals * np.exp(-B_vals * r_t)
