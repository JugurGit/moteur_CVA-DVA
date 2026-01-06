"""
Pricer de zéro-coupon sous HW1F++ :

P(t,T) = A(t,T) * exp( -B(t,T) * r_t ),

avec
  B(t,T) = (1 - e^{-kappa (T - t)}) / kappa,
  ln A(t,T) = - ∫_t^T theta(s) * (1 - e^{-kappa (T - s)}) ds  +  0.5 * Var(∫_t^T r_u du),

et pour un taux court de type OU,
  Var(∫_t^T r_u du) = (sigma^2 / (2 kappa^3)) * ( 2 kappa Δ + 4 e^{-kappa Δ} - e^{-2 kappa Δ} - 3 ),
  où Δ = T - t.

On calcule l’intégrale avec une règle des trapèzes (pas adaptatif), pour rester précis et robuste.
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
    ts: TermStructure  

    # ---- Fonctions affines de base ------------------------------------------

    def B(self, t: float, T: float) -> float:
        a = self.hw.kappa
        if T < t:
            raise ValueError("B(t,T): requires T >= t")
        Δ = T - t
        if a == 0:
            return Δ  
        return (1.0 - np.exp(-a * Δ)) / a

    def var_integrated_ou(self, delta: float) -> float:
        """Variance de ∫_0^Δ r_u du pour la partie *bruit* d’un OU ; dépend uniquement de Δ."""
        a, sig = self.hw.kappa, self.hw.sigma
        if delta <= 0.0:
            return 0.0
        e1 = np.exp(-a * delta)
        e2 = np.exp(-2.0 * a * delta)
        return (sig * sig) / (2.0 * a**3) * (2.0 * a * delta + 4.0 * e1 - e2 - 3.0)

    def _integral_theta_weighted(self, t: float, T: float) -> float:
        """
        I(t,T) = ∫_t^T theta(s) * (1 - e^{-kappa (T - s)}) ds

        Intégrale "theta pondérée" qui intervient dans ln A(t,T).
        On l’évalue par une règle des trapèzes avec un nombre de sous-pas
        qui augmente avec Δ = T - t.
        """
        if self.hw.theta_fn is None:
            raise RuntimeError("ZCAnalyticHW: theta_fn n'est pas dans HW model.")
        if T <= t:
            return 0.0

        a = self.hw.kappa
        Δ = T - t

        n = int(max(64, np.ceil(80.0 * Δ)))
        grid = np.linspace(t, T, n + 1)

        # Poids (1 - exp(-kappa (T - s))) évalués sur la grille
        weights = 1.0 - np.exp(-a * (T - grid))

        # Theta évalué sur la grille
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

    # ---- Prix du bond --------------------------------------------------------

    def P(self, t: float, r_t: float, T: float) -> float:
        """Prix du zéro-coupon P(t,T) sous HW1F++."""
        # CAS PARTICULIER : recollage exact à t=0 via la courbe initiale
        if abs(t) < 1e-14:
            return float(self.ts.discount_factor(T))
        return self.A(t, T) * np.exp(-self.B(t, T) * r_t)

    def P_vector(self, t: float, r_t: float, T_list: Iterable[float]) -> np.ndarray:
        T_arr = np.asarray(list(T_list), dtype=float)
        A_vals = np.array([self.A(t, T) for T in T_arr], dtype=float)
        B_vals = np.array([self.B(t, T) for T in T_arr], dtype=float)
        return A_vals * np.exp(-B_vals * r_t)
