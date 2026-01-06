"""
Modèle Hull–White 1 facteur (HW1F) en version '++' avec theta(t) dépendant du temps,
ajusté à une courbe initiale via l’identité standard HW++ :

    theta(t) = f(0,t) + (1/kappa) * d/dt f(0,t)
               + (sigma^2 / (2*kappa^2)) * (1 - e^{-2*kappa t})

On fournit :
- fit_theta_to_curve(ts, grid) -> construit une fonction theta(t)
- simulate_rates(...) : simule r_t avec une moyenne qui varie dans le temps (step OU stable)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .termstructure.base_curve import TermStructure
from ..core.timegrid import TimeGrid


@dataclass
class HW1FModel:
    kappa: float
    sigma: float
    theta_fn: Callable[[float], float] | None = None

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("HW1FModel: kappa doit être > 0")
        if self.sigma < 0:
            raise ValueError("HW1FModel: sigma doit être >= 0")

    # -------- Ajustement de theta(t) à une courbe initiale -------------------

    def fit_theta_to_curve(self, ts: TermStructure, grid: TimeGrid) -> Callable[[float], float]:
        """
        Construit θ(t) à partir du forward instantané f(0,t) fourni par la TermStructure,
        via l'identité HW++ standard :

            theta(t) = f(0,t) + (1/kappa) * df/dt
                       + (sigma^2 / (2*kappa^2)) * (1 - e^{-2*kappa t})

        On calcule df/dt avec une différence finie centrée robuste,
        Renvoie une fonction theta(t) utilisable à la fois en pricing et en simulation.
        """
        a = self.kappa
        sig = self.sigma

        def df_dt(t: float) -> float:
            base = 1e-6
            h = max(base, 5e-5 * max(1.0, t))

            t_minus = max(0.0, t - h)
            t_plus = t + h

            # Différence finie (quasi) centrée : (f(t+h)-f(t-h))/(2h) si t-h>0
            f_minus = ts.inst_forward(t_minus)
            f_plus = ts.inst_forward(t_plus)
            denom = t_plus - t_minus
            return (f_plus - f_minus) / denom

        def theta(t: float) -> float:
            # f(0,t)
            f = ts.inst_forward(t)
            # df/dt
            slope = df_dt(t)
            # Terme d'ajustement dû à la convexité (dépend de sigma et kappa)
            adj = (sig * sig) / (2.0 * a * a) * (1.0 - np.exp(-2.0 * a * t))
            return f + (1.0 / a) * slope + adj

        # On stocke la fonction theta pour réutilisation (simulation/pricing)
        self.theta_fn = theta
        return theta

    # -------- Simulation de r_t avec une moyenne variable (theta(t)) ---------

    def simulate_rates(self, N: int, grid: TimeGrid, r0: float, rng: np.random.Generator) -> np.ndarray:
        """
        Simule r_t sur la grille via un pas de type OU avec variance exacte et moyenne variable θ(t).

        Comme θ(t) varie sur [t_k, t_{k+1}], on utilise une convolution "midpoint" standard :
            r_{k+1} = r_k * e^{-a dt} + (1 - e^{-a dt}) * theta(t_k + dt/2)
                      + sigma * sqrt((1 - e^{-2 a dt}) / (2 a)) * Z


        Retourne : tableau (N, K+1) contenant r à chaque point de grille (t0 inclus).
        """
        if self.theta_fn is None:
            self.theta_fn = lambda t: r0

        a, sig = self.kappa, self.sigma
        times = grid.times
        K = grid.K

        r = np.empty((N, K + 1), dtype=float)
        r[:, 0] = r0

        for k in range(K):
            dt = times[k + 1] - times[k]
            ead = np.exp(-a * dt)

            # Contribution de la moyenne sur le pas (approximation au milieu)
            mean_conv = (1.0 - ead) * self.theta_fn(times[k] + 0.5 * dt)

            # Volatilité du pas (variance exacte du OU)
            vol_step = sig * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))

            Z = rng.standard_normal(size=N)
            r[:, k + 1] = r[:, k] * ead + mean_conv + vol_step * Z

        return r
