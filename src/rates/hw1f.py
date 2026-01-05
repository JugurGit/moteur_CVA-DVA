"""
Hull–White 1F '++' model with time-dependent theta(t), fitted to a given
initial term structure through the standard HW++ relation:

    theta(t) = f(0,t) + (1/kappa) * d/dt f(0,t) + (sigma^2 / (2*kappa^2)) * (1 - e^{-2*kappa t})

We provide:
- fit_theta_to_curve(ts, grid) -> callable theta(t)
- simulate_rates(...) : simulate r_t with time-varying mean using a stable OU step
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
    # theta_fn will be set after fit to curve; if None, assume flat theta=const
    theta_fn: Callable[[float], float] | None = None

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("HW1FModel: kappa must be > 0")
        if self.sigma < 0:
            raise ValueError("HW1FModel: sigma must be >= 0")

    # -------- Fit theta(t) to an initial term structure ----------------------

    def fit_theta_to_curve(self, ts: TermStructure, grid: TimeGrid) -> Callable[[float], float]:
        """
        Build θ(t) from the instantaneous forward f(0,t) of the provided term structure,
        using the standard HW++ identity:

            theta(t) = f(0,t) + (1/kappa) * df/dt + (sigma^2 / (2*kappa^2)) * (1 - e^{-2*kappa t})

        We compute df/dt with a robust centered finite difference and clamp near t=0.
        Returns a callable theta(t) to be used both in pricing and in simulation.
        """
        a = self.kappa
        sig = self.sigma

        def df_dt(t: float) -> float:
            # Pas adaptatif : plus petit près de 0, un peu plus grand ensuite
            # On borne par le bas pour éviter h=0 en machine.
            base = 1e-6
            h = max(base, 5e-5 * max(1.0, t))
            t_minus = max(0.0, t - h)
            t_plus = t + h
            f_minus = ts.inst_forward(t_minus)
            f_plus = ts.inst_forward(t_plus)
            denom = t_plus - t_minus
            return (f_plus - f_minus) / denom



        def theta(t: float) -> float:
            f = ts.inst_forward(t)
            slope = df_dt(t)
            adj = (sig * sig) / (2.0 * a * a) * (1.0 - np.exp(-2.0 * a * t))
            return f + (1.0 / a) * slope + adj

        # store and return
        self.theta_fn = theta
        return theta

    # -------- Simulation of r_t with time-varying mean -----------------------

    def simulate_rates(self, N: int, grid: TimeGrid, r0: float, rng: np.random.Generator) -> np.ndarray:
        """
        Simulate r_t on the grid using the OU exact-variance step with time-varying mean θ(t).
        For theta varying over the step, we use the standard 'midpoint' convolution:

            r_{k+1} = r_k * e^{-a dt} + (1 - e^{-a dt}) * theta(t_k + dt/2)
                      + sigma * sqrt((1 - e^{-2 a dt}) / (2 a)) * Z

        This is accurate and stable for small dt and smooth theta(t).

        Returns: array (N, K+1) with r at each grid time (including t0).
        """
        if self.theta_fn is None:
            # fallback: flat theta equal to r0 (keeps r stationary if sigma=0)
            self.theta_fn = lambda t: r0

        a, sig = self.kappa, self.sigma
        times = grid.times
        K = grid.K

        r = np.empty((N, K + 1), dtype=float)
        r[:, 0] = r0

        for k in range(K):
            dt = times[k + 1] - times[k]
            ead = np.exp(-a * dt)
            mean_conv = (1.0 - ead) * self.theta_fn(times[k] + 0.5 * dt)
            vol_step = sig * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))
            Z = rng.standard_normal(size=N)
            r[:, k + 1] = r[:, k] * ead + mean_conv + vol_step * Z

        return r
