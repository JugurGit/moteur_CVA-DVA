"""
Log-OU intensity model for default risk:

    dx_t = kappa*(theta - x_t) dt + sigma dZ_t,   lambda_t = exp(x_t) > 0

We simulate x on a uniform grid using the *exact Gaussian transition* of OU,
then map to lambda, integrate by trapezoid to get survival S and marginal PD.

Shapes:
- returns lambda_, S, PD with shapes (N, K+1), (N, K+1), (N, K+1)
  where PD[:,0] = 0 and for k>=1, PD[:,k] = S[:,k-1] - S[:,k].
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..core.timegrid import TimeGrid


@dataclass
class LogOUIntensity:
    kappa: float      # speed of mean reversion
    sigma: float      # vol of x = log(lambda)
    theta: float      # long-run mean of x
    x0: float         # initial x(0) = log(lambda_0)

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("LogOUIntensity: kappa must be > 0")
        if self.sigma < 0:
            raise ValueError("LogOUIntensity: sigma must be >= 0")

    @staticmethod
    def theta_from_lambda_bar(lambda_bar: float, kappa: float, sigma: float) -> float:
        """
        Calibrate theta so that E[lambda_t] (long-run) ≈ lambda_bar:
            E[lambda]_{∞} = exp( theta + sigma^2 / (4 kappa) )  =>  theta = log(lambda_bar) - sigma^2/(4 kappa)
        """
        if lambda_bar <= 0:
            raise ValueError("lambda_bar must be > 0")
        return float(np.log(lambda_bar) - (sigma * sigma) / (4.0 * kappa))

    def simulate(self, N: int, grid: TimeGrid, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate lambda, survival S, and marginal PD on the provided grid.

        Returns
        -------
        lambda_ : (N, K+1)
        S       : (N, K+1), with S[:,0] = 1
        PD      : (N, K+1), with PD[:,0] = 0 and PD[:,k] = S[:,k-1] - S[:,k]
        """
        a, sig = self.kappa, self.sigma
        times = grid.times
        K = grid.K

        x = np.empty((N, K + 1), dtype=float)
        lam = np.empty_like(x)
        S = np.empty_like(x)
        PD = np.zeros_like(x)

        # init
        x[:, 0] = self.x0
        lam[:, 0] = np.exp(x[:, 0])
        S[:, 0] = 1.0

        # precompute OU variance for each step (uniform grid -> constant)
        dt = times[1] - times[0]  # uniform by construction
        e2a = np.exp(-2.0 * a * dt)
        ead = np.exp(-a * dt)
        var_step = (sig * sig) * (1.0 - e2a) / (2.0 * a)  # Var[x_{k+1} | x_k]
        std_step = np.sqrt(var_step)

        # cumulative integral A_k = ∫_0^{t_k} lambda_s ds (trapz incremental)
        A = np.zeros(N, dtype=float)

        for k in range(K):
            # OU exact transition for x
            mean_next = self.theta + (x[:, k] - self.theta) * ead
            Z = rng.standard_normal(size=N)
            x[:, k + 1] = mean_next + std_step * Z

            # map to lambda
            lam[:, k + 1] = np.exp(x[:, k + 1])

            # trapezoid increment for ∫ lambda ds over [t_k, t_{k+1}]
            A += 0.5 * (lam[:, k] + lam[:, k + 1]) * dt
            S[:, k + 1] = np.exp(-A)

            # marginal PD on (t_k, t_{k+1}]
            PD[:, k + 1] = S[:, k] - S[:, k + 1]

        return lam, S, PD
