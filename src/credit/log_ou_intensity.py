"""
Modèle d’intensité log-OU pour le risque de défaut :

    dx_t = kappa*(theta - x_t) dt + sigma dZ_t,   lambda_t = exp(x_t) > 0

On simule x sur une grille uniforme via la *transition gaussienne exacte* de l’OU,
puis on passe à lambda, et on intègre (trapèzes) pour obtenir :
- la survie S(t)
- la probabilité marginale de défaut PD sur chaque pas

Formes (shapes) :
- on renvoie lambda_, S, PD de tailles (N, K+1), (N, K+1), (N, K+1)
  avec PD[:,0] = 0 et pour k>=1 : PD[:,k] = S[:,k-1] - S[:,k].
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..core.timegrid import TimeGrid


@dataclass
class LogOUIntensity:
    kappa: float      # vitesse de retour à la moyenne
    sigma: float      # volatilité de x = log(lambda)
    theta: float      # moyenne de long-terme de x
    x0: float         # état initial x(0) = log(lambda_0)

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("LogOUIntensity: kappa must be > 0")
        if self.sigma < 0:
            raise ValueError("LogOUIntensity: sigma must be >= 0")

    @staticmethod
    def theta_from_lambda_bar(lambda_bar: float, kappa: float, sigma: float) -> float:
        """
        Calibre theta pour que E[lambda_t] (en régime stationnaire) ≈ lambda_bar :
            E[lambda]_{∞} = exp( theta + sigma^2 / (4 kappa) )
        donc :
            theta = log(lambda_bar) - sigma^2/(4 kappa)
        """
        if lambda_bar <= 0:
            raise ValueError("lambda_bar must be > 0")
        return float(np.log(lambda_bar) - (sigma * sigma) / (4.0 * kappa))

    def simulate(
        self, N: int, grid: TimeGrid, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simule lambda(t), la survie S(t) et la PD marginale sur la grille donnée.

        Renvoie
        -------
        lambda_ : (N, K+1)
        S       : (N, K+1), avec S[:,0] = 1
        PD      : (N, K+1), avec PD[:,0] = 0 et PD[:,k] = S[:,k-1] - S[:,k]
        """
        a, sig = self.kappa, self.sigma
        times = grid.times
        K = grid.K

        x = np.empty((N, K + 1), dtype=float)
        lam = np.empty_like(x)
        S = np.empty_like(x)
        PD = np.zeros_like(x)

        # initialisation
        x[:, 0] = self.x0
        lam[:, 0] = np.exp(x[:, 0])
        S[:, 0] = 1.0

        # pré-calcul : variance de transition OU sur un pas (grille uniforme -> constant)
        dt = times[1] - times[0]  # uniforme par construction
        e2a = np.exp(-2.0 * a * dt)
        ead = np.exp(-a * dt)
        var_step = (sig * sig) * (1.0 - e2a) / (2.0 * a)  # Var[x_{k+1} | x_k]
        std_step = np.sqrt(var_step)

        # intégrale cumulée A_k = ∫_0^{t_k} lambda_s ds (mise à jour par incrément trapèzes)
        A = np.zeros(N, dtype=float)

        for k in range(K):
            # transition exacte OU pour x
            mean_next = self.theta + (x[:, k] - self.theta) * ead
            Z = rng.standard_normal(size=N)
            x[:, k + 1] = mean_next + std_step * Z

            # passage à l'intensité lambda = exp(x)
            lam[:, k + 1] = np.exp(x[:, k + 1])

            # incrément trapèze pour ∫ lambda ds sur [t_k, t_{k+1}]
            A += 0.5 * (lam[:, k] + lam[:, k + 1]) * dt
            S[:, k + 1] = np.exp(-A)

            # PD marginale sur (t_k, t_{k+1}]
            PD[:, k + 1] = S[:, k] - S[:, k + 1]

        return lam, S, PD
