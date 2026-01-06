"""
Entités crédit (contrepartie, banque) 

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .log_ou_intensity import LogOUIntensity
from ..products.swap import Swap


@dataclass
class CreditParams:
    """Conteneur de paramètres du modèle log-OU pour l'intensité λ(t)."""
    kappa_lambda: float   # vitesse de retour à la moyenne
    sigma_lambda: float   # volatilité
    theta_lambda: float   # niveau de long-terme (sur x=log(λ))
    x0: float             # état initial x(0)=log(λ(0))


def build_credit_params_from_spread(
    spread0: float, LGD: float, kappa_lambda: float, sigma_lambda: float
) -> CreditParams:
    """
    Déduit (theta, x0) cohérents avec un spread initial.

    Hypothèse simplifiée : lambda_bar = spread0 / LGD (niveau moyen d'intensité visé).
    """
    if LGD <= 0 or LGD > 1:
        raise ValueError("LGD must be in (0,1]")
    if spread0 < 0:
        raise ValueError("spread0 must be >= 0")

    lambda_bar = spread0 / LGD
    theta = LogOUIntensity.theta_from_lambda_bar(lambda_bar, kappa_lambda, sigma_lambda)
    x0 = float(np.log(lambda_bar))  

    return CreditParams(
        kappa_lambda=kappa_lambda,
        sigma_lambda=sigma_lambda,
        theta_lambda=theta,
        x0=x0,
    )


@dataclass
class Counterparty:
    """
    Contrepartie : paramètres crédit + produit (swap) qui génère l'exposition.
    """
    cid: str
    LGD: float
    spread0: float              

    # paramètres du modèle log-OU (x=log(λ))
    kappa_lambda: float
    sigma_lambda: float
    theta_lambda: float
    x0: float

    swap: Swap                  # instrument d'exposition (EPE/ENE)

    @classmethod
    def from_spread(
        cls,
        cid: str,
        LGD: float,
        spread0: float,
        kappa_lambda: float,
        sigma_lambda: float,
        swap: Swap,
    ) -> "Counterparty":
        """
        Constructeur "user-friendly" : on passe (LGD, spread0, kappa, sigma)
        et on calcule (theta, x0) automatiquement.
        """
        cp = build_credit_params_from_spread(spread0, LGD, kappa_lambda, sigma_lambda)
        return cls(
            cid=cid,
            LGD=LGD,
            spread0=spread0,
            kappa_lambda=cp.kappa_lambda,
            sigma_lambda=cp.sigma_lambda,
            theta_lambda=cp.theta_lambda,
            x0=cp.x0,
            swap=swap,
        )

    def make_model(self) -> LogOUIntensity:
        """Instancie le modèle log-OU d'intensité associé à la contrepartie."""
        return LogOUIntensity(
            kappa=self.kappa_lambda,
            sigma=self.sigma_lambda,
            theta=self.theta_lambda,
            x0=self.x0,
        )


@dataclass
class Bank:
    """
    Banque : même logique que Counterparty mais sans produit associé.
    Sert typiquement pour simuler l'intensité propre (DVA).
    """
    LGD: float
    spread0: float
    kappa_lambda: float
    sigma_lambda: float
    theta_lambda: float
    x0: float

    @classmethod
    def from_spread(
        cls, LGD: float, spread0: float, kappa_lambda: float, sigma_lambda: float
    ) -> "Bank":
        """Constructeur pratique : calcule (theta, x0) à partir de (LGD, spread0, kappa, sigma)."""
        cp = build_credit_params_from_spread(spread0, LGD, kappa_lambda, sigma_lambda)
        return cls(
            LGD=LGD,
            spread0=spread0,
            kappa_lambda=cp.kappa_lambda,
            sigma_lambda=cp.sigma_lambda,
            theta_lambda=cp.theta_lambda,
            x0=cp.x0,
        )

    def make_model(self) -> LogOUIntensity:
        """Instancie le modèle log-OU d'intensité pour la banque."""
        return LogOUIntensity(
            kappa=self.kappa_lambda,
            sigma=self.sigma_lambda,
            theta=self.theta_lambda,
            x0=self.x0,
        )
