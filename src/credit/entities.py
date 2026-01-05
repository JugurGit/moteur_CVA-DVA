"""
Credit entities (Counterparty, Bank) holding LGD, spread0, and log-OU params.

We calibrate lambda_bar from spread0 and LGD, and compute theta accordingly:
    lambda_bar = spread0 / LGD
    theta = log(lambda_bar) - sigma^2/(4*kappa)

x0 is set to log(lambda_bar) by default (consistent initialization).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .log_ou_intensity import LogOUIntensity
from ..products.swap import Swap


@dataclass
class CreditParams:
    kappa_lambda: float
    sigma_lambda: float
    theta_lambda: float
    x0: float


def build_credit_params_from_spread(
    spread0: float, LGD: float, kappa_lambda: float, sigma_lambda: float
) -> CreditParams:
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
    cid: str
    LGD: float
    spread0: float              # in decimal per year (e.g., 0.015 for 150 bps)
    kappa_lambda: float
    sigma_lambda: float
    theta_lambda: float
    x0: float
    swap: Swap                  # product driving exposure

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
        return LogOUIntensity(
            kappa=self.kappa_lambda,
            sigma=self.sigma_lambda,
            theta=self.theta_lambda,
            x0=self.x0,
        )


@dataclass
class Bank:
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
        return LogOUIntensity(
            kappa=self.kappa_lambda,
            sigma=self.sigma_lambda,
            theta=self.theta_lambda,
            x0=self.x0,
        )
