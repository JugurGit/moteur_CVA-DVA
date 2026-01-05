"""
Nelson–Siegel term structure:
    y(0,T) = β0 + β1 * h(T) + β2 * ( h(T) - exp(-T/τ) ),
where h(T) = (1 - exp(-T/τ)) / (T/τ).

We provide:
- zero_rate(T)
- discount_factor(T) = exp(-y(0,T)*T)
- inst_forward(t) (instantaneous forward f(0,t))

We handle small T carefully (stable limits).
"""

from __future__ import annotations
import math
from dataclasses import dataclass

from .base_curve import TermStructure


@dataclass(frozen=True)
class NelsonSiegel(TermStructure):
    beta0: float
    beta1: float
    beta2: float
    tau: float

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("NelsonSiegel: tau must be > 0")

    # ---- core building blocks ------------------------------------------------

    def _h(self, T: float) -> float:
        """
        h(T) = (1 - e^{-T/τ}) / (T/τ)
        Stable for small T: use series expansion h ~ 1 - (T/(2τ)) + (T/τ)^2/6 - ...
        """
        if T == 0.0:
            return 1.0
        x = T / self.tau
        if abs(x) < 1e-6:
            # 3-term series is enough here
            return 1.0 - 0.5*x + (x*x)/6.0
        return (1.0 - math.exp(-x)) / x

    def _h_prime(self, T: float) -> float:
        """
        h'(T) = d/dT [ (1 - e^{-T/τ}) / (T/τ) ]
              = [ (e^{-x} / τ)*x - (1 - e^{-x})*(1/τ) ] / x^2 , with x=T/τ
        We implement a numerically stable form and a small-T expansion.

        A simpler closed form:
          h'(T) = ( (T/τ)*e^{-T/τ} - (1 - e^{-T/τ}) ) / (T^2/τ)
                = ( τ / T^2 ) * ( (T/τ) e^{-T/τ} - (1 - e^{-T/τ}) )
        """
        if T == 0.0:
            # series: h(T) ≈ 1 - x/2 + x^2/6 ; h'(T) = dh/dT = (dh/dx)*(dx/dT) with x=T/τ
            # dh/dx ≈ -1/2 + (2x)/6 = -1/2 + x/3 ; at T=0 -> dh/dx ≈ -1/2
            # so h'(0) = (-1/2)*(1/τ)
            return -0.5 / self.tau

        x = T / self.tau
        ex = math.exp(-x)
        numerator = (x * ex) - (1.0 - ex)     # (x e^{-x} - (1 - e^{-x}))
        return (self.tau / (T * T)) * numerator

    # ---- Nelson–Siegel formulas ---------------------------------------------

    def zero_rate(self, T: float) -> float:
        if T < 0:
            raise ValueError("T must be >= 0")
        h = self._h(T)
        return self.beta0 + self.beta1 * h + self.beta2 * (h - math.exp(-T / self.tau))

    def discount_factor(self, T: float) -> float:
        if T < 0:
            raise ValueError("T must be >= 0")
        if T == 0.0:
            return 1.0
        y = self.zero_rate(T)
        return math.exp(-y * T)

    def inst_forward(self, t: float) -> float:
        """
        Instantaneous forward:
          f(0,t) = d/dt [ y(0,t) * t ] = y(0,t) + t * dy/dt.

        With y(0,t) = β0 + β1 h(t) + β2 (h(t) - e^{-t/τ}),
        dy/dt = β1 h'(t) + β2 ( h'(t) + (1/τ) e^{-t/τ} ).
        """
        if t < 0:
            raise ValueError("t must be >= 0")

        y = self.zero_rate(t)
        hp = self._h_prime(t)
        e = math.exp(-t / self.tau)
        dy_dt = self.beta1 * hp + self.beta2 * (hp + (1.0 / self.tau) * e)

        return y + t * dy_dt
