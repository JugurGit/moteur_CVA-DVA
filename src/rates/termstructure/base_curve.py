"""
Term structure base interface.

We work in *years*. Implementations must provide:
- zero_rate(T): y(0,T) in decimal (e.g., 0.025 for 2.5%).
- discount_factor(T): DF(0,T) = exp(-y(0,T)*T).
- inst_forward(t): instantaneous forward f(0,t).

Notes
-----
- All inputs are floats in years, T>=0.
- At T=0, we define DF(0)=1, and use right limits for zero/forward if needed.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math


class TermStructure(ABC):
    @abstractmethod
    def zero_rate(self, T: float) -> float:
        """Zero-coupon continuously-compounded yield y(0,T)."""
        raise NotImplementedError

    @abstractmethod
    def discount_factor(self, T: float) -> float:
        """DF(0,T) = exp(-y(0,T)*T)."""
        raise NotImplementedError

    @abstractmethod
    def inst_forward(self, t: float) -> float:
        """Instantaneous forward rate f(0,t)."""
        raise NotImplementedError

    # ---- small helpers (optional default impls) ------------------------------

    def df_from_zero(self, T: float) -> float:
        """Default DF from zero rate (continuous compounding)."""
        if T < 0:
            raise ValueError("T must be >= 0")
        if T == 0.0:
            return 1.0
        y = self.zero_rate(T)
        return math.exp(-y * T)
