"""
Plain-vanilla fixed-for-float IRS (single currency), priced under HW1F++.

Sign convention:
- direction = "payer_fix": pay fixed, receive float → V = Float - K * Annuity
- direction = "receiver_fix": receive fixed, pay float → V = K * Annuity - Float
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np

from .schedule import Schedule
from ..rates.zc_pricer import ZCAnalyticHW


Direction = Literal["payer_fix", "receiver_fix"]


@dataclass
class Swap:
    notional: float
    direction: Direction
    coupon: float            # K (decimal per year, e.g., 0.025 for 2.5%)
    schedule: Schedule

    # -------- Helpers on remaining legs --------------------------------------

    def _annuity(self, t: float, r_t: float, zc: ZCAnalyticHW) -> float:
        """
        A(t) = sum_{j} α_j * P(t, T_j) over remaining payments T_j > t.
        """
        T, A = self.schedule.remaining_after(t)
        if T.size == 0:
            return 0.0
        P = zc.P_vector(t, r_t, T)
        return float(np.dot(self.schedule.accruals[: P.size], P))

    def _float_leg(self, t: float, r_t: float, zc: ZCAnalyticHW) -> float:
        """
        Value of remaining floating leg at time t (ignores fixing lag/couru),
        standard result under absence of spread: Float(t) = P(t, T_start_next) - P(t, T_maturity).
        Here T_start_next is the last paid date (≤ t) replaced by t; with our simple schedule we
        approximate by P(t, t_first_remaining - dt) ≈ 1 (if t < first payment),
        and more generally we use the standard simplification:

        If t is between two coupon dates and no spread, the PV of float leg equals
        1 - P(t, T_last) only at inception. For general t, the exact expression uses the
        current floating coupon accrual. For EPE/ENE monthly buckets, the common clean
        approximation is:
            Float(t) ≈ P(t, T_prev) - P(t, T_last)
        where T_prev is the previous coupon date (or t_start).
        We implement a robust version:
            - if t < first payment: Float ≈ 1 - P(t, T_last)
            - else: use previous schedule date as T_prev.
        """
        pay_times = self.schedule.pay_times
        if pay_times.size == 0:
            return 0.0

        T_last = float(pay_times[-1])
        P_last = zc.P(t, r_t, T_last)

        # Find previous coupon date before t
        prev_idx = np.searchsorted(pay_times, t, side="right") - 1
        if prev_idx < 0:
            # before first coupon → approximation 1 - P(t, T_last)
            T_prev = t  # P(t,t)=1
            P_prev = 1.0
        else:
            T_prev = float(pay_times[prev_idx])
            # if T_prev <= t, use discount from t to T_prev (should be <1 but close if near)
            P_prev = zc.P(t, r_t, T_prev)

        return float(P_prev - P_last)

    # -------- Public API ------------------------------------------------------

    def par_rate(self, t: float, r_t: float, zc: ZCAnalyticHW) -> float:
        """
        K_par(t) = Float(t) / Annuity(t), provided Annuity(t) > 0.
        """
        A = self._annuity(t, r_t, zc)
        if A <= 0.0:
            return 0.0
        F = self._float_leg(t, r_t, zc)
        return F / A

    def mtm(self, t: float, r_t: float, zc: ZCAnalyticHW) -> float:
        """
        V(t) = N * [ Float(t) - K * Annuity(t) ]   if payer_fix
             = N * [ K * Annuity(t) - Float(t) ]   if receiver_fix
        """
        A = self._annuity(t, r_t, zc)
        F = self._float_leg(t, r_t, zc)
        if self.direction == "payer_fix":
            v = F - self.coupon * A
        elif self.direction == "receiver_fix":
            v = self.coupon * A - F
        else:
            raise ValueError("Swap.direction must be 'payer_fix' or 'receiver_fix'")
        return self.notional * v

    # Convenience: vectorized MTM across many (t, r_t) pairs (optional)
    def mtm_vector(self, t: float, r_vec: np.ndarray, zc: ZCAnalyticHW) -> np.ndarray:
        """
        Compute MTM for a vector of short rates r_vec at the same time t.
        """
        return np.array([self.mtm(t, r, zc) for r in r_vec], dtype=float)

# ---- ADD: roll/age a swap to a new as-of (t_star) --------------------------

from .schedule import ExplicitSchedule

def roll_swap(swap: "Swap", t_star: float) -> "Swap":
    """
    Revaloriser le MÊME swap à une nouvelle date as-of t_star:
    on conserve notional/direction/coupon, et on "déduis" t_star
    aux dates de paiement futures (T_j > t_star).
    """
    pt = np.asarray(swap.schedule.pay_times, dtype=float)
    mask = pt > (t_star + 1e-14)
    pt_new = pt[mask] - t_star
    sched_new = ExplicitSchedule.from_pay_times(pt_new)
    return Swap(
        notional=swap.notional,
        direction=swap.direction,
        coupon=swap.coupon,
        schedule=sched_new,
    )
