"""
Exposure engine for vanilla IRS (no CSA).

Given:
- rate paths r[n,k] on grid t_k (HW1F++),
- a Swap (notional, direction, coupon, schedule),
- a ZCAnalyticHW pricer (providing A(t,T), B(t,T), P(t,T)=A*exp(-B*r_t)),

we compute scenario-wise MTM V_{n,k}, then EPE_k = mean(max(V,0)), ENE_k = mean(max(-V,0)).

Vectorization strategy
----------------------
For each grid time t_k, we precompute the affine bond coefficients (A_kj, B_kj)
for all remaining payment dates T_j > t_k, plus:
- (A_prev_k, B_prev_k) for the "previous" coupon date T_prev (or t_start),
- (A_last_k, B_last_k) for the final maturity T_last.

Then for all scenarios n:
    P_kj(n) = A_kj * exp(-B_kj * r[n,k])   # broadcasts over j
    Annuity_k(n) = sum_j alpha_j * P_kj(n)
    Float_k(n)   = P(t_k, T_prev_k) - P(t_k, T_last)

Finally:
    V_k(n) = N * [ Float_k(n) - K * Annuity_k(n) ]   (payer_fix)
           = N * [ K * Annuity_k(n) - Float_k(n) ]   (receiver_fix)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..core.timegrid import TimeGrid
from ..rates.zc_pricer import ZCAnalyticHW
from ..products.swap import Swap


@dataclass
class _CoeffAtTime:
    # For remaining cashflows at t_k:
    A_pay: np.ndarray   # shape (m_k,)   for T_j > t_k
    B_pay: np.ndarray   # shape (m_k,)
    alpha: np.ndarray   # shape (m_k,)   accruals corresponding to those T_j
    # For float leg:
    A_prev: float       # for T_prev (or t_start≈t with P=1)
    B_prev: float
    A_last: float       # for T_last
    B_last: float


class ExposureEngine:
    def __init__(self, zc: ZCAnalyticHW) -> None:
        self.zc = zc
        self._cache: Dict[Tuple[float, Tuple[float, ...]], _CoeffAtTime] = {}

    # ---------- Precompute A,B coeffs for speed (per t_k) ---------------------

    def _precompute_coeffs_for_time(self, t: float, swap: Swap) -> _CoeffAtTime:
        """
        Build the A,B arrays for remaining fixed-leg dates T_j > t and the scalars for
        the float leg using the robust approximation:
            Float(t) ≈ 1 - P(t, T_last)
        i.e. set P_prev(t) = 1.0  (A_prev=1, B_prev=0) for all t.

        This avoids calling A(t, T_prev) with T_prev < t (undefined in our pricer),
        and is accurate enough for monthly EPE/ENE buckets.
        """
        # Remaining fixed leg (T_j > t)
        T_pay, alpha = swap.schedule.remaining_after(t)
        if T_pay.size:
            A_pay = np.array([self.zc.A(t, T) for T in T_pay], dtype=float)
            B_pay = np.array([self.zc.B(t, T) for T in T_pay], dtype=float)
        else:
            A_pay = np.array([], dtype=float)
            B_pay = np.array([], dtype=float)

        # Float leg: use Float(t) ≈ 1 - P(t, T_last)
        pay_times = swap.schedule.pay_times
        if pay_times.size == 0:
            # No cashflows at all
            A_prev = 1.0; B_prev = 0.0
            A_last = 1.0; B_last = 0.0
        else:
            T_last = float(pay_times[-1])

            # IMPORTANT: if t is after maturity, exposure is 0 → set P_last = 1
            if T_last <= t + 1e-14:
                A_prev = 1.0; B_prev = 0.0
                A_last = 1.0; B_last = 0.0
            else:
                A_last = self.zc.A(t, T_last)
                B_last = self.zc.B(t, T_last)
                A_prev = 1.0
                B_prev = 0.0

        return _CoeffAtTime(
            A_pay=A_pay,
            B_pay=B_pay,
            alpha=alpha,
            A_prev=A_prev,
            B_prev=B_prev,
            A_last=A_last,
            B_last=B_last,
        )

    def _coeffs(self, t: float, swap: Swap) -> _CoeffAtTime:
        key = (t, tuple(swap.schedule.pay_times.tolist()))
        c = self._cache.get(key)
        if c is None:
            c = self._precompute_coeffs_for_time(t, swap)
            self._cache[key] = c
        return c

    # ---------- Core API: EPE/ENE from rate paths ----------------------------

    def epe_ene(self, rates: np.ndarray, grid: TimeGrid, swap: Swap) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute EPE/ENE on the grid for a single swap given simulated short-rate paths.

        Parameters
        ----------
        rates : (N, K+1) array of short rates r_{n,k}
        grid  : TimeGrid
        swap  : Swap

        Returns
        -------
        EPE : (K+1,) expected positive exposure per grid time
        ENE : (K+1,) expected negative exposure per grid time
        """
        rates = np.asarray(rates, dtype=float)
        N, Kp1 = rates.shape
        K = Kp1 - 1

        EPE = np.zeros(Kp1, dtype=float)
        ENE = np.zeros(Kp1, dtype=float)

        notional = swap.notional
        K_coupon = swap.coupon
        payer = (swap.direction == "payer_fix")

        times = grid.times

        for k in range(Kp1):
            t = float(times[k])

            # ---- SPECIAL CASE t=0: utiliser zc.P(0, r0, T) pour un recollage exact
            if k == 0:
                # Jambe fixe restante à t=0 (déterministe)
                T_pay, alpha = swap.schedule.remaining_after(t)
                if T_pay.size == 0:
                    annuity_n = np.zeros(N, dtype=float)
                else:
                    # P(0,T) identique pour tous les scénarios
                    P0 = np.array([self.zc.P(0.0, rates[0, 0], T) for T in T_pay], dtype=float)
                    A0 = float(np.dot(alpha, P0))
                    annuity_n = np.full(N, A0, dtype=float)

                # Jambe flottante ≈ 1 - P(0, T_last)
                pay_times = swap.schedule.pay_times
                if pay_times.size == 0:
                    float_n = np.zeros(N, dtype=float)
                else:
                    T_last = float(pay_times[-1])
                    P_last0 = self.zc.P(0.0, rates[0, 0], T_last)
                    float_n = np.full(N, 1.0 - P_last0, dtype=float)

                # MTM et expositions
                if payer:
                    v_n = notional * (float_n - K_coupon * annuity_n)
                else:
                    v_n = notional * (K_coupon * annuity_n - float_n)

                Epos = np.maximum(v_n, 0.0)
                Eneg = np.maximum(-v_n, 0.0)
                EPE[k] = float(Epos.mean())
                ENE[k] = float(Eneg.mean())
                continue  # prochain k

            # ---- Cas générique t>0 : version vectorisée avec A,B pré-calculés
            coeffs = self._coeffs(t, swap)

            # Jambe fixe (annuity)
            if coeffs.A_pay.size == 0:
                annuity_n = np.zeros(N, dtype=float)
            else:
                exp_term = np.exp(-np.outer(rates[:, k], coeffs.B_pay))  # (N, m_k)
                P_kj = coeffs.A_pay[None, :] * exp_term                  # (N, m_k)
                annuity_n = P_kj @ coeffs.alpha                          # (N,)

            # Jambe flottante ≈ 1 - P(t, T_last) (on a A_prev=1, B_prev=0 dans _precompute_)
            if coeffs.B_prev == 0.0:
                P_prev = np.ones(N, dtype=float)  # =1
            else:
                P_prev = coeffs.A_prev * np.exp(-coeffs.B_prev * rates[:, k])

            P_last = coeffs.A_last * np.exp(-coeffs.B_last * rates[:, k])
            float_n = P_prev - P_last

            # MTM et expositions
            if payer:
                v_n = notional * (float_n - K_coupon * annuity_n)
            else:
                v_n = notional * (K_coupon * annuity_n - float_n)

            Epos = np.maximum(v_n, 0.0)
            Eneg = np.maximum(-v_n, 0.0)
            EPE[k] = float(Epos.mean())
            ENE[k] = float(Eneg.mean())

        return EPE, ENE

    def mtm_paths(self, rates: np.ndarray, grid: TimeGrid, swap: Swap) -> np.ndarray:
        """
        Return V[n,k] for audit/plots (no netting, no CSA).
        """
        rates = np.asarray(rates, dtype=float)
        N, Kp1 = rates.shape
        times = grid.times

        V = np.zeros((N, Kp1), dtype=float)
        payer = (swap.direction == "payer_fix")
        notional = swap.notional
        K_coupon = swap.coupon

        for k in range(Kp1):
            t = float(times[k])

            if k == 0:
                # Jambe fixe à t=0 (déterministe)
                T_pay, alpha = swap.schedule.remaining_after(t)
                if T_pay.size == 0:
                    annuity_n = np.zeros(N, dtype=float)
                else:
                    P0 = np.array([self.zc.P(0.0, rates[0, 0], T) for T in T_pay], dtype=float)
                    A0 = float(np.dot(alpha, P0))
                    annuity_n = np.full(N, A0, dtype=float)

                # Jambe flottante ≈ 1 - P(0, T_last)
                pay_times = swap.schedule.pay_times
                if pay_times.size == 0:
                    float_n = np.zeros(N, dtype=float)
                else:
                    T_last = float(pay_times[-1])
                    P_last0 = self.zc.P(0.0, rates[0, 0], T_last)
                    float_n = np.full(N, 1.0 - P_last0, dtype=float)

                if payer:
                    V[:, k] = notional * (float_n - K_coupon * annuity_n)
                else:
                    V[:, k] = notional * (K_coupon * annuity_n - float_n)
                continue

            # t>0 : version vectorisée
            coeffs = self._coeffs(t, swap)

            if coeffs.A_pay.size == 0:
                annuity_n = np.zeros(N, dtype=float)
            else:
                P_kj = coeffs.A_pay[None, :] * np.exp(-np.outer(rates[:, k], coeffs.B_pay))
                annuity_n = P_kj @ coeffs.alpha

            if coeffs.B_prev == 0.0:
                P_prev = np.ones(N, dtype=float)  # =1
            else:
                P_prev = coeffs.A_prev * np.exp(-coeffs.B_prev * rates[:, k])

            P_last = coeffs.A_last * np.exp(-coeffs.B_last * rates[:, k])
            float_n = P_prev - P_last

            if payer:
                V[:, k] = notional * (float_n - K_coupon * annuity_n)
            else:
                V[:, k] = notional * (K_coupon * annuity_n - float_n)

        return V
