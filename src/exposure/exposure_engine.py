"""
Moteur d’exposition pour un swap de taux vanilla (IRS) sans CSA.

Étant donnés :
- des trajectoires de taux courts r[n,k] sur la grille t_k (HW1F++),
- un Swap (notional, direction, coupon, calendrier),
- un pricer ZCAnalyticHW (qui fournit A(t,T), B(t,T), P(t,T)=A*exp(-B*r_t)),

on calcule la MTM scénario par scénario V_{n,k}, puis :
- EPE_k = moyenne(max(V,0))
- ENE_k = moyenne(max(-V,0))

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
    # Pour les cashflows fixes restants à t_k :
    A_pay: np.ndarray   # shape (m_k,)   pour T_j > t_k
    B_pay: np.ndarray   # shape (m_k,)
    alpha: np.ndarray   # shape (m_k,)   accruals associés à ces T_j
    # Pour la jambe flottante :
    A_prev: float       # pour T_prev (ou t_start≈t avec P=1)
    B_prev: float
    A_last: float       # pour T_last
    B_last: float


class ExposureEngine:
    def __init__(self, zc: ZCAnalyticHW) -> None:
        self.zc = zc
        self._cache: Dict[Tuple[float, Tuple[float, ...]], _CoeffAtTime] = {}

    # ---------- Pré-calcul des coeffs A,B pour accélérer (par t_k) ------------

    def _precompute_coeffs_for_time(self, t: float, swap: Swap) -> _CoeffAtTime:
        """
        Construit les tableaux A,B pour les dates fixes restantes T_j > t et les scalaires
        utiles à la jambe flottante en utilisant l’approximation robuste :
            Float(t) ≈ 1 - P(t, T_last)
        c.-à-d. on impose P_prev(t) = 1.0 (A_prev=1, B_prev=0) pour tout t.
        """
        # Jambe fixe restante (T_j > t)
        T_pay, alpha = swap.schedule.remaining_after(t)
        if T_pay.size:
            A_pay = np.array([self.zc.A(t, T) for T in T_pay], dtype=float)
            B_pay = np.array([self.zc.B(t, T) for T in T_pay], dtype=float)
        else:
            A_pay = np.array([], dtype=float)
            B_pay = np.array([], dtype=float)

        # Jambe flottante : on approxime Float(t) ≈ 1 - P(t, T_last)
        pay_times = swap.schedule.pay_times
        if pay_times.size == 0:
            A_prev = 1.0; B_prev = 0.0
            A_last = 1.0; B_last = 0.0
        else:
            T_last = float(pay_times[-1])

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

    # ---------- API principale : EPE/ENE à partir des paths de taux -----------

    def epe_ene(self, rates: np.ndarray, grid: TimeGrid, swap: Swap) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcule EPE/ENE sur la grille pour un seul swap, à partir de trajectoires
        simulées de taux court.

        Paramètres
        ----------
        rates : (N, K+1) tableau des taux courts r_{n,k}
        grid  : TimeGrid
        swap  : Swap

        Renvoie
        -------
        EPE : (K+1,) exposition positive moyenne à chaque date de grille
        ENE : (K+1,) exposition négative moyenne à chaque date de grille
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

            # ---- CAS PARTICULIER t=0 : utiliser zc.P(0, r0, T) pour un recollage exact
            if k == 0:
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

            if coeffs.A_pay.size == 0:
                annuity_n = np.zeros(N, dtype=float)
            else:
                exp_term = np.exp(-np.outer(rates[:, k], coeffs.B_pay))  # (N, m_k)
                P_kj = coeffs.A_pay[None, :] * exp_term                  # (N, m_k)
                annuity_n = P_kj @ coeffs.alpha                          # (N,)

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
        Renvoie V[n,k] pour audit / plots (pas de netting, pas de CSA).
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
                T_pay, alpha = swap.schedule.remaining_after(t)
                if T_pay.size == 0:
                    annuity_n = np.zeros(N, dtype=float)
                else:
                    P0 = np.array([self.zc.P(0.0, rates[0, 0], T) for T in T_pay], dtype=float)
                    A0 = float(np.dot(alpha, P0))
                    annuity_n = np.full(N, A0, dtype=float)

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
