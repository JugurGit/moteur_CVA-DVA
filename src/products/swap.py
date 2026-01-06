"""
Swap de taux vanilla fixe-contre-flottant (une seule devise), pricé sous HW1F++.

Convention de signe :
- direction = "payer_fix"    : paye le fixe, reçoit le flottant → V = Float - K * Annuity
- direction = "receiver_fix" : reçoit le fixe, paye le flottant → V = K * Annuity - Float
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
    coupon: float            # K (taux annuel en décimal, ex : 0.025 pour 2.5%)
    schedule: Schedule

    # -------- Helpers sur les jambes restantes --------------------------------

    def _annuity(self, t: float, r_t: float, zc: ZCAnalyticHW) -> float:
        """
        Annuity(t) = somme des accruals actualisés sur les paiements restants :
            A(t) = Σ_j α_j * P(t, T_j)   pour tous les T_j > t.
        """
        T, A = self.schedule.remaining_after(t)
        if T.size == 0:
            return 0.0
        P = zc.P_vector(t, r_t, T)
        return float(np.dot(self.schedule.accruals[: P.size], P))

    def _float_leg(self, t: float, r_t: float, zc: ZCAnalyticHW) -> float:
        """
        Valeur de la jambe flottante restante à la date t.

        Hypothèses / simplifications (pas de spread, pas de fixing lag, pas de coupon couru) :
        - Pour un calcul EPE/ENE en buckets mensuels, on utilise une approximation "clean" :
              Float(t) ≈ P(t, T_prev) - P(t, T_last)
          où T_prev est la date de coupon précédente (ou t_start).

        """
        pay_times = self.schedule.pay_times
        if pay_times.size == 0:
            return 0.0

        T_last = float(pay_times[-1])
        P_last = zc.P(t, r_t, T_last)

        # Cherche l'index de la dernière date de coupon <= t
        prev_idx = np.searchsorted(pay_times, t, side="right") - 1
        if prev_idx < 0:
            # Avant le 1er coupon → approximation 1 - P(t, T_last)
            T_prev = t  # P(t,t)=1
            P_prev = 1.0
        else:
            T_prev = float(pay_times[prev_idx])
            # Si T_prev <= t, la fonction P(t, T_prev) est bien définie
            P_prev = zc.P(t, r_t, T_prev)

        return float(P_prev - P_last)


    def par_rate(self, t: float, r_t: float, zc: ZCAnalyticHW) -> float:
        """
        Taux par (à la date t) :
            K_par(t) = Float(t) / Annuity(t)   (si Annuity(t) > 0).
        """
        A = self._annuity(t, r_t, zc)
        if A <= 0.0:
            return 0.0
        F = self._float_leg(t, r_t, zc)
        return F / A

    def mtm(self, t: float, r_t: float, zc: ZCAnalyticHW) -> float:
        """
        MTM du swap à la date t :
            V(t) = N * [ Float(t) - K * Annuity(t) ]   si payer_fix
                 = N * [ K * Annuity(t) - Float(t) ]   si receiver_fix
        """
        A = self._annuity(t, r_t, zc)
        F = self._float_leg(t, r_t, zc)

        if self.direction == "payer_fix":
            v = F - self.coupon * A
        elif self.direction == "receiver_fix":
            v = self.coupon * A - F
        else:
            raise ValueError("Swap.direction doit être 'payer_fix' ou 'receiver_fix'")

        return self.notional * v

    def mtm_vector(self, t: float, r_vec: np.ndarray, zc: ZCAnalyticHW) -> np.ndarray:
        """
        Calcule la MTM pour un vecteur de taux courts r_vec à la même date t.
        """
        return np.array([self.mtm(t, r, zc) for r in r_vec], dtype=float)


from .schedule import ExplicitSchedule

def roll_swap(swap: "Swap", t_star: float) -> "Swap":
    """
    Revalorise le *même* swap à une nouvelle date as-of t_star :
    - on conserve notional / direction / coupon
    - on ne garde que les dates de paiement futures (T_j > t_star)
    - on "recentre" le calendrier : T'_j = T_j - t_star (temps depuis le nouvel as-of)
    Cela permet par exemple de relancer une simulation à partir d’un run existant
    en repartant d’une date intermédiaire.
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
