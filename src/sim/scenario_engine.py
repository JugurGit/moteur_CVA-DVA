"""
Simulateur de portefeuille (sans WWR, sans dépendances entre risques) :
- On simule une seule fois les trajectoires de taux HW1F++
- Pour la banque : on simule un crédit Log-OU et on moyenne PD/S sur les scénarios
- Pour chaque contrepartie :
    * on simule un crédit Log-OU (indépendant), on moyenne PD/S
    * on calcule EPE/ENE à partir des trajectoires de taux
    * on calcule les jambes CVA/DVA et les agrégats

Retourne des dictionnaires structurés prêts pour export.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np

from ..core.timegrid import TimeGrid
from ..rates.hw1f import HW1FModel
from ..rates.zc_pricer import ZCAnalyticHW
from ..rates.df_curve import DFCurveOnGrid
from ..products.swap import Swap
from ..credit.entities import Counterparty, Bank
from ..credit.log_ou_intensity import LogOUIntensity
from ..exposure.exposure_engine import ExposureEngine
from ..xva.cva_dva import CVAEngine


@dataclass
class Simulator:
    grid: TimeGrid
    rate_model: HW1FModel
    zc_pricer: ZCAnalyticHW
    bank: Bank
    rng: np.random.Generator
    df_curve_on_grid: DFCurveOnGrid

    # caches internes (pour éviter de re-simuler inutilement)
    _rates_paths: np.ndarray | None = None          
    _bank_PD: np.ndarray | None = None             
    _bank_S: np.ndarray | None = None               


    def _ensure_rate_paths(self, N: int, r0: float) -> np.ndarray:
        """Garantit que les chemins de taux (N scénarios) sont simulés et en cache."""
        if self._rates_paths is None or self._rates_paths.shape[0] != N:
            self._rates_paths = self.rate_model.simulate_rates(N, self.grid, r0, self.rng)
        return self._rates_paths

    def _simulate_bank_credit(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Renvoie (PD_bank_mean, S_bank_mean), chacun de shape (K+1,).
        Les grandeurs sont moyennées sur les N scénarios.
        """
        if self._bank_PD is not None and self._bank_S is not None:
            return self._bank_PD, self._bank_S

        lou_b = self.bank.make_model()
        lam_b, S_b, PD_b = lou_b.simulate(N, self.grid, self.rng)
        self._bank_PD = PD_b.mean(axis=0)
        self._bank_S  = S_b.mean(axis=0)
        return self._bank_PD, self._bank_S

    # ---------- API publique --------------------------------------------------

    def run_for_counterparty(self, cpty: Counterparty, N: int) -> Dict[str, Any]:
        """
        Exécute tout le pipeline pour une contrepartie.

        Retourne un dict avec :
          - 'cid', 'LGD_cpty', 'LGD_bank'
          - 'DF' (K+1,), 'EPE', 'ENE', 'PD_cpty', 'S_cpty', 'PD_bank'
          - 'CVA_leg', 'DVA_leg', 'CVA', 'DVA'
        """
        Kp1 = self.grid.K + 1

        # 1) Taux : une simulation HW++ (mise en cache, partagée entre contreparties)
        r0 = self.zc_pricer.ts.inst_forward(0.0)  # cohérent HW++ à t=0
        rates = self._ensure_rate_paths(N, r0)     # (N, K+1)

        # 2) Facteurs d’actualisation DF(0,t_k) sur la grille
        DF = self.df_curve_on_grid.values()        # (K+1,)

        # 3) Crédit contrepartie : Log-OU indépendant, puis moyenne sur scénarios
        lou_i = cpty.make_model()
        lam_i, S_i, PD_i = lou_i.simulate(N, self.grid, self.rng)
        PD_cpty = PD_i.mean(axis=0)                # (K+1,)
        S_cpty  = S_i.mean(axis=0)                 # (K+1,)

        # Crédit banque : simulé une fois (cache) puis réutilisé
        PD_bank, S_bank = self._simulate_bank_credit(N)   # (K+1,), (K+1,)

        # 4) Exposition : EPE/ENE à partir des chemins de taux + pricer ZC
        engine = ExposureEngine(self.zc_pricer)
        EPE, ENE = engine.epe_ene(rates, self.grid, cpty.swap)  # (K+1,)

        # 5) CVA / DVA : agrégation par buckets (marges non conditionnelles)
        cvares = CVAEngine.compute_all(
            DF=DF,
            LGD_cpty=cpty.LGD,
            LGD_bank=self.bank.LGD,
            EPE=EPE,
            ENE=ENE,
            PD_cpty=PD_cpty,     # PD marginale non conditionnelle
            PD_bank=PD_bank,     # PD marginale non conditionnelle
            S_cpty=S_cpty,       # survie cpty (le moteur décalera S_{k-1})
        )

        return {
            "cid": cpty.cid,
            "LGD_cpty": cpty.LGD,
            "LGD_bank": self.bank.LGD,
            "DF": DF, "EPE": EPE, "ENE": ENE,
            "PD_cpty": PD_cpty, "S_cpty": S_cpty, "PD_bank": PD_bank,
            "CVA_leg": cvares.cva_leg, "DVA_leg": cvares.dva_leg,
            "CVA": cvares.cva, "DVA": cvares.dva,
        }

    def run_portfolio(self, counterparties: List[Counterparty], N: int) -> Dict[str, Any]:
        """
        Exécute le pipeline pour toutes les contreparties et calcule des totaux agrégés.

        Retourne
        -------
        dict avec :
          - 'per_counterparty' : liste de dicts (voir run_for_counterparty)
          - 'totals' : {'CVA_total', 'DVA_total', 'CVA_legs_sum', 'DVA_legs_sum', 'DF'}
          - 'meta'   : {'N', 'Kp1', 'n_counterparties'}
        """
        if len(counterparties) == 0:
            raise ValueError("run_portfolio: liste des contreparties vide")

        # On force la génération des taux + crédit banque une seule fois
        r0 = self.zc_pricer.ts.inst_forward(0.0)
        self._ensure_rate_paths(N, r0)
        self._simulate_bank_credit(N)
        DF = self.df_curve_on_grid.values()
        Kp1 = DF.shape[0]

        per_cpty: List[Dict[str, Any]] = []
        cva_legs_sum = np.zeros(Kp1, dtype=float)
        dva_legs_sum = np.zeros(Kp1, dtype=float)
        CVA_total = 0.0
        DVA_total = 0.0

        for c in counterparties:
            res = self.run_for_counterparty(c, N)
            per_cpty.append(res)

            # Agrégation des jambes (bucket par bucket)
            cva_legs_sum += res["CVA_leg"]
            dva_legs_sum += res["DVA_leg"]

            # Agrégation des scalaires
            CVA_total += res["CVA"]
            DVA_total += res["DVA"]

        return {
            "per_counterparty": per_cpty,
            "totals": {
                "CVA_total": CVA_total,
                "DVA_total": DVA_total,
                "CVA_legs_sum": cva_legs_sum,
                "DVA_legs_sum": dva_legs_sum,
                "DF": DF,
            },
            "meta": {"N": N, "Kp1": Kp1, "n_counterparties": len(counterparties)},
        }
