"""
Moteur CVA / DVA (par buckets puis agrégation).

Entrées (pour une contrepartie i) :
- DF[k]                : facteurs d’actualisation DF(0, t_k), shape (K+1,)
- LGD_cpty             : float dans [0,1]
- LGD_bank             : float dans [0,1]
- EPE[k], ENE[k]       : expositions issues de ExposureEngine, shape (K+1,)
- PD_cpty[k]           : PD marginale *non conditionnelle* sur (t_{k-1}, t_k], PD[0]=0
- S_cpty[k]            : survie jusqu’à t_k, S[0]=1
- PD_bank[k]           : PD marginale *non conditionnelle* banque sur (t_{k-1}, t_k], PD_bank[0]=0

Formules (buckets discrets) :
  CVA_leg[k] = DF[k] * LGD_cpty * EPE[k] * PD_cpty[k]
  DVA_leg[k] = DF[k] * LGD_bank * ENE[k] * S_cpty[k-1] * PD_bank[k]
avec CVA_leg[0] = DVA_leg[0] = 0 par convention.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np


def _as_1d(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    if a.ndim != 1:
        raise ValueError(f"{name} doit être 1D")
    return a


@dataclass
class CVAResults:
    cva_leg: np.ndarray
    dva_leg: np.ndarray
    cva: float
    dva: float


class CVAEngine:
    @staticmethod
    def cva_by_bucket(DF, LGD_cpty: float, EPE, PD_cpty) -> np.ndarray:
        """Calcule la jambe CVA par bucket : DF * LGD_cpty * EPE * PD_cpty."""
        DF = _as_1d(DF, "DF")
        EPE = _as_1d(EPE, "EPE")
        PD  = _as_1d(PD_cpty, "PD_cpty")
        if not (DF.shape == EPE.shape == PD.shape):
            raise ValueError("DF, EPE, PD_cpty doivent être de la même forme (K+1,)")

        LGD_cpty = float(LGD_cpty)
        if not (0.0 <= LGD_cpty <= 1.0):
            raise ValueError("LGD_cpty doit être compris dans [0,1]")

        leg = DF * LGD_cpty * EPE * PD
        if leg.size > 0:
            leg[0] = 0.0  # convention : pas de contribution au temps 0
        return leg

    @staticmethod
    def dva_by_bucket(DF, LGD_bank: float, ENE, PD_bank, S_cpty, use_S_prev: bool = True) -> np.ndarray:
        """
        Calcule la jambe DVA par bucket : DF * LGD_bank * ENE * S_prev * PD_bank.
        Par défaut on utilise S_prev[k] = S[k-1] (survie “avant” le bucket).
        """
        DF = _as_1d(DF, "DF")
        ENE = _as_1d(ENE, "ENE")
        PD_b = _as_1d(PD_bank, "PD_bank")
        S    = _as_1d(S_cpty, "S_cpty")
        if not (DF.shape == ENE.shape == PD_b.shape == S.shape):
            raise ValueError("DF, ENE, PD_bank, S_cpty doivent être de la même forme (K+1,)")

        LGD_bank = float(LGD_bank)
        if not (0.0 <= LGD_bank <= 1.0):
            raise ValueError("LGD_bank doit être compris dans [0,1]")

        # S_prev sert à approximer la survie juste avant l’intervalle (t_{k-1}, t_k]
        if use_S_prev:
            S_prev = np.empty_like(S)
            S_prev[0] = 1.0
            if S.size > 1:
                S_prev[1:] = S[:-1]
        else:
            S_prev = S

        leg = DF * LGD_bank * ENE * S_prev * PD_b
        if leg.size > 0:
            leg[0] = 0.0  # convention : pas de contribution au temps 0
        return leg

    @staticmethod
    def aggregate(leg: np.ndarray) -> float:
        """Agrège une jambe (par buckets) en un scalaire, en clipant à 0 pour éviter les artefacts négatifs."""
        leg = _as_1d(leg, "leg")
        leg = np.maximum(leg, 0.0)
        return float(np.sum(leg))

    @staticmethod
    def compute_all(
        DF, LGD_cpty: float, LGD_bank: float,
        EPE, ENE, PD_cpty, PD_bank, S_cpty
    ) -> CVAResults:
        """Pipeline complet : jambes CVA/DVA + agrégats scalaires."""
        cva_leg = CVAEngine.cva_by_bucket(DF, LGD_cpty, EPE, PD_cpty)
        dva_leg = CVAEngine.dva_by_bucket(DF, LGD_bank, ENE, PD_bank, S_cpty, use_S_prev=True)
        return CVAResults(
            cva_leg=cva_leg,
            dva_leg=dva_leg,
            cva=CVAEngine.aggregate(cva_leg),
            dva=CVAEngine.aggregate(dva_leg),
        )
