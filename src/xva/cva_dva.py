"""
CVA / DVA engine (bucketed and aggregated).

Inputs (per counterparty i):
- DF[k]                : discount factors DF(0, t_k), shape (K+1,)
- LGD_cpty             : float in [0,1]
- LGD_bank             : float in [0,1]
- EPE[k], ENE[k]       : exposures from ExposureEngine, shape (K+1,)
- PD_cpty[k]           : *marginal non-conditional* PD on (t_{k-1}, t_k], PD[0]=0
- S_cpty[k]            : survival up to t_k, S[0]=1
- PD_bank[k]           : bank marginal non-conditional PD on (t_{k-1}, t_k], PD_bank[0]=0

Formulas (discrete buckets):
  CVA_leg[k] = DF[k] * LGD_cpty  * EPE[k] * PD_cpty[k]
  DVA_leg[k] = DF[k] * LGD_bank  * ENE[k] * S_cpty[k-1] * PD_bank[k]
with CVA_leg[0] = DVA_leg[0] = 0 by convention.

Notes:
- Arrays are *not* discounted exposures; DF enters only in the CVA/DVA legs.
- We accept any (K+1,) shape and enforce k=0 legs to 0.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np


def _as_1d(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return a


@dataclass
class CVAResults:
    cva_leg: np.ndarray   # (K+1,)
    dva_leg: np.ndarray   # (K+1,)
    cva: float
    dva: float


class CVAEngine:
    @staticmethod
    def cva_by_bucket(DF, LGD_cpty: float, EPE, PD_cpty) -> np.ndarray:
        DF = _as_1d(DF, "DF")
        EPE = _as_1d(EPE, "EPE")
        PD  = _as_1d(PD_cpty, "PD_cpty")
        if not (DF.shape == EPE.shape == PD.shape):
            raise ValueError("DF, EPE, PD_cpty must have the same shape (K+1,)")

        LGD_cpty = float(LGD_cpty)
        if not (0.0 <= LGD_cpty <= 1.0):
            raise ValueError("LGD_cpty must be in [0,1]")

        leg = DF * LGD_cpty * EPE * PD
        # convention: k=0 leg = 0 (PD[0] must be 0 but we enforce anyway)
        if leg.size > 0:
            leg[0] = 0.0
        return leg

    @staticmethod
    def dva_by_bucket(DF, LGD_bank: float, ENE, PD_bank, S_cpty, use_S_prev: bool = True) -> np.ndarray:
        DF = _as_1d(DF, "DF")
        ENE = _as_1d(ENE, "ENE")
        PD_b = _as_1d(PD_bank, "PD_bank")
        S    = _as_1d(S_cpty, "S_cpty")
        if not (DF.shape == ENE.shape == PD_b.shape == S.shape):
            raise ValueError("DF, ENE, PD_bank, S_cpty must have same shape (K+1,)")

        LGD_bank = float(LGD_bank)
        if not (0.0 <= LGD_bank <= 1.0):
            raise ValueError("LGD_bank must be in [0,1]")

        # S_prev[k] = S[k-1], with S_prev[0]=1 by convention
        if use_S_prev:
            S_prev = np.empty_like(S)
            S_prev[0] = 1.0
            if S.size > 1:
                S_prev[1:] = S[:-1]
        else:
            # allow caller to pass already-shifted survival in S
            S_prev = S

        leg = DF * LGD_bank * ENE * S_prev * PD_b
        if leg.size > 0:
            leg[0] = 0.0
        return leg

    @staticmethod
    def aggregate(leg: np.ndarray) -> float:
        leg = _as_1d(leg, "leg")
        # ensure non-negative up to tiny num noise
        leg = np.maximum(leg, 0.0)
        return float(np.sum(leg))

    @staticmethod
    def compute_all(
        DF, LGD_cpty: float, LGD_bank: float,
        EPE, ENE, PD_cpty, PD_bank, S_cpty
    ) -> CVAResults:
        cva_leg = CVAEngine.cva_by_bucket(DF, LGD_cpty, EPE, PD_cpty)
        dva_leg = CVAEngine.dva_by_bucket(DF, LGD_bank, ENE, PD_bank, S_cpty, use_S_prev=True)
        return CVAResults(
            cva_leg=cva_leg,
            dva_leg=dva_leg,
            cva=CVAEngine.aggregate(cva_leg),
            dva=CVAEngine.aggregate(dva_leg),
        )
