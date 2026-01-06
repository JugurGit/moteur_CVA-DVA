# src/xva/shapley_explain.py
from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, List, Tuple

import numpy as np


def _as_1d(a, name: str) -> np.ndarray:
    """Normalise un input en array 1D float."""
    a = np.asarray(a, dtype=float).ravel()
    if a.ndim != 1:
        raise ValueError(f"{name} doit être 1D")
    return a


def _shift_survival(S_cpty: np.ndarray) -> np.ndarray:
    """
    Construit S_prev[k] = S[k-1] (avec S_prev[0]=1).
    Utile pour la DVA : on pèse la PD banque sur l’intervalle (t_{k-1}, t_k]
    par la survie de la contrepartie "juste avant" le bucket.
    """
    S = _as_1d(S_cpty, "S_cpty")
    S_prev = np.empty_like(S)
    S_prev[0] = 1.0
    if S.size > 1:
        S_prev[1:] = S[:-1]
    return S_prev


def cva_leg(DF, LGD_cpty: float, EPE, PD_cpty) -> np.ndarray:
    """Jambe CVA par bucket : DF * LGD_cpty * EPE * PD_cpty (avec leg[0]=0)."""
    DF = _as_1d(DF, "DF")
    EPE = _as_1d(EPE, "EPE")
    PD = _as_1d(PD_cpty, "PD_cpty")
    if not (DF.shape == EPE.shape == PD.shape):
        raise ValueError("DF, EPE, PD_cpty doivent avoir la même forme")
    leg = DF * float(LGD_cpty) * EPE * PD
    if leg.size:
        leg[0] = 0.0  # convention : pas de bucket au temps 0
    return leg


def dva_leg(DF, LGD_bank: float, ENE, PD_bank, S_cpty) -> np.ndarray:
    """Jambe DVA par bucket : DF * LGD_bank * ENE * S_prev * PD_bank (avec leg[0]=0)."""
    DF = _as_1d(DF, "DF")
    ENE = _as_1d(ENE, "ENE")
    PDb = _as_1d(PD_bank, "PD_bank")
    S_prev = _shift_survival(S_cpty)
    if not (DF.shape == ENE.shape == PDb.shape == S_prev.shape):
        raise ValueError("DF, ENE, PD_bank, S_cpty doivent avoir la même forme")
    leg = DF * float(LGD_bank) * ENE * S_prev * PDb
    if leg.size:
        leg[0] = 0.0  # convention : pas de bucket au temps 0
    return leg


def shapley_vector(
    base_inputs: Dict[str, np.ndarray],
    new_inputs: Dict[str, np.ndarray],
    feature_names: List[str],
    value_fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Décomposition de Shapley (vectorielle) entre un état "base" et un état "new".
    - value_fn renvoie un vecteur (K+1,) (ex: jambe CVA par bucket).
    - On moyenne la contribution marginale de chaque feature sur toutes les permutations.
    """
    if set(feature_names) != set(base_inputs.keys()) or set(feature_names) != set(new_inputs.keys()):
        raise ValueError("Doit avoir mêmes features dans base et new")

    # Valeur de référence et valeur finale
    v0 = _as_1d(value_fn(dict(base_inputs)), "v0")
    v1 = _as_1d(value_fn(dict(new_inputs)), "v1")
    if v0.shape != v1.shape:
        raise ValueError("value_fn(base) et value_fn(new) doivent avoir la même forme")
    delta = v1 - v0  # variation totale à expliquer

    # Contributions Shapley : même shape que v0/v1 (décomposition bucket par bucket)
    contribs = {f: np.zeros_like(v0) for f in feature_names}

    perms = list(itertools.permutations(feature_names))
    for perm in perms:
        state = dict(base_inputs)               # état courant (on part du base)
        v_prev = _as_1d(value_fn(state), "v_prev")
        for f in perm:
            state[f] = new_inputs[f]
            v_curr = _as_1d(value_fn(state), "v_curr")
            contribs[f] += (v_curr - v_prev)
            v_prev = v_curr

    m = float(math.factorial(len(feature_names)))
    for f in feature_names:
        contribs[f] /= m

    return contribs, delta


def shapley_cva_legs(
    DF0, EPE0, PD0,
    DF1, EPE1, PD1,
    LGD_cpty: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Shapley sur la jambe CVA : features = (DF, EPE, PD_cpty)."""
    feats = ["DF", "EPE", "PD_cpty"]
    base = {"DF": _as_1d(DF0, "DF0"), "EPE": _as_1d(EPE0, "EPE0"), "PD_cpty": _as_1d(PD0, "PD0")}
    new  = {"DF": _as_1d(DF1, "DF1"), "EPE": _as_1d(EPE1, "EPE1"), "PD_cpty": _as_1d(PD1, "PD1")}

    def vf(inp: Dict[str, np.ndarray]) -> np.ndarray:
        return cva_leg(inp["DF"], LGD_cpty, inp["EPE"], inp["PD_cpty"])

    return shapley_vector(base, new, feats, vf)


def shapley_dva_legs(
    DF0, ENE0, PD_bank0, S_cpty0,
    DF1, ENE1, PD_bank1, S_cpty1,
    LGD_bank: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Shapley sur la jambe DVA : features = (DF, ENE, PD_bank, S_cpty)."""
    feats = ["DF", "ENE", "PD_bank", "S_cpty"]
    base = {
        "DF": _as_1d(DF0, "DF0"),
        "ENE": _as_1d(ENE0, "ENE0"),
        "PD_bank": _as_1d(PD_bank0, "PD_bank0"),
        "S_cpty": _as_1d(S_cpty0, "S_cpty0"),
    }
    new = {
        "DF": _as_1d(DF1, "DF1"),
        "ENE": _as_1d(ENE1, "ENE1"),
        "PD_bank": _as_1d(PD_bank1, "PD_bank1"),
        "S_cpty": _as_1d(S_cpty1, "S_cpty1"),
    }

    def vf(inp: Dict[str, np.ndarray]) -> np.ndarray:
        return dva_leg(inp["DF"], LGD_bank, inp["ENE"], inp["PD_bank"], inp["S_cpty"])

    return shapley_vector(base, new, feats, vf)
