"""
Utilitaires : RNG, helpers numériques, vérifications de cohérence.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


# ===== Aléatoire / Seeds ======================================================

def make_rng(seed: int | None = None) -> np.random.Generator:
    """
    Crée un générateur NumPy (Generator) avec une graine optionnelle.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


# ===== Numérique ==============================================================

def trapz_increment(y_prev: float, y_curr: float, dt: float) -> float:
    """
    Incrément trapèze pour approximer ∫ y dt sur un pas [t, t+dt].

    Retourne : 0.5 * (y_prev + y_curr) * dt
    """
    return 0.5 * (y_prev + y_curr) * dt


def trapz(x: np.ndarray, y: np.ndarray) -> float:
    """
    Règle des trapèzes pour approximer ∫ y(x) dx sur toute la grille.

    Wrapper léger autour de numpy.trapz, avec assertions sur forme/type.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("trapz attend des tableaux 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("trapz: x et y doivent avoir la même longueur")
    return float(np.trapz(y, x))


# ===== Contrôles / assertions =================================================

def assert_finite(name: str, arr: np.ndarray) -> None:
    """Lève une erreur si arr contient au moins un NaN/Inf."""
    arr = np.asarray(arr)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: contient NaN/Inf")


def assert_shape(name: str, arr: np.ndarray, expected: Tuple[int, ...]) -> None:
    """
    Vérifie la forme exacte (shape). 
    """
    arr = np.asarray(arr)
    if arr.ndim != len(expected):
        raise ValueError(f"{name}: attendu {len(expected)}D, obtenu {arr.ndim}D")
    for i, (got, exp) in enumerate(zip(arr.shape, expected)):
        if exp != -1 and got != exp:
            raise ValueError(f"{name}: axe {i} attendu {exp}, obtenu {got}")


def is_monotone_decreasing(x: Iterable[float]) -> bool:
    """True si x est non-croissant (strictement décroissant ou constant)."""
    prev = None
    for v in x:
        if prev is not None and v > prev + 1e-14:
            return False
        prev = v
    return True


# ===== Petits helpers =========================================================

@dataclass(frozen=True)
class Stats1D:
    mean: float
    std: float
    min: float
    max: float


def stats1d(a: np.ndarray) -> Stats1D:
    """Stats descriptives rapides (log/debug)."""
    a = np.asarray(a, dtype=float).ravel()
    return Stats1D(
        mean=float(np.mean(a)),
        std=float(np.std(a, ddof=0)),
        min=float(np.min(a)),
        max=float(np.max(a)),
    )
