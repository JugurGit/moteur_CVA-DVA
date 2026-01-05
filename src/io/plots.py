"""
Plot helpers (PNG) for exposures, credit, and CVA/DVA legs.

Each function saves ONE figure per call and closes it (no memory leak).
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_exposure_plot(times: np.ndarray, EPE: np.ndarray, ENE: np.ndarray, outpath: str, title: str = "Exposure"):
    _ensure_dir(os.path.dirname(outpath))
    plt.figure(figsize=(8, 4.5))
    plt.plot(times, EPE, label="EPE")
    plt.plot(times, ENE, label="ENE")
    plt.xlabel("Time (years)")
    plt.ylabel("Exposure (currency)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_credit_plot(times: np.ndarray, PD: np.ndarray, S: np.ndarray, outpath: str, title: str = "Credit (bucket PD & Survival)"):
    _ensure_dir(os.path.dirname(outpath))
    plt.figure(figsize=(8, 4.5))
    # PD is bucketed at k; show from k=1
    if PD.shape[0] == times.shape[0]:
        t_pd = times
    else:
        t_pd = np.arange(PD.shape[0])
    plt.bar(t_pd[1:], PD[1:], width=(times[1]-times[0]) if times.size>1 else 0.02, alpha=0.4, label="PD (marginal)")
    plt.plot(times, S, label="Survival S(t)")
    plt.xlabel("Time (years)")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_xva_legs_plot(times: np.ndarray, CVA_leg: np.ndarray, DVA_leg: np.ndarray, outpath: str, title: str = "CVA/DVA legs"):
    _ensure_dir(os.path.dirname(outpath))
    plt.figure(figsize=(8, 4.5))
    plt.plot(times, CVA_leg, label="CVA_leg")
    plt.plot(times, DVA_leg, label="DVA_leg")
    plt.xlabel("Time (years)")
    plt.ylabel("Value (currency)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_totals_legs_plot(times: np.ndarray, CVA_legs_sum: np.ndarray, DVA_legs_sum: np.ndarray, outpath: str, title: str = "Portfolio CVA/DVA legs (sum)"):
    _ensure_dir(os.path.dirname(outpath))
    plt.figure(figsize=(8, 4.5))
    plt.plot(times, CVA_legs_sum, label="Σ CVA_leg")
    plt.plot(times, DVA_legs_sum, label="Σ DVA_leg")
    plt.xlabel("Time (years)")
    plt.ylabel("Value (currency)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
