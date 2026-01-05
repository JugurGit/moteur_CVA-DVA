# app_lib/runner.py
from __future__ import annotations
import pathlib
import time
from typing import List

import numpy as np

from src.core.timegrid import TimeGrid
from src.sim.rng import RNG
from src.rates.termstructure.nelson_siegel import NelsonSiegel
from src.rates.hw1f import HW1FModel
from src.rates.zc_pricer import ZCAnalyticHW
from src.rates.df_curve import DFCurveOnGrid
from src.products.schedule import Schedule
from src.products.swap import Swap, roll_swap
from src.credit.entities import Counterparty, Bank
from src.sim.scenario_engine import Simulator
from src.io.outputs import export_everything, write_meta_json, write_matrix_csv
from src.xva.shapley_explain import shapley_cva_legs, shapley_dva_legs
from src.io.plots import (
    save_exposure_plot, save_credit_plot, save_xva_legs_plot, save_totals_legs_plot
)

def build_portfolio_deterministic(
    ns: NelsonSiegel,
    zc: ZCAnalyticHW,
    grid: TimeGrid,
    rng: np.random.Generator,
    n_counterparties: int = 20,   # --- NEW
) -> List[Counterparty]:
    if n_counterparties <= 0:
        raise ValueError("n_counterparties must be >= 1")

    sched = Schedule(0.0, 5.0, 2)  # 5Y semi-annual
    r0 = ns.inst_forward(0.0)
    tmp = Swap(notional=1_000_000, direction="payer_fix", coupon=0.0, schedule=sched)
    Kpar = tmp.par_rate(0.0, r0, zc)

    directions = ["payer_fix", "receiver_fix"]
    coupon_bps = np.array([0.0, +0.00125, -0.00125, +0.0025, -0.0025])

    width = max(2, len(str(n_counterparties)))  # CPTY_01, ... et au-delà si >99

    cptys: List[Counterparty] = []
    for i in range(1, n_counterparties + 1):
        cid = f"CPTY_{i:0{width}d}"
        notional = int(rng.choice([1_000_000, 2_000_000, 3_000_000]))
        direction = str(rng.choice(directions))
        coupon = float(Kpar + rng.choice(coupon_bps))
        swap = Swap(notional=notional, direction=direction, coupon=coupon, schedule=sched)

        LGD = float(rng.uniform(0.35, 0.55))
        spread0 = float(rng.uniform(0.012, 0.025))

        kappa_l = float(1.0 + rng.normal(0.0, 0.15))
        kappa_l = max(0.5, min(1.5, kappa_l))
        sigma_l = float(0.35 + rng.normal(0.0, 0.05))
        sigma_l = max(0.20, min(0.55, sigma_l))

        cpty = Counterparty.from_spread(
            cid=cid, LGD=LGD, spread0=spread0,
            kappa_lambda=kappa_l, sigma_lambda=sigma_l,
            swap=swap
        )
        cptys.append(cpty)
    return cptys

def run_pipeline_and_export(
    N: int,
    seed: int,
    outdir: pathlib.Path,
    make_plots: bool,
    include_mar_and_shapley: bool,

    # --- NEW
    n_counterparties: int = 20,
) -> pathlib.Path:
    outdir.mkdir(parents=True, exist_ok=True)

    rng = RNG(seed=seed).gen

    grid = TimeGrid(T=5.0, dt=1/12)
    ns = NelsonSiegel(beta0=0.02, beta1=-0.01, beta2=-0.02, tau=2.5)
    hw = HW1FModel(kappa=0.60, sigma=0.01)
    hw.fit_theta_to_curve(ns, grid)
    zc = ZCAnalyticHW(hw, ns)

    df_on_grid = DFCurveOnGrid(ts=ns, grid=grid)
    bank = Bank.from_spread(LGD=0.60, spread0=0.0080, kappa_lambda=1.0, sigma_lambda=0.35)

    # --- NEW: portfolio paramétrable
    cptys = build_portfolio_deterministic(ns, zc, grid, rng, n_counterparties=n_counterparties)

    sim = Simulator(grid=grid, rate_model=hw, zc_pricer=zc, bank=bank, rng=rng, df_curve_on_grid=df_on_grid)
    portfolio_out = sim.run_portfolio(cptys, N=N)

    export_everything(str(outdir), portfolio_out)

    if make_plots:
        times = grid.times
        figs_dir = outdir / "figs"
        figs_dir.mkdir(exist_ok=True)

        totals = portfolio_out["totals"]
        save_totals_legs_plot(times, totals["CVA_legs_sum"], totals["DVA_legs_sum"], str(figs_dir / "portfolio_legs.png"))

        for res in portfolio_out["per_counterparty"]:
            cid = res["cid"]
            cdir = figs_dir / cid
            cdir.mkdir(exist_ok=True)

            save_exposure_plot(times, res["EPE"], res["ENE"], str(cdir / f"{cid}_exposure.png"))
            save_credit_plot(times, res["PD_cpty"], res["S_cpty"], str(cdir / f"{cid}_credit.png"))
            save_xva_legs_plot(times, res["CVA_leg"], res["DVA_leg"], str(cdir / f"{cid}_xva_legs.png"))

    # --- NEW: stocker n_counterparties dans run_meta.json
    meta = {
        "N": N,
        "seed": seed,
        "n_counterparties": int(n_counterparties),
        "grid": {"T": grid.T, "dt": grid.dt, "Kp1": grid.K + 1},
        "NS_params": {"beta0": ns.beta0, "beta1": ns.beta1, "beta2": ns.beta2, "tau": ns.tau},
        "HW1F": {"kappa": hw.kappa, "sigma": hw.sigma},
        "bank": {"LGD": bank.LGD, "spread0": bank.spread0, "kappa_lambda": bank.kappa_lambda, "sigma_lambda": bank.sigma_lambda},
        "notes": "No CSA, no netting, no dependence/WWR. Float leg approx 1 - P(t, T_last).",
    }
    write_meta_json(str(outdir / "run_meta.json"), meta)

    if include_mar_and_shapley:
        t_star = 0.25
        k_star = grid.index_of_time(t_star)
        DF_jan = df_on_grid.values()
        df_0_tstar = float(DF_jan[k_star])

        ns_mar = NelsonSiegel(beta0=ns.beta0, beta1=ns.beta1, beta2=ns.beta2, tau=ns.tau)
        hw_mar = HW1FModel(kappa=hw.kappa, sigma=hw.sigma)
        hw_mar.fit_theta_to_curve(ns_mar, grid)
        zc_mar = ZCAnalyticHW(hw_mar, ns_mar)
        df_on_grid_mar = DFCurveOnGrid(ts=ns_mar, grid=grid)

        bank_mar = Bank.from_spread(LGD=bank.LGD, spread0=bank.spread0, kappa_lambda=bank.kappa_lambda, sigma_lambda=bank.sigma_lambda)

        cptys_mar = []
        for c in cptys:
            swap_rolled = roll_swap(c.swap, t_star)
            c_mar = Counterparty.from_spread(
                cid=c.cid,
                LGD=c.LGD,
                spread0=c.spread0,
                kappa_lambda=c.kappa_lambda,
                sigma_lambda=c.sigma_lambda,
                swap=swap_rolled,
            )
            cptys_mar.append(c_mar)

        sim_mar = Simulator(
            grid=grid, rate_model=hw_mar, zc_pricer=zc_mar,
            bank=bank_mar,
            rng=np.random.default_rng(seed + 1),
            df_curve_on_grid=df_on_grid_mar,
        )

        outdir_mar = outdir.parent / f"{outdir.name}_MAR"
        outdir_mar.mkdir(exist_ok=True)
        portfolio_out_mar = sim_mar.run_portfolio(cptys_mar, N=N)
        export_everything(str(outdir_mar), portfolio_out_mar)

        cva_jan = float(portfolio_out["totals"]["CVA_total"])
        dva_jan = float(portfolio_out["totals"]["DVA_total"])
        cva_mar = float(portfolio_out_mar["totals"]["CVA_total"])
        dva_mar = float(portfolio_out_mar["totals"]["DVA_total"])

        delta_cva_pvjan = df_0_tstar * cva_mar - cva_jan
        delta_dva_pvjan = df_0_tstar * dva_mar - dva_jan

        rows = [
            ["t_star_years", f"{t_star:.12g}"],
            ["DF_Jan_0_tstar", f"{df_0_tstar:.12g}"],
            ["CVA_Jan", f"{cva_jan:.12g}"],
            ["CVA_Mar", f"{cva_mar:.12g}"],
            ["Delta_CVA_PVJan", f"{delta_cva_pvjan:.12g}"],
            ["DVA_Jan", f"{dva_jan:.12g}"],
            ["DVA_Mar", f"{dva_mar:.12g}"],
            ["Delta_DVA_PVJan", f"{delta_dva_pvjan:.12g}"],
        ]
        write_matrix_csv(str(outdir / "xva_compare_pvjan.csv"), headers=["metric", "value"], rows=rows)

        shap_dir = outdir / "shapley_per_counterparty"
        shap_dir.mkdir(exist_ok=True)

        jan_by_cid = {r["cid"]: r for r in portfolio_out["per_counterparty"]}
        mar_by_cid = {r["cid"]: r for r in portfolio_out_mar["per_counterparty"]}

        for cid, r0 in jan_by_cid.items():
            r1 = mar_by_cid[cid]

            cva_contrib, cva_delta = shapley_cva_legs(
                DF0=r0["DF"], EPE0=r0["EPE"], PD0=r0["PD_cpty"],
                DF1=r1["DF"], EPE1=r1["EPE"], PD1=r1["PD_cpty"],
                LGD_cpty=r0["LGD_cpty"],
            )
            cva_check = (cva_contrib["DF"] + cva_contrib["EPE"] + cva_contrib["PD_cpty"]) - cva_delta

            rows_cva = []
            Kp1 = len(cva_delta)
            for k in range(Kp1):
                rows_cva.append([
                    k,
                    f"{cva_contrib['DF'][k]:.12g}",
                    f"{cva_contrib['EPE'][k]:.12g}",
                    f"{cva_contrib['PD_cpty'][k]:.12g}",
                    f"{cva_delta[k]:.12g}",
                    f"{cva_check[k]:.12g}",
                ])
            rows_cva.append([
                "TOTAL",
                f"{cva_contrib['DF'].sum():.12g}",
                f"{cva_contrib['EPE'].sum():.12g}",
                f"{cva_contrib['PD_cpty'].sum():.12g}",
                f"{cva_delta.sum():.12g}",
                f"{cva_check.sum():.12g}",
            ])
            write_matrix_csv(
                str(shap_dir / f"shapley_cva_{cid}.csv"),
                headers=["k", "phi_DF", "phi_EPE", "phi_PD_cpty", "delta_leg", "check(phi_sum-delta)"],
                rows=rows_cva,
            )

            dva_contrib, dva_delta = shapley_dva_legs(
                DF0=r0["DF"], ENE0=r0["ENE"], PD_bank0=r0["PD_bank"], S_cpty0=r0["S_cpty"],
                DF1=r1["DF"], ENE1=r1["ENE"], PD_bank1=r1["PD_bank"], S_cpty1=r1["S_cpty"],
                LGD_bank=r0["LGD_bank"],
            )
            dva_check = (dva_contrib["DF"] + dva_contrib["ENE"] + dva_contrib["PD_bank"] + dva_contrib["S_cpty"]) - dva_delta

            rows_dva = []
            for k in range(Kp1):
                rows_dva.append([
                    k,
                    f"{dva_contrib['DF'][k]:.12g}",
                    f"{dva_contrib['ENE'][k]:.12g}",
                    f"{dva_contrib['PD_bank'][k]:.12g}",
                    f"{dva_contrib['S_cpty'][k]:.12g}",
                    f"{dva_delta[k]:.12g}",
                    f"{dva_check[k]:.12g}",
                ])
            rows_dva.append([
                "TOTAL",
                f"{dva_contrib['DF'].sum():.12g}",
                f"{dva_contrib['ENE'].sum():.12g}",
                f"{dva_contrib['PD_bank'].sum():.12g}",
                f"{dva_contrib['S_cpty'].sum():.12g}",
                f"{dva_delta.sum():.12g}",
                f"{dva_check.sum():.12g}",
            ])
            write_matrix_csv(
                str(shap_dir / f"shapley_dva_{cid}.csv"),
                headers=["k", "phi_DF", "phi_ENE", "phi_PD_bank", "phi_S_cpty", "delta_leg", "check(phi_sum-delta)"],
                rows=rows_dva,
            )

    return outdir

def default_run_name() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")
