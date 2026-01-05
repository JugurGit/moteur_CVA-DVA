"""
Main runner: build a 20-counterparty portfolio, simulate, and export CSVs.

Usage:
  python -m main --N 10000 --out data/run1
"""

from __future__ import annotations
import os, argparse, datetime as dt
import numpy as np

from src.core.timegrid import TimeGrid
from src.sim.rng import RNG
from src.rates.termstructure.nelson_siegel import NelsonSiegel
from src.rates.hw1f import HW1FModel
from src.rates.zc_pricer import ZCAnalyticHW
from src.rates.df_curve import DFCurveOnGrid
from src.products.schedule import Schedule
from src.products.swap import Swap
from src.credit.entities import Counterparty, Bank
from src.sim.scenario_engine import Simulator
from src.io.outputs import export_everything, write_meta_json
from src.io.plots import (
    save_exposure_plot, save_credit_plot, save_xva_legs_plot, save_totals_legs_plot
)
import csv
from src.products.swap import roll_swap
from src.xva.shapley_explain import shapley_cva_legs, shapley_dva_legs
from src.io.outputs import write_matrix_csv



def build_portfolio(ns, zc, grid, rng) -> list[Counterparty]:
    """
    Create 20 counterparties:
      - LGD ~ U[35%, 55%]
      - spread0 ~ U[120, 250] bps
      - random payer/receiver
      - coupon = par ± {0, ±12.5, ±25} bps
      - notionals random in {1, 2, 3} millions
    """
    sched = Schedule(0.0, 5.0, 2)  # 5Y semi-annual
    r0 = ns.inst_forward(0.0)
    tmp = Swap(notional=1_000_000, direction="payer_fix", coupon=0.0, schedule=sched)
    Kpar = tmp.par_rate(0.0, r0, zc)

    cptys: list[Counterparty] = []
    directions = ["payer_fix", "receiver_fix"]
    coupon_bps = np.array([0.0, +0.00125, -0.00125, +0.0025, -0.0025])  # 0, ±12.5, ±25 bps

    for i in range(1, 21):
        cid = f"CPTY_{i:02d}"
        notional = int(np.random.default_rng().choice([1_000_000, 2_000_000, 3_000_000]))
        direction = np.random.default_rng().choice(directions)

        coupon = float(Kpar + np.random.default_rng().choice(coupon_bps))
        swap = Swap(notional=notional, direction=direction, coupon=coupon, schedule=sched)

        LGD = float(np.random.default_rng().uniform(0.35, 0.55))
        spread0 = float(np.random.default_rng().uniform(0.012, 0.025))  # 120–250 bps

        # credit params (log-OU) — plausible IG/HY-ish
        kappa_l = 1.0 + float(np.random.default_rng().normal(0.0, 0.15))   # ~ N(1, 0.15)
        kappa_l = max(0.5, min(1.5, kappa_l))
        sigma_l = 0.35 + float(np.random.default_rng().normal(0.0, 0.05))  # ~ N(0.35, 0.05)
        sigma_l = max(0.20, min(0.55, sigma_l))

        cpty = Counterparty.from_spread(
            cid=cid,
            LGD=LGD,
            spread0=spread0,
            kappa_lambda=kappa_l,
            sigma_lambda=sigma_l,
            swap=swap,
        )
        cptys.append(cpty)
    return cptys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10000, help="Number of Monte Carlo scenarios")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: data/run_YYYYMMDD_HHMMSS)")
    parser.add_argument("--seed", type=int, default=12345, help="Global RNG seed")
    # dans main(): parser
    parser.add_argument("--plots", action="store_true", help="Save PNG plots for exposures/credit/CVA-DVA")
    args = parser.parse_args()

    # Output directory
    outdir = args.out
    if outdir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join("data", f"run_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    # RNG
    rng = RNG(seed=args.seed).gen

    # Grid & curve
    grid = TimeGrid(T=5.0, dt=1/12)  # 5Y monthly buckets
    ns = NelsonSiegel(beta0=0.02, beta1=-0.01, beta2=-0.02, tau=2.5)

    # HW++ fit
    hw = HW1FModel(kappa=0.60, sigma=0.01)
    hw.fit_theta_to_curve(ns, grid)
    zc = ZCAnalyticHW(hw, ns)

    # DF grid
    df_on_grid = DFCurveOnGrid(ts=ns, grid=grid)

    # Bank (LGD fixed 60%)
    bank = Bank.from_spread(LGD=0.60, spread0=0.0080, kappa_lambda=1.0, sigma_lambda=0.35)

    # Portfolio (20 cptys)
    cptys = build_portfolio(ns, zc, grid, rng)

    # Simulator
    sim = Simulator(grid=grid, rate_model=hw, zc_pricer=zc, bank=bank, rng=rng, df_curve_on_grid=df_on_grid)

    # Run
    portfolio_out = sim.run_portfolio(cptys, N=args.N)

    # Export
    export_everything(outdir, portfolio_out)

        # ==========================
    # 2nd snapshot: "Mar" as-of
    # ==========================
    t_star = 0.25  # 3 months
    k_star = grid.index_of_time(t_star)

    # DF(0, t*) from Jan snapshot, used to bring Mar values back to PV Jan
    DF_jan = df_on_grid.values()
    df_0_tstar = float(DF_jan[k_star])

    # --- (optional) define Mar market snapshot (here: same params; replace by Mar params if needed)
    ns_mar = NelsonSiegel(beta0=ns.beta0, beta1=ns.beta1, beta2=ns.beta2, tau=ns.tau)
    hw_mar = HW1FModel(kappa=hw.kappa, sigma=hw.sigma)
    hw_mar.fit_theta_to_curve(ns_mar, grid)
    zc_mar = ZCAnalyticHW(hw_mar, ns_mar)
    df_on_grid_mar = DFCurveOnGrid(ts=ns_mar, grid=grid)

    # Bank as-of Mar (here same; replace spread0 etc by Mar values)
    bank_mar = Bank.from_spread(LGD=bank.LGD, spread0=bank.spread0, kappa_lambda=bank.kappa_lambda, sigma_lambda=bank.sigma_lambda)

    # Roll the SAME swaps to as-of t_star, keep cpty identity/credit params (spread can be updated here too)
    cptys_mar = []
    for c in cptys:
        swap_rolled = roll_swap(c.swap, t_star)
        c_mar = Counterparty.from_spread(
            cid=c.cid,
            LGD=c.LGD,
            spread0=c.spread0,              # <-- change to Mar spread if you want credit move
            kappa_lambda=c.kappa_lambda,
            sigma_lambda=c.sigma_lambda,
            swap=swap_rolled,
        )
        cptys_mar.append(c_mar)

    # New simulator (fresh caches!)
    sim_mar = Simulator(
        grid=grid,
        rate_model=hw_mar,
        zc_pricer=zc_mar,
        bank=bank_mar,
        rng=np.random.default_rng(args.seed + 1),   # separate RNG stream for clarity
        df_curve_on_grid=df_on_grid_mar,
    )

    outdir_mar = os.path.join(os.path.dirname(outdir), os.path.basename(outdir) + "_MAR")
    os.makedirs(outdir_mar, exist_ok=True)

    portfolio_out_mar = sim_mar.run_portfolio(cptys_mar, N=args.N)
    export_everything(outdir_mar, portfolio_out_mar)

    # ==========================
    # Compare: PV Jan deltas
    # ==========================
    cva_jan = float(portfolio_out["totals"]["CVA_total"])
    dva_jan = float(portfolio_out["totals"]["DVA_total"])
    cva_mar = float(portfolio_out_mar["totals"]["CVA_total"])
    dva_mar = float(portfolio_out_mar["totals"]["DVA_total"])

    delta_cva_pvjan = df_0_tstar * cva_mar - cva_jan
    delta_dva_pvjan = df_0_tstar * dva_mar - dva_jan

    print("---- XVA snapshot compare (PV Jan) ----")
    print(f"DF_Jan(0,t*) = {df_0_tstar:.12g}   (t*={t_star}y, k*={k_star})")
    print(f"CVA_Jan      = {cva_jan:.12g}")
    print(f"CVA_Mar      = {cva_mar:.12g}   (as-of Mar)")
    print(f"ΔCVA (PVJan) = {delta_cva_pvjan:.12g}")
    print(f"DVA_Jan      = {dva_jan:.12g}")
    print(f"DVA_Mar      = {dva_mar:.12g}   (as-of Mar)")
    print(f"ΔDVA (PVJan) = {delta_dva_pvjan:.12g}")

    # Write a tiny CSV summary
    summary_path = os.path.join(outdir, "xva_compare_pvjan.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["t_star_years", f"{t_star:.12g}"])
        w.writerow(["DF_Jan_0_tstar", f"{df_0_tstar:.12g}"])
        w.writerow(["CVA_Jan", f"{cva_jan:.12g}"])
        w.writerow(["CVA_Mar", f"{cva_mar:.12g}"])
        w.writerow(["Delta_CVA_PVJan", f"{delta_cva_pvjan:.12g}"])
        w.writerow(["DVA_Jan", f"{dva_jan:.12g}"])
        w.writerow(["DVA_Mar", f"{dva_mar:.12g}"])
        w.writerow(["Delta_DVA_PVJan", f"{delta_dva_pvjan:.12g}"])
    print(f"✅ Compare CSV written: {summary_path}")

    # ==========================
    # Shapley explain (per cpty)
    # ==========================
    shap_dir = os.path.join(outdir, "shapley_per_counterparty")
    os.makedirs(shap_dir, exist_ok=True)

    jan_by_cid = {r["cid"]: r for r in portfolio_out["per_counterparty"]}
    mar_by_cid = {r["cid"]: r for r in portfolio_out_mar["per_counterparty"]}

    for cid, r0 in jan_by_cid.items():
        r1 = mar_by_cid[cid]

        # --- CVA: (DF, PD_cpty, EPE)
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
            os.path.join(shap_dir, f"shapley_cva_{cid}.csv"),
            headers=["k", "phi_DF", "phi_EPE", "phi_PD_cpty", "delta_leg", "check(phi_sum-delta)"],
            rows=rows_cva,
        )

        # --- DVA: (DF, ENE, PD_bank, S_cpty)
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
            os.path.join(shap_dir, f"shapley_dva_{cid}.csv"),
            headers=["k", "phi_DF", "phi_ENE", "phi_PD_bank", "phi_S_cpty", "delta_leg", "check(phi_sum-delta)"],
            rows=rows_dva,
        )

    print(f"✅ Shapley per-counterparty CSVs written under: {shap_dir}")

    # après export_everything(outdir, portfolio_out)
    if args.plots:
        times = grid.times
        figs_dir = os.path.join(outdir, "figs")
        os.makedirs(figs_dir, exist_ok=True)

        # Totaux (legs agrégés)
        totals = portfolio_out["totals"]
        save_totals_legs_plot(
            times=times,
            CVA_legs_sum=totals["CVA_legs_sum"],
            DVA_legs_sum=totals["DVA_legs_sum"],
            outpath=os.path.join(figs_dir, "portfolio_legs.png"),
            title="Portfolio CVA/DVA legs (sum)",
        )

        # Par contrepartie
        for res in portfolio_out["per_counterparty"]:
            cid = res["cid"]
            cdir = os.path.join(figs_dir, cid)
            os.makedirs(cdir, exist_ok=True)

            save_exposure_plot(
                times=times, EPE=res["EPE"], ENE=res["ENE"],
                outpath=os.path.join(cdir, f"{cid}_exposure.png"),
                title=f"{cid} — EPE/ENE"
            )
            save_credit_plot(
                times=times, PD=res["PD_cpty"], S=res["S_cpty"],
                outpath=os.path.join(cdir, f"{cid}_credit.png"),
                title=f"{cid} — Credit (PD buckets & Survival)"
            )
            save_xva_legs_plot(
                times=times, CVA_leg=res["CVA_leg"], DVA_leg=res["DVA_leg"],
                outpath=os.path.join(cdir, f"{cid}_xva_legs.png"),
                title=f"{cid} — CVA/DVA legs"
            )


    # Meta
    meta = {
        "N": args.N,
        "seed": args.seed,
        "grid": {"T": grid.T, "dt": grid.dt, "Kp1": grid.K + 1},
        "NS_params": {"beta0": ns.beta0, "beta1": ns.beta1, "beta2": ns.beta2, "tau": ns.tau},
        "HW1F": {"kappa": hw.kappa, "sigma": hw.sigma},
        "bank": {"LGD": bank.LGD, "spread0": bank.spread0, "kappa_lambda": bank.kappa_lambda, "sigma_lambda": bank.sigma_lambda},
        "notes": "No CSA, no netting, no dependence/WWR. Float leg approx 1 - P(t, T_last).",
    }
    write_meta_json(os.path.join(outdir, "run_meta.json"), meta)

    print(f"✅ Done. Files written under: {outdir}")


if __name__ == "__main__":
    main()
