import numpy as np

from src.core.timegrid import TimeGrid
from src.rates.termstructure.nelson_siegel import NelsonSiegel
from src.rates.hw1f import HW1FModel
from src.rates.zc_pricer import ZCAnalyticHW
from src.rates.df_curve import DFCurveOnGrid
from src.products.schedule import Schedule
from src.products.swap import Swap
from src.credit.entities import Counterparty, Bank
from src.sim.rng import RNG
from src.sim.scenario_engine import Simulator

# Courbe & HW++
ns = NelsonSiegel(beta0=0.02, beta1=-0.01, beta2=-0.02, tau=2.5)
grid = TimeGrid(T=5.0, dt=1/12)
hw = HW1FModel(kappa=0.60, sigma=0.01); hw.fit_theta_to_curve(ns, grid)
zc = ZCAnalyticHW(hw, ns)

# DF sur grille
df_on_grid = DFCurveOnGrid(ts=ns, grid=grid)

# Banque (LGD=60%, spread 80 bps)
bank = Bank.from_spread(LGD=0.60, spread0=0.0080, kappa_lambda=1.0, sigma_lambda=0.35)

# 2 contreparties pour test (payer/receiver, off-par ±25 bps)
sched = Schedule(0.0, 5.0, 2)
# par rate à t=0
r0 = ns.inst_forward(0.0)
swap_tmp = Swap(notional=1_000_000, direction="payer_fix", coupon=0.0, schedule=sched)
Kpar = swap_tmp.par_rate(0.0, r0, zc)

c1 = Counterparty.from_spread(
    cid="CPTY_01", LGD=0.45, spread0=0.0150, kappa_lambda=1.0, sigma_lambda=0.35,
    swap=Swap(notional=1_000_000, direction="payer_fix",    coupon=Kpar+0.0025, schedule=sched)
)
c2 = Counterparty.from_spread(
    cid="CPTY_02", LGD=0.40, spread0=0.0200, kappa_lambda=1.2, sigma_lambda=0.40,
    swap=Swap(notional=2_000_000, direction="receiver_fix", coupon=Kpar-0.0025, schedule=sched)
)

# RNG + Simulator
rng = RNG(seed=123).gen
sim = Simulator(grid=grid, rate_model=hw, zc_pricer=zc, bank=bank, rng=rng, df_curve_on_grid=df_on_grid)

out = sim.run_portfolio([c1, c2], N=5000)
print("Totals:", out["totals"]["CVA_total"], out["totals"]["DVA_total"])
print("Legs shapes:", out["totals"]["CVA_legs_sum"].shape, out["totals"]["DVA_legs_sum"].shape)
print("Per-cpty count:", len(out["per_counterparty"]))
