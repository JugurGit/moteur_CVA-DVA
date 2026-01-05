# pages/1_📊_Dashboard.py
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.style import apply_page_config, apply_css
from app_lib.state import data_dir, sidebar_run_selector, require_outdir
from app_lib.io import list_runs, load_totals, guess_times_from_meta

apply_page_config(title="Dashboard — XVA", icon="📊")
apply_css()

runs = list_runs(str(data_dir()))
outdir = sidebar_run_selector(runs)
require_outdir(outdir)

st.title("📊 Dashboard")

totals = load_totals(outdir)
cva_total = float(totals.get("CVA_total", np.nan))
dva_total = float(totals.get("DVA_total", np.nan))
CVA_legs_sum = totals.get("CVA_legs_sum", None)
DVA_legs_sum = totals.get("DVA_legs_sum", None)
DF = totals.get("DF", None)

Kp1 = 61
if isinstance(CVA_legs_sum, np.ndarray):
    Kp1 = CVA_legs_sum.shape[0]
elif isinstance(DF, np.ndarray):
    Kp1 = DF.shape[0]
times = guess_times_from_meta(outdir, int(Kp1))

c1, c2, c3 = st.columns(3)
c1.metric("CVA total", f"{cva_total:,.6g}")
c2.metric("DVA total", f"{dva_total:,.6g}")
if isinstance(DF, np.ndarray):
    c3.metric("DF(0, 5Y)", f"{float(DF[-1]):.6g}")
else:
    c3.metric("DF(0, 5Y)", "—")

st.subheader("Legs agrégés (portfolio)")
if isinstance(CVA_legs_sum, np.ndarray) and isinstance(DVA_legs_sum, np.ndarray):
    df_plot = pd.DataFrame({
        "t": times,
        "Σ CVA_leg": CVA_legs_sum,
        "Σ DVA_leg": DVA_legs_sum,
    })
    st.line_chart(df_plot.set_index("t"))
else:
    st.warning("Impossible de charger cva_legs_sum.csv / dva_legs_sum.csv.")

png = outdir / "figs" / "portfolio_legs.png"
if png.exists():
    st.image(str(png), caption="PNG exporté (si plots activés)")

st.divider()
st.subheader("Downloads rapides")

cand = [
    outdir / "meta.json",
    outdir / "run_meta.json",
    outdir / "xva_compare_pvjan.csv",
    outdir / "totals" / "totals.csv",
]
for f in cand:
    if f.exists():
        st.download_button(f"⬇️ {f.name}", data=f.read_bytes(), file_name=f.name, use_container_width=True)
