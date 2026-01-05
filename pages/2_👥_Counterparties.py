# pages/2_👥_Counterparties.py
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.style import apply_page_config, apply_css
from app_lib.state import data_dir, sidebar_run_selector, require_outdir
from app_lib.io import list_runs, find_counterparties_in_run, load_cpty_tables, load_totals, guess_times_from_meta

apply_page_config(title="Counterparties — XVA", icon="👥")
apply_css()

runs = list_runs(str(data_dir()))
outdir = sidebar_run_selector(runs)
require_outdir(outdir)

st.title("👥 Contreparties")

cids = find_counterparties_in_run(outdir)
if not cids:
    st.warning("Aucune contrepartie trouvée (exposures_*.csv absent).")
    st.stop()

cid = st.selectbox("Choisir une contrepartie", cids, index=0)
tables = load_cpty_tables(outdir, cid)

# Try infer times length
totals = load_totals(outdir)
CVA_legs_sum = totals.get("CVA_legs_sum", None)
DF = totals.get("DF", None)
Kp1 = 61
if isinstance(CVA_legs_sum, np.ndarray):
    Kp1 = CVA_legs_sum.shape[0]
elif isinstance(DF, np.ndarray):
    Kp1 = DF.shape[0]
times = guess_times_from_meta(outdir, int(Kp1))

cA, cB, cC = st.columns([1.2, 1.2, 1.0])
with cC:
    st.write("Meta (cpty)")
    st.dataframe(tables["meta_json"], use_container_width=True)

with cA:
    st.write("Exposures (EPE/ENE)")
    if "exposures" in tables:
        df_e = tables["exposures"].copy()
        df_e["t"] = times[: len(df_e)]
        st.line_chart(df_e[["t", "EPE", "ENE"]].set_index("t"))
        st.dataframe(df_e.head(10), use_container_width=True)
    else:
        st.info("exposures_{cid}.csv absent")

with cB:
    st.write("Credit (PD / Survival)")
    if "credit" in tables:
        df_c = tables["credit"].copy()
        df_c["t"] = times[: len(df_c)]
        st.line_chart(df_c[["t", "S_cpty"]].set_index("t"))
        st.bar_chart(df_c.set_index("t")["PD_cpty"])
        st.dataframe(df_c.head(10), use_container_width=True)
    else:
        st.info("credit_{cid}.csv absent")

st.subheader("XVA legs")
if "xva" in tables:
    df_x = tables["xva"].copy()
    df_x_num = df_x[pd.to_numeric(df_x["k"], errors="coerce").notna()].copy()
    df_x_num["k"] = df_x_num["k"].astype(int)
    df_x_num["t"] = times[: len(df_x_num)]
    st.line_chart(df_x_num[["t", "CVA_leg", "DVA_leg"]].set_index("t"))
    st.dataframe(df_x.tail(8), use_container_width=True)

st.divider()
figs_dir = outdir / "figs" / cid
if figs_dir.exists():
    img1 = figs_dir / f"{cid}_exposure.png"
    img2 = figs_dir / f"{cid}_credit.png"
    img3 = figs_dir / f"{cid}_xva_legs.png"
    cols = st.columns(3)
    if img1.exists(): cols[0].image(str(img1), caption="Exposure PNG")
    if img2.exists(): cols[1].image(str(img2), caption="Credit PNG")
    if img3.exists(): cols[2].image(str(img3), caption="XVA legs PNG")
