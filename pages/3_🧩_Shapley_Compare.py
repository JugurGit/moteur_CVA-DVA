# pages/3_🧩_Shapley_Compare.py
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path  # <-- NEW

from app_lib.style import apply_page_config, apply_css
from app_lib.state import data_dir, sidebar_run_selector, require_outdir
from app_lib.io import (
    list_runs, find_counterparties_in_run, load_shapley, load_compare,
    load_totals, guess_times_from_meta
)

apply_page_config(title="Shapley & Compare — XVA", icon="🧩")
apply_css()

runs = list_runs(str(data_dir()))

# --- NEW: default run should NOT end with "MAR"
def _run_name(x) -> str:
    return Path(str(x)).name

def _is_mar_run(x) -> bool:
    return _run_name(x).strip().upper().endswith("MAR")

# stable sort: non-MAR first, MAR last
if runs:
    runs = sorted(runs, key=_is_mar_run)

outdir = sidebar_run_selector(runs)
require_outdir(outdir)

st.title("🧩 Shapley & Compare (Jan/Mar)")

cids = find_counterparties_in_run(outdir)
if not cids:
    st.warning("Aucune contrepartie trouvée.")
    st.stop()

# times
totals = load_totals(outdir)
DF = totals.get("DF", None)
Kp1 = DF.shape[0] if isinstance(DF, np.ndarray) else 61
times = guess_times_from_meta(outdir, int(Kp1))

cid = st.selectbox("Contrepartie (Shapley)", cids, index=0)
df_cva, df_dva = load_shapley(outdir, cid)

col1, col2 = st.columns(2)
with col1:
    st.write("Shapley — CVA legs (DF / EPE / PD_cpty)")
    if df_cva is None:
        st.info("Pas de fichier shapley_cva_*.csv trouvé dans shapley_per_counterparty/.")
    else:
        df_num = df_cva[pd.to_numeric(df_cva["k"], errors="coerce").notna()].copy()
        df_num["k"] = df_num["k"].astype(int)
        df_num["t"] = times[: len(df_num)]
        st.area_chart(df_num[["t", "phi_DF", "phi_EPE", "phi_PD_cpty"]].set_index("t"))
        st.dataframe(df_cva.tail(10), use_container_width=True)

with col2:
    st.write("Shapley — DVA legs (DF / ENE / PD_bank / S_cpty)")
    if df_dva is None:
        st.info("Pas de fichier shapley_dva_*.csv trouvé dans shapley_per_counterparty/.")
    else:
        df_num = df_dva[pd.to_numeric(df_dva["k"], errors="coerce").notna()].copy()
        df_num["k"] = df_num["k"].astype(int)
        df_num["t"] = times[: len(df_num)]
        st.area_chart(df_num[["t", "phi_DF", "phi_ENE", "phi_PD_bank", "phi_S_cpty"]].set_index("t"))
        st.dataframe(df_dva.tail(10), use_container_width=True)

st.divider()
st.subheader("Compare Jan/Mar (PV Jan)")
comp = load_compare(outdir)
if comp is None:
    st.info("xva_compare_pvjan.csv absent (activer l’option snapshot Mar + compare + shapley).")
else:
    st.dataframe(comp, use_container_width=True)
