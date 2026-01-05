# pages/6_📈_Portfolio_Tracking.py
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.style import apply_page_config, apply_css
from app_lib.state import data_dir, sidebar_run_selector, require_outdir
from app_lib.io import (
    list_runs, load_totals, load_portfolio_table,
    # load_totals_timeseries, load_cpty_metric_across_runs  # removed
)

apply_page_config(title="Portfolio Tracking — XVA", icon="📈")
apply_css()

DATA = data_dir()
runs = list_runs(str(DATA))

outdir = sidebar_run_selector(runs)
require_outdir(outdir)

st.title("📈 Portfolio Tracking")
st.caption("Ranking, deltas run-vs-run, historique des runs, export CSV.")

# -----------------------------
# Totals (current run)
# -----------------------------
totals = load_totals(outdir)
cva_total = float(totals.get("CVA_total", np.nan))
dva_total = float(totals.get("DVA_total", np.nan))
net_total = dva_total - cva_total if np.isfinite(cva_total) and np.isfinite(dva_total) else np.nan

c1, c2, c3 = st.columns(3)
c1.metric("CVA total", f"{cva_total:,.6g}")
c2.metric("DVA total", f"{dva_total:,.6g}")
c3.metric("Net (DVA − CVA)", f"{net_total:,.6g}")

st.divider()

# -----------------------------
# Ranking table (current run)
# -----------------------------
st.subheader("🏁 Portfolio ranking (par contrepartie)")

df = load_portfolio_table(str(outdir))
if df.empty:
    st.warning("Impossible de construire la table portfolio (meta_*.json manquants ?).")
    st.stop()

left, right = st.columns([1.1, 0.9])

with left:
    search = st.text_input("Recherche cid", value="")
    metric = st.radio(
        "Classer par",
        ["CVA", "DVA", "Net_DVA_minus_CVA", "Abs_Net", "Gross_CVA_plus_DVA"],
        horizontal=True,
        index=0,
    )
with right:
    topn = st.slider("Top N", min_value=2, max_value=min(50, len(df)), value=min(20, len(df)))
    ascending = st.checkbox("Tri ascendant", value=False)

df_view = df.copy()
if search.strip():
    df_view = df_view[df_view["cid"].str.contains(search.strip(), case=False, na=False)]

df_view = df_view.sort_values(metric, ascending=ascending).reset_index(drop=True)

st.dataframe(
    df_view[["cid", "CVA", "DVA", "Net_DVA_minus_CVA", "Gross_CVA_plus_DVA", "Abs_Net", "LGD_cpty"]],
    use_container_width=True,
    height=420,
)

# Charts top N
st.markdown("#### 📊 Top movers (current run)")
df_top = df_view.head(topn)

cA, cB = st.columns(2)
with cA:
    st.write(f"Top {topn} — {metric}")
    chart = df_top.set_index("cid")[metric]
    st.bar_chart(chart)

with cB:
    st.write("Scatter CVA vs DVA")
    df_sc = df_view[["cid", "CVA", "DVA"]].copy()
    st.dataframe(df_sc.head(topn), use_container_width=True)

csv_bytes = df_view.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Télécharger ranking CSV",
    data=csv_bytes,
    file_name=f"{outdir.name}_portfolio_ranking.csv",
    use_container_width=True,
)

st.divider()

# -----------------------------
# Compare run vs run (deltas)
# -----------------------------
st.subheader("🔁 Compare run vs run (Δ par contrepartie)")

other_runs = [r for r in runs if r != outdir.name]
if not other_runs:
    st.info("Aucun autre run disponible pour comparer.")
else:
    compare_run = st.selectbox("Run de comparaison", other_runs, index=0)
    outdir2 = DATA / compare_run

    df2 = load_portfolio_table(str(outdir2))
    if df2.empty:
        st.warning("Le run de comparaison ne contient pas meta_*.json.")
    else:
        merged = df.merge(df2, on="cid", how="inner", suffixes=("_base", "_cmp"))
        merged["dCVA"] = merged["CVA_cmp"] - merged["CVA_base"]
        merged["dDVA"] = merged["DVA_cmp"] - merged["DVA_base"]
        merged["dNet"] = merged["Net_DVA_minus_CVA_cmp"] - merged["Net_DVA_minus_CVA_base"]
        merged["abs_dCVA"] = merged["dCVA"].abs()
        merged["abs_dDVA"] = merged["dDVA"].abs()

        col1, col2, col3 = st.columns(3)
        col1.metric("ΔCVA (sum)", f"{merged['dCVA'].sum():,.6g}")
        col2.metric("ΔDVA (sum)", f"{merged['dDVA'].sum():,.6g}")
        col3.metric("ΔNet (sum)", f"{merged['dNet'].sum():,.6g}")

        sort_key = st.radio("Trier deltas par", ["abs_dCVA", "abs_dDVA", "dCVA", "dDVA", "dNet"], horizontal=True)
        merged = merged.sort_values(sort_key, ascending=False)

        st.dataframe(
            merged[[
                "cid",
                "CVA_base", "CVA_cmp", "dCVA",
                "DVA_base", "DVA_cmp", "dDVA",
                "Net_DVA_minus_CVA_base", "Net_DVA_minus_CVA_cmp", "dNet",
                "LGD_cpty_base"
            ]],
            use_container_width=True,
            height=420
        )

        st.download_button(
            "⬇️ Télécharger deltas CSV",
            data=merged.to_csv(index=False).encode("utf-8"),
            file_name=f"{outdir.name}_vs_{compare_run}_deltas.csv",
            use_container_width=True,
        )
