# pages/3_🧩_Shapley_Compare.py
import io
import zipfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.style import apply_page_config, apply_css
from app_lib.state import data_dir, sidebar_run_selector, require_outdir
from app_lib.io import (
    list_runs, find_counterparties_in_run, load_shapley, load_compare,
    load_totals, guess_times_from_meta
)

apply_page_config(title="Sensibilités", icon="🧩")
apply_css()

runs = list_runs(str(data_dir()))

# --- default run should NOT end with "MAR"
def _run_name(x) -> str:
    return Path(str(x)).name

def _is_mar_run(x) -> bool:
    return _run_name(x).strip().upper().endswith("MAR")

# stable sort: non-MAR first, MAR last
if runs:
    runs = sorted(runs, key=_is_mar_run)

outdir = sidebar_run_selector(runs)
require_outdir(outdir)

st.title("🧩 Analyse de sensibilités (Jan/Mar)")

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


def _zip_two_csv_bytes(path_a: Path, path_b: Path, name_a: str, name_b: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name_a, path_a.read_text(encoding="utf-8"))
        zf.writestr(name_b, path_b.read_text(encoding="utf-8"))
    return buf.getvalue()


def _shapley_area_chart(
    df_num: pd.DataFrame,
    value_cols: list[str],
    *,
    x_col: str = "t",
    x_title: str = "Temps t (années)",
    y_title: str = "Contribution (phi)",
) -> alt.Chart:
    """
    Graphique aire empilée (style st.area_chart) avec titres d'axes personnalisés.
    """
    df_plot = df_num[[x_col] + value_cols].copy()

    df_long = df_plot.melt(
        id_vars=[x_col],
        var_name="leg",
        value_name="phi",
    )

    # Renommage optionnel pour une légende plus propre
    leg_map = {
        "phi_DF": "DF",
        "phi_EPE": "EPE",
        "phi_PD_cpty": "PD contrepartie",
        "phi_ENE": "ENE",
        "phi_PD_bank": "PD banque",
        "phi_S_cpty": "Survie cpty (S)",
    }
    df_long["leg"] = df_long["leg"].map(leg_map).fillna(df_long["leg"])

    chart = (
        alt.Chart(df_long)
        .mark_area()
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_title),
            y=alt.Y("phi:Q", title=y_title, stack="zero"),
            color=alt.Color("leg:N", title="Composante"),
            tooltip=[
                alt.Tooltip(f"{x_col}:Q", title="t", format=".4f"),
                alt.Tooltip("leg:N", title="Composante"),
                alt.Tooltip("phi:Q", title="Contribution", format=".6g"),
            ],
        )
        .properties(height=320)
    )
    return chart


shap_dir = outdir / "shapley_per_counterparty"
f_cva = shap_dir / f"shapley_cva_{cid}.csv"
f_dva = shap_dir / f"shapley_dva_{cid}.csv"
f_cmp = outdir / "xva_compare_pvjan.csv"

st.subheader("⬇️ Téléchargements")
d1, d2, d3, d4 = st.columns([1.0, 1.0, 1.0, 1.0], gap="small")

with d1:
    if f_cva.exists():
        st.download_button(
            "⬇️ Shapley CVA (CSV)",
            data=f_cva.read_bytes(),
            file_name=f_cva.name,
            mime="text/csv",
            use_container_width=True,
            key=f"dl_shap_cva_{cid}_{outdir.name}",
        )
    else:
        st.caption("Shapley CVA indisponible")

with d2:
    if f_dva.exists():
        st.download_button(
            "⬇️ Shapley DVA (CSV)",
            data=f_dva.read_bytes(),
            file_name=f_dva.name,
            mime="text/csv",
            use_container_width=True,
            key=f"dl_shap_dva_{cid}_{outdir.name}",
        )
    else:
        st.caption("Shapley DVA indisponible")

with d3:
    if f_cva.exists() and f_dva.exists():
        zip_bytes = _zip_two_csv_bytes(
            f_cva, f_dva,
            name_a=f_cva.name,
            name_b=f_dva.name,
        )
        st.download_button(
            "⬇️ Shapley (ZIP)",
            data=zip_bytes,
            file_name=f"{outdir.name}_{cid}_shapley.zip",
            mime="application/zip",
            use_container_width=True,
            key=f"dl_shap_zip_{cid}_{outdir.name}",
        )
    else:
        st.caption("ZIP indisponible")

with d4:
    if f_cmp.exists():
        st.download_button(
            "⬇️ Comparatif Jan/Mar (CSV)",
            data=f_cmp.read_bytes(),
            file_name=f_cmp.name,
            mime="text/csv",
            use_container_width=True,
            key=f"dl_compare_{outdir.name}",
        )
    else:
        st.caption("Comparatif indisponible")

st.divider()


col1, col2 = st.columns(2)

with col1:
    st.write("Shapley — CVA legs (DF / EPE / PD_cpty)")
    if df_cva is None:
        st.info("Pas de fichier shapley_cva_*.csv trouvé dans shapley_per_counterparty/.")
    else:
        df_num = df_cva[pd.to_numeric(df_cva["k"], errors="coerce").notna()].copy()
        if df_num.empty:
            st.info("Données shapley CVA vides après filtrage numérique sur k.")
        else:
            df_num["k"] = df_num["k"].astype(int)
            df_num["t"] = times[: len(df_num)]
            chart = _shapley_area_chart(
                df_num,
                ["phi_DF", "phi_EPE", "phi_PD_cpty"],
                x_title="Temps t (années)",
                y_title="Contribution au CVA leg (phi)",
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(df_cva.tail(10), use_container_width=True)

with col2:
    st.write("Shapley — DVA legs (DF / ENE / PD_bank / S_cpty)")
    if df_dva is None:
        st.info("Pas de fichier shapley_dva_*.csv trouvé dans shapley_per_counterparty/.")
    else:
        df_num = df_dva[pd.to_numeric(df_dva["k"], errors="coerce").notna()].copy()
        if df_num.empty:
            st.info("Données shapley DVA vides après filtrage numérique sur k.")
        else:
            df_num["k"] = df_num["k"].astype(int)
            df_num["t"] = times[: len(df_num)]
            chart = _shapley_area_chart(
                df_num,
                ["phi_DF", "phi_ENE", "phi_PD_bank", "phi_S_cpty"],
                x_title="Temps t (années)",
                y_title="Contribution au DVA leg (phi)",
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(df_dva.tail(10), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Compare table
# ---------------------------------------------------------------------
st.subheader("Comparatif Jan/Mar (PV Jan)")
comp = load_compare(outdir)
if comp is None:
    st.info("xva_compare_pvjan.csv absent (activer l’option snapshot Mar + compare + shapley).")
else:
    st.dataframe(comp, use_container_width=True)
