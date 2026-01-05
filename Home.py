# Home.py
import streamlit as st

from app_lib.style import apply_page_config, apply_css
from app_lib.state import data_dir, sidebar_run_selector, require_outdir
from app_lib.io import list_runs

apply_page_config(title="XVA HW1F++ — Multi-pages", icon="📊")
apply_css()

st.title("📊 XVA HW1F++ — Interface multi-pages")
st.caption("Dashboard / drilldown contreparties / Shapley & compare / lancer un run / code browser")

runs = list_runs(str(data_dir()))
outdir = sidebar_run_selector(runs)

st.markdown("---")
if outdir is None:
    st.info("Va sur la page 🚀 Run_from_UI pour créer un run, ou lance main.py pour générer ./data/run_...")
else:
    st.success(f"Run sélectionné : `{outdir.name}`")
    st.write("Tu peux naviguer avec les pages à gauche :")
    st.markdown(
        """
- 📊 **Dashboard** : CVA/DVA totaux, legs agrégés, downloads.
- 👥 **Counterparties** : EPE/ENE, PD/Survival, legs par contrepartie.
- 🧩 **Shapley & Compare** : contributions DF/EPE/PD… et comparaison Jan/Mar si présente.
- 🚀 **Run from UI** : lancer une simulation depuis Streamlit.
- 🧾 **Code & Artefacts** : lecture des fichiers Python + listing du run.
"""
    )
