# pages/4_🚀_Run_from_UI.py
import streamlit as st

from app_lib.style import apply_page_config, apply_css
from app_lib.state import data_dir, set_selected_run
from app_lib.io import list_runs
from app_lib.runner import run_pipeline_and_export, default_run_name

apply_page_config(title="Run from UI — XVA", icon="🚀")
apply_css()

st.title("🚀 Lancer un nouveau run (depuis Streamlit)")

DATA = data_dir()
DATA.mkdir(exist_ok=True)

with st.sidebar:
    st.subheader("Paramètres")

    # --- NEW: nombre de contreparties
    n_cptys = st.number_input(
        "Nombre de contreparties",
        min_value=1,
        max_value=500,
        value=20,
        step=1,
        help="Taille du portfolio simulé (CPTY_01..). Attention: plus c'est grand, plus c'est long.",
    )

    N = st.number_input("Nombre de scénarios (N)", min_value=100, max_value=200000, value=5000, step=500)
    seed = st.number_input("Seed", min_value=0, max_value=2_000_000_000, value=12345, step=1)
    make_plots = st.checkbox("Générer PNG (plots)", value=False)
    include_mar = st.checkbox("Inclure snapshot Mar + compare + Shapley", value=True)
    run_name = st.text_input("Nom du run (optionnel)", value=default_run_name())

# Helper: clear cached file scans
def _clear_run_caches():
    try:
        list_runs.clear()
    except Exception:
        pass
    try:
        st.cache_data.clear()
    except Exception:
        pass

if st.button("🚀 Run", use_container_width=True):
    outdir = DATA / run_name.strip()

    with st.status("Simulation en cours…", expanded=True) as status:
        st.write("Monte-Carlo + exports…")
        run_pipeline_and_export(
            N=int(N),
            seed=int(seed),
            outdir=outdir,
            make_plots=make_plots,
            include_mar_and_shapley=include_mar,

            # --- NEW
            n_counterparties=int(n_cptys),
        )
        status.update(label=f"✅ Run terminé: {outdir.name}", state="complete")

    _clear_run_caches()

    set_selected_run(outdir.name)
    st.success(f"Run créé et sélectionné : {outdir.name}")
    st.rerun()

st.markdown("---")
st.subheader("Runs existants")

c1, c2 = st.columns([0.35, 0.65])
with c1:
    if st.button("🔄 Refresh runs", use_container_width=True):
        _clear_run_caches()
        st.rerun()

runs = list_runs(str(DATA))
st.write(runs if runs else "Aucun run.")
