# app_lib/state.py
from __future__ import annotations
import pathlib
import streamlit as st

def project_root() -> pathlib.Path:
    # app_lib/state.py -> app_lib -> ROOT
    return pathlib.Path(__file__).resolve().parents[1]

def data_dir() -> pathlib.Path:
    return project_root() / "data"

def get_selected_run() -> str | None:
    return st.session_state.get("selected_run", None)

def set_selected_run(run_name: str) -> None:
    st.session_state["selected_run"] = run_name

def selected_outdir() -> pathlib.Path | None:
    r = get_selected_run()
    if not r:
        return None
    return data_dir() / r

def sidebar_run_selector(runs: list[str], *, show_refresh: bool = True) -> pathlib.Path | None:
    st.sidebar.title("⚙️ Run selection")

    if show_refresh:
        if st.sidebar.button("🔄 Refresh runs", use_container_width=True):
            # IMPORTANT: clear cached directory scans / CSV reads
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

    if not runs:
        st.sidebar.warning("Aucun run détecté dans ./data.")
        return None

    current = get_selected_run()
    if current not in runs:
        current = runs[0]
        set_selected_run(current)

    picked = st.sidebar.selectbox("Run", runs, index=runs.index(current))
    set_selected_run(picked)
    return selected_outdir()

def require_outdir(outdir):
    if outdir is None or (not outdir.exists()):
        st.info("Sélectionne un run existant dans la sidebar (ou crée-en un via la page 🚀 Run).")
        st.stop()
