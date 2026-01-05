# pages/5_🧾_Code_Artefacts.py
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from app_lib.style import apply_page_config, apply_css
from app_lib.state import project_root

try:
    from pages.code_docs import render_doc_panel as _render_doc_panel_manual
    _DOC_IMPORT_ERROR = None
except Exception as e:
    _render_doc_panel_manual = None
    _DOC_IMPORT_ERROR = repr(e)

apply_page_config(title="Documentation du code — CVA-DVA", icon="🧾")
apply_css()

ROOT = project_root()
EXCLUDE_PROJECT = {
    ".git", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".streamlit_app_cache", "data"
}

# "Text-like" files we can preview without extra deps
TEXT_EXTS = {
    ".py": "python",
    ".ipynb": "ipynb",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".csv": "csv",
    ".log": "text",
}

MAX_PREVIEW_BYTES = 1_200_000  # ~1.2MB

DOCS_REGISTRY_PATH = ROOT / "pages" / "docs_registry.json"


# =============================================================================
# Small utils
# =============================================================================
def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n/1024:.1f} KB"
    return f"{n/(1024*1024):.2f} MB"


def _fmt_dt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _read_text_safely(path: Path, max_bytes: int = MAX_PREVIEW_BYTES) -> Tuple[str, bool]:
    """
    Returns (text, truncated).
    Decodes as UTF-8 with replacement; truncates by bytes if too large.
    """
    b = path.read_bytes()
    truncated = False
    if len(b) > max_bytes:
        b = b[:max_bytes]
        truncated = True
    text = b.decode("utf-8", errors="replace")
    return text, truncated


def _language_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".ipynb":
        return "json"
    return TEXT_EXTS.get(ext, "text")


# =============================================================================
# Project file scanner (cached)
# =============================================================================
@st.cache_data(show_spinner=False)
def _scan_files_project(root: str) -> List[Dict[str, Any]]:
    """
    Scan project files once and cache.
    Returns list of dicts with metadata.
    """
    root_p = Path(root)
    out: List[Dict[str, Any]] = []

    for dirpath, dirnames, filenames in os.walk(root_p):
        # exclude dirs
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_PROJECT]

        for fn in filenames:
            p = Path(dirpath) / fn
            ext = p.suffix.lower()

            if ext not in TEXT_EXTS:
                continue

            try:
                stt = p.stat()
                size = int(stt.st_size)
                mtime = float(stt.st_mtime)
            except OSError:
                continue

            rel = p.relative_to(root_p).as_posix()
            folder = str(Path(rel).parent).replace("\\", "/")
            if folder == ".":
                folder = ""

            n_lines: Optional[int] = None
            if ext != ".ipynb" and size <= 400_000:
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    n_lines = txt.count("\n") + 1 if txt else 0
                except Exception:
                    n_lines = None

            out.append(
                {
                    "rel": rel,
                    "path": str(p),
                    "ext": ext,
                    "folder": folder,
                    "size": size,
                    "mtime": mtime,
                    "lines": n_lines,
                }
            )

    out.sort(key=lambda d: d["rel"])
    return out


@st.cache_data(show_spinner=False)
def _load_docs_registry(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# =============================================================================
# Notebook renderer (simple)
# =============================================================================
def _render_notebook(path: Path) -> None:
    raw, truncated = _read_text_safely(path)
    if truncated:
        st.warning(
            f"Notebook trop volumineux : preview tronquée à {_fmt_bytes(MAX_PREVIEW_BYTES)}. "
            "Tu peux le télécharger pour l’ouvrir complet.",
            icon="⚠️",
        )

    try:
        nb = json.loads(raw)
    except Exception:
        st.error("Impossible de parser ce .ipynb (JSON invalide).", icon="❌")
        st.code(raw, language="json")
        return

    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        st.error("Format .ipynb inattendu (cells manquant).", icon="❌")
        st.code(raw, language="json")
        return

    show_outputs = st.checkbox("Afficher outputs", value=False, key="ca_show_nb_outputs")
    st.divider()

    for i, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type", "")
        src = cell.get("source", "")

        if isinstance(src, list):
            src_text = "".join(src)
        else:
            src_text = str(src)

        if cell_type == "markdown":
            if src_text.strip():
                st.markdown(src_text)
        elif cell_type == "code":
            st.markdown(f"**In [{i}]**")
            st.code(src_text, language="python")

            if show_outputs:
                outs = cell.get("outputs", [])
                if isinstance(outs, list) and outs:
                    for out in outs:
                        otype = out.get("output_type", "")
                        if otype == "stream":
                            txt = out.get("text", "")
                            if isinstance(txt, list):
                                txt = "".join(txt)
                            st.code(str(txt), language="text")
                        elif otype in ("execute_result", "display_data"):
                            data = out.get("data", {})
                            txt = None
                            if isinstance(data, dict):
                                txt = data.get("text/plain", None)
                            if txt is not None:
                                if isinstance(txt, list):
                                    txt = "".join(txt)
                                st.code(str(txt), language="text")
                        elif otype == "error":
                            tb = out.get("traceback", [])
                            if isinstance(tb, list):
                                st.code("\n".join(tb), language="text")


# =============================================================================
# UI
# =============================================================================
st.title("🧾 Documentation du code")
st.caption("Preview + documentation manuelle (docs_registry.json).")

# --- Force manual doc only
if _render_doc_panel_manual is None:
    st.error(
        "Documentation manuelle indisponible : impossible d’importer `pages/code_docs.py`.\n\n"
        f"Erreur: `{_DOC_IMPORT_ERROR}`",
        icon="❌",
    )
    st.stop()

# --- Files restricted to docs_registry.json only
all_files = _scan_files_project(str(ROOT))
docs_registry = _load_docs_registry(str(DOCS_REGISTRY_PATH))
allowed_rels = set(docs_registry.keys())

all_files = [d for d in all_files if d["rel"] in allowed_rels]
all_files.sort(key=lambda d: d["rel"])

if not docs_registry:
    st.warning(f"`docs_registry.json` introuvable ou vide: {DOCS_REGISTRY_PATH}", icon="⚠️")

if not all_files:
    st.info("Aucun fichier (présent dans docs_registry.json) n’a été trouvé dans le projet.", icon="ℹ️")
    st.stop()

rels = [d["rel"] for d in all_files]

# -----------------------------
# Selection state + navigation
# -----------------------------
if "ca_px_choice" not in st.session_state:
    st.session_state["ca_px_choice"] = rels[0]
if st.session_state["ca_px_choice"] not in rels:
    st.session_state["ca_px_choice"] = rels[0]

nav1, nav2, nav3, nav4 = st.columns([0.18, 0.18, 1.0, 0.28], gap="small")
with nav1:
    if st.button("⬅️ Prev", use_container_width=True, key="ca_px_prev"):
        i = rels.index(st.session_state["ca_px_choice"])
        st.session_state["ca_px_choice"] = rels[max(0, i - 1)]
with nav2:
    if st.button("Next ➡️", use_container_width=True, key="ca_px_next"):
        i = rels.index(st.session_state["ca_px_choice"])
        st.session_state["ca_px_choice"] = rels[min(len(rels) - 1, i + 1)]
with nav4:
    st.caption(f"{rels.index(st.session_state['ca_px_choice']) + 1} / {len(rels)}")

choice = st.selectbox(
    "Select file",
    rels,
    index=rels.index(st.session_state["ca_px_choice"]),
    key="ca_px_select",
)
st.session_state["ca_px_choice"] = choice

path = ROOT / choice
meta = next(d for d in all_files if d["rel"] == choice)

# -----------------------------
# Header + actions
# -----------------------------
st.write(f"**{choice}**")
st.caption(
    f"Ext: `{meta['ext']}`  |  Size: {_fmt_bytes(meta['size'])}  |  Modified: {_fmt_dt(meta['mtime'])}"
    + (f"  |  Lines: {meta['lines']}" if meta.get("lines") is not None else "")
)

try:
    file_bytes = path.read_bytes()
    st.download_button(
        "⬇️ Download file",
        data=file_bytes,
        file_name=path.name,
        mime="text/plain",
        use_container_width=False,
        key="ca_px_dl",
    )
except Exception:
    st.warning("Impossible de préparer le téléchargement (droits/IO).", icon="⚠️")

# -----------------------------
# Preview + manual doc (always)
# -----------------------------
left, right = st.columns([1.35, 0.85], gap="large")
ext = path.suffix.lower()

with left:
    st.subheader("Preview")

    if ext == ".ipynb":
        view_mode = st.radio(
            "Notebook view",
            ["Rendered", "Raw JSON"],
            horizontal=True,
            index=0,
            key="ca_px_nb_view",
        )
        if view_mode == "Rendered":
            _render_notebook(path)
        else:
            raw, truncated = _read_text_safely(path)
            if truncated:
                st.warning(
                    f"Preview tronquée à {_fmt_bytes(MAX_PREVIEW_BYTES)} (notebook trop volumineux).",
                    icon="⚠️",
                )
            st.code(raw, language="json")
    else:
        try:
            content, truncated = _read_text_safely(path)
            if truncated:
                st.warning(
                    f"Preview tronquée à {_fmt_bytes(MAX_PREVIEW_BYTES)} (fichier volumineux).",
                    icon="⚠️",
                )
            lang = _language_for(path)
            st.code(content, language=lang)
        except Exception as e:
            st.error(f"Erreur de lecture : {e}", icon="❌")

with right:
    _render_doc_panel_manual(choice, path)
