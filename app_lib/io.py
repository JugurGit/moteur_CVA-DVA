# app_lib/io.py
from __future__ import annotations
import json
import pathlib
import re
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def list_runs(data_dir: str) -> List[str]:
    p = pathlib.Path(data_dir)
    if not p.exists():
        return []
    runs = []
    for d in p.iterdir():
        if d.is_dir() and (d.name.startswith("run_") or d.name.endswith("_MAR")):
            runs.append(d.name)
    runs.sort(reverse=True)
    return runs

def _safe_read_json(path: pathlib.Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def guess_times_from_meta(outdir: pathlib.Path, Kp1: int) -> np.ndarray:
    rm = outdir / "run_meta.json"
    if rm.exists():
        meta = _safe_read_json(rm) or {}
        grid = meta.get("grid", {})
        T = float(grid.get("T", 5.0))
        dt = float(grid.get("dt", T / max(1, (Kp1 - 1))))
        return np.arange(Kp1, dtype=float) * dt
    if Kp1 <= 1:
        return np.array([0.0])
    return np.linspace(0.0, 5.0, Kp1)

def find_counterparties_in_run(outdir: pathlib.Path) -> List[str]:
    p = outdir / "per_counterparty"
    if not p.exists():
        return []
    cids = []
    for f in p.glob("exposures_*.csv"):
        m = re.match(r"exposures_(.+)\.csv", f.name)
        if m:
            cids.append(m.group(1))
    cids.sort()
    return cids

def load_totals(outdir: pathlib.Path) -> Dict[str, Any]:
    totals_dir = outdir / "totals"
    res: Dict[str, Any] = {}

    # totals.csv
    f_tot = totals_dir / "totals.csv"
    if f_tot.exists():
        df = read_csv(str(f_tot))
        for _, row in df.iterrows():
            res[str(row["metric"])] = float(row["value"])

    # legs + DF
    for name, key in [
        ("cva_legs_sum.csv", "CVA_legs_sum"),
        ("dva_legs_sum.csv", "DVA_legs_sum"),
        ("df_curve.csv", "DF"),
    ]:
        f = totals_dir / name
        if f.exists():
            df = read_csv(str(f))
            res[key] = df.iloc[:, 1].astype(float).to_numpy()

    return res

def load_cpty_tables(outdir: pathlib.Path, cid: str) -> Dict[str, pd.DataFrame]:
    p = outdir / "per_counterparty"
    out: Dict[str, pd.DataFrame] = {}

    for base in ["exposures", "credit", "xva"]:
        f = p / f"{base}_{cid}.csv"
        if f.exists():
            out[base] = read_csv(str(f))

    meta_f = p / f"meta_{cid}.json"
    out["meta_json"] = pd.DataFrame([_safe_read_json(meta_f) or {}])

    return out

def load_shapley(outdir: pathlib.Path, cid: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    p = outdir / "shapley_per_counterparty"
    if not p.exists():
        return None, None
    f_cva = p / f"shapley_cva_{cid}.csv"
    f_dva = p / f"shapley_dva_{cid}.csv"
    df_cva = read_csv(str(f_cva)) if f_cva.exists() else None
    df_dva = read_csv(str(f_dva)) if f_dva.exists() else None
    return df_cva, df_dva

def load_compare(outdir: pathlib.Path) -> Optional[pd.DataFrame]:
    f = outdir / "xva_compare_pvjan.csv"
    if f.exists():
        return read_csv(str(f))
    return None

# --- ADD at end of app_lib/io.py ---------------------------------------------
import datetime as dt

def _parse_run_timestamp(run_name: str) -> dt.datetime | None:
    # run_YYYYMMDD_HHMMSS
    try:
        if run_name.startswith("run_") and len(run_name) >= 19:
            stamp = run_name.split("_", 1)[1]
            # stamp = YYYYMMDD_HHMMSS or sometimes with suffix; take first 15 chars
            stamp = stamp[:15]
            return dt.datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def load_portfolio_table(outdir: str) -> pd.DataFrame:
    """
    Build a per-counterparty table from per_counterparty/meta_*.json (fast).
    Columns: cid, CVA, DVA, LGD_cpty, LGD_bank (+ optional).
    """
    outdir_p = pathlib.Path(outdir)
    p = outdir_p / "per_counterparty"
    if not p.exists():
        return pd.DataFrame()

    rows = []
    for f in sorted(p.glob("meta_*.json")):
        d = _safe_read_json(f) or {}
        if not d:
            continue
        rows.append({
            "cid": d.get("cid", f.stem.replace("meta_", "")),
            "CVA": float(d.get("CVA", 0.0)),
            "DVA": float(d.get("DVA", 0.0)),
            "LGD_cpty": float(d.get("LGD_cpty", float("nan"))),
            "LGD_bank": float(d.get("LGD_bank", float("nan"))),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Net_DVA_minus_CVA"] = df["DVA"] - df["CVA"]
    df["Gross_CVA_plus_DVA"] = df["DVA"] + df["CVA"]
    df["Abs_Net"] = (df["Net_DVA_minus_CVA"]).abs()
    return df.sort_values("CVA", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_totals_timeseries(data_dir: str) -> pd.DataFrame:
    """
    For all runs in ./data, read totals/totals.csv and produce time series.
    """
    data_p = pathlib.Path(data_dir)
    if not data_p.exists():
        return pd.DataFrame()

    runs = []
    for d in data_p.iterdir():
        if not d.is_dir():
            continue
        if not (d.name.startswith("run_") or d.name.endswith("_MAR")):
            continue
        ts = _parse_run_timestamp(d.name.replace("_MAR", ""))
        totals_dir = d / "totals" / "totals.csv"
        if totals_dir.exists():
            try:
                tdf = pd.read_csv(totals_dir)
                m = {str(r["metric"]): float(r["value"]) for _, r in tdf.iterrows()}
                runs.append({
                    "run": d.name,
                    "timestamp": ts,
                    "CVA_total": m.get("CVA_total", float("nan")),
                    "DVA_total": m.get("DVA_total", float("nan")),
                })
            except Exception:
                pass

    df = pd.DataFrame(runs)
    if df.empty:
        return df

    # If timestamp missing, keep lexical
    if "timestamp" in df.columns:
        df = df.sort_values(["timestamp", "run"], ascending=[True, True])
    else:
        df = df.sort_values("run")

    df["Net_DVA_minus_CVA"] = df["DVA_total"] - df["CVA_total"]
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_cpty_metric_across_runs(data_dir: str, cid: str) -> pd.DataFrame:
    """
    Track one counterparty across runs (reads meta_{cid}.json in each run).
    """
    data_p = pathlib.Path(data_dir)
    rows = []
    for d in data_p.iterdir():
        if not d.is_dir():
            continue
        if not (d.name.startswith("run_") or d.name.endswith("_MAR")):
            continue
        ts = _parse_run_timestamp(d.name.replace("_MAR", ""))
        meta_f = d / "per_counterparty" / f"meta_{cid}.json"
        if meta_f.exists():
            dd = _safe_read_json(meta_f) or {}
            rows.append({
                "run": d.name,
                "timestamp": ts,
                "CVA": float(dd.get("CVA", float("nan"))),
                "DVA": float(dd.get("DVA", float("nan"))),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Net_DVA_minus_CVA"] = df["DVA"] - df["CVA"]
    if "timestamp" in df.columns:
        df = df.sort_values(["timestamp", "run"], ascending=[True, True])
    else:
        df = df.sort_values("run")
    return df.reset_index(drop=True)
