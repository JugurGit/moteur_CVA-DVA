"""
CSV/JSON exports for the XVA simulation.

We keep it dependency-free (no pandas). All arrays are written as simple CSVs.
Directory is created if missing.
"""

from __future__ import annotations
import os, csv, json
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_vector_csv(filepath: str, header: str, vec: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(filepath))
    vec = np.asarray(vec, dtype=float).ravel()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([header])
        for v in vec:
            w.writerow([f"{v:.12g}"])


def write_matrix_csv(filepath: str, headers: List[str], rows: List[List[Any]]) -> None:
    _ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)


def write_meta_json(filepath: str, meta: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------- High-level exports -----------------------------------------------

def export_per_counterparty_tables(outdir: str, per_cpty: List[Dict[str, Any]]) -> None:
    """
    For each counterparty 'res' (from Simulator.run_for_counterparty):
      - exposures_{cid}.csv: t_k, EPE_k, ENE_k
      - credit_{cid}.csv   : t_k, PD_k, S_k
      - xva_{cid}.csv      : t_k, CVA_leg_k, DVA_leg_k, and totals on last row
    """
    _ensure_dir(outdir)

    # times can be taken from lengths (assume uniform grid length)
    Kp1 = len(per_cpty[0]["EPE"])
    # We don't have the grid object here; reconstruct times as 0..K with dt=unknown
    # => we write an index column "k". If you want actual times, pass them in the dict.
    idx = list(range(Kp1))

    for res in per_cpty:
        cid = res["cid"]

        rows_expo = [[k, f"{res['EPE'][k]:.12g}", f"{res['ENE'][k]:.12g}"] for k in idx]
        write_matrix_csv(
            os.path.join(outdir, f"exposures_{cid}.csv"),
            headers=["k", "EPE", "ENE"],
            rows=rows_expo,
        )

        rows_credit = [[k, f"{res['PD_cpty'][k]:.12g}", f"{res['S_cpty'][k]:.12g}"] for k in idx]
        write_matrix_csv(
            os.path.join(outdir, f"credit_{cid}.csv"),
            headers=["k", "PD_cpty", "S_cpty"],
            rows=rows_credit,
        )

        rows_xva = [[k, f"{res['CVA_leg'][k]:.12g}", f"{res['DVA_leg'][k]:.12g}"] for k in idx]
        # add a final total line (with k = "TOTAL")
        rows_xva.append(["TOTAL", f"{res['CVA']:.12g}", f"{res['DVA']:.12g}"])

        write_matrix_csv(
            os.path.join(outdir, f"xva_{cid}.csv"),
            headers=["k", "CVA_leg", "DVA_leg"],
            rows=rows_xva,
        )

        # also dump a tiny meta json per cpty (useful for the report)
        meta = {
            "cid": cid,
            "LGD_cpty": res["LGD_cpty"],
            "LGD_bank": res["LGD_bank"],
            "CVA": res["CVA"],
            "DVA": res["DVA"],
        }
        write_meta_json(os.path.join(outdir, f"meta_{cid}.json"), meta)


def export_totals(outdir: str, totals: Dict[str, Any]) -> None:
    """
    Write:
      - totals.csv          : CVA_total, DVA_total
      - cva_legs_sum.csv    : k, CVA_leg_sum_k
      - dva_legs_sum.csv    : k, DVA_leg_sum_k
      - df_curve.csv        : k, DF_k
    """
    _ensure_dir(outdir)
    write_matrix_csv(
        os.path.join(outdir, "totals.csv"),
        headers=["metric", "value"],
        rows=[["CVA_total", f"{totals['CVA_total']:.12g}"],
              ["DVA_total", f"{totals['DVA_total']:.12g}"]],
    )

    Kp1 = len(totals["CVA_legs_sum"])
    idx = list(range(Kp1))

    rows_cva = [[k, f"{totals['CVA_legs_sum'][k]:.12g}"] for k in idx]
    write_matrix_csv(
        os.path.join(outdir, "cva_legs_sum.csv"),
        headers=["k", "CVA_leg_sum"],
        rows=rows_cva,
    )

    rows_dva = [[k, f"{totals['DVA_legs_sum'][k]:.12g}"] for k in idx]
    write_matrix_csv(
        os.path.join(outdir, "dva_legs_sum.csv"),
        headers=["k", "DVA_leg_sum"],
        rows=rows_dva,
    )

    rows_df = [[k, f"{totals['DF'][k]:.12g}"] for k in idx]
    write_matrix_csv(
        os.path.join(outdir, "df_curve.csv"),
        headers=["k", "DF"],
        rows=rows_df,
    )


def export_everything(outdir: str, portfolio_out: Dict[str, Any]) -> None:
    """
    Convenience: write both per-counterparty files and totals.
    """
    export_per_counterparty_tables(
        os.path.join(outdir, "per_counterparty"),
        portfolio_out["per_counterparty"],
    )
    export_totals(
        os.path.join(outdir, "totals"),
        portfolio_out["totals"],   # <-- argument 'totals' manquant corrigé
    )
    write_meta_json(
        os.path.join(outdir, "meta.json"),
        portfolio_out["meta"],
    )

