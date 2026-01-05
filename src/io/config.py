# src/io/config.py
from __future__ import annotations
import pathlib
import yaml

def load_settings(path: str | pathlib.Path) -> dict:
    """
    Charge le fichier YAML de configuration et renvoie un dict.
    Utilise yaml.safe_load. Lève FileNotFoundError si absent.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # validations minimales
    required = ["seed", "horizon_years", "dt_years", "n_scenarios",
                "rates", "discount_curve", "credit_defaults", "bank", "portfolio"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required key in settings: '{k}'")
    return cfg
