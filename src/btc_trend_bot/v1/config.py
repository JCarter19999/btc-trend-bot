from pathlib import Path
from typing import Any
import yaml

REQUIRED = ("system", "exchange", "strategy", "exit", "costs", "model", "risk", "retraining", "storage")

def load_v1_config(path: str | Path = "config/v1.yaml") -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("v1 config must be a mapping")
    missing = [section for section in REQUIRED if section not in cfg]
    if missing:
        raise ValueError(f"Missing v1 config sections: {missing}")
    if cfg["system"]["strategy_version"] != "selective_long_5m_v1":
        raise ValueError("v1 entry logic is frozen under selective_long_5m_v1")
    if cfg["exit"]["collision_policy"] != "stop_first":
        raise ValueError("v1 intrabar collision policy must be stop_first")
    return cfg
