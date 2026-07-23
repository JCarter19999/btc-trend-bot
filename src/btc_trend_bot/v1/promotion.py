import json, os, shutil, sqlite3
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_GATES = ("out_of_sample", "walk_forward", "cost_sensitivity", "drawdown", "trade_count", "calibration", "paper", "shadow")

def promote(model_dir: str | Path, champion_dir: str | Path, conn: sqlite3.Connection, approved_by: str, reason: str = "") -> None:
    source = Path(model_dir)
    manifest = json.loads((source / "model_manifest.json").read_text(encoding="utf-8"))
    failed = [gate for gate in REQUIRED_GATES if manifest.get("gates", {}).get(gate) is not True]
    if failed:
        raise ValueError(f"Promotion blocked; failed or missing gates: {failed}")
    destination = Path(champion_dir)
    temporary = destination.with_name(destination.name + ".tmp")
    backup = destination.with_name(destination.name + ".previous")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    if backup.exists():
        shutil.rmtree(backup)
    old_model_id = None
    if destination.exists():
        try:
            old_model_id = json.loads((destination / "model_manifest.json").read_text(encoding="utf-8")).get("model_id")
        except Exception:
            old_model_id = None
        os.replace(destination, backup)
    os.replace(temporary, destination)
    conn.execute("INSERT INTO promotion_events(created_at,old_model_id,new_model_id,approved_by,reason) VALUES(?,?,?,?,?)", (datetime.now(timezone.utc).isoformat(), old_model_id, manifest["model_id"], approved_by, reason))
    conn.commit()
