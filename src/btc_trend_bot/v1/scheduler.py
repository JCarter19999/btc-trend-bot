import json, time
from datetime import datetime, timezone
from .config import load_v1_config
from .retraining import should_retrain
from .storage import connect, get_meta

def main() -> None:
    cfg = load_v1_config()
    conn = connect(cfg["storage"]["sqlite_path"])
    while True:
        now = datetime.now(timezone.utc)
        count = int(get_meta(conn, "newly_matured_since_training", 0))
        eligible, reason = should_retrain(now, count, cfg["retraining"])
        details = json.dumps({"reason": reason, "newly_matured": count})
        conn.execute("INSERT INTO scheduler_runs(created_at,job_name,status,details_json) VALUES(?,?,?,?)", (now.isoformat(), "retraining_check", "ELIGIBLE" if eligible else "SKIPPED", details))
        conn.commit()
        time.sleep(3600)

if __name__ == "__main__":
    main()
