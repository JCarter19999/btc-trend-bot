import json, sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidates(candidate_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, strategy_version TEXT NOT NULL, payload_json TEXT NOT NULL, label_status TEXT NOT NULL DEFAULT 'PENDING');
CREATE TABLE IF NOT EXISTS model_versions(model_id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL, manifest_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decisions(decision_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, created_at TEXT NOT NULL, model_id TEXT NOT NULL, action TEXT NOT NULL, payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS risk_events(id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, reason_code TEXT NOT NULL, payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scheduler_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, job_name TEXT NOT NULL, status TEXT NOT NULL, details_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS promotion_events(id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, old_model_id TEXT, new_model_id TEXT NOT NULL, approved_by TEXT NOT NULL, reason TEXT);
"""

def connect(path: str | Path) -> sqlite3.Connection:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(destination)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    return conn

def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute("INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
    conn.commit()

def get_meta(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return default if row is None else json.loads(row[0])
