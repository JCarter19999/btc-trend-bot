from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  bar_timestamp TEXT,
  mode TEXT NOT NULL,
  status TEXT NOT NULL,
  message TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
  client_order_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  bar_timestamp TEXT NOT NULL,
  side TEXT NOT NULL,
  requested_size TEXT NOT NULL,
  order_id TEXT,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


class AuditStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def log_run(self, created_at: str, bar_timestamp: str | None, mode: str, status: str,
                message: str, payload: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO runs(created_at,bar_timestamp,mode,status,message,payload_json) VALUES(?,?,?,?,?,?)",
            (created_at, bar_timestamp, mode, status, message, json.dumps(payload or {}, sort_keys=True, default=str)),
        )
        self.conn.commit()

    def order_exists(self, client_order_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM orders WHERE client_order_id=?", (client_order_id,)
        ).fetchone()
        return row is not None

    def log_order(self, *, client_order_id: str, created_at: str, bar_timestamp: str,
                  side: str, requested_size: str, order_id: str | None,
                  status: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO orders(client_order_id,created_at,bar_timestamp,side,requested_size,order_id,status,payload_json) VALUES(?,?,?,?,?,?,?,?)",
            (client_order_id, created_at, bar_timestamp, side, requested_size, order_id, status,
             json.dumps(payload, sort_keys=True, default=str)),
        )
        self.conn.commit()

    def get_state(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()
