from __future__ import annotations

import json
import os
import urllib.request


def notify(message: str) -> None:
    """Send an optional generic webhook notification; never fail the trading run."""
    url = os.getenv("BTC_BOT_WEBHOOK_URL", "").strip()
    if not url:
        return
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"content": message, "text": message}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8):
            pass
    except Exception:
        return
