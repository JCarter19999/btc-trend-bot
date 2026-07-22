from __future__ import annotations
import hashlib, hmac, time
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlencode
import requests

@dataclass
class BinanceClient:
    base_url: str = "https://api.binance.com"
    api_key: str | None = None
    api_secret: str | None = None
    timeout_seconds: float = 15.0

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
        params = dict(params or {})
        headers: dict[str, str] = {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise RuntimeError("Signed Binance request requires API credentials")
            params.setdefault("timestamp", int(time.time() * 1000))
            query = urlencode(params)
            params["signature"] = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            headers["X-MBX-APIKEY"] = self.api_key
        response = requests.request(method, self.base_url + path, params=params, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def server_time(self) -> int:
        return int(self._request("GET", "/api/v3/time")["serverTime"])

    def klines(self, symbol: str, interval: str, start_time: int | None = None, end_time: int | None = None, limit: int = 1000) -> list[list[Any]]:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        return self._request("GET", "/api/v3/klines", params)

    def iter_klines(self, symbol: str, interval: str, start_time: int, end_time: int | None = None) -> Iterable[list[Any]]:
        cursor = start_time
        while True:
            batch = self.klines(symbol, interval, cursor, end_time, 1000)
            if not batch:
                break
            yield from batch
            next_cursor = int(batch[-1][0]) + 1
            if next_cursor <= cursor or len(batch) < 1000:
                break
            cursor = next_cursor

    def book_ticker(self, symbol: str) -> dict[str, Any]:
        return self._request("GET", "/api/v3/ticker/bookTicker", {"symbol": symbol})

    def account(self) -> dict[str, Any]:
        return self._request("GET", "/api/v3/account", {"recvWindow": 5000}, signed=True)

    def market_order(self, symbol: str, side: str, quantity: str, client_order_id: str) -> dict[str, Any]:
        return self._request("POST", "/api/v3/order", {"symbol": symbol, "side": side, "type": "MARKET", "quantity": quantity, "newClientOrderId": client_order_id, "recvWindow": 5000}, signed=True)
