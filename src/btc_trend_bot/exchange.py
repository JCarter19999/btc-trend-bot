from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Protocol


@dataclass(frozen=True)
class Balances:
    quote_available: Decimal
    base_available: Decimal


@dataclass(frozen=True)
class OrderResult:
    client_order_id: str
    order_id: str | None
    success: bool
    raw: dict[str, Any]


class SpotBroker(Protocol):
    def balances(self, base_currency: str, quote_currency: str) -> Balances: ...
    def market_buy(self, product_id: str, quote_size: Decimal, client_order_id: str) -> OrderResult: ...
    def market_sell(self, product_id: str, base_size: Decimal, client_order_id: str) -> OrderResult: ...
    def get_order(self, order_id: str) -> dict[str, Any]: ...


def floor_decimal(value: Decimal, increment: Decimal) -> Decimal:
    if increment <= 0:
        raise ValueError("increment must be positive")
    return (value / increment).to_integral_value(rounding=ROUND_DOWN) * increment


def _as_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        return response.to_dict()
    try:
        return dict(response)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise TypeError(f"Unsupported Coinbase response type: {type(response)!r}") from exc


class CoinbaseAdvancedBroker:
    """Thin adapter around Coinbase's official Advanced Trade Python SDK."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        try:
            from coinbase.rest import RESTClient
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "coinbase-advanced-py is required for live trading. Install the production dependencies."
            ) from exc
        self._client = RESTClient(timeout=timeout_seconds, rate_limit_headers=True)

    def balances(self, base_currency: str, quote_currency: str) -> Balances:
        response = _as_dict(self._client.get_accounts(limit=250))
        values: dict[str, Decimal] = {}
        for account in response.get("accounts", []):
            currency = str(account.get("currency", "")).upper()
            available = account.get("available_balance", {})
            values[currency] = values.get(currency, Decimal("0")) + Decimal(str(available.get("value", "0")))
        return Balances(
            quote_available=values.get(quote_currency.upper(), Decimal("0")),
            base_available=values.get(base_currency.upper(), Decimal("0")),
        )

    def market_buy(self, product_id: str, quote_size: Decimal, client_order_id: str) -> OrderResult:
        raw = _as_dict(
            self._client.market_order_buy(
                client_order_id=client_order_id,
                product_id=product_id,
                quote_size=format(quote_size, "f"),
            )
        )
        success = bool(raw.get("success", False))
        order_id = (raw.get("success_response") or {}).get("order_id")
        return OrderResult(client_order_id, order_id, success, raw)

    def market_sell(self, product_id: str, base_size: Decimal, client_order_id: str) -> OrderResult:
        raw = _as_dict(
            self._client.market_order_sell(
                client_order_id=client_order_id,
                product_id=product_id,
                base_size=format(base_size, "f"),
            )
        )
        success = bool(raw.get("success", False))
        order_id = (raw.get("success_response") or {}).get("order_id")
        return OrderResult(client_order_id, order_id, success, raw)

    def get_order(self, order_id: str) -> dict[str, Any]:
        return _as_dict(self._client.get_order(order_id))
