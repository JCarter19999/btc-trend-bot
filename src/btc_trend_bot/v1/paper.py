from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class PaperAccount:
    cash: float = 500.0
    btc: float = 0.0
    average_entry: float = 0.0

    def equity(self, mark_price: float) -> float:
        return self.cash + self.btc * mark_price

    def buy(self, price: float, notional: float, fee_bps: float) -> dict:
        if self.btc > 0:
            raise ValueError("Only one position is allowed")
        if notional > self.cash:
            raise ValueError("Insufficient paper cash")
        fee = notional * fee_bps / 10000
        quantity = (notional - fee) / price
        self.cash -= notional
        self.btc = quantity
        self.average_entry = price
        return {"side": "BUY", "price": price, "quantity": quantity, "fee": fee, "timestamp": datetime.now(timezone.utc).isoformat()}

    def sell_all(self, price: float, fee_bps: float) -> dict:
        if self.btc <= 0:
            raise ValueError("No paper BTC position")
        gross = self.btc * price
        fee = gross * fee_bps / 10000
        quantity = self.btc
        pnl = gross - fee - quantity * self.average_entry
        self.cash += gross - fee
        self.btc = 0.0
        self.average_entry = 0.0
        return {"side": "SELL", "price": price, "quantity": quantity, "fee": fee, "net_pnl": pnl, "timestamp": datetime.now(timezone.utc).isoformat()}
