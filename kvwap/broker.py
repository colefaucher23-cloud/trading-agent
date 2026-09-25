"""Order routing: a local simulator and Alpaca (free paper trading)."""
from __future__ import annotations

import itertools
import os
import uuid
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import pandas as pd


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    qty: int
    status: str  # "filled", "accepted", "rejected", "dry_run", ...
    filled_qty: int = 0
    avg_price: Optional[float] = None
    message: str = ""


class Broker(Protocol):
    def submit_market_order(self, symbol: str, side: str, qty: int) -> OrderResult: ...
    def refresh(self, order: OrderResult) -> OrderResult: ...


class SimulatedBroker:
    """Fills market orders instantly at ``price_fn()`` plus a half-spread cost.

    ``cost_bps`` is charged against you (buys fill higher, sells lower); see the
    slippage-modeling skill for how to pick it for a given name.
    """

    def __init__(self, price_fn: Callable[[], float], cost_bps: float = 0.0):
        self.price_fn = price_fn
        self.cost_bps = cost_bps
        self._ids = itertools.count(1)

    def submit_market_order(self, symbol: str, side: str, qty: int) -> OrderResult:
        px = self.price_fn() * (1.0 + (1 if side == "buy" else -1) * self.cost_bps * 1e-4)
        return OrderResult(f"sim-{next(self._ids)}", symbol, side, qty, "filled", qty, px)

    def refresh(self, order: OrderResult) -> OrderResult:
        return order


class DryRunBroker:
    """Logs what would be sent and fills nothing."""

    def __init__(self):
        self._ids = itertools.count(1)

    def submit_market_order(self, symbol: str, side: str, qty: int) -> OrderResult:
        return OrderResult(f"dry-{next(self._ids)}", symbol, side, qty, "dry_run", 0, None, "not sent (dry run)")

    def refresh(self, order: OrderResult) -> OrderResult:
        return order


class AlpacaBroker:
    """Alpaca Trading API v2. Defaults to the free paper-trading endpoint.

    Live trading needs both ``live=True`` and the environment variable
    ``KVWAP_ALLOW_LIVE=yes`` so a config typo cannot route real orders.
    """

    PAPER = "https://paper-api.alpaca.markets"
    LIVE = "https://api.alpaca.markets"

    def __init__(self, live: bool = False, key_id: Optional[str] = None, secret: Optional[str] = None):
        if live and os.environ.get("KVWAP_ALLOW_LIVE") != "yes":
            raise RuntimeError("live trading requested but KVWAP_ALLOW_LIVE=yes is not set")
        self.base = self.LIVE if live else self.PAPER
        self.key_id = key_id or os.environ.get("APCA_API_KEY_ID")
        self.secret = secret or os.environ.get("APCA_API_SECRET_KEY")
        if not self.key_id or not self.secret:
            raise RuntimeError("set APCA_API_KEY_ID and APCA_API_SECRET_KEY (free paper account at alpaca.markets)")

    def _req(self, method: str, path: str, **kw):
        import requests

        headers = {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret}
        resp = requests.request(method, self.base + path, headers=headers, timeout=30, **kw)
        if resp.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {path}: {resp.status_code} {resp.text}")
        return resp.json()

    def submit_market_order(self, symbol: str, side: str, qty: int) -> OrderResult:
        body = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": f"kvwap-{uuid.uuid4().hex[:20]}",
        }
        try:
            o = self._req("POST", "/v2/orders", json=body)
        except RuntimeError as exc:
            return OrderResult("", symbol, side, qty, "rejected", message=str(exc))
        return self._to_result(o)

    def refresh(self, order: OrderResult) -> OrderResult:
        if not order.order_id:
            return order
        return self._to_result(self._req("GET", f"/v2/orders/{order.order_id}"))

    @staticmethod
    def _to_result(o: dict) -> OrderResult:
        avg = o.get("filled_avg_price")
        return OrderResult(
            order_id=o["id"],
            symbol=o["symbol"],
            side=o["side"],
            qty=int(float(o["qty"])),
            status=o["status"],
            filled_qty=int(float(o.get("filled_qty") or 0)),
            avg_price=float(avg) if avg else None,
        )

    def calendar(self, day) -> Optional[dict]:
        """Exchange session for ``day`` ({'open': 'HH:MM', 'close': 'HH:MM'}) or None if closed."""
        d = pd.Timestamp(day).strftime("%Y-%m-%d")
        rows = self._req("GET", "/v2/calendar", params={"start": d, "end": d})
        return rows[0] if rows and rows[0].get("date") == d else None
