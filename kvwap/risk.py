"""Pre-trade risk controls for child orders (see the risk-management skill)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskLimits:
    # child order <= max_participation * forecast bin volume (in shares)
    max_participation: float = 0.10
    # multiply the data feed's volume to approximate consolidated volume
    # (e.g. ~40 for Alpaca's IEX-only feed); used only for the participation cap
    volume_scale: float = 1.0
    max_child_notional: Optional[float] = None
    max_parent_notional: Optional[float] = None
    # if this file exists, the agent stops sending orders immediately
    kill_switch_file: str = "STOP"
    # ignore the participation cap on the last bin so the order completes
    final_sweep: bool = False


@dataclass
class RiskDecision:
    qty: int
    reasons: list


def kill_switch_engaged(limits: RiskLimits) -> bool:
    return bool(limits.kill_switch_file) and os.path.exists(limits.kill_switch_file)


def check_parent(quantity: int, ref_price: Optional[float], limits: RiskLimits) -> None:
    if quantity <= 0:
        raise ValueError("parent order quantity must be positive")
    if limits.max_parent_notional is not None and ref_price is not None:
        notional = quantity * ref_price
        if notional > limits.max_parent_notional:
            raise ValueError(
                f"parent order notional {notional:,.0f} exceeds max_parent_notional {limits.max_parent_notional:,.0f}"
            )


def clamp_child(
    qty: int,
    forecast_bin_volume: float,
    ref_price: Optional[float],
    limits: RiskLimits,
    last_bin: bool = False,
) -> RiskDecision:
    """Apply participation and notional caps; the excess rolls to later bins."""
    reasons = []
    q = max(int(qty), 0)
    if not (last_bin and limits.final_sweep):
        cap = int(limits.max_participation * forecast_bin_volume * limits.volume_scale)
        if q > cap:
            reasons.append(f"participation cap {cap} (forecast bin volume {forecast_bin_volume:,.0f})")
            q = cap
    if limits.max_child_notional is not None and ref_price:
        cap = int(limits.max_child_notional // ref_price)
        if q > cap:
            reasons.append(f"child notional cap {cap}")
            q = cap
    return RiskDecision(max(q, 0), reasons)
