"""Load an AgentConfig (plus data/broker choices) from a TOML file."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field

from .agent import AgentConfig, ParentOrder
from .data import Session
from .risk import RiskLimits


@dataclass
class RunConfig:
    agent: AgentConfig
    data_source: str = "yahoo"
    data_options: dict = field(default_factory=dict)
    broker: str = "dry-run"  # dry-run | alpaca-paper | alpaca-live
    journal_path: str = "journal/executions.jsonl"


def load_config(path: str) -> RunConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    order = ParentOrder(**raw["order"])
    strat = dict(raw.get("strategy", {}))
    if strat.get("lam") in (0, "none", "None", False):
        strat["lam"] = None
    session = Session(**raw.get("session", {}))
    risk = RiskLimits(**raw.get("risk", {}))
    agent = AgentConfig(order=order, session=session, risk=risk, **strat)
    data = dict(raw.get("data", {}))
    source = data.pop("source", "yahoo")
    br = raw.get("broker", {})
    return RunConfig(
        agent=agent,
        data_source=source,
        data_options=data,
        broker=br.get("name", "dry-run"),
        journal_path=raw.get("journal", {}).get("path", "journal/executions.jsonl"),
    )
