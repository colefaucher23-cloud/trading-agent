"""Append-only JSONL execution journal (see the trade-journal skill)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Journal:
    def __init__(self, path: Optional[str] = None, echo: bool = True):
        self.path = path
        self.echo = echo
        self.events: list = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(self, event: str, ts=None, **fields) -> dict:
        rec = {"ts": str(ts) if ts is not None else datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        self.events.append(rec)
        if self.path:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec, default=_default) + "\n")
        if self.echo:
            brief = {k: v for k, v in rec.items() if k not in ("ts", "event", "params", "schedule")}
            print(f"[{rec['ts']}] {event}: {json.dumps(brief, default=_default)}", flush=True)
        return rec
