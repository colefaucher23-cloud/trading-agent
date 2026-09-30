"""Extract energy-candidate rows from a clone of github.com/rreichel3/US-Stock-Symbols.

That repository republishes the Nasdaq stock screener (NYSE, Nasdaq and NYSE American
listings, including sector, industry, market cap and last-day volume) every night.
We keep only the rows that could plausibly be energy securities so the universe build
is reproducible from a small committed file:

    python3 extract_candidates.py /path/to/us-stock-symbols > listings_candidates.json

Selection is deliberately loose (sector tag, industry, energy keywords in the name);
build_universe.py makes the actual include/exclude decisions.
"""
import json
import pathlib
import re
import subprocess
import sys

SRC = pathlib.Path(sys.argv[1])

# Industries (Nasdaq taxonomy) that contain energy companies outside the "Energy" sector tag.
INDUSTRIES = {
    ("Consumer Discretionary", "Oil and Gas Field Machinery"),
    ("Consumer Discretionary", "Marine Transportation"),
    ("Consumer Discretionary", "Transportation Services"),
    ("Consumer Discretionary", "Other Specialty Stores"),
    ("Utilities", "Natural Gas Distribution"),
    ("Utilities", "Oil/Gas Transmission"),
    ("Utilities", "Oil & Gas Production"),
    ("Utilities", "Electric Utilities: Central"),
    ("Basic Materials", "Other Metals and Minerals"),
    ("Basic Materials", "Precious Metals"),
    ("Industrials", "Mining & Quarrying of Nonmetallic Minerals (No Fuels)"),
    ("Industrials", "Major Chemicals"),
    ("Industrials", "Industrial Machinery/Components"),
    ("Industrials", "Steel/Iron Ore"),
}
KEYWORDS = re.compile(
    r"oil|gas\b|petrol|energy|drill|pipeline|midstream|tanker|lng|uranium|nuclear|coal|refin|"
    r"offshore|crude|royalt|mlp|infrastructure|natural resources|fuel|gaslog", re.I)

rows = []
for ex in ("nasdaq", "nyse", "amex"):
    for r in json.loads((SRC / ex / f"{ex}_full_tickers.json").read_text()):
        r["exchange"] = ex
        sector, industry = r.get("sector") or "", r.get("industry") or ""
        if sector == "Energy" or (sector, industry) in INDUSTRIES or KEYWORDS.search(r.get("name") or ""):
            rows.append({k: r.get(k) for k in (
                "symbol", "name", "exchange", "sector", "industry", "country", "marketCap",
                "lastsale", "volume", "ipoyear")})

git = lambda *a: subprocess.run(["git", "-C", str(SRC), *a], capture_output=True, text=True).stdout.strip()
out = {
    "source": "https://github.com/rreichel3/US-Stock-Symbols (Nasdaq stock screener, NYSE/Nasdaq/NYSE American)",
    "source_commit": git("rev-parse", "HEAD"),
    "source_commit_time": git("log", "-1", "--format=%cI"),
    "total_listed_rows": sum(len(json.loads((SRC / ex / f"{ex}_full_tickers.json").read_text()))
                             for ex in ("nasdaq", "nyse", "amex")),
    "candidate_rows": len(rows),
    "rows": sorted(rows, key=lambda r: r["symbol"]),
}
json.dump(out, sys.stdout, indent=0)
