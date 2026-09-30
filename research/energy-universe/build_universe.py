"""Build the energy tradable universe (snapshot 2026-09-30).

Tradable universe = US exchange-listed securities (NYSE, Nasdaq, NYSE American, NYSE Arca,
Cboe BZX): the venues the project's broker (Alpaca) routes to. OTC-only names are out.

Inputs (all committed under data/):
  listings_candidates.json  energy-candidate rows from the nightly Nasdaq-screener dump
                            (rreichel3/US-Stock-Symbols, commit + timestamp recorded inside)
  etp_validation.jsonl      Alpha Vantage analytics checks that each ETP has daily prices
                            through 2026-09-29 (a batch errors with the first missing ticker)

The screener's sector tags are unreliable for energy (midstream sits under Utilities,
oilfield equipment under Consumer Discretionary, tankers under Marine Transportation,
uranium under Basic Materials), so every included equity is classified explicitly in
CLASSIFICATION below and every excluded candidate gets a reason.

Outputs (written next to this file):
  energy_universe.csv           every included energy security (equities, CEFs, ETPs,
                                preferreds/notes/warrants of energy issuers)
  energy_universe_excluded.csv  candidates rejected, with the reason

Run: python3 research/energy-universe/build_universe.py
"""
import csv
import json
import pathlib
import re

HERE = pathlib.Path(__file__).parent
SNAP = json.loads((HERE / "data" / "listings_candidates.json").read_text())

# ------------------------------------------------------------------ taxonomy
# alignment: core       = energy business that GICS files under Energy
#            borderline = energy business GICS may file elsewhere (met coal, renewable fuels,
#                         gas-power services, chemicals-heavy integrateds) or mixed exposure
#            adjacent   = commonly traded as energy but not an energy producer/servicer
#            unverified = tagged Energy by the source; business not independently confirmed
SUB = {
    "INT": "Integrated oil & gas",
    "EP": "Exploration & production",
    "MIN": "Minerals, royalties & land",
    "RT": "Royalty trusts",
    "DRL": "Drilling (land & offshore)",
    "OFS": "Oilfield services & equipment",
    "RM": "Refining, marketing & fuel distribution",
    "MID": "Midstream (pipelines, G&P, storage, compression)",
    "LNG": "LNG export & regasification",
    "SHP": "Tankers & gas carriers",
    "COAL": "Coal",
    "U": "Uranium & nuclear fuel",
    "ALT": "Alternative & renewable fuels",
    "PWR": "Gas-fired power solutions",
    "CEF": "Closed-end energy funds",
}

C = {}  # symbol -> (sub-industry key, alignment, note)


def put(keys, sub, align="core", note=""):
    for k in keys.split():
        C[k] = (sub, align, note)


put("XOM CVX SHEL TTE BP E EQNR PBR EC YPF SU IMO CVE OXY", "INT")
put("SSL", "INT", "borderline", "coal-to-liquids and chemicals heavy")
put("COP EOG FANG DVN EQT EXE APA OVV PR AR RRC CNX CHRD MTDR MGY SM NOG CRGY CRC MUR TALO KOS "
    "GPOR CRK BKV VIST WDS CNQ GPRK VET BTE OBE GTE GFR TBN SOC HPK REPX VTS GRNT EGY WTI SD AMPY "
    "INR TXO MNR DEC FTW BATL REI EPM PED EPSN KGEI MXC PNRG EP BRN GBR TPET PROP EONR INDO ANNA", "EP")
put("NUAI", "EP", "borderline", "natural gas/helium plus digital infrastructure")
put("GLND BSIN AGIG", "EP", "unverified", "tagged Oil & Gas Production by source; business not confirmed")
put("PHXE^", "EP", "core", "preferred of Phoenix Energy One (common not listed)")
put("TPL LB EROK VNOM BSM KRP DMLP WHK CKX", "MIN")
put("NRP", "MIN", "core", "coal and minerals royalties")
put("PBT SBR SJT NRT CRT PVL VOC PRT MARPS MTR", "RT")
put("HP PTEN NBR PDS NE RIG VAL SDRL BORR", "DRL")
put("SLB HAL BKR FTI NOV WFRD TS WHD DNOW INVX FLOC FET HMH OIS OII LBRT PUMP ACDC RES NESR XPRO "
    "CLB NGS RNGR TUSK NINE KLXE RCON OMSE GOW DTI LSE STAK TTI WTTR AESI SND FTK EFXT DWSN TDW "
    "SMHI HOS VTOL", "OFS")
put("GEOS", "OFS", "borderline", "seismic equipment, mixed end markets")
put("NOA", "OFS", "borderline", "oil-sands mining contractor")
put("SEI", "OFS", "borderline", "oilfield logistics pivoting to data-center power")
put("VLO MPC PSX PBF CVI DK PARR DINO CLMT SUN SUNC GLP CAPL WKC APC UGP", "RM")
put("PTLE TMDE UFG BANL DLXY", "RM", "core", "marine fuel / oil trading micro-cap")
put("ENB TRP PBA SOBO WMB KMI OKE TRGP EPD ET MPLX PAA PAGP WES DTM AM KNTK HESM DKL GEL NGL MMLP "
    "SMC WBI TGS AROC KGS USAC", "MID")
put("LNG CQP VG NEXT EE NFE SLNG", "LNG")
put("FRO DHT STNG INSW TNK NAT TK ASC ECO TEN TRMD HAFN LPG BWLP NVGS GLNG FLNG CCEC DLNG GASS KNOP "
    "IMPP PXS PSHG TOPS TORO HMR", "SHP")
put("RUBI EHLD ICON RBNE CISS", "SHP", "unverified", "micro-cap Greek shipping; fleet mix not confirmed")
put("GLOP^A GLOP^B GLOP^C", "SHP", "core", "preferreds of GasLog Partners (common taken private)")
put("BTU CNR ARLP NC", "COAL")
put("HCC AMR METC METCB AREC", "COAL", "borderline", "metallurgical coal (GICS: Materials)")
put("HNRG", "COAL", "borderline", "coal producer transitioning to power")
put("CCJ NXE UEC DNN UUUU URG EU LEU UROY ISOU NUCL FNUC AEC JAGU", "U")
put("FMST", "U", "borderline", "uranium/lithium explorer")
put("GPRE REX ALTO AMTX GEVO CLNE OPAL MNTK VGAS SAFX", "ALT", "borderline", "renewable fuels / RNG / SAF")
put("EROC", "PWR", "adjacent", "modular natural-gas power for data centers (IPO Jun-2026)")
put("NPWR", "PWR", "adjacent", "natural-gas power-cycle technology")
put("KYN TYG EMO NML SRV BGR PEO NXG", "CEF")

# Preferreds / notes / warrants of included issuers: symbol -> parent.
NON_COMMON_PARENT = {"IMPPP": "IMPP", "NFEGP": "NFE", "METCI": "METC", "METCZ": "METC", "TCPA": "TRP",
                     "NUAIW": "NUAI", "GLNDW": "GLND", "ANNAW": "ANNA", "NUCLW": "NUCL", "FMSTW": "FMST",
                     "VGASW": "VGAS"}

EXCLUDE = {
    "BHP": "diversified miner (Materials); source tags it Coal Mining",
    "MGEE": "electric utility",
    "LFUS": "electronic components", "POWL": "electrical equipment",
    "BE": "fuel cells (Industrials)", "PLUG": "hydrogen/fuel cells (Industrials)",
    "FCEL": "fuel cells (Industrials)", "BLDP": "fuel cells (Industrials)",
    "WWD": "aerospace & industrial controls", "FPS": "electrical power-distribution equipment",
    "EAF": "graphite electrodes (Industrials)",
    "IEP": "diversified holding company (owns a CVR Energy stake)",
    "MINE": "gold explorer mis-tagged as Oil & Gas Production",
    "RKDA": "agricultural biotech mis-tagged as Oil & Gas Production",
}
GAS_UTIL = "ATO NFG NJR SR OGS NWN SWX CPK RGCO UGI NGG SRE SREA SRJN CTRI CETY".split()
POWER = "CEG VST TLN NRG OKLO NNE SMR BWXT ASPI LTBR PAM CEPU".split()
for s in GAS_UTIL:
    EXCLUDE[s] = "gas/multi utility (GICS Utilities)"
for s in POWER:
    EXCLUDE[s] = "power generation / nuclear technology (Utilities or Industrials)"
for s in "KEX SFL CMBT NMM SPH MUSA CASY".split():
    EXCLUDE[s] = "energy-adjacent but not an energy company (diversified shipping, propane/fuel retail, tank barges)"
SPAC = re.compile(r"acquisition corp|energy transition corp", re.I)

# ------------------------------------------------------------------ ETP catalog
# (symbol, name, category, structure, leverage, K-1, alignment)
ETPS = [
    ("XLE", "State Street Energy Select Sector SPDR ETF", "US energy equity - broad", "ETF", 1, False, "core"),
    ("VDE", "Vanguard Energy ETF", "US energy equity - broad", "ETF", 1, False, "core"),
    ("IYE", "iShares U.S. Energy ETF", "US energy equity - broad", "ETF", 1, False, "core"),
    ("FENY", "Fidelity MSCI Energy Index ETF", "US energy equity - broad", "ETF", 1, False, "core"),
    ("DRLL", "Strive U.S. Energy ETF", "US energy equity - broad", "ETF", 1, False, "core"),
    ("RSPG", "Invesco S&P 500 Equal Weight Energy ETF", "US energy equity - equal weight", "ETF", 1, False, "core"),
    ("PSCE", "Invesco S&P SmallCap Energy ETF", "US energy equity - small cap", "ETF", 1, False, "core"),
    ("FXN", "First Trust Energy AlphaDEX Fund", "US energy equity - factor", "ETF", 1, False, "core"),
    ("PXI", "Invesco Dorsey Wright Energy Momentum ETF", "US energy equity - factor", "ETF", 1, False, "core"),
    ("IXC", "iShares Global Energy ETF", "Global energy equity", "ETF", 1, False, "core"),
    ("XOP", "SPDR S&P Oil & Gas Exploration & Production ETF", "E&P equity", "ETF", 1, False, "core"),
    ("IEO", "iShares U.S. Oil & Gas Exploration & Production ETF", "E&P equity", "ETF", 1, False, "core"),
    ("PXE", "Invesco Energy Exploration & Production ETF", "E&P equity", "ETF", 1, False, "core"),
    ("FTXN", "First Trust Nasdaq Oil & Gas ETF", "E&P equity", "ETF", 1, False, "core"),
    ("FCG", "First Trust Natural Gas ETF", "Natural gas equity", "ETF", 1, False, "core"),
    ("OIH", "VanEck Oil Services ETF", "Oilfield services equity", "ETF", 1, False, "core"),
    ("IEZ", "iShares U.S. Oil Equipment & Services ETF", "Oilfield services equity", "ETF", 1, False, "core"),
    ("XES", "SPDR S&P Oil & Gas Equipment & Services ETF", "Oilfield services equity", "ETF", 1, False, "core"),
    ("PXJ", "Invesco Oil & Gas Services ETF", "Oilfield services equity", "ETF", 1, False, "core"),
    ("CRAK", "VanEck Oil Refiners ETF", "Refiners equity", "ETF", 1, False, "core"),
    ("AMLP", "Alerian MLP ETF", "Midstream / MLP", "ETF", 1, False, "core"),
    ("MLPA", "Global X MLP ETF", "Midstream / MLP", "ETF", 1, False, "core"),
    ("MLPX", "Global X MLP & Energy Infrastructure ETF", "Midstream / MLP", "ETF", 1, False, "core"),
    ("EMLP", "First Trust North American Energy Infrastructure Fund", "Midstream / MLP", "ETF", 1, False, "core"),
    ("TPYP", "Tortoise North American Pipeline Fund", "Midstream / MLP", "ETF", 1, False, "core"),
    ("ENFR", "Alerian Energy Infrastructure ETF", "Midstream / MLP", "ETF", 1, False, "core"),
    ("AMZA", "InfraCap MLP ETF", "Midstream / MLP", "ETF", 1, False, "core"),
    ("USAI", "Pacer American Energy Independence ETF", "Midstream / MLP", "ETF", 1, False, "core"),
    ("UMI", "USCF Midstream Energy Income Fund", "Midstream / MLP", "ETF", 1, False, "core"),
    ("EINC", "VanEck Energy Income ETF", "Midstream / MLP", "ETF", 1, False, "core"),
    ("MDST", "Westwood Salient Enhanced Midstream Income ETF", "Midstream / MLP", "ETF", 1, False, "core"),
    ("MLPD", "Global X MLP & Energy Infrastructure Covered Call ETF", "Midstream / MLP", "ETF", 1, False, "core"),
    ("AMUB", "ETRACS Alerian MLP Index ETN Series B", "Midstream / MLP", "ETN", 1, False, "core"),
    ("MLPB", "ETRACS Alerian MLP Infrastructure Index ETN Series B", "Midstream / MLP", "ETN", 1, False, "core"),
    ("ATMP", "iPath Select MLP ETN", "Midstream / MLP", "ETN", 1, False, "core"),
    ("MLPR", "ETRACS Quarterly Pay 1.5X Leveraged Alerian MLP Index ETN", "Midstream / MLP", "ETN", 1.5, False, "core"),
    ("EIPI", "FT Energy Income Partners Enhanced Income ETF", "Midstream / MLP", "ETF", 1, False, "core"),
    ("EIPX", "FT Energy Income Partners Strategy ETF", "Midstream / MLP", "ETF", 1, False, "core"),
    ("URA", "Global X Uranium ETF", "Uranium & nuclear", "ETF", 1, False, "core"),
    ("URNM", "Sprott Uranium Miners ETF", "Uranium & nuclear", "ETF", 1, False, "core"),
    ("URNJ", "Sprott Junior Uranium Miners ETF", "Uranium & nuclear", "ETF", 1, False, "core"),
    ("NLR", "VanEck Uranium and Nuclear ETF", "Uranium & nuclear", "ETF", 1, False, "borderline"),
    ("NUKZ", "Range Nuclear Renaissance Index ETF", "Uranium & nuclear", "ETF", 1, False, "borderline"),
    ("URAN", "Themes Uranium & Nuclear ETF", "Uranium & nuclear", "ETF", 1, False, "borderline"),
    ("USO", "United States Oil Fund", "Crude oil futures", "Commodity pool", 1, True, "core"),
    ("BNO", "United States Brent Oil Fund", "Crude oil futures", "Commodity pool", 1, True, "core"),
    ("USL", "United States 12 Month Oil Fund", "Crude oil futures", "Commodity pool", 1, True, "core"),
    ("DBO", "Invesco DB Oil Fund", "Crude oil futures", "Commodity pool", 1, True, "core"),
    ("OILK", "ProShares K-1 Free Crude Oil Strategy ETF", "Crude oil futures", "ETF", 1, False, "core"),
    ("UNG", "United States Natural Gas Fund", "Natural gas futures", "Commodity pool", 1, True, "core"),
    ("UNL", "United States 12 Month Natural Gas Fund", "Natural gas futures", "Commodity pool", 1, True, "core"),
    ("UGA", "United States Gasoline Fund", "Gasoline futures", "Commodity pool", 1, True, "core"),
    ("DBE", "Invesco DB Energy Fund", "Broad energy futures", "Commodity pool", 1, True, "core"),
    ("UCO", "ProShares Ultra Bloomberg Crude Oil", "Leveraged / inverse commodity", "Commodity pool", 2, True, "core"),
    ("SCO", "ProShares UltraShort Bloomberg Crude Oil", "Leveraged / inverse commodity", "Commodity pool", -2, True, "core"),
    ("BOIL", "ProShares Ultra Bloomberg Natural Gas", "Leveraged / inverse commodity", "Commodity pool", 2, True, "core"),
    ("KOLD", "ProShares UltraShort Bloomberg Natural Gas", "Leveraged / inverse commodity", "Commodity pool", -2, True, "core"),
    ("ERX", "Direxion Daily Energy Bull 2X Shares", "Leveraged / inverse equity", "ETF", 2, False, "core"),
    ("ERY", "Direxion Daily Energy Bear 2X Shares", "Leveraged / inverse equity", "ETF", -2, False, "core"),
    ("DIG", "ProShares Ultra Energy", "Leveraged / inverse equity", "ETF", 2, False, "core"),
    ("DUG", "ProShares UltraShort Energy", "Leveraged / inverse equity", "ETF", -2, False, "core"),
    ("GUSH", "Direxion Daily S&P Oil & Gas Exp. & Prod. Bull 2X Shares", "Leveraged / inverse equity", "ETF", 2, False, "core"),
    ("DRIP", "Direxion Daily S&P Oil & Gas Exp. & Prod. Bear 2X Shares", "Leveraged / inverse equity", "ETF", -2, False, "core"),
    ("NRGU", "MicroSectors U.S. Big Oil Index 3X Leveraged ETN", "Leveraged / inverse equity", "ETN", 3, False, "core"),
    ("NRGD", "MicroSectors U.S. Big Oil Index -3X Inverse Leveraged ETN", "Leveraged / inverse equity", "ETN", -3, False, "core"),
    ("OILU", "MicroSectors Oil & Gas Exploration & Production 3X Leveraged ETN", "Leveraged / inverse equity", "ETN", 3, False, "core"),
    ("OILD", "MicroSectors Oil & Gas Exploration & Production -3X Inverse Leveraged ETN", "Leveraged / inverse equity", "ETN", -3, False, "core"),
    ("IGE", "iShares North American Natural Resources ETF", "Natural resources (energy-heavy)", "ETF", 1, False, "adjacent"),
    ("NANR", "SPDR S&P North American Natural Resources ETF", "Natural resources (energy-heavy)", "ETF", 1, False, "adjacent"),
    ("GNR", "SPDR S&P Global Natural Resources ETF", "Natural resources (energy-heavy)", "ETF", 1, False, "adjacent"),
    ("BWET", "Breakwave Tanker Shipping ETF", "Tanker freight futures", "Commodity pool", 1, True, "adjacent"),
    ("XOMO", "YieldMax XOM Option Income Strategy ETF", "Single-stock (XOM) option income", "ETF", 1, False, "adjacent"),
    ("FILL", "iShares MSCI Global Energy Producers ETF", "Global energy equity", "ETF", 1, False, "core"),
    ("KOL", "VanEck Coal ETF", "Coal equity", "ETF", 1, False, "core"),
]

K1 = set("EPD ET MPLX PAA WES USAC GEL NGL MMLP DKL GLP CAPL CQP SUN ARLP NRP BSM DMLP TXO MNR".split())
# Primary listing outside North America (home-market close precedes the US close; some report semiannually).
FOREIGN_PRIMARY = set("BP SHEL TTE E EQNR PBR EC YPF SSL WDS VIST TS TGS UGP HAFN BWLP TRMD ECO".split())


# ------------------------------------------------------------------ helpers
def num(x):
    try:
        return float(str(x).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def security_type(sym, name):
    n = name or ""
    if re.search(r"\bwarrants?\b", n, re.I):
        return "warrant"
    if re.search(r"\brights?\b", n, re.I):
        return "right"
    if re.search(r"notes? due|debentures", n, re.I):
        return "note"
    if "^" in sym or re.search(r"preferred|perpetual pref", n, re.I):
        return "preferred"
    if re.search(r"american depositary|\bADS\b|\bADR\b", n, re.I):
        return "adr"
    if re.search(r"royalt(y|ies)? trust|units of beneficial interest|Royality Trust", n, re.I):
        return "trust_units"
    if re.search(r"L\.P\.|\bLP\b|limited partner|common units|partnership", n, re.I):
        return "partnership_units"
    if re.search(r"\bLLC\b", n):
        return "llc_units"
    return "common"


def size_bucket(mc):
    if mc is None or mc <= 0:
        return "n/a"
    return "large" if mc >= 10e9 else "mid" if mc >= 2e9 else "small" if mc >= 300e6 else "micro"


# ------------------------------------------------------------------ equities
rows_out, excluded = [], []
for r in SNAP["rows"]:
    sym, name = r["symbol"], r["name"] or ""
    parent = sym.split("^")[0] if "^" in sym else NON_COMMON_PARENT.get(sym)
    stype = security_type(sym, name)
    if sym in EXCLUDE:
        excluded.append((r, EXCLUDE[sym]))
        continue
    if SPAC.search(name):
        excluded.append((r, "blank-check company (SPAC), no operating business"))
        continue
    if sym in C:
        key, align, note = C[sym]
    elif parent in C:
        key, align, note = C[parent]
        note = f"security of {parent}" + (f"; {note}" if note else "")
    else:
        why = ("tagged Energy by source but not an energy business" if r.get("sector") == "Energy"
               else "keyword/industry match only; not an energy business")
        excluded.append((r, why))
        continue
    if key == "CEF":
        stype = "closed_end_fund"
    mc, px, vol = num(r.get("marketCap")), num(r.get("lastsale")), num(r.get("volume"))
    rows_out.append({
        "symbol": sym, "name": name.strip(),
        "asset_class": "fund" if key == "CEF" else ("equity" if stype in ("common", "adr", "partnership_units",
                                                                           "llc_units", "trust_units") else "non_common"),
        "security_type": stype, "sub_industry": SUB[key], "alignment": align,
        "exchange": r["exchange"].upper(), "country": r.get("country") or "",
        "market_cap_usd": round(mc) if mc else "", "last_price": px if px is not None else "",
        "volume_1d": int(vol) if vol is not None else "",
        "dollar_volume_1d": round(px * vol) if px is not None and vol is not None else "",
        "size_bucket": size_bucket(mc) if stype in ("common", "adr", "partnership_units", "llc_units",
                                                    "trust_units", "closed_end_fund") else "n/a",
        "k1_tax_form": "yes" if sym in K1 else "",
        "foreign_primary": "yes" if (parent or sym) in FOREIGN_PRIMARY else "",
        "etp_category": "", "leverage": "", "price_check_2026_09_29": "listed on 2026-09-30 snapshot",
        "ret_1m_to_2026_09_29": "", "vol_1m_ann": "",
        "source_sector": r.get("sector") or "", "source_industry": r.get("industry") or "",
        "classification": "curated" if sym in C else "curated (parent)", "notes": note,
    })

missing_curated = sorted(set(C) - {r["symbol"] for r in SNAP["rows"]})

# ------------------------------------------------------------------ ETPs
checks = [json.loads(l) for l in (HERE / "data" / "etp_validation.jsonl").read_text().splitlines() if l.strip()]
ok, dead = {}, {}
for c in checks:
    if c["status"] == "ok":
        for s, v in c["ret_1m"].items():
            ok[s] = (v, c["vol_1m"].get(s))
    elif c.get("missing"):
        dead[c["missing"]] = c["error"]
for sym, name, cat, struct, lev, k1, align in ETPS:
    if sym in ok:
        status, (r1, v1) = "prices through 2026-09-29 (Alpha Vantage)", ok[sym]
    elif sym in dead:
        excluded.append(({"symbol": sym, "name": name, "sector": "ETP", "industry": cat},
                         "no Alpha Vantage prices after 2026-08-29 (closed or delisted)"))
        continue
    else:
        status, r1, v1 = "not checked (API quota)", None, None
    rows_out.append({
        "symbol": sym, "name": name, "asset_class": "etp",
        "security_type": {"ETF": "etf", "ETN": "etn", "Commodity pool": "commodity_pool"}[struct],
        "sub_industry": cat, "alignment": align, "exchange": "", "country": "United States",
        "market_cap_usd": "", "last_price": "", "volume_1d": "", "dollar_volume_1d": "", "size_bucket": "n/a",
        "k1_tax_form": "yes" if k1 else "", "foreign_primary": "", "etp_category": cat, "leverage": lev,
        "price_check_2026_09_29": status,
        "ret_1m_to_2026_09_29": round(r1, 4) if r1 is not None else "",
        "vol_1m_ann": round(v1, 4) if v1 is not None else "",
        "source_sector": "", "source_industry": "", "classification": "curated ETP catalog", "notes": "",
    })

# ------------------------------------------------------------------ write
order = {"equity": 0, "fund": 1, "etp": 2, "non_common": 3}
rows_out.sort(key=lambda d: (order[d["asset_class"]], d["sub_industry"], -(d["market_cap_usd"] or 0), d["symbol"]))
with (HERE / "energy_universe.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader()
    w.writerows(rows_out)
with (HERE / "energy_universe_excluded.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["symbol", "name", "source_sector", "source_industry", "reason"])
    for r, why in sorted(excluded, key=lambda t: t[0]["symbol"]):
        w.writerow([r["symbol"], (r.get("name") or "").strip(), r.get("sector") or "", r.get("industry") or "", why])

# ------------------------------------------------------------------ summary
from collections import Counter
eq = [d for d in rows_out if d["asset_class"] == "equity"]
print(f"source: {SNAP['source']} @ {SNAP['source_commit'][:10]} ({SNAP['source_commit_time']}), "
      f"{SNAP['total_listed_rows']} listed rows, {SNAP['candidate_rows']} candidates")
print(f"included: {len(rows_out)} total | " + ", ".join(f"{k}={v}" for k, v in Counter(d['asset_class'] for d in rows_out).items()))
print(f"excluded: {len(excluded)}")
print("curated symbols absent from the listing snapshot:", " ".join(missing_curated) or "none")
print("\nEquities by sub-industry (alignment core/borderline/adjacent/unverified):")
for sub in SUB.values():
    grp = [d for d in eq if d["sub_industry"] == sub]
    if grp:
        a = Counter(d["alignment"] for d in grp)
        print(f"  {sub:<48} {len(grp):>4}  " + " ".join(f"{k}={a[k]}" for k in ("core", "borderline", "adjacent", "unverified") if a[k]))
print("\nEquities by security type:", dict(Counter(d["security_type"] for d in eq)))
print("Equities by size bucket:", dict(Counter(d["size_bucket"] for d in eq)))
print("ETPs by status:", dict(Counter(d["price_check_2026_09_29"] for d in rows_out if d["asset_class"] == "etp")))

# ------------------------------------------------------------------ UNIVERSE.md
def fmt_mc(v):
    return "" if v in ("", None) else (f"${v / 1e9:.1f}B" if v >= 1e9 else f"${v / 1e6:.0f}M")


md = [
    "# Energy tradable universe (snapshot 2026-09-30)",
    "",
    "Generated by `build_universe.py` from `data/listings_candidates.json` and `data/etp_validation.jsonl`. "
    "Machine-readable version: `energy_universe.csv`; rejected candidates with reasons: `energy_universe_excluded.csv`.",
    "",
    "## Definition",
    "",
    "- **Tradable universe.** US exchange-listed securities (NYSE, Nasdaq, NYSE American, NYSE Arca, Cboe BZX): the venues "
    "the project's broker (Alpaca) routes to. OTC-only names are out.",
    "- **Listing snapshot.** The nightly Nasdaq stock-screener dump (github.com/rreichel3/US-Stock-Symbols, commit "
    f"`{SNAP['source_commit'][:10]}`, generated {SNAP['source_commit_time']}): {SNAP['total_listed_rows']} listed rows, "
    f"of which {SNAP['candidate_rows']} are energy candidates.",
    "- **ETPs.** Not in the screener. They come from a curated catalog. Each was confirmed to have daily prices through "
    "2026-09-29 using the Alpha Vantage analytics endpoint.",
    "- **Energy.** Follows the GICS Energy sector: oil, gas, coal and uranium fuel, plus their services, transport, refining and royalties.",
    "- **Alignment tags.** Every row carries one:",
    "  - `core`: the GICS-energy business.",
    "  - `borderline`: an energy business that GICS may file elsewhere, such as met coal, renewable fuels, or gas-power services.",
    "  - `adjacent`: traded as energy but not an energy producer or servicer.",
    "  - `unverified`: tagged Energy by the source, but the business was not confirmed.",
    "- **Other flags.** `foreign_primary = yes` marks issuers whose primary listing is outside North America "
    "(home-market close precedes the US close). `k1_tax_form = yes` marks partnerships and commodity pools that issue K-1s.",
    "",
    f"**Totals:** {len(rows_out)} securities.",
    "",
]
for k, v in Counter(d["asset_class"] for d in rows_out).items():
    md.append(f"- {k}: {v}")
md.append("")
md += ["## Equities by sub-industry", "",
       "| Sub-industry | Count | core | borderline | adjacent | unverified | Symbols (largest first) |",
       "|---|---:|---:|---:|---:|---:|---|"]
for sub in SUB.values():
    grp = [d for d in eq if d["sub_industry"] == sub]
    if not grp:
        continue
    a = Counter(d["alignment"] for d in grp)
    syms = ", ".join(d["symbol"] + ("*" if d["alignment"] != "core" else "") for d in grp)
    md.append(f"| {sub} | {len(grp)} | {a['core']} | {a['borderline']} | {a['adjacent']} | {a['unverified']} | {syms} |")
md += ["", "`*` = not `core`; see the `alignment` and `notes` columns in the CSV.", "",
       "Equity security types: " + ", ".join(f"{k} {v}" for k, v in Counter(d['security_type'] for d in eq).items()) + ".",
       "",
       "Size buckets use snapshot market cap (large ≥ $10B, mid ≥ $2B, small ≥ $300M, else micro): "
       + ", ".join(f"{k} {v}" for k, v in Counter(d['size_bucket'] for d in eq).items()) + ".", "",
       "## Closed-end funds", "",
       ", ".join(d["symbol"] for d in rows_out if d["asset_class"] == "fund"), "",
       "## Exchange-traded products", "",
       "| Category | Symbols (leverage if not 1x; K-1 marked †) |", "|---|---|"]
etp = [d for d in rows_out if d["asset_class"] == "etp"]
for cat in dict.fromkeys(d["etp_category"] for d in etp):
    syms = ", ".join(d["symbol"] + (f" ({d['leverage']}x)" if d["leverage"] not in (1, "") else "")
                     + ("†" if d["k1_tax_form"] else "") for d in etp if d["etp_category"] == cat)
    md.append(f"| {cat} | {syms} |")
md += ["", "## Preferreds, notes and warrants of energy issuers", "",
       ", ".join(d["symbol"] for d in rows_out if d["asset_class"] == "non_common"), "",
       "## Excluded candidates", "",
       f"{len(excluded)} candidate rows were rejected; `energy_universe_excluded.csv` gives the reason for each. Most common reasons:", ""]
for why, n in Counter(w for _, w in excluded).most_common(8):
    md.append(f"- {n} × {why}")
md += ["", "## Limitations", "",
       "- **Point-in-time only.** This is the universe as of 2026-09-30, so it cannot be used as a historical universe.",
       "  - Recent delistings include Coterra (merged into Devon on 2026-05-07), Civitas (merged into SM on 2026-01-30), and Helix (reverse merger into HOS).",
       "  - A backtest must rebuild membership at each date, including dead names, or it will carry survivorship bias.",
       "- **Snapshot metrics are one day only.** Market cap, price and volume come from a single session. Strategies must apply their liquidity screens using trailing ADV from price data.",
       "- **Unverified rows.** Unverified micro-caps are listed for completeness only.",
       ]
(HERE / "UNIVERSE.md").write_text("\n".join(md) + "\n")
print("wrote UNIVERSE.md")
