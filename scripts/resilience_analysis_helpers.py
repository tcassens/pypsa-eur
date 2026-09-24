# SPDX-FileCopyrightText: 2026 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT

"""
Shared helper functions for the near-optimal resilience analysis scripts.

Sections
--------
1. Constants
2. Capacities        (carrier extraction, aggregation, concentration metrics)
3. Net load          (stats from solved network)
4. Prices            (marginal price stats from solved network)
5. Shedding          (cost, time-series stats)
6. Scoring           (delta, CVaR, S, BCR)
7. Emissions         (profile match)
8. Directions        (load direction JSON)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from mga_helpers import extract_net_load


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

VOLL_ELEC = 10_000  # EUR/MWh
VOLL_HEAT = 50_000  # EUR/MWh (from prepare_network.py)

DIRECTION_DIMS = ["renewables", "storage", "heating", "tes", "backup", "nuclear"]


# ---------------------------------------------------------------------------
# 2. Capacities
# ---------------------------------------------------------------------------

def build_carrier_lookup(tech_categories: dict) -> dict:
    """
    Build a flat carrier → tech mapping from the projection config.

    Parameters
    ----------
    tech_categories : dict mapping tech name → list of carrier strings.

    Returns
    -------
    dict mapping carrier string → tech name.
    """
    return {
        carrier: tech
        for tech, carriers in tech_categories.items()
        for carrier in carriers
    }


def extract_carrier(name: str, carrier_lookup: dict) -> str | None:
    """
    Match a component name against known carriers by suffix (longest match first).

    Parameters
    ----------
    name : str  Component name, e.g. 'BE0 0 urban central gas boiler'.
    carrier_lookup : dict  Output of build_carrier_lookup.

    Returns
    -------
    Matched carrier string, or None if no match.
    """
    # strip a myopic per-horizon vintage-year suffix (e.g. "-2040")
    name = re.sub(r"-\d{4}$", "", name)
    for carrier in sorted(carrier_lookup, key=len, reverse=True):
        if name.endswith(carrier):
            return carrier
    return None


def parse_caps(caps: pd.DataFrame, carrier_lookup: dict) -> pd.DataFrame:
    """
    Add carrier, tech, and node columns to a capacities DataFrame.

    Parameters
    ----------
    caps : pd.DataFrame  Output of extract_optimal_capacities (component, name, attribute, value).
    carrier_lookup : dict  Output of build_carrier_lookup.

    Returns
    -------
    caps with additional columns: carrier, tech, node.
    """
    caps = caps.copy()
    caps["carrier"] = caps["name"].apply(lambda n: extract_carrier(n, carrier_lookup))
    caps["tech"] = caps["carrier"].map(carrier_lookup)
    caps["node"] = caps.apply(
        lambda r: r["name"][:-len(r["carrier"]) - 1] if r["carrier"] else None,
        axis=1,
    )
    return caps


def concentration_stats(
    caps: pd.DataFrame,
    tech_categories: dict,
    capital_costs: pd.Series | None = None,
) -> dict:
    """
    Compute Gini per tech category and HHI across categories.

    Parameters
    ----------
    caps : pd.DataFrame  Output of parse_caps.
    tech_categories : dict  Projection config tech categories.
    capital_costs : pd.Series | None
        Capital cost per component name (EUR/MW or EUR/MWh), indexed by name.
        When provided, shares and HHI are investment-weighted (value × capital_cost),
        matching the direction vector definition. Gini uses raw physical capacity.

    Returns
    -------
    dict with keys: gini_{tech} per category, hhi, share_{tech} per category.
    """
    carrier_lookup = build_carrier_lookup(tech_categories)
    if "tech" not in caps.columns:
        caps = parse_caps(caps, carrier_lookup)

    if capital_costs is not None:
        caps = caps.copy()
        caps["investment"] = caps["value"] * caps["name"].map(capital_costs).fillna(0.0)
    else:
        caps = caps.copy()
        caps["investment"] = caps["value"]

    inv_totals = {}
    ginis = {}
    for tech in tech_categories:
        sub = caps[caps["tech"] == tech]
        inv_totals[tech] = float(sub["investment"].sum())
        ginis[f"gini_{tech}"] = gini(sub["investment"].values)

    total_all = sum(inv_totals.values())
    shares = {
        f"share_{tech}": inv_totals[tech] / total_all if total_all > 0 else 0.0
        for tech in tech_categories
    }

    return {
        **ginis,
        **shares,
        "hhi": hhi(np.array(list(inv_totals.values()))),
    }

def carrier_concentration_stats(
    caps: pd.DataFrame,
    carrier_lookup: dict,
    network_hash: str,
    direction_hash: str,
    capital_costs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute investment-weighted shares and Gini per carrier.

    Parameters
    ----------
    caps : pd.DataFrame  Output of parse_caps with investment column added.
    carrier_lookup : dict  Output of build_carrier_lookup.
    network_hash : str
    direction_hash : str

    Returns
    -------
    DataFrame with one row per carrier: network_hash, direction_hash,
    carrier, tech, total_investment, share, gini.
    """
    caps = caps.merge(capital_costs, on="name", how="left")
    caps["investment"] = caps["value"] * caps["capital_cost"].fillna(0.0)
    rows = []
    total = caps["investment"].sum()
    for carrier, tech in carrier_lookup.items():
        sub = caps[caps["carrier"] == carrier]
        if sub.empty:
            continue
        total_inv = float(sub["investment"].sum())
        rows.append({
            "network_hash":     network_hash,
            "direction_hash":   direction_hash,
            "carrier":          carrier,
            "tech":             tech,
            "total_investment": total_inv,
            "share":            total_inv / total if total > 0 else 0.0,
            "gini":             gini(sub["value"].values),
        })
    return pd.DataFrame(rows)


def jsd_vs_reference(
    caps: pd.DataFrame,
    ref_caps: pd.DataFrame,
    tech_categories: dict,
    capital_costs: pd.Series | None = None,
) -> dict:
    """
    Compute Jensen-Shannon divergence between candidate and reference investment
    shares per tech category.

    Parameters
    ----------
    caps : pd.DataFrame      Candidate capacities (output of parse_caps).
    ref_caps : pd.DataFrame  Reference (cost-optimal) capacities (output of parse_caps).
    tech_categories : dict   Projection config tech categories.
    capital_costs : pd.Series | None
        Capital cost per component name. When provided, JSD is computed on
        investment-weighted shares (matching the direction vector definition).

    Returns
    -------
    dict with keys: jsd_{tech} per category.
    """
    def _investment(df):
        if capital_costs is not None:
            return df["value"] * df["name"].map(capital_costs).fillna(0.0)
        return df["value"]

    result = {}
    for tech in tech_categories:
        p = _investment(caps[caps["tech"] == tech]).values
        q = _investment(ref_caps[ref_caps["tech"] == tech]).values
        if p.sum() > 0 and q.sum() > 0:
            result[f"jsd_{tech}"] = jsd(capacity_shares(p), capacity_shares(q))
        else:
            result[f"jsd_{tech}"] = np.nan
    return result


def capacity_shares(values: np.ndarray) -> np.ndarray:
    """Fractional shares summing to 1."""
    tot = values.sum()
    return values / tot if tot > 0 else np.zeros_like(values)


def gini(values: np.ndarray) -> float:
    """Gini coefficient (0=equal, 1=concentrated)."""
    x = np.sort(np.asarray(values, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return (n + 1 - 2 * np.cumsum(x).sum() / x.sum()) / n


def hhi(shares: np.ndarray) -> float:
    """HHI = sum(s²) on fractional shares."""
    s = np.asarray(shares, float)
    s = s / s.sum() if s.sum() > 0 else s
    return float((s ** 2).sum())


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2, bounded [0,1]). Safe at zero support."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


# ---------------------------------------------------------------------------
# 3. Net load
# ---------------------------------------------------------------------------

def netload_stats(n: pypsa.Network, top_n: int = 100) -> pd.DataFrame:
    """
    Net load peak and top-N mean per carrier (elec, heat).

    Parameters
    ----------
    n : pypsa.Network  Solved network.
    top_n : int        Number of peak hours for top-N mean.

    Returns
    -------
    DataFrame with columns: carrier, peak, top{n}_mean, annual_mean.
    """
    nl = extract_net_load(n, heating=True)
    rows = []
    for carrier, series in nl.items():
        s = pd.Series(series)
        rows.append({
            "carrier":           carrier,
            "peak":              float(s.max()),
            f"top{top_n}_mean":  float(s.nlargest(top_n).mean()),
            "annual_mean":       float(s.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Prices
# ---------------------------------------------------------------------------

PRICE_CARRIERS = {
    "elec":  ["AC", "low voltage"],
    "heat":  ["urban central heat", "rural heat", "urban decentral heat"],
    "h2":    ["H2"],
    "co2":   ["co2 stored"],
}


def price_stats(n: pypsa.Network) -> pd.DataFrame:
    """
    Marginal price percentiles per carrier group (elec, heat, h2, co2).
    Carrier groups match the validation CSV outputs from test_operations.

    Returns
    -------
    DataFrame with columns: carrier, mean, p10, p25, p50, p75, p90.
    """
    rows = []
    for label, carriers in PRICE_CARRIERS.items():
        buses = n.buses[n.buses.carrier.isin(carriers)].index
        if buses.empty or not hasattr(n.buses_t, "marginal_price"):
            continue
        cols = buses.intersection(n.buses_t.marginal_price.columns)
        if cols.empty:
            continue
        prices = n.buses_t.marginal_price[cols].mean(axis=1)
        rows.append({
            "carrier": label,
            "mean": float(prices.mean()),
            "p10":  float(prices.quantile(0.10)),
            "p25":  float(prices.quantile(0.25)),
            "p50":  float(prices.quantile(0.50)),
            "p75":  float(prices.quantile(0.75)),
            "p90":  float(prices.quantile(0.90)),
        })
    return pd.DataFrame(rows)

def price_stats_from_series(s: pd.Series, name: str) -> dict:
    """
    Price percentiles from a pre-aggregated price Series.
    Used in analyse_cost_optimal_validation and analyse_mga_validation
    where prices are read from CSV (already spatially aggregated).

    Parameters
    ----------
    s : pd.Series  Spatially aggregated price time series.
    name : str     Prefix for output keys, e.g. 'elec_price'.

    Returns
    -------
    dict with keys: {name}_mean, {name}_p50, {name}_p90, {name}_p99.
    """
    return {
        f"{name}_mean": float(s.mean()),
        f"{name}_p50":  float(s.quantile(0.5)),
        f"{name}_p90":  float(s.quantile(0.9)),
        f"{name}_p99":  float(s.quantile(0.99)),
    }


# ---------------------------------------------------------------------------
# 5. Shedding
# ---------------------------------------------------------------------------

def shed_cost(load_mwh: float, heat_mwh: float) -> float:
    """VOLL-weighted shedding cost. Only legal cross-carrier combiner."""
    return load_mwh * VOLL_ELEC + heat_mwh * VOLL_HEAT


def shedding_stats(load_shed: pd.Series, heat_shed: pd.Series) -> dict:
    """
    Time-series shedding statistics for one (d,w) or (d,c,w) pair.

    Parameters
    ----------
    load_shed : pd.Series  Hourly electricity shedding (MWh), aggregated across nodes.
    heat_shed : pd.Series  Hourly heat shedding (MWh), aggregated across nodes.

    Returns
    -------
    dict with annual totals, peak, shedding hours, and VOLL-weighted cost.
    """
    load_total = float(load_shed.sum())
    heat_total = float(heat_shed.sum())
    return {
        "load_shed_mwh":   load_total,
        "heat_shed_mwh":   heat_total,
        "load_shed_peak":  float(load_shed.max()),
        "heat_shed_peak":  float(heat_shed.max()),
        "load_shed_hours": int((load_shed > 0).sum()),
        "heat_shed_hours": int((heat_shed > 0).sum()),
        "shed_cost":       shed_cost(load_total, heat_total),
    }


# ---------------------------------------------------------------------------
# 6. Scoring
# ---------------------------------------------------------------------------

def delta_shed_cost(cand_stats: dict, base_shed_cost: float) -> float:
    """
    δ = baseline shed_cost − candidate shed_cost (positive = improvement).
    Shedding-cost basis only (v1). See DECISIONS.md for total-cost variant.

    Parameters
    ----------
    cand_stats : dict    Output of shedding_stats for the candidate.
    base_shed_cost : float  Baseline shed_cost for the same (d, w).
    """
    return base_shed_cost - cand_stats["shed_cost"]


def cvar(values: np.ndarray, alpha: float = 0.33) -> tuple[float, int]:
    """
    CVaR_alpha = mean over ceil(alpha*n) smallest values (worst outcomes).
    δ positive = improvement, so the worst outcomes are the smallest δ.
    At |W|=3, alpha=0.33 → single worst year.

    Returns
    -------
    (cvar_value, index_of_worst)
    """
    v = np.asarray(values, float)
    k = max(1, int(np.ceil(alpha * len(v))))
    order = np.argsort(v)
    worst_idx = int(order[0])
    return float(v[order[:k]].mean()), worst_idx


def configuration_score(deltas: np.ndarray, alpha: float = 0.33) -> tuple[float, int]:
    """
    S = mean_w(δ) + CVaR_alpha(δ). Lower is better.
    δ = shedding cost saved (positive = improvement), negated for S.

    Returns
    -------
    (S, worst_w_index)
    """
    d = np.asarray(deltas, float)
    cv, worst_idx = cvar(d, alpha)
    return float(-d.mean() + (-cv)), worst_idx


def bcr(delta_shed: float, slack_spent: float) -> float:
    """
    BCR = ΔShedCost / SlackSpent. NaN if no slack spent.
    Conditional-on-stress metric — do not phrase as 'pays for itself'.
    """
    if slack_spent <= 0:
        return float("nan")
    return delta_shed / slack_spent


# ---------------------------------------------------------------------------
# 7. Emissions
# ---------------------------------------------------------------------------

def profile_match(cand: pd.Series, opt: pd.Series) -> dict:
    """
    Compare candidate emission profile to cost-optimal.
    Under a binding CO₂ cap, level_ratio ≈ 1; signal is shape_corr and jsd.

    Parameters
    ----------
    cand : pd.Series  Candidate hourly emissions.
    opt : pd.Series   Cost-optimal hourly emissions.

    Returns
    -------
    dict with keys: emis_shape_corr, emis_level_ratio, emis_jsd.
    """
    c = cand.values.astype(float)
    o = opt.values.astype(float)
    shape_corr = float(np.corrcoef(c, o)[0, 1]) if c.std() > 0 and o.std() > 0 else np.nan
    level_ratio = float(c.sum() / o.sum()) if o.sum() != 0 else np.nan
    cc = c - min(0.0, c.min())
    oo = o - min(0.0, o.min())
    pjsd = jsd(cc, oo) if cc.sum() > 0 and oo.sum() > 0 else np.nan
    return {"emis_shape_corr": shape_corr, "emis_level_ratio": level_ratio, "emis_jsd": pjsd}


# ---------------------------------------------------------------------------
# 8. Directions
# ---------------------------------------------------------------------------

def load_direction(directions_dir: Path, direction_hash: str) -> dict:
    """
    Load direction vector from {directions_dir}/{direction_hash}.json.

    Parameters
    ----------
    directions_dir : Path  Directory containing direction JSON files.
    direction_hash : str   16-character direction hash.

    Returns
    -------
    dict keyed by DIRECTION_DIMS with float values.
    """
    path = Path(directions_dir) / f"{direction_hash}.json"
    with open(path) as f:
        v = json.load(f)
    missing = set(DIRECTION_DIMS) - set(v)
    if missing:
        raise ValueError(f"Direction {direction_hash} missing dims: {missing}")
    return {k: v[k] for k in DIRECTION_DIMS}