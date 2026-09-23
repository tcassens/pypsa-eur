# SPDX-FileCopyrightText: 2025 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT

"""Helper functions for MGA (Modeling to Generate Alternatives) workflows."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import json

logger = logging.getLogger(__name__)

def export_mga_capacities(n, snapshots, cache_dir, network_hash, direction_hash, check_only=False):
    """
    Export or check optimal capacities for an MGA solution.

    This function is designed to be used as mga_extra_functionality callback
    in pypsa-mga optimization.

    Parameters
    ----------
    n : pypsa.Network or None
        Solved PyPSA network (None when check_only=True)
    snapshots : pd.DatetimeIndex
        Snapshots used in optimization
    cache_dir : str
        Cache directory for storing results
    network_hash : str
        Network configuration hash
    direction_hash : str
        Direction vector hash
    check_only : bool, default False
        If True, only check if capacities file exists and return bool.
        If False, extract and export capacities.

    Returns
    -------
    bool (only when check_only=True)
        True if capacities file exists, False otherwise
    """
    # Construct output path
    caps_dir = Path(cache_dir) / "caps"
    caps_file = caps_dir / f"caps_{network_hash}_{direction_hash}.csv"

    if check_only:
        # Just check if file exists
        return caps_file.exists()

    # Create output directory
    caps_dir.mkdir(parents=True, exist_ok=True)

    # Extract capacities from solved network
    capacities = extract_optimal_capacities(n)

    # Save to file
    capacities.to_csv(caps_file, index=False)

    logger.info(
        f"Exported {len(capacities)} capacities to {caps_file.name}"
    )


def apply_mga_extra_functionality(n, snapshots, config, custom_extra_functionality, planning_horizons):
    """
    Run PyPSA-Eur's real extra_functionality() (CO2 budget, battery/TES ratios,
    solar potential, etc.) for one MGA direction.

    Used as the `extra_functionality` callback for
    `n.optimize.optimize_mga_in_multiple_directions`. Each direction is solved
    in a freshly reloaded Network in a spawned worker process, so `n.config`/
    `n.params`are never set there; this attaches them first.

    Parameters
    ----------
    n : pypsa.Network
        The (per-worker) network being solved for this direction.
    snapshots : pd.DatetimeIndex
    config : dict
        Full snakemake.config, as passed to solve_second_network.solve_network.
    custom_extra_functionality : str | list
        snakemake.params.custom_extra_functionality (a path, or [] if unset).
    planning_horizons : str | None
    """
    from types import SimpleNamespace

    from solve_second_network import extra_functionality

    n.config = config
    n.params = SimpleNamespace(custom_extra_functionality=custom_extra_functionality)
    extra_functionality(n, snapshots, planning_horizons=planning_horizons)


def export_mga_information(n, snapshots, cache_dir, network_hash, direction_hash, wildcards=None, slack=None, check_only=False):
    """
    Export or check all per-direction MGA solution outputs.

    Sole mga_extra_functionality callback. Writes caps, info, netload, and
    emissions artifacts per direction. Net load and emissions are design-year
    (w=d) quantities; per-stress-year (w!=d) versions come from validation script.

    Budget dual is guarded (n.model availability in spawned worker is uncertain);
    net load and emissions are not (failures indicate real bugs).

    Parameters
    ----------
    n : pypsa.Network or None
        Solved PyPSA network (None when check_only=True).
    snapshots : pd.DatetimeIndex
    cache_dir : str
    network_hash : str
    direction_hash : str
    wildcards : dict, optional
        Used to extract wildcards for logging.
    slack : float, optional
        Used to slack for logging.
    check_only : bool, default False
        If True, check if all four output files exist.

    Returns
    -------
    bool (only when check_only=True)
        True if all four files exist.
    """
    caps_dir = Path(cache_dir) / "caps"
    info_dir = Path(cache_dir) / "info"
    netload_dir = Path(cache_dir) / "netload"
    emissions_dir = Path(cache_dir) / "emissions"

    caps_file = caps_dir / f"caps_{network_hash}_{direction_hash}.csv"
    info_file = info_dir / f"info_{network_hash}_{direction_hash}.json"
    netload_file = netload_dir / f"netload_{network_hash}_{direction_hash}.csv"
    emissions_file = emissions_dir / f"emissions_{network_hash}_{direction_hash}.csv"

    if check_only:
        return (
            caps_file.exists()
            and info_file.exists()
            and netload_file.exists()
            and emissions_file.exists()
        )

    for d in (caps_dir, info_dir, netload_dir, emissions_dir):
        d.mkdir(parents=True, exist_ok=True)

    capacities = extract_optimal_capacities(n)
    capacities.to_csv(caps_file, index=False)
    logger.info(f"Exported {len(capacities)} capacities to {caps_file.name}")

    nl = extract_net_load(n, heating=True)
    netload_df = pd.DataFrame({"elec": nl["elec"], "heat": nl["heat"]})
    netload_df.to_csv(netload_file)
    logger.info(f"Exported net load to {netload_file.name}")

    emissions = extract_emissions(n)
    emissions.to_frame(name="co2_flow").to_csv(emissions_file)
    logger.info(f"Exported emissions to {emissions_file.name}")

    obj = extract_objective(n)

    try:
        mga_dual = extract_mga_dual(n)
        logger.info(f"budget dual for {direction_hash}: {mga_dual}")
    except Exception as e:
        logger.warning(f"budget dual NOT extractable for {direction_hash}: {e}")
        mga_dual = float("nan")

    info = {
        "network_hash": network_hash,
        "direction_hash": direction_hash,
        "config_name": wildcards.get("run", "") if wildcards else "",
        "design_year": wildcards.get("planning_horizons", "") if wildcards else "",
        "clusters": wildcards.get("clusters", "") if wildcards else "",
        "opts": wildcards.get("opts", "") if wildcards else "",
        "sector_opts": wildcards.get("sector_opts", "") if wildcards else "",
        "slack": slack,
        "capex": round(obj["capex"], 2),
        "opex": round(obj["opex"], 2),
        "total_cost": round(obj["total"], 2),
        "budget_constraint_dual": round(mga_dual, 4),
    }
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Exported MGA info to {info_file.name}")


def extract_emissions(n):
    """
    Extract CO2 flow into 'co2 atmosphere' bus per timestep.

    Aggregates flow from links connected to co2 atmosphere via bus1 (process),
    bus2 (emitters), or bus3 (CHP). HVC-to-air links release annual total in
    one arbitrary snapshot due to LP degeneracy; detected and smoothed to flat
    average to avoid corrupting hour-resolved analysis.

    Returns
    -------
    pd.Series (snapshots)
        Net CO2 flow into atmosphere, with HVC spikes smoothed.
    """
    process_i = n.links.query('bus1 == "co2 atmosphere"').index
    emitters_i = n.links.query('bus2 == "co2 atmosphere"').index
    chp_i = n.links.query('bus3 == "co2 atmosphere"').index

    if process_i.empty and emitters_i.empty and chp_i.empty:
        raise ValueError("No CO2 atmosphere links found; check bus configuration.")

    weights = n.snapshot_weightings.objective
    total_weight = weights.sum()
    flow = pd.Series(0.0, index=n.snapshots)

    # Process emissions: detect and smooth single-hour dumps (HVC degeneracy)
    for name in process_i:
        series = n.links_t.p1[name]
        weighted_total = (series * weights).sum()
        if weighted_total == 0:
            continue
        peak_share = (series.abs() * weights).max() / abs(weighted_total)
        if peak_share > 0.5:
            flow += weighted_total / total_weight
        else:
            flow += series

    # Emitters and CHP: already flat, no smoothing needed
    if not emitters_i.empty:
        flow += n.links_t.p2[emitters_i].sum(axis=1)
    if not chp_i.empty:
        flow += n.links_t.p3[chp_i].sum(axis=1)

    return flow



def extract_mga_dual(n):
    """Extract the dual to the near-optimality (slack / MGA budget) constraint from a solved PyPSA network."""
    if not hasattr(n, "model") or n.model is None:
        raise RuntimeError(
            "n.model is not available -- extract_mga_dual must be called "
            "immediately after the MGA solve, on the live model."
        )
    if "budget" not in n.model.dual:
        import numpy as np
        logger.warning("No 'budget' constraint found on this model; returning NaN.")
        return np.nan
    return float(n.model.dual["budget"].item())


def extract_net_load(n, heating=True):
    """
    Extract instantaneous net load time series from a solved PyPSA network.

    Electricity net load = elec demand + HP elec draw - VRE potential.
    Heat net load = heat demand - HP availability, clipped to zero (no transmission).
    Heat is assessed per-bus then aggregated; electricity aggregates across buses.

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network.
    heating : bool, default True
        If True, extract both elec and heat net load; if False, elec only.

    Returns
    -------
    dict
        Keys 'elec' (always), 'heat' (if heating=True); each is a pd.Series.
    """
    # Electricity demand
    elec_loads = n.loads[n.loads.carrier == "electricity"]
    elec_loads = elec_loads[elec_loads.index.isin(n.loads_t.p_set.columns)]
    if elec_loads.empty:
        raise ValueError("No 'electricity' loads with p_set found; check carriers.")
    elec_demand = n.loads_t.p_set[elec_loads.index].sum(axis=1)

    # VRE potential
    vre_carriers = ["onwind", "offwind-ac", "offwind-dc", "offwind-float",
                    "ror", "solar", "solar-hsat", "solar rooftop"]
    vre = n.generators[n.generators.carrier.isin(vre_carriers)]
    if vre.empty:
        raise ValueError("No VRE generators matched; check n.generators.carrier.")
    vre_potential = sum(
        n.generators_t.p_max_pu[g] * n.generators.loc[g, "p_nom_opt"]
        for g in vre.index if g in n.generators_t.p_max_pu.columns
    )

    if not heating:
        return {"elec": elec_demand - vre_potential}
    
    # Heat demand by bus
    heat_loads = n.loads[n.loads.carrier.str.contains("heat", case=False, na=False)]
    heat_loads = heat_loads[heat_loads.index.isin(n.loads_t.p_set.columns)]
    if heat_loads.empty:
        raise ValueError("No 'heat' loads with p_set found; check carriers.")
    heat_demand_by_bus = (
        n.loads_t.p_set[heat_loads.index]
        .T.groupby(heat_loads.bus).sum().T
    )     

    # Heat pumps (bus0=heat, bus1=elec)
    hp = n.links[n.links.carrier.str.contains("heat pump", case=False, na=False)]
    if hp.empty:
        raise ValueError("No heat-pump links found; check n.links.carrier.")
    
    elec_hp_draw = pd.Series(0.0, index=n.snapshots)
    heat_supplied_by_bus = pd.DataFrame(0.0, index=n.snapshots, columns=heat_demand_by_bus.columns)
    availability_by_bus = pd.DataFrame(0.0, index=n.snapshots, columns=heat_demand_by_bus.columns)

    # Merit-order dispatch per heat bus (ground-source before air-source by efficiency)
    for heat_bus in heat_demand_by_bus.columns:
        remaining_demand = heat_demand_by_bus[heat_bus].copy()
        bus_hps = hp[hp.bus0 == heat_bus]
        if bus_hps.empty:
            continue

        # Sort by efficiency (lowest elec/heat first)
        order = n.links_t.efficiency[bus_hps.index].mean().sort_values().index

        for hp_name in order:
            eff = n.links_t.efficiency[hp_name]
            p_nom_heat = n.links.at[hp_name, "p_nom_opt"]
            heat_provided = np.minimum(remaining_demand, p_nom_heat)
            elec_hp_draw += heat_provided * eff
            heat_supplied_by_bus[heat_bus] += heat_provided
            remaining_demand -= heat_provided

        availability_by_bus[heat_bus] = n.links.loc[bus_hps.index, "p_nom_opt"].sum()
    
    elec_net_load = elec_demand + elec_hp_draw - vre_potential
    heat_net_load_by_bus = (heat_demand_by_bus - availability_by_bus).clip(lower=0)
    heat_net_load = heat_net_load_by_bus.sum(axis=1)
    return {"elec": elec_net_load, "heat": heat_net_load}


def extract_objective(n):
    """
    Extract realized system cost from a solved PyPSA network, split capex/opex.

    Uses expanded_capex() (newly-built capacity) + opex. Do not read n.objective
    on an MGA solve (it holds the direction objective, not cost).

    Returns
    -------
    dict with keys: capex, opex, total.
    """
    capex = float(n.statistics.expanded_capex().sum())
    opex = float(n.statistics.opex().sum())
    return {"capex": capex, "opex": opex, "total": capex + opex}

def extract_optimal_capacities(n):
    """
    Extract all optimal capacities from a solved PyPSA network.

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: component, name, attribute, value.
    """
    capacities = []

    # Generator p_nom_opt
    if "p_nom_opt" in n.generators.columns:
        for name, gen in n.generators.iterrows():
            capacities.append({
                "component": "Generator",
                "name": name,
                "attribute": "p_nom_opt",
                "value": gen.get("p_nom_opt", 0.0),
            })

    # StorageUnit p_nom_opt
    if "p_nom_opt" in n.storage_units.columns:
        for name, su in n.storage_units.iterrows():
            capacities.append({
                "component": "StorageUnit",
                "name": name,
                "attribute": "p_nom_opt",
                "value": su.get("p_nom_opt", 0.0),
            })

    # Store e_nom_opt
    if "e_nom_opt" in n.stores.columns:
        for name, store in n.stores.iterrows():
            capacities.append({
                "component": "Store",
                "name": name,
                "attribute": "e_nom_opt",
                "value": store.get("e_nom_opt", 0.0),
            })

    # Link p_nom_opt
    if "p_nom_opt" in n.links.columns:
        for name, link in n.links.iterrows():
            capacities.append({
                "component": "Link",
                "name": name,
                "attribute": "p_nom_opt",
                "value": link.get("p_nom_opt", 0.0),
            })

    # Line s_nom_opt
    if "s_nom_opt" in n.lines.columns:
        for name, line in n.lines.iterrows():
            capacities.append({
                "component": "Line",
                "name": name,
                "attribute": "s_nom_opt",
                "value": line.get("s_nom_opt", 0.0),
            })

    return pd.DataFrame(capacities)


