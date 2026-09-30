# SPDX-FileCopyrightText: 2026 Till Cassens
#
# SPDX-License-Identifier: MIT

"""
Re-solve a horizon with its MGA dimension totals pinned to the Chebyshev centre.

The input is the solved network the MGA was run on (the parent ``o`` node) and
the centre JSON from ``compute_chebyshev_centre.py``. The model is built as in
``compute_near_opt_batch.py`` (transmission fixed, same ``extra_functionality``),
then each dimension total ``sum capital_cost * nom`` over the extendable assets
(the expression behind ``coord_*``) is pinned to the centre by an equality, and
the normal cost objective is minimised. Everything within a dimension (split
across nodes and technologies), storage, heating and dispatch is re-optimised.

The network is exported like a cost-optimal myopic solve, so the next horizon
can be built on it with ``add_brownfield``.
"""

import json
import logging

import pypsa
import yaml

from _benchmark import memory_logger
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from compute_near_opt import fill_dimension_weights, load_dimensions_from_config
from mga_helpers import apply_mga_extra_functionality
from solve_second_network import fix_networks

logger = logging.getLogger(__name__)


def fixed_transmission(m, n):
    """
    Components that ``fix_networks`` switched from extendable to fixed, as
    ``{component: (nominal attr, index)}``.
    """
    fixed = {}
    for comp, attr in {"Line": "s_nom", "Link": "p_nom"}.items():
        ext_n = n.components[comp].static[f"{attr}_extendable"]
        ext_m = m.components[comp].static[f"{attr}_extendable"]
        fixed[comp] = (attr, ext_n.index[ext_n & ~ext_m])
    return fixed


def restore_extendable(m, n, fixed):
    """
    Undo ``fix_networks`` on the static data after solving (``*_opt`` stays at
    the fixed value), so ``add_brownfield`` treats transmission as it does for
    a cost-optimal network (it filters small extendable assets).
    """
    for comp, (attr, idx) in fixed.items():
        c, c_n = m.components[comp].static, n.components[comp].static
        c.loc[idx, attr] = c_n.loc[idx, attr]
        c.loc[idx, f"{attr}_extendable"] = True


def add_centre_constraints(m, dimensions, centre):
    """Pin each dimension total (EUR, as ``coord_*``) to the centre."""
    for k, weights in dimensions.items():
        lhs = m.optimize.build_linexpr_from_weights(weights).sum()
        m.model.add_constraints(lhs == centre[k], name=f"chebyshev-{k}")


def dimension_nom_totals(m, dimensions):
    """Total optimised nominal capacity per dimension (MW or MWh), extendables only."""
    totals = {}
    for k, weights in dimensions.items():
        total = 0.0
        for comp, attrs in weights.items():
            c = m.components[comp]
            for attr, coeffs in attrs.items():
                idx = c.extendables.intersection(list(coeffs))  # same assets as the pin
                total += float(c.static.loc[idx, f"{attr}_opt"].sum())
        totals[k] = total
    return totals


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_chebyshev_network",
            opts="",
            clusters="2",
            sector_opts="",
            planning_horizons="2030",
            pathway="c2030",
            configfiles="config/pathway_mga/mini_myopic_dk.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    planning_horizons = snakemake.wildcards.get("planning_horizons", None)
    mga_config = snakemake.config["near-opt"]

    with open(snakemake.input.centre) as f:
        centre_info = json.load(f)
    centre = centre_info["centre"]
    if str(centre_info["planning_horizons"]) != str(planning_horizons):
        raise ValueError(
            f"Centre is for horizon {centre_info['planning_horizons']}, "
            f"not {planning_horizons}"
        )

    # Same setup as compute_near_opt_batch.py
    logger.info(f"Loading network from {snakemake.input.network}")
    n = pypsa.Network(snakemake.input.network)
    m = n.copy()
    fix_networks(m, n)  # transmission fixed, as in MGA
    fixed = fixed_transmission(m, n)

    # get slack from config, for the summary and checks only (no budget constraint is added)
    slack_config = mga_config.get("slack", {})
    if slack_config.get("relative", True):
        slack = slack_config.get("value", 0.05)
    else:
        slack = float(slack_config.get("value", 0.0)) / float(
            n.statistics.capex().sum() + n.statistics.opex().sum()
        )
    logger.info(f"Using relative slack: {slack}")

    dimensions = load_dimensions_from_config(mga_config.get("projection", {}))
    dimensions = fill_dimension_weights(m, dimensions)
    if set(dimensions) != set(centre):
        raise ValueError(
            f"Dimensions {sorted(dimensions)} do not match centre {sorted(centre)}"
        )
    logger.info(f"Pinning dimensions {list(dimensions)} to the Chebyshev centre")

    # C* as in _add_near_opt_constraint
    optimal_cost = float(n.statistics.capex().sum() + n.statistics.opex().sum())

    solving = snakemake.config["solving"]
    solver_name = solving["solver"]["name"]
    solver_options = solving["solver_options"].get(solving["solver"]["options"], {})
    if solver_name == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)

    with memory_logger(
        filename=getattr(snakemake.log, "memory", None),
        interval=solving.get("mem_logging_frequency", 30),
    ) as mem:
        m.optimize.create_model(multi_investment_periods=False)  # same as MGA
        apply_mga_extra_functionality(
            m,
            m.snapshots,
            config=snakemake.config,
            custom_extra_functionality=snakemake.params.custom_extra_functionality,
            planning_horizons=planning_horizons,
        )  # CO2 budget etc., same as MGA
        add_centre_constraints(m, dimensions, centre)
        status, condition = m.optimize.solve_model(
            solver_name=solver_name,
            solver_options=solver_options,
            log_fn=snakemake.log.solver,
        )  # normal cost objective
    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    if "infeasible" in condition:
        m.model.print_infeasibilities()
    if status != "ok":  # c network is a parent for later horizons, fail loudly
        raise RuntimeError(
            f"Chebyshev re-solve failed: status '{status}', condition '{condition}'"
        )

    # Summary: projected coords vs centre, cost vs budget
    coords = m.optimize.project_solved(dimensions)
    fixed_cost = float(m.statistics.installed_capex().sum())
    cost = float(m.objective) + fixed_cost  # LHS of the MGA budget constraint
    budget = (1 + slack) * optimal_cost
    deviation = {k: coords[k] - centre[k] for k in dimensions}
    rel_dev = max(
        abs(d) / max(abs(centre[k]), 1.0) for k, d in deviation.items()
    )
    logger.info(f"Max relative deviation from centre: {rel_dev:.3g}")
    logger.info(
        f"Cost {cost:.6g} vs C* {optimal_cost:.6g} (budget {budget:.6g}, "
        f"{cost / optimal_cost - 1:.4%} above C*)"
    )
    if cost > budget * (1 + 1e-6):
        logger.warning("Chebyshev re-solve exceeds the near-optimal budget")

    summary = {
        "planning_horizons": planning_horizons,
        "pathway": snakemake.wildcards.get("pathway", None),
        "network_hash": centre_info["network_hash"],  # MGA the centre came from
        "status": status,
        "condition": condition,
        "centre": centre,
        "coord": coords.to_dict(),  # EUR, same expression as coord_*
        "deviation": deviation,
        "max_rel_deviation": rel_dev,
        "nom_total": dimension_nom_totals(m, dimensions),  # MW / MWh, information
        "radius": centre_info["radius"],
        "cost": cost,  # objective + fixed cost, as in the budget constraint
        "optimal_cost": optimal_cost,  # C* of the parent network
        "slack": slack,
        "budget": budget,  # (1 + slack) * C*
        "cost_above_optimal": cost / optimal_cost - 1,
        "within_budget": bool(cost <= budget * (1 + 1e-6)),
    }
    with open(snakemake.output.summary, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info(f"Summary written to {snakemake.output.summary}")

    # Export like a cost-optimal myopic solve (solve_network.py)
    restore_extendable(m, n, fixed)
    m.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    m.export_to_netcdf(snakemake.output.network)

    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            m.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
