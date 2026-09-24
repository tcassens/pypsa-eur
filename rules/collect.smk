# SPDX-FileCopyrightText: 2025 Aleksander Grochowicz & Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import yaml
import pandas as pd
from pathlib import Path

def design_years(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return list(data.keys())

def test_years(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return list(data.keys())

def network_year(config):
    if config["run"]["fixed_network"].get("enable", False):
        return config["run"]["fixed_network"]["scenario"]
    else:
        return None


def get_mga_directions(near_opt_file):
    """Return unique direction hashes from a near_opt_solutions CSV. Returns [] if file missing."""
    file_path = Path(near_opt_file)
    if not file_path.exists():
        return []
    try:
        df = pd.read_csv(file_path)
        return df["dir_hash"].unique().tolist() if "dir_hash" in df.columns else []
    except Exception as e:
        print(f"Warning: Could not read MGA directions from {near_opt_file}: {e}")
        return []


def get_network_hash_for_near_opt(near_opt_file):
    """Return network hash from the _network_hash.txt file alongside near_opt CSV. Returns None if missing."""
    near_opt_path = Path(near_opt_file)
    if not near_opt_path.exists():
        return None
    hash_file = near_opt_path.parent / (near_opt_path.stem + "_network_hash.txt")
    try:
        return hash_file.read_text().strip()
    except Exception:
        return None



localrules:
    all,
    cluster_networks,
    prepare_elec_networks,
    prepare_sector_networks,
    solve_elec_networks,
    solve_sector_networks,
    test_networks,
    compute_mga_solutions,
    validate_mga_solutions,


rule process_costs:
    input:
        lambda w: (
            expand(
                resources(
                    f"costs_{config_provider('costs', 'year')(w)}_processed.csv"
                ),
                run=config["run"]["name"],
            )
            if config_provider("foresight")(w) == "overnight"
            else expand(
                resources("costs_{planning_horizons}_processed.csv"),
                **config["scenario"],
                run=config["run"]["name"],
            )
        ),


rule cluster_networks:
    message:
        "Collecting clustered network files"
    input:
        expand(
            resources("networks/base_s_{clusters}.nc"),
            **config["scenario"],
            run=config["run"]["name"],
        ),


rule prepare_elec_networks:
    message:
        "Collecting prepared electricity network files"
    input:
        expand(
            resources("networks/base_s_{clusters}_elec_{opts}.nc"),
            **config["scenario"],
            run=config["run"]["name"],
        ),


rule prepare_sector_networks:
    message:
        "Collecting prepared sector-coupled network files"
    input:
        expand(
            resources(
                "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
            ),
            **config["scenario"],
            run=config["run"]["name"],
        ),


rule solve_elec_networks:
    message:
        "Collecting solved electricity network files"
    input:
        expand(
            RESULTS + "networks/base_s_{clusters}_elec_{opts}.nc",
            **config["scenario"],
            run=config["run"]["name"],
        ),


rule solve_sector_networks:
    message:
        "Collecting solved sector-coupled network files"
    input:
        expand(
            RESULTS
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"],
            run=config["run"]["name"],
        ),


rule solve_sector_networks_perfect:
    message:
        "Collecting solved sector-coupled network files with perfect foresight"
    input:
        expand(
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
            **config["scenario"],
            run=config["run"]["name"],
        ),

rule test_networks:
    input:
        lambda w: expand(
            "results/" + config["run"]["prefix"] + "/{design_year}/validation/{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}/{suffix}",
            design_year=design_years(config["run"]["stress_tests"]["design_years"]),
            operational_year=test_years(config["run"]["stress_tests"]["stress_years"]),
            suffix=["load_shedding.csv", "heat_shedding.csv", "net_load.csv", "emissions.csv", "elec_prices.csv", "heat_prices.csv", "h2_prices.csv", "co2_prices.csv", "objective.json"],
            **config["scenario"],
        ),

rule compute_mga_solutions:
    """Compute all near-optimal (MGA) solutions for specified design years."""
    input:
        lambda w: expand(
            "results/" + config["run"]["prefix"] + "/{design_year}/near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            design_year=design_years(config["run"]["stress_tests"]["design_years"]),
            **config["scenario"],
        ) if config.get("near-opt", {}).get("enable", False) else [],


rule collect_mga_summaries:
    """Collect MGA validation summaries for all design years."""
    input:
        lambda w: [
            f"results/{config['run']['prefix']}/{design_year}/validation/_summary_{scenario}.csv"
            for design_year in design_years(config["run"]["stress_tests"]["design_years"])
            for scenario in expand("base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}", **config["scenario"])
        ] if config.get("near-opt", {}).get("validation", {}).get("enable", False) else [],


rule validate_mga_solutions:
    """Collect all per-direction validation outputs for testing.

    NOTE: reads near_opt.csv and network_hash.txt at DAG-build time (no checkpoint
    gating). Requires these files to already exist. Use collect_resilience_analysis
    for a fully automatic end-to-end run; use this rule only for targeted testing
    when near_opt.csv is already populated.
    """
    input:
        lambda w: [
            f"results/{config['run']['prefix']}/{design_year}/validation/mga_{network_hash}_{dir_hash}_{operational_year}_{scenario}/{suffix}"
            for design_year in design_years(config["run"]["stress_tests"]["design_years"])
            for operational_year in test_years(config["run"]["stress_tests"]["stress_years"])
            for scenario in expand("base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}", **config["scenario"])
            for near_opt_file in [f"results/{config['run']['prefix']}/{design_year}/near_opt/{scenario}.csv"]
            for network_hash in [get_network_hash_for_near_opt(near_opt_file) or ""]
            for dir_hash in get_mga_directions(near_opt_file)
            for suffix in ["load_shedding.csv", "heat_shedding.csv", "net_load.csv", "emissions.csv", "elec_prices.csv", "heat_prices.csv", "h2_prices.csv", "co2_prices.csv", "objective.json"]
            if network_hash
        ] if config.get("near-opt", {}).get("validation", {}).get("enable", False) else [],
        
rule collect_resilience_analysis:
    """Collect all resilience analysis outputs for all design years."""
    input:
        lambda w: [
            f"results/{config['run']['prefix']}/{design_year}/resilience/{filename}"
            for design_year in design_years(config["run"]["stress_tests"]["design_years"])
            for scenario in expand("base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}", **config["scenario"])
            for filename in [
                f"baseline_shedding_{scenario}.csv",
                f"baseline_analysis_{scenario}.csv",
                f"cost_opt_summary_{scenario}.json",
                f"cost_opt_caps_{scenario}.csv",
                f"cost_opt_netload_{scenario}.csv",
                f"cost_opt_prices_{scenario}.csv",
                f"cost_opt_emissions_{scenario}.csv",
                f"mga_candidates_{scenario}.csv",
                f"mga_validation_atomic_{scenario}.csv",
                f"mga_validation_rollup_{scenario}.csv",
            ]
        ] if config.get("near-opt", {}).get("enable", False) else [],


rule collect_optimal_pathway_mga:
    """Collect cost-optimal summaries and MGA candidates for the myopic pathway, per design year and horizon (no stress-test/validation outputs)."""
    input:
        lambda w: [
            f"results/{config['run']['prefix']}/{design_year}/resilience/{filename}"
            for design_year in design_years(config["run"]["stress_tests"]["design_years"])
            for scenario in expand("base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}", **config["scenario"])
            for filename in [
                f"cost_opt_summary_{scenario}.json",
                f"cost_opt_caps_{scenario}.csv",
                f"mga_candidates_{scenario}.csv",
            ]
        ] if config.get("near-opt", {}).get("enable", False) else [],


def balance_map_paths(kind, w):
    """
    kind = "static" or "interactive"
    """
    cfg_key = "balance_map" if kind == "static" else "balance_map_interactive"

    return expand(
        RESULTS
        + f"maps/{kind}/base_s_{{clusters}}_{{opts}}_{{sector_opts}}_{{planning_horizons}}"
        f"-balance_map_{{carrier}}.{'pdf'if kind== 'static' else 'html'}",
        **config["scenario"],
        run=config["run"]["name"],
        carrier=config_provider("plotting", cfg_key, "bus_carriers")(w),
    )


rule plot_balance_maps:
    message:
        "Plotting energy balance maps"
    input:
        static=lambda w: balance_map_paths("static", w),
        interactive=lambda w: balance_map_paths("interactive", w),


rule plot_balance_maps_static:
    input:
        lambda w: balance_map_paths("static", w),


rule plot_balance_maps_interactive:
    input:
        lambda w: balance_map_paths("interactive", w),


# rule plot_power_networks_clustered:
#    message:
#        "Plotting clustered power network topology"
#    input:
#        expand(
#            resources("maps/power-network-s-{clusters}.pdf"),
#            **config["scenario"],
#            run=config["run"]["name"],
#        ),
