# SPDX-FileCopyrightText: Aleksander Grochowicz 2025-2026
#
# SPDX-License-Identifier: MIT

import hashlib
import json
import yaml

# Prefer parallel aggregation over monolithic compute when both could produce the output
ruleorder: aggregate_near_opt > compute_near_opt

# test_operations only exists for overnight foresight
if config["foresight"] == "overnight":
    ruleorder: validation_mga > test_operations

with open(config["run"]["stress_tests"]["design_years"]) as f:
    DESIGN_YEARS = list(yaml.safe_load(f).keys())

with open(config["run"]["stress_tests"]["stress_years"]) as f:
    STRESS_YEARS = list(yaml.safe_load(f).keys())

# Pathways (see the pathway section at the end of this file)
PATHWAY_C = r"([oc]\d{4}-)*c\d{4}"  # ends in a c step
PATHWAY_O = r"([oc]\d{4}-)*c\d{4}(-[oc]\d{4})*-o\d{4}"  # contains a c step, ends in an o step
PATHWAY_SUFFIX = r"(_" + PATHWAY_O + ")?"  # MGA on "" = cost-opt network, "_c2030-o2040" = o pathway network



rule compute_near_opt:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        total_directions=lambda w: (
            (2 * len(config_provider("near-opt", "projection")(w)) if config_provider("near-opt", "approx", "minmax")(w) else 0)
            + config_provider("near-opt", "approx", "iterations")(w)
        ),
        slack=config_provider("near-opt", "slack", "value"),
    message:
        "Computing near-optimal solutions for {wildcards.run} "
        "({params.total_directions} directions, slack={params.slack})"
    input:
        network=RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
    output:
        near_opt_solutions=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        network_hash=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_network_hash.txt",
    log:
        python=RESULTS + "logs/mga/compute_near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    benchmark:
        RESULTS + "benchmarks/mga/compute_near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
    threads: lambda wildcards: config_provider("near-opt", "approx", "max_parallel")(wildcards)
    resources:
        mem_mb=memory,
        runtime=lambda wildcards: (
            lambda rt: int(rt[:-1]) * 60 if isinstance(rt, str) and rt.endswith('h') else int(rt)
        )(config_provider("solving", "runtime", default=360)(wildcards)) * (
            config_provider("near-opt", "approx", "iterations")(wildcards)
            + (2 * len(config_provider("near-opt", "projection")(wildcards)) if config_provider("near-opt", "approx", "minmax")(wildcards) else 0)
        ),
    shadow:
        shadow_config
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/compute_near_opt.py"

def val_mga(suffix):  # validation_mga
    return ("results/" + config["run"]["prefix"] +
            "/{design_year}/validation/mga_{network_hash}_{dir_hash}_{operational_year}"
            "_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}/" + suffix)

rule validation_mga:
    wildcard_constraints:
        network_hash=r"[a-zA-Z0-9_]+",
        dir_hash=r"[a-zA-Z0-9_]+",
        design_year=r"weather_year_\d+_\d+H",
        operational_year=r"weather_year_\d+_\d+H",
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    message:
        "Validating near-optimal solution {wildcards.dir_hash} | "
        "design {wildcards.design_year} | stress {wildcards.operational_year} | "
        "network {wildcards.network_hash}"
    input:
        network="results/" + config["run"]["prefix"] +
            "/{design_year}/networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        weather_network="resources/" + config["run"]["prefix"] +
            "/{operational_year}/networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        mga_capacities=lambda w: config_provider("near-opt", "cache_dir")(w) +
            f"/caps/caps_{w.network_hash}_{w.dir_hash}.csv",
        near_opt_solutions="results/" + config["run"]["prefix"] +
            "/{design_year}/near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
    output:
        load_shedding=val_mga("load_shedding.csv"),
        heat_shedding=val_mga("heat_shedding.csv"),
        net_load=val_mga("net_load.csv"),
        emissions=val_mga("emissions.csv"),
        elec_prices=val_mga("elec_prices.csv"),
        heat_prices=val_mga("heat_prices.csv"),
        h2_prices=val_mga("h2_prices.csv"),
        co2_prices=val_mga("co2_prices.csv"),
        objective=val_mga("objective.json"),
        metadata=val_mga("metadata.json"),
    shadow:
        None
    log:
        solver="results/" + config["run"]["prefix"] +
            "/{design_year}/logs/mga/validation_mga_{network_hash}_{dir_hash}_{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_solver.log",
        memory="results/" + config["run"]["prefix"] +
            "/{design_year}/logs/mga/validation_mga_{network_hash}_{dir_hash}_{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_memory.log",
        python="results/" + config["run"]["prefix"] +
            "/{design_year}/logs/mga/validation_mga_{network_hash}_{dir_hash}_{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    retries: 3
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=8 * 60,
    benchmark:
        "results/" + config["run"]["prefix"] +
            "/{design_year}/benchmarks/mga/validation_mga_{network_hash}_{dir_hash}_{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/test_mga_operations.py"


rule collect_mga_validation:
    """Aggregate all MGA validation results for a design year into a single summary CSV.

    Scans the validation directory for existing results — missing validations
    (infeasible after all retries) appear as rows with status=infeasible.
    """
    input:
        near_opt_solutions=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        network_hash=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_network_hash.txt",
    output:
        summary=RESULTS + "validation/_summary_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
    log:
        python=RESULTS + "logs/mga/collect_mga_validation/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/collect_mga_validation.py"


# ========== Near-Opt Multi-Node Parallelization ==========
# Distributes near-opt computation across multiple SLURM nodes using checkpoints.
# =========================================================


checkpoint generate_near_opt_directions:
    """Generate all near-opt directions as individual JSON files + manifest."""
    wildcard_constraints:
        pathway_suffix=PATHWAY_SUFFIX,
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        total_directions=lambda w: (
            (2 * len(config_provider("near-opt", "projection")(w)) if config_provider("near-opt", "approx", "minmax")(w) else 0)
            + config_provider("near-opt", "approx", "iterations")(w)
        ),
        slack=config_provider("near-opt", "slack", "value"),
    message:
        "Generating near-optimal direction files for {wildcards.run} "
        "({params.total_directions} directions, slack={params.slack})"
    input:
        network=RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}.nc",
        reference=RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",  # cost-opt network, C* of the budget
    output:
        directions_dir=directory(
            RESULTS + "near_opt/directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}/"
        ),
        manifest=RESULTS + "near_opt/directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}_manifest.json",
    log:
        python=RESULTS + "logs/mga/generate_near_opt_directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}_python.log",
    benchmark:
        RESULTS + "benchmarks/mga/generate_near_opt_directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/generate_near_opt_directions.py"


def _get_near_opt_batches(wildcards):
    """Group directions from manifest into batches of size max_parallel."""
    checkpoint_output = checkpoints.generate_near_opt_directions.get(**wildcards).output
    with open(checkpoint_output.manifest) as f:
        manifest = json.load(f)
    all_directions = manifest["directions"]
    max_parallel = config_provider("near-opt", "approx", "max_parallel")(wildcards)
    batches = {}
    for i in range(0, len(all_directions), max_parallel):
        batch_dirs = all_directions[i : i + max_parallel]
        batch_content = "_".join(sorted(batch_dirs))
        batch_hash = hashlib.md5(batch_content.encode()).hexdigest()[:8]
        batches[batch_hash] = batch_dirs
    return batches


def _get_batch_direction_files(wildcards):
    """Get direction JSON files for this specific batch."""
    batches = _get_near_opt_batches(wildcards)
    batch_dirs = batches[wildcards.batch_hash]
    return expand(
        RESULTS + "near_opt/directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}/{dir_hash}.json",
        dir_hash=batch_dirs,
        run=wildcards.run,
        clusters=wildcards.clusters,
        opts=wildcards.opts,
        sector_opts=wildcards.sector_opts,
        planning_horizons=wildcards.planning_horizons,
        pathway_suffix=wildcards.pathway_suffix,
    )


def _get_all_batch_results(wildcards):
    """Get all batch result files (triggers checkpoint resolution)."""
    batches = _get_near_opt_batches(wildcards)
    return expand(
        RESULTS + "near_opt/batches/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}/batch_{batch_hash}.csv",
        batch_hash=list(batches.keys()),
        run=wildcards.run,
        clusters=wildcards.clusters,
        opts=wildcards.opts,
        sector_opts=wildcards.sector_opts,
        planning_horizons=wildcards.planning_horizons,
        pathway_suffix=wildcards.pathway_suffix,
    )


rule compute_near_opt_batch:
    """Solve one batch of near-opt directions on one SLURM node."""
    wildcard_constraints:
        batch_hash=r"[a-f0-9]{8}",
        pathway_suffix=PATHWAY_SUFFIX,
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        max_parallel=config_provider("near-opt", "approx", "max_parallel"),
        slack=config_provider("near-opt", "slack", "value"),
    message:
        "Solving near-optimal batch {wildcards.batch_hash} for {wildcards.run} "
        "({params.max_parallel} directions, slack={params.slack})"
    input:
        network=RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}.nc",
        direction_files=_get_batch_direction_files,
        manifest=RESULTS + "near_opt/directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}_manifest.json",  # remaining slack
    output:
        batch_result=temp(RESULTS + "near_opt/batches/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}/batch_{batch_hash}.csv"),
    log:
        python=RESULTS + "logs/mga/compute_near_opt_batch/batch_{batch_hash}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}_python.log",
    benchmark:
        RESULTS + "benchmarks/mga/compute_near_opt_batch/batch_{batch_hash}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}"
    threads: lambda wildcards: config_provider("near-opt", "approx", "max_parallel")(wildcards)
    resources:
        mem_mb=memory,
        runtime=config_provider("solving", "runtime", default="12h"),
    retries: 1
    shadow:
        shadow_config
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/compute_near_opt_batch.py"


checkpoint aggregate_near_opt:
    """Combine all near-opt batch results into the final CSV (parallel equivalent of compute_near_opt).

    Declared as a checkpoint so that downstream input functions (_get_mga_info_files,
    _get_mga_validation_load_shedding) are re-evaluated after this rule completes,
    allowing Snakemake to resolve dynamic inputs that depend on the solved directions.

    Also verifies cache completeness before writing outputs: if any info files are
    missing (e.g. because a batch solve failed silently), this rule fails early with
    a clear error rather than letting the pipeline proceed with missing cache files.
    """
    wildcard_constraints:
        pathway_suffix=PATHWAY_SUFFIX,
    params:
        solving=config_provider("solving"),
        cache_dir=config_provider("near-opt", "cache_dir"),
    message:
        "Aggregating near-optimal batch results for {wildcards.run}"
    input:
        batch_results=_get_all_batch_results,
        manifest=RESULTS + "near_opt/directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}_manifest.json",
    output:
        near_opt_solutions=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}.csv",
        network_hash=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}_network_hash.txt",
    log:
        python=RESULTS + "logs/mga/aggregate_near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}_python.log",
    benchmark:
        RESULTS + "benchmarks/mga/aggregate_near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}"
    run:
        from pathlib import Path

        logger.info(f"Aggregating {len(input.batch_results)} batches")

        all_points = [pd.read_csv(f) for f in input.batch_results]
        combined = pd.concat(all_points, ignore_index=True)
        logger.info(f"Total solutions: {len(combined)}")

        with open(input.manifest) as f:
            manifest = json.load(f)
        network_hash = manifest["network_hash"]

        # Verify cache completeness before writing outputs — if any info file is
        # missing the batch solve failed silently; fail here with a clear message
        # rather than letting downstream analysis rules hit MissingInputException.
        direction_hashes = combined["dir_hash"].unique().tolist() if "dir_hash" in combined.columns else []
        cache_dir = params.cache_dir
        missing = [
            dh for dh in direction_hashes
            if not Path(f"{cache_dir}/info/info_{network_hash}_{dh}.json").exists()
        ]
        if missing:
            raise RuntimeError(
                f"Cache incomplete after batch solve: {len(missing)}/{len(direction_hashes)} "
                f"info files missing for network {network_hash}. "
                f"Re-run compute_near_opt_batch to populate. First missing: {missing[0]}"
            )

        combined.to_csv(output.near_opt_solutions, index=False)
        Path(output.network_hash).write_text(network_hash)
        logger.info("Aggregation complete")



# ========== Near-Opt Resilience Analysis ==========
# Computes resilience metrics for cost-optimal runs, MGA candidates, as well as their validation runs.
# =========================================================

rule analyse_cost_optimal_validation:
    """Analyse cost-optimal validation runs to produce baseline shedding statistics."""
    wildcard_constraints:
        run=r"weather_year_\d+_\d+H",
    input:
        val_dirs=expand(
            "results/" + config["run"]["prefix"] + "/{{run}}/validation/{operational_year}_base_s_{{clusters}}_{{opts}}_{{sector_opts}}_{{planning_horizons}}/objective.json",
            operational_year=STRESS_YEARS,
        ),
        network_hash="results/" + config["run"]["prefix"] + "/{run}/near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_network_hash.txt",
    output:
        baseline_shedding="results/" + config["run"]["prefix"] + "/{run}/resilience/baseline_shedding_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        baseline_analysis="results/" + config["run"]["prefix"] + "/{run}/resilience/baseline_analysis_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
    log:
        python="results/" + config["run"]["prefix"] + "/{run}/logs/mga/analyse_cost_optimal_validation/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    benchmark:
        "results/" + config["run"]["prefix"] + "/{run}/benchmarks/mga/analyse_cost_optimal_validation/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/analyse_cost_optimal_validation.py"


def _get_mga_info_files(wildcards):
    """Return paths to per-direction info JSON files in the MGA cache.

    Gated on the aggregate_near_opt checkpoint: Snakemake guarantees this function
    is only evaluated after that checkpoint has successfully completed (which in turn
    requires all batch solves to have written their cache files). This eliminates
    the stale-state fragility of the previous os.path.exists approach.
    """
    import pandas as pd
    cp_out = checkpoints.aggregate_near_opt.get(pathway_suffix="", **wildcards).output  # cost-opt MGA
    network_hash = open(cp_out.network_hash).read().strip()
    near_opt = pd.read_csv(cp_out.near_opt_solutions)
    direction_hashes = near_opt["dir_hash"].unique().tolist() if "dir_hash" in near_opt.columns else []
    cache_dir = config_provider("near-opt", "cache_dir")(wildcards)
    return [f"{cache_dir}/info/info_{network_hash}_{dh}.json" for dh in direction_hashes]


def _get_mga_validation_load_shedding(wildcards):
    """Return paths to per-direction, per-stress-year load_shedding validation files.

    Gated on the aggregate_near_opt checkpoint (same rationale as _get_mga_info_files).
    """
    import pandas as pd
    cp_out = checkpoints.aggregate_near_opt.get(pathway_suffix="", **wildcards).output  # cost-opt MGA
    network_hash = open(cp_out.network_hash).read().strip()
    near_opt = pd.read_csv(cp_out.near_opt_solutions)
    direction_hashes = near_opt["dir_hash"].unique().tolist() if "dir_hash" in near_opt.columns else []
    return expand(
        val_mga("load_shedding.csv"),
        design_year=wildcards.run,
        network_hash=network_hash,
        dir_hash=direction_hashes,
        operational_year=STRESS_YEARS,
        clusters=wildcards.clusters,
        opts=wildcards.opts,
        sector_opts=wildcards.sector_opts,
        planning_horizons=wildcards.planning_horizons,
    )


rule analyse_mga_validation:
    """Analyse MGA validation runs to produce per-candidate resilience metrics."""
    wildcard_constraints:
        run=r"weather_year_\d+_\d+H",
    input:
        info_files=_get_mga_info_files,
        load_shedding=_get_mga_validation_load_shedding,
        baseline_shedding=RESULTS + "resilience/baseline_shedding_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        baseline_analysis=RESULTS + "resilience/baseline_analysis_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        cost_opt_summary=RESULTS + "resilience/cost_opt_summary_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.json",
        near_opt_solutions=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        network_hash=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_network_hash.txt",
    output:
        atomic=RESULTS + "resilience/mga_validation_atomic_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        rollup=RESULTS + "resilience/mga_validation_rollup_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
    params:
        cache_dir=config_provider("near-opt", "cache_dir"),
    log:
        python=RESULTS + "logs/mga/analyse_mga_validation/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    benchmark:
        RESULTS + "benchmarks/mga/analyse_mga_validation/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/analyse_mga_validation.py"


rule analyse_cost_opt:
    input:
        network=RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
    output:
        summary=RESULTS + "resilience/cost_opt_summary_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.json",
        caps=RESULTS + "resilience/cost_opt_caps_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        netload_stats=RESULTS + "resilience/cost_opt_netload_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        price_stats=RESULTS + "resilience/cost_opt_prices_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        emissions=RESULTS + "resilience/cost_opt_emissions_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        capital_costs=RESULTS + "resilience/cost_opt_capital_costs_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
    log:
        python=RESULTS + "logs/mga/analyse_cost_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    benchmark:
        RESULTS + "benchmarks/mga/analyse_cost_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
    message:
        "Analysing cost-optimal network for {wildcards.run}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/analyse_cost_opt.py"


rule analyse_mga_candidates:
    wildcard_constraints:
        run=r"weather_year_\d+_\d+H",
    input:
        info_files=_get_mga_info_files,
        cost_opt_caps=RESULTS + "resilience/cost_opt_caps_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        cost_opt_summary=RESULTS + "resilience/cost_opt_summary_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.json",
        cost_opt_emissions=RESULTS + "resilience/cost_opt_emissions_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        capital_costs=RESULTS + "resilience/cost_opt_capital_costs_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        near_opt_solutions=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        network_hash=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_network_hash.txt",
    output:
        candidates=RESULTS + "resilience/mga_candidates_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
    params:
        cache_dir=config_provider("near-opt", "cache_dir"),
    log:
        python=RESULTS + "logs/mga/analyse_mga_candidates/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    benchmark:
        RESULTS + "benchmarks/mga/analyse_mga_candidates/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
    message:
        "Analysing MGA candidates for {wildcards.run}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/analyse_mga_candidates.py"


# ========== Pathways: o (cost-optimum) / c (Chebyshev centre) ==========
# A pathway network is named by its steps, one per horizon, appended to the file name,
# e.g. base_s_2___2040_c2030-o2040.nc = centre 2030 -> cost-opt 2040.
# ...-oY is brownfield on the previous step, ...-cY is ...-oY re-solved at the Chebyshev
# centre of its MGA. All-o pathways are the existing cost-opt networks (no suffix).
# =========================================================

def _get_pathway_parent(wildcards):
    """Pathway suffix and horizon of the parent: ...-cY -> ...-oY (same horizon), ...-oY -> ... (previous horizon)."""
    planning_horizons = [str(h) for h in config["scenario"]["planning_horizons"]]
    steps = wildcards.pathway.split("-")
    if [s[1:] for s in steps] != planning_horizons[: len(steps)] or steps[-1][1:] != wildcards.planning_horizons:
        raise ValueError(f"Pathway '{wildcards.pathway}' does not match horizons {planning_horizons}")
    parent = steps[:-1] + ["o" + steps[-1][1:]] if steps[-1].startswith("c") else steps[:-1]
    pathway_suffix = "_" + "-".join(parent) if any(s.startswith("c") for s in parent) else ""  # all-o = cost-opt network
    return pathway_suffix, parent[-1][1:]


def _get_parent_network(wildcards):
    """Solved network of the parent of a pathway network."""
    pathway_suffix, planning_horizons = _get_pathway_parent(wildcards)
    return expand(
        RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}.nc",
        pathway_suffix=pathway_suffix,
        run=wildcards.run,
        clusters=wildcards.clusters,
        opts=wildcards.opts,
        sector_opts=wildcards.sector_opts,
        planning_horizons=planning_horizons,
    )[0]


def _get_parent_near_opt(wildcards):
    """MGA results (near-opt CSV and network hash) of the parent o network of a c pathway."""
    pathway_suffix, planning_horizons = _get_pathway_parent(wildcards)
    near_opt = expand(
        RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}",
        pathway_suffix=pathway_suffix,
        run=wildcards.run,
        clusters=wildcards.clusters,
        opts=wildcards.opts,
        sector_opts=wildcards.sector_opts,
        planning_horizons=planning_horizons,
    )[0]
    return {"near_opt": near_opt + ".csv", "network_hash": near_opt + "_network_hash.txt"}


def _get_parent_manifest(wildcards):
    """MGA manifest of the parent o network of a c pathway (remaining slack, gives C* of the budget)."""
    pathway_suffix, planning_horizons = _get_pathway_parent(wildcards)
    return expand(
        RESULTS + "near_opt/directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}{pathway_suffix}_manifest.json",
        pathway_suffix=pathway_suffix,
        run=wildcards.run,
        clusters=wildcards.clusters,
        opts=wildcards.opts,
        sector_opts=wildcards.sector_opts,
        planning_horizons=planning_horizons,
    )[0]


rule compute_chebyshev_centre:
    """Chebyshev centre of the MGA on the parent o network of a c pathway."""
    wildcard_constraints:
        pathway=PATHWAY_C,
    message:
        "Computing Chebyshev centre for pathway {wildcards.pathway} ({wildcards.run})"
    input:
        unpack(_get_parent_near_opt),
    output:
        centre=RESULTS + "near_opt/chebyshev_centre_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}.json",
    log:
        python=RESULTS + "logs/mga/compute_chebyshev_centre/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}_python.log",
    benchmark:
        RESULTS + "benchmarks/mga/compute_chebyshev_centre/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}"
    localrule: True
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/compute_chebyshev_centre.py"


rule solve_chebyshev_network:
    """Re-solve the parent o network with the MGA dimensions pinned to the Chebyshev centre."""
    wildcard_constraints:
        pathway=PATHWAY_C,
    params:
        solving=config_provider("solving"),
        custom_extra_functionality=input_custom_extra_functionality,
    message:
        "Solving Chebyshev-centre network for pathway {wildcards.pathway} ({wildcards.run})"
    input:
        network=_get_parent_network,
        centre=RESULTS + "near_opt/chebyshev_centre_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}.json",
        manifest=_get_parent_manifest,  # remaining slack of the parent MGA -> C* of the budget
    output:
        network=RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}.nc",
        config=RESULTS + "configs/config.base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}.yaml",
        summary=RESULTS + "near_opt/chebyshev_centre_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}_summary.json",
    log:
        solver=RESULTS + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}_solver.log",
        memory=RESULTS + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}_memory.log",
        python=RESULTS + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}_python.log",
    benchmark:
        RESULTS + "benchmarks/solve_chebyshev_network/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}"
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_chebyshev_network.py"


# add_brownfield / solve_sector_network_myopic only exist for myopic foresight
if config["foresight"] == "myopic":

    use rule add_brownfield as add_brownfield_pathway with:
        wildcard_constraints:
            pathway=PATHWAY_O,
        message:
            "Adding brownfield constraints for pathway {wildcards.pathway} ({wildcards.run})"
        input:
            unpack(input_profile_tech_brownfield),
            network=resources("networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"),
            network_p=_get_parent_network,
        output:
            resources("networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}_brownfield.nc"),
        log:
            logs("add_brownfield_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}.log"),
        benchmark:
            benchmarks("add_brownfield/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}")

    use rule solve_sector_network_myopic as solve_sector_network_myopic_pathway with:
        wildcard_constraints:
            pathway=PATHWAY_O,
        message:
            "Solving sector-coupled network with myopic foresight for pathway {wildcards.pathway} ({wildcards.run})"
        input:
            network=resources("networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}_brownfield.nc"),
        output:
            network=RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}.nc",
            config=RESULTS + "configs/config.base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}.yaml",
        log:
            solver=RESULTS + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}_solver.log",
            memory=RESULTS + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}_memory.log",
            python=RESULTS + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}_python.log",
        benchmark:
            RESULTS + "benchmarks/solve_sector_network/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{pathway}"
