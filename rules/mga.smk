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
        network=RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
    output:
        directions_dir=directory(
            RESULTS + "near_opt/directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}/"
        ),
        manifest=RESULTS + "near_opt/directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_manifest.json",
    log:
        python=RESULTS + "logs/mga/generate_near_opt_directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    benchmark:
        RESULTS + "benchmarks/mga/generate_near_opt_directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
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
        RESULTS + "near_opt/directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}/{dir_hash}.json",
        dir_hash=batch_dirs,
        run=wildcards.run,
        clusters=wildcards.clusters,
        opts=wildcards.opts,
        sector_opts=wildcards.sector_opts,
        planning_horizons=wildcards.planning_horizons,
    )


def _get_all_batch_results(wildcards):
    """Get all batch result files (triggers checkpoint resolution)."""
    batches = _get_near_opt_batches(wildcards)
    return expand(
        RESULTS + "near_opt/batches/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}/batch_{batch_hash}.csv",
        batch_hash=list(batches.keys()),
        run=wildcards.run,
        clusters=wildcards.clusters,
        opts=wildcards.opts,
        sector_opts=wildcards.sector_opts,
        planning_horizons=wildcards.planning_horizons,
    )


rule compute_near_opt_batch:
    """Solve one batch of near-opt directions on one SLURM node."""
    wildcard_constraints:
        batch_hash=r"[a-f0-9]{8}",
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
        network=RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        direction_files=_get_batch_direction_files,
    output:
        batch_result=temp(RESULTS + "near_opt/batches/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}/batch_{batch_hash}.csv"),
    log:
        python=RESULTS + "logs/mga/compute_near_opt_batch/batch_{batch_hash}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    benchmark:
        RESULTS + "benchmarks/mga/compute_near_opt_batch/batch_{batch_hash}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
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
    params:
        solving=config_provider("solving"),
        cache_dir=config_provider("near-opt", "cache_dir"),
    message:
        "Aggregating near-optimal batch results for {wildcards.run}"
    input:
        batch_results=_get_all_batch_results,
        manifest=RESULTS + "near_opt/directions/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_manifest.json",
    output:
        near_opt_solutions=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        network_hash=RESULTS + "near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_network_hash.txt",
    log:
        python=RESULTS + "logs/mga/aggregate_near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    benchmark:
        RESULTS + "benchmarks/mga/aggregate_near_opt/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
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
    cp_out = checkpoints.aggregate_near_opt.get(**wildcards).output
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
    cp_out = checkpoints.aggregate_near_opt.get(**wildcards).output
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
