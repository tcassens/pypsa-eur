# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT
"""
Generate near-optimal direction files for checkpoint-based parallelization.

Creates one JSON file per direction + manifest for snakemake discovery.
This enables multi-node parallel solving where each batch of directions
runs on a separate SLURM node.
"""

import logging
import json
import numpy as np
import pandas as pd
import pypsa
from pathlib import Path

from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from solve_second_network import fix_networks
from mga_helpers import remaining_slack
from pypsa.optimization.mga import hash_direction, hash_mga

# Import direction generation functions from compute_near_opt
from compute_near_opt import (
    load_dimensions_from_config,
    fill_dimension_weights,
    generate_minmax_directions,
    generate_random_directions,
    generate_halton_directions,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "generate_near_opt_directions",
            opts="",
            clusters="50",
            configfiles="config/mini-sector_droughts.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # Load network (needed for dimensions)
    logger.info(f"Loading network from {snakemake.input.network}")
    n = pypsa.Network(snakemake.input.network)
    m = n.copy()

    # Fix network to prevent transmission expansion
    logger.info("Fixing network capacities")
    fix_networks(m, n)

    # Load near-opt configuration
    mga_config = snakemake.config.get("near-opt", {})
    if not mga_config:
        raise ValueError("No 'near-opt' configuration found in config file")

    # Compute slack (needed for network hash)
    slack_config = mga_config.get("slack", {})
    if slack_config.get("relative", True):
        slack = slack_config.get("value", 0.05)
        logger.info(f"Using relative slack: {slack}")
    else:
        # Absolute slack in currency units — use statistics to match compute_near_opt_batch.py
        absolute_slack = float(slack_config.get("value", 0.0))
        objective_value = float(n.statistics.capex().sum() + n.statistics.opex().sum())
        if objective_value == 0:
            raise ValueError("Network objective is zero.")
        slack = float(absolute_slack) / objective_value
        logger.info(f"Using absolute slack: {absolute_slack} (relative: {slack:.4f})")

    # Budget relative to the cost-opt network of this horizon (unchanged for cost-opt networks)
    slack = remaining_slack(n, slack, snakemake.input.network, snakemake.input.reference)

    # Load and fill dimensions
    logger.info("Loading projection dimensions from config")
    config_projection = mga_config.get("projection", {})
    dimensions = load_dimensions_from_config(config_projection)
    dimensions = fill_dimension_weights(m, dimensions)

    dimension_names = list(dimensions.keys())
    logger.info(f"Near-opt dimensions: {dimension_names}")

    # Generate directions
    approx_config = mga_config.get("approx", {})
    all_directions = []

    # Min-max directions (if enabled)
    if approx_config.get("minmax", False):
        logger.info("Generating min-max directions")
        minmax_dirs = generate_minmax_directions(dimension_names)
        all_directions.append(minmax_dirs)
        logger.info(f"Generated {len(minmax_dirs)} min-max directions")

    # Additional directions
    n_iterations = approx_config.get("iterations", 0)
    if n_iterations > 0:
        direction_method = approx_config.get("directions", "random-uniform")

        if direction_method == "random-uniform":
            logger.info(f"Generating {n_iterations} random directions")
            seed = approx_config.get("seed", 123)
            random_dirs = generate_random_directions(dimension_names, n_iterations, seed)
            all_directions.append(random_dirs)
        elif direction_method == "halton":
            logger.info(f"Generating {n_iterations} Halton directions")
            halton_dirs = generate_halton_directions(dimension_names, n_iterations)
            all_directions.append(halton_dirs)
        else:
            raise ValueError(f"Unknown direction method: {direction_method}")

    # Combine all directions
    if not all_directions:
        raise ValueError("No directions generated. Enable minmax or set iterations > 0")

    directions_df = pd.concat(all_directions, ignore_index=True)
    logger.info(f"Total directions to explore: {len(directions_df)}")

    # Compute network hash (for matching with MGA cache)
    logger.info("Computing network hash")
    network_hash = hash_mga(
        m,
        dimensions,
        slack,
        snapshots=None,  # Uses all snapshots
        multi_investment_periods=False,
    )
    logger.info(f"Network hash: {network_hash}")

    # Create output directory
    output_dir = Path(snakemake.output.directions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each direction as JSON
    direction_hashes = []
    for idx, row in directions_df.iterrows():
        # Use pypsa-mga's hash_direction for consistency
        dir_hash = hash_direction(row)
        direction_hashes.append(dir_hash)

        # Save direction as JSON
        filepath = output_dir / f"{dir_hash}.json"
        row.to_json(filepath)

    logger.info(f"Created {len(direction_hashes)} direction files")

    # Save manifest
    manifest = {
        "network_hash": network_hash,
        "slack": slack,  # relative to this network, read by compute_near_opt_batch
        "directions": direction_hashes,
        "total": len(direction_hashes),
        "dimensions": dimension_names,
    }

    with open(snakemake.output.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest saved: {len(direction_hashes)} directions, network_hash={network_hash}")
    logger.info("Direction generation complete ✓")
