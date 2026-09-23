# SPDX-FileCopyrightText: 2025 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT

"""
Compute near-optimal solutions using Modeling to Generate Alternatives (MGA).

This script performs MGA optimization to explore the space of near-optimal solutions
for energy system planning. It relies on pypsa-mga for the core MGA functionality.
"""

import logging
import numpy as np
import pandas as pd
import pypsa
from pathlib import Path
from functools import partial

from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from mga_helpers import apply_mga_extra_functionality, export_mga_capacities, export_mga_information
from solve_second_network import fix_networks
from pypsa.optimization.mga import hash_direction, hash_mga

logger = logging.getLogger(__name__)


def load_dimensions_from_config(config_projection):
    """
    Load MGA dimensions from config projection specification.

    Parameters
    ----------
    config_projection : dict
        Projection configuration with dimension categories

    Returns
    -------
    dict
        Dimensions in pypsa-mga format
    """
    dimensions = {}

    for category, specs in config_projection.items():
        # Each category becomes a dimension
        category_dict = {}

        for spec in specs:
            carrier = spec['carrier']
            component = spec['component']
            attribute = spec['attribute']
            weight_attr = spec['weight']

            # Build nested structure: {component: {attribute: {carrier: weight}}}
            if component not in category_dict:
                category_dict[component] = {}
            if attribute not in category_dict[component]:
                category_dict[component][attribute] = {}

            # For now, we use the carrier name as placeholder
            # The actual weight will be filled from the network
            category_dict[component][attribute][carrier] = weight_attr

        dimensions[category] = category_dict

    return dimensions


def fill_dimension_weights(n, dimensions):
    """
    Fill dimension weights with actual values from the network.

    Parameters
    ----------
    n : pypsa.Network
        Network object
    dimensions : dict
        Dimensions structure with weight attribute names

    Returns
    -------
    dict
        Dimensions with actual numeric weights
    """
    filled_dimensions = {}

    for dim_name, components in dimensions.items():
        filled_components = {}

        for component, attributes in components.items():
            comp_df = n.df(component)
            filled_attributes = {}

            for attribute, carriers in attributes.items():
                filled_carriers = {}

                for carrier, weight_attr in carriers.items():
                    # Find components matching this carrier
                    matching = comp_df[comp_df['carrier'] == carrier].index

                    if len(matching) > 0:
                        # Get weight values for matching components
                        for comp_name in matching:
                            weight_value = comp_df.loc[comp_name, weight_attr]
                            filled_carriers[comp_name] = weight_value

                if filled_carriers:
                    filled_attributes[attribute] = filled_carriers

            if filled_attributes:
                filled_components[component] = filled_attributes

        if filled_components:
            filled_dimensions[dim_name] = filled_components

    return filled_dimensions


def generate_minmax_directions(dimension_names):
    """
    Generate unit vectors in positive and negative directions.

    Parameters
    ----------
    dimension_names : list
        List of dimension names

    Returns
    -------
    pd.DataFrame
        DataFrame with direction vectors (one per row)
    """
    directions = []

    # Create unit vectors in each dimension
    for i, dim in enumerate(dimension_names):
        # Positive direction
        pos_dir = np.zeros(len(dimension_names))
        pos_dir[i] = 1.0
        directions.append(pos_dir)

        # Negative direction
        neg_dir = np.zeros(len(dimension_names))
        neg_dir[i] = -1.0
        directions.append(neg_dir)

    return pd.DataFrame(directions, columns=dimension_names)


def generate_random_directions(dimension_names, n_directions, seed=None):
    """
    Generate random uniformly distributed directions on unit sphere.

    Parameters
    ----------
    dimension_names : list
        List of dimension names
    n_directions : int
        Number of directions to generate
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with direction vectors (one per row)
    """
    from pypsa.optimization.mga import generate_directions_random

    directions = generate_directions_random(
        keys=dimension_names,
        n_directions=n_directions,
        seed=seed
    )

    return directions


def generate_halton_directions(dimension_names, n_directions):
    """
    Generate directions using Halton sequence.

    Parameters
    ----------
    dimension_names : list
        List of dimension names
    n_directions : int
        Number of directions to generate

    Returns
    -------
    pd.DataFrame
        DataFrame with direction vectors (one per row)
    """
    from pypsa.optimization.mga import generate_directions_halton

    directions = generate_directions_halton(
        keys=dimension_names,
        n_directions=n_directions
    )

    return directions


def combine_results(directions_df, coordinates_df):
    """
    Combine directions and coordinates into single DataFrame with dir_hash.

    Parameters
    ----------
    directions_df : pd.DataFrame
        Direction vectors
    coordinates_df : pd.DataFrame
        Coordinate values

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns: dir_hash, dir_*, coord_*
    """
    # Compute hash for each direction using pypsa-mga's hash function
    dir_hashes = []
    for idx, row in directions_df.iterrows():
        dir_hash = hash_direction(row)
        dir_hashes.append(dir_hash)

    # Rename columns
    directions_renamed = directions_df.copy()
    directions_renamed.columns = [f"dir_{col}" for col in directions_df.columns]

    coordinates_renamed = coordinates_df.copy()
    coordinates_renamed.columns = [f"coord_{col}" for col in coordinates_df.columns]

    # Combine
    result = pd.DataFrame({'dir_hash': dir_hashes})
    result = pd.concat([result, directions_renamed, coordinates_renamed], axis=1)

    return result


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "compute_near_opt",
            opts="",
            clusters="5",
            configfiles="config/test/config.overnight.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # Load network
    logger.info(f"Loading network from {snakemake.input.network}")
    n = pypsa.Network(snakemake.input.network)
    m = n.copy()

    # Fix network to prevent transmission expansion
    logger.info("Fixing network capacities")
    fix_networks(m, n)

    planning_horizons = snakemake.wildcards.get("planning_horizons", None)

    # Load MGA configuration
    mga_config = snakemake.config.get("near-opt", {})
    if not mga_config:
        raise ValueError("No 'near-opt' configuration found in config file")

    # Compute slack
    slack_config = mga_config.get("slack", {})
    if slack_config.get("relative", True):
        slack = slack_config.get("value", 0.05)
        logger.info(f"Using relative slack: {slack}")
    else:
        absolute_slack = float(slack_config.get("value", 0.0))
        objective_value = float(n.statistics.capex().sum() + n.statistics.opex().sum())
        if objective_value == 0:
            raise ValueError("Network objective is zero.")
        slack = float(absolute_slack) / objective_value
        logger.info(f"Using absolute slack: {absolute_slack} (relative: {slack:.4f})")

    # Load and fill dimensions
    logger.info("Loading projection dimensions from config")
    config_projection = mga_config.get("projection", {})
    dimensions = load_dimensions_from_config(config_projection)
    dimensions = fill_dimension_weights(m, dimensions)

    dimension_names = list(dimensions.keys())
    logger.info(f"MGA dimensions: {dimension_names}")

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

    # Run MGA optimization with caching
    logger.info("Running MGA optimization")
    max_parallel = approx_config.get("max_parallel", 4)
    cache_dir = mga_config.get("cache_dir", None)

    # Get solver configuration
    solver_config = snakemake.config.get("solving", {})
    solver_name = solver_config.get("solver", {}).get("name", "gurobi")
    solver_options = solver_config.get("solver_options", {}).get(
        solver_config.get("solver", {}).get("options", "default"), {}
    )

    logger.info(f"Using solver: {solver_name}")

    successful_directions, successful_coordinates = m.optimize.optimize_mga_in_multiple_directions(
        directions=directions_df,
        dimensions=dimensions,
        cache_dir=cache_dir,
        mga_extra_functionality=partial(export_mga_information, wildcards=dict(snakemake.wildcards), slack=slack_config),
        extra_functionality=partial(
            apply_mga_extra_functionality,
            config=snakemake.config,
            custom_extra_functionality=snakemake.params.custom_extra_functionality,
            planning_horizons=planning_horizons,
        ),
        snapshots=None,
        multi_investment_periods=False,
        slack=slack,
        model_kwargs=None,
        max_parallel=max_parallel,
        solver_name=solver_name,
        solver_options=solver_options,
    )

    logger.info(f"Successfully solved {len(successful_directions)} out of {len(directions_df)} directions")

    # Compute network hash for storage
    logger.info("Computing network hash")
    network_hash = hash_mga(
        m,
        dimensions,
        slack,
        snapshots=None,  # Uses all snapshots
        multi_investment_periods=False,
    )
    logger.info(f"Network hash: {network_hash}")

    # Combine and export results
    logger.info("Combining results")
    combined_results = combine_results(successful_directions, successful_coordinates)

    logger.info(f"Exporting results to {snakemake.output.near_opt_solutions}")
    combined_results.to_csv(snakemake.output.near_opt_solutions, index=False)

    # Save network hash to separate file
    logger.info(f"Saving network hash to {snakemake.output.network_hash}")
    Path(snakemake.output.network_hash).write_text(network_hash)

    logger.info("MGA computation complete")
