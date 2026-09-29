# SPDX-FileCopyrightText: 2026 Till Cassens
#
# SPDX-License-Identifier: MIT

"""
Compute the Chebyshev centre of the near-optimal space spanned by an MGA run.

The points are the MGA coordinates (``coord_*`` columns of the near-opt CSV,
annualised EUR per dimension, ``sum capital_cost * p_nom`` over the extendable
assets). The centre is the centre of the largest ball inside the convex hull
of these points (inner approximation of the near-optimal set), so it is a
convex combination of feasible near-optimal solutions. The cost-optimal point
is not added: it almost always lies in the interior of the hull, and loading
the network only for that is slow.
"""

import json
import logging

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)

logger = logging.getLogger(__name__)


def load_mga_points(near_opt_csv):
    """
    Read the MGA coordinates from the near-opt CSV: one row per direction
    (index ``dir_hash``), one column per dimension.
    """
    return (
        pd.read_csv(near_opt_csv, index_col="dir_hash")
        .filter(regex="^coord_")
        .rename(columns=lambda c: c.removeprefix("coord_"))
    )


def chebyshev_centre(points, range_tol=1e-6):
    """
    Chebyshev centre of the convex hull of ``points``.

    Standard recipe (as in the scipy ``HalfspaceIntersection`` docs): the
    hull facets ``A x + b <= 0`` (unit-norm rows) from ``ConvexHull``, then
    ``max r  s.t.  A x + r <= -b`` with ``linprog``. Dimensions whose range is
    below ``range_tol`` times the largest range (e.g. ``tes`` ~ 0) are dropped,
    since qhull fails on flat input, and kept at their mean value.

    Returns
    -------
    centre : pd.Series
        Centre per dimension (EUR).
    radius : float
        Radius of the largest ball inside the hull (EUR).
    """
    ranges = points.max() - points.min()
    kept = ranges.index[ranges > range_tol * ranges.max()]  # flat dims break qhull
    dropped = ranges.index.difference(kept)
    if len(dropped):
        logger.info(f"Dropping zero-range dimensions {list(dropped)}")

    hull = ConvexHull(points[kept].to_numpy() / 1e6) # MEUR/a 
    A, b = hull.equations[:, :-1], hull.equations[:, -1]  # inside: A x + b <= 0, |A_i| = 1
    c = np.zeros(len(kept) + 1)  # variables [x, r]
    c[-1] = -1  # max r
    res = linprog(
        c,
        A_ub=np.hstack([A, np.ones((len(A), 1))]),  # ball of radius r inside every facet
        b_ub=-b,
        bounds=[(None, None)] * len(kept) + [(0, None)],  # x free, r >= 0
        method="highs",
    )
    if res.status != 0 or not res.x[-1] > 0:
        raise RuntimeError(f"Chebyshev centre LP failed: {res.message}")

    centre = points.mean()  # dropped dims keep their constant value
    centre[kept] = res.x[:-1] * 1e6  # back to EUR/a
    return centre, res.x[-1] * 1e6


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "compute_chebyshev_centre",
            opts="",
            clusters="2",
            sector_opts="",
            planning_horizons="2030",
            configfiles="config/pathway_mga/myopic_dk.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    mga_config = snakemake.config["near-opt"]
    option = mga_config["pathway"]["option"]
    if option != "chebyshev":  # only option so far
        raise ValueError(f"near-opt.pathway.option '{option}' is not supported")

    with open(snakemake.input.network_hash) as f:
        network_hash = f.read().strip()  # traces the centre back to its MGA run

    points = load_mga_points(snakemake.input.near_opt)  # one point per direction, EUR/a per dim
    logger.info(f"Loaded {len(points)} MGA points in dimensions {list(points.columns)}")

    centre, radius = chebyshev_centre(points)
    logger.info(f"Chebyshev centre (EUR): {centre.to_dict()}, radius {radius:.4g} EUR")

    out = {
        "option": option,
        "planning_horizons": snakemake.wildcards.planning_horizons,
        "network_hash": network_hash,
        "unit": "EUR (annualised capital cost, sum capital_cost * nom over extendables)",
        "centre": centre.to_dict(),  # pinned by the re-solve
        "radius": radius,
        "n_points": len(points),
        "min": points.min().to_dict(),  # MGA range per dim
        "max": points.max().to_dict(),
    }
    with open(snakemake.output.centre, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Centre written to {snakemake.output.centre}")
