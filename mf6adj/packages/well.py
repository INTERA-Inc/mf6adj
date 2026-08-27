"""Well (WEL) terms for the adjoint solution.

A well rate is already a flow, so a performance measure follows it through the
adjoint state alone, unless the package scales the rate it is given. An
auxiliary multiplier does that, and MODFLOW 6 applies it where it forms the
right-hand side rather than folding it into the rate it keeps.
"""

import numpy as np


def rate_factor(nnodes, groups):
    """Return the flow each cell's well rate produces, per unit of rate.

    Parameters
    ----------
    nnodes : int
        Number of cells in the grid.
    groups : iterable
        One stored group per well package, each holding ``q``, ``rhs``,
        ``nodelist``, and, where the package has one, ``auxmult``.

    Returns
    -------
    numpy.ndarray
        Factor for every cell. A well given a nonzero rate takes the factor
        from the right-hand side, which carries whatever the package applied. A
        well given a rate of zero produces no flow whatever the multiplier is,
        so the multiplier is taken on its own there. A cell holding no well
        keeps one, so its sensitivity is that of a unit flow.
    """
    factor = np.ones(nnodes)
    for group in groups:
        if "q" not in group:
            continue
        rate = group["q"][:]
        applied = -group["rhs"][:]
        nodes = group["nodelist"][:] - 1
        given = rate != 0.0
        factor[nodes[given]] = applied[given] / rate[given]
        if not given.all():
            multiplier = (
                group["auxmult"][:] if "auxmult" in group else np.ones(rate.shape[0])
            )
            factor[nodes[~given]] = multiplier[~given]
    return factor
