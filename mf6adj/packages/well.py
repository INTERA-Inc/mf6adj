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
        Factor for every cell. A cell holding no well keeps one, so its
        sensitivity is that of a unit flow.

    Notes
    -----
    A cell can hold more than one well, from one package or from several, and
    the field carries one value per cell. The flow the cell produces is summed
    and divided by the rate it was given, so the factor is the response to
    changing the rate of the cell as a whole, shared out as the rates already
    are. Where every well in a cell is given a rate of zero, that ratio says
    nothing, and the multipliers are averaged instead.
    """
    applied = np.zeros(nnodes)
    given = np.zeros(nnodes)
    multiplier = np.zeros(nnodes)
    wells = np.zeros(nnodes)
    for group in groups:
        if "q" not in group:
            continue
        rate = group["q"][:]
        nodes = group["nodelist"][:] - 1
        scale = group["auxmult"][:] if "auxmult" in group else np.ones(rate.shape[0])
        np.add.at(applied, nodes, -group["rhs"][:])
        np.add.at(given, nodes, rate)
        np.add.at(multiplier, nodes, scale)
        np.add.at(wells, nodes, 1.0)

    factor = np.ones(nnodes)
    pumped = given != 0.0
    factor[pumped] = applied[pumped] / given[pumped]
    idle = (wells > 0.0) & ~pumped
    factor[idle] = multiplier[idle] / wells[idle]
    return factor
