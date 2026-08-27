"""Recharge (RCH) terms for the adjoint solution.

A well rate is already a flow, but recharge is a rate over the cell area, and a
package may scale it further by an auxiliary multiplier. MODFLOW 6 applies both
where it forms the right-hand side rather than folding them into the rate it
keeps, so the flow a recharge value produces is read from there.
"""

import numpy as np


def rate_factor(area, groups):
    """Return the flow each cell's recharge rate produces, per unit of rate.

    Parameters
    ----------
    area : numpy.ndarray
        Horizontal area of every cell in the grid.
    groups : iterable
        One stored group per recharge package, each holding ``recharge``,
        ``rhs``, ``nodelist``, and, where the package has one, ``auxmult``.

    Returns
    -------
    numpy.ndarray
        Factor for every cell. A cell no package recharges keeps the area
        alone, so its sensitivity is that of a rate over that area.

    Notes
    -----
    A cell can be recharged more than once, by a package that lists it twice or
    by two packages that cover it, and the field carries one value per cell.
    The flow the cell produces is summed and divided by the rate it was given,
    which carries the area and the multiplier together and is the response to
    changing the rate of the cell as a whole. Where every rate given to a cell
    is zero, that ratio says nothing, and the area times the averaged
    multiplier stands instead.
    """
    applied = np.zeros_like(area)
    given = np.zeros_like(area)
    multiplier = np.zeros_like(area)
    entries = np.zeros_like(area)
    for group in groups:
        if "recharge" not in group:
            continue
        rate = group["recharge"][:]
        nodes = group["nodelist"][:] - 1
        scale = group["auxmult"][:] if "auxmult" in group else np.ones(rate.shape[0])
        np.add.at(applied, nodes, -group["rhs"][:])
        np.add.at(given, nodes, rate)
        np.add.at(multiplier, nodes, scale)
        np.add.at(entries, nodes, 1.0)

    factor = area.copy()
    recharged = given != 0.0
    factor[recharged] = applied[recharged] / given[recharged]
    idle = (entries > 0.0) & ~recharged
    factor[idle] = area[idle] * multiplier[idle] / entries[idle]
    return factor
