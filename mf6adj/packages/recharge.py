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
        Factor for every cell. A cell recharged at a nonzero rate takes the
        factor from the right-hand side, which carries the area and the
        multiplier together. A cell recharged at zero produces no flow whatever
        either is, so the area and the multiplier are taken separately there. A
        cell no package recharges keeps the area alone.
    """
    factor = area.copy()
    for group in groups:
        if "recharge" not in group:
            continue
        rate = group["recharge"][:]
        applied = -group["rhs"][:]
        nodes = group["nodelist"][:] - 1
        given = rate != 0.0
        factor[nodes[given]] = applied[given] / rate[given]
        if not given.all():
            multiplier = (
                group["auxmult"][:] if "auxmult" in group else np.ones(rate.shape[0])
            )
            idle = nodes[~given]
            factor[idle] = area[idle] * multiplier[~given]
    return factor
