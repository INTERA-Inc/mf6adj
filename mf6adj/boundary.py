"""Head-dependent boundaries whose flow has a corner at a threshold.

A drain without a drainage depth switches on and off at its elevation, so its
flow is not differentiable there. The derivative is minus the conductance
approaching from above and zero approaching from below, and a converged solution
leaves drains sitting on that corner because a drain pins the head at its
elevation.
"""

import numpy as np

DRAIN_CORNER_TOL = 1.0e-6
"""Default head above a drain elevation below which the entry is on the corner."""


def drain_corner_entries(group, head, tol=DRAIN_CORNER_TOL):
    """Return the drain entries whose activation state is unresolved.

    Parameters
    ----------
    group : h5py.Group
        Forward-solution group for one drain package.
    head : ndarray
        Simulated head for every node.
    tol : float, optional
        Head above the drain elevation below which the entry is on the corner.
        A value of zero disables the check.

    Returns
    -------
    ndarray
        Boolean mask, true where the entry sits on the corner.
    """
    if tol <= 0.0 or "elev" not in group:
        return np.zeros(group["hcof"].shape[0], dtype=bool)

    nodes = group["nodelist"][:] - 1
    hcof = group["hcof"][:]
    above = head[nodes] - group["elev"][:]
    # an active drain this close to its elevation carries no flow and turns off
    # under any drawdown, so its conductance is not a defensible derivative
    return (hcof != 0.0) & (above <= tol)


def drop_corner_entries(group, hcof, head, tol=DRAIN_CORNER_TOL):
    """Zero the drain entries that sit on the corner.

    Parameters
    ----------
    group : h5py.Group
        Forward-solution group for one drain package.
    hcof : ndarray
        Boundary hcof for the package.
    head : ndarray
        Simulated head for every node.
    tol : float, optional
        Head above the drain elevation below which the entry is on the corner.

    Returns
    -------
    tuple[ndarray, int]
        Adjusted hcof and the number of entries that were zeroed.
    """
    corner = drain_corner_entries(group, head, tol)
    ncorner = int(corner.sum())
    if ncorner == 0:
        return hcof, 0
    adjusted = hcof.copy()
    adjusted[corner] = 0.0
    return adjusted, ncorner
