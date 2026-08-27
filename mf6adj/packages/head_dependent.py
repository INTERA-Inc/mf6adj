"""Head-dependent boundaries: their derivatives, and the corner a drain has.

A general-head, drain, or river boundary carries a flow that follows both the
head in the cell and the values the boundary is given, so a performance measure
has a sensitivity to each.

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


def lam_drhs_dbnd(
    lamb: np.ndarray,
    head: np.ndarray,
    sp_dict: dict,
    direct_weights: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return adjoint-weighted derivatives with respect to boundary terms.

    The ``sp_dict["bound"]`` array is interpreted positionally with
    ``bound[0]`` as boundary head and optional ``bound[1]`` as boundary
    conductance.

    Parameters
    ----------
    lamb : ndarray
        Adjoint state array.
    head : ndarray
        Head array.
    sp_dict : dict
        Stress-package data for a single time step containing at least
        ``node`` and ``bound`` arrays, and ``auxmult`` where the package
        scales its values by an auxiliary multiplier.
    direct_weights : dict
        Node to weight for the entries this measure takes from this package.
        A boundary the measure does not name has no direct contribution.

    Returns
    -------
    tuple[ndarray, ndarray]
        Two arrays containing derivatives with respect to the head-like
        boundary term and the conductance-like boundary term,
        respectively.
    """
    result_head = np.zeros_like(lamb)
    result_cond = np.zeros_like(lamb)

    # for id in sp_dict:
    # A cell can carry more than one boundary from the same package - a
    # lake connected both vertically and horizontally to the same cell, or
    # two river reaches in one cell - and each one contributes to that
    # cell's residual, so the derivatives accumulate rather than overwrite.
    auxmult = sp_dict.get("auxmult")
    for i, (node, bound) in enumerate(zip(sp_dict["node"], sp_dict["bound"])):
        n = node - 1
        # A package may scale the values it is given by an auxiliary
        # multiplier, which MODFLOW applies to the conductance where it forms
        # its terms. The conductance sensitivity therefore carries it, and so
        # does the head sensitivity, which is the conductance itself.
        multiplier = 1.0 if auxmult is None else float(auxmult[i])
        boundcond = 1e10
        if len(bound) > 1:
            boundcond = bound[1] * multiplier
        # the second item in bound should be cond
        result_head[n] += lamb[n] * boundcond
        # Add the direct effect, only where the measure sums this
        # package's flux, and weighted the way the measure weights it
        weight = direct_weights.get(n, 0.0)
        if weight != 0.0:
            result_head[n] += weight * boundcond
        # the first item in bound should be head
        lam_drhs_dcond = lamb[n] * bound[0]
        lam_dadcond_h = -1.0 * lamb[n] * head[n]
        result_cond[n] += (lam_drhs_dcond + lam_dadcond_h) * multiplier
        if weight != 0.0:
            result_cond[n] += weight * (bound[0] - head[n]) * multiplier

    return result_head, result_cond
