"""Horizontal flow barrier (HFB) terms for the adjoint solution.

A barrier adds no equations and exchanges no water. What it does is change the
conductance of the connections it sits on, and MODFLOW writes that changed
conductance into the array the flow equations are assembled from rather than
keeping it apart. Two things follow from that.

A sensitivity to hydraulic conductivity is the derivative of that conductance,
so it has to carry the barrier or it answers for a connection the model does
not have. And the barrier itself is a term of the model, so a measure has a
sensitivity to the hydraulic characteristic it is given.

A barrier in series always lowers the conductance, because two conductances in
series carry less than either. Given as a multiplier it does whatever the
multiplier says, so it can raise the conductance as well, and a multiplier of
zero closes the connection.

MODFLOW keeps the conductance each connection had before the barrier was
applied. Both derivatives below are ratios of what it ended up with to what it
started from, so neither the flow area nor the geometry behind it is formed
again here, and a change to how MODFLOW computes them does not reach either.
"""

import numpy as np

# MODFLOW names the package HFB in memory whatever the name file calls it, and
# a model holds one
MEMORY_PATH = "HFB"


def forward_terms(gwf, gwf_name: str, nconn: int) -> tuple[dict, int]:
    """Return the barrier terms of a model for one time step.

    Writing `a` for the conductance a connection has without the barrier, `c`
    for the conductance of the barrier, and `h` for the hydraulic
    characteristic it is given, MODFLOW puts the two in series where `h` is
    positive, `cond = a c / (a + c)` with `c = h A` over the face area `A`. The
    two derivatives that follow are

        d(cond)/da = (cond / a)**2,
        d(cond)/dh = cond (a - cond) / (h a),

    the second of which is the face area where the barrier is weak, and falls
    away as it strengthens.

    A hydraulic characteristic of zero or less means something else: MODFLOW
    reads it as a multiplier on the conductance, `cond = -a h`, whose
    derivatives are `cond / a` and `-a`. Nothing bounds a multiplier by one, so
    this form can raise the conductance, and a multiplier of zero closes the
    connection.

    Parameters
    ----------
    gwf : modflowapi.ModflowApi
        MODFLOW 6 groundwater-flow instance.
    gwf_name : str
        Name of the groundwater-flow model.
    nconn : int
        Number of connections in the symmetric arrays, which is the length of
        the conductance array the factor is applied alongside.

    Returns
    -------
    tuple[dict, int]
        The barrier terms, and the number of barriers whose effect MODFLOW is
        not carrying in the conductance and which are therefore in neither
        derivative. ``hfb_factor`` is one for every connection no barrier sits
        on, and ``hfb_dconddhc`` is the derivative of the conductance with
        respect to the hydraulic characteristic, one per barrier.
    """

    def value(name, *components):
        return gwf.get_value(gwf.get_var_address(name, *components)).copy()

    factor = np.ones(int(nconn))
    nhfb = int(value("NHFB", gwf_name, MEMORY_PATH)[0])
    if nhfb == 0:
        return {
            "hfb_factor": factor,
            "hfb_dconddhc": np.zeros(0),
            "hfb_hydchr": np.zeros(0),
            "hfb_noden": np.zeros(0, dtype=int),
            "hfb_nodem": np.zeros(0, dtype=int),
        }, 0

    # the arrays are sized for the most barriers the package may hold, which is
    # not always how many it holds now
    idxloc = value("IDXLOC", gwf_name, MEMORY_PATH)[:nhfb] - 1
    hydchr = value("HYDCHR", gwf_name, MEMORY_PATH)[:nhfb]
    csatsav = value("CSATSAV", gwf_name, MEMORY_PATH)[:nhfb]
    noden = value("NODEN", gwf_name, MEMORY_PATH)[:nhfb] - 1
    nodem = value("NODEM", gwf_name, MEMORY_PATH)[:nhfb] - 1

    jas = value("JAS", gwf_name, "CON") - 1
    condsat = value("CONDSAT", gwf_name, "NPF")
    icelltype = value("ICELLTYPE", gwf_name, "NPF")
    is_newton = int(value("INEWTON", gwf_name)[0]) != 0

    # MODFLOW folds the barrier into the conductance under the Newton-Raphson
    # formulation, and where neither cell converts between confined and
    # unconfined. Everywhere else it applies the barrier once per iteration
    # against the saturated thickness of the moment, and leaves the stored
    # conductance alone, so there is nothing there to recover either
    # derivative from.
    folded = is_newton | ((icelltype[noden] == 0) & (icelltype[nodem] == 0))

    dconddhc = np.zeros(nhfb)
    for ihfb in range(nhfb):
        if not folded[ihfb]:
            continue
        unbarriered = float(csatsav[ihfb])
        if unbarriered <= 0.0:
            continue
        iconn = int(jas[int(idxloc[ihfb])])
        barriered = float(condsat[iconn])
        ratio = barriered / unbarriered
        if hydchr[ihfb] > 0.0:
            factor[iconn] = ratio * ratio
            dconddhc[ihfb] = (
                barriered * (unbarriered - barriered) / (hydchr[ihfb] * unbarriered)
            )
        else:
            factor[iconn] = ratio
            dconddhc[ihfb] = -unbarriered

    return {
        "hfb_dconddhc": dconddhc,
        "hfb_factor": factor,
        "hfb_hydchr": hydchr,
        "hfb_noden": noden,
        "hfb_nodem": nodem,
    }, int(nhfb - np.count_nonzero(folded))


def sensitivity(dconddhc, noden, nodem, head, lamb) -> np.ndarray:
    """Return the sensitivity of a measure to each hydraulic characteristic.

    A barrier sits on one connection, so what it changes is the flow between
    the two cells that connection joins, and the measure follows it through the
    adjoint state of each. The form is the one the conductivity sensitivity
    takes for a connection, with the derivative of the conductance with respect
    to the hydraulic characteristic in place of the one with respect to
    conductivity.

    Parameters
    ----------
    dconddhc : ndarray
        Derivative of the conductance with respect to the hydraulic
        characteristic, one per barrier.
    noden, nodem : ndarray
        Zero-based nodes the barrier lies between.
    head : ndarray
        Simulated head for every node.
    lamb : ndarray
        Adjoint state for every node.

    Returns
    -------
    ndarray
        Sensitivity to the hydraulic characteristic of each barrier.
    """
    return dconddhc * (head[nodem] - head[noden]) * (lamb[noden] - lamb[nodem])
