"""Horizontal flow barrier (HFB) terms for the adjoint solution.

A barrier adds no equations and exchanges no water. What it does is change the
conductance of the connections it sits on, and MODFLOW writes that changed
conductance into the array the flow equations are assembled from rather than
keeping it apart. A sensitivity to hydraulic conductivity is the derivative of
that conductance, so it has to carry the barrier or it answers for a connection
the model does not have.

A barrier in series always lowers the conductance, because two conductances in
series carry less than either. Given as a multiplier it does whatever the
multiplier says, so it can raise the conductance as well, and a multiplier of
zero closes the connection.

MODFLOW keeps the conductance each connection had before the barrier was
applied, which is what makes the derivative recoverable without forming the
barrier again here.
"""

import numpy as np

# MODFLOW names the package HFB in memory whatever the name file calls it, and
# a model holds one
MEMORY_PATH = "HFB"


def conductance_factor(gwf, gwf_name: str, nconn: int) -> tuple[np.ndarray, int]:
    """Return what a barrier does to the derivative of conductance with respect to K.

    Writing `a` for the conductance a connection has without the barrier and
    `c` for the conductance of the barrier itself, MODFLOW puts the two in
    series, `cond = a c / (a + c)`, so

        d(cond)/da = (c / (a + c))**2 = (cond / a)**2,

    which is one where there is no barrier and, since two conductances in
    series carry less than either, below one wherever a barrier sits.

    A hydraulic characteristic of zero or less means something else: MODFLOW
    reads it as a multiplier on the conductance, `cond = -a hydchr`, whose
    derivative is `cond / a` without the square. Nothing bounds a multiplier by
    one, so this form can raise the conductance and the derivative with it, and
    a multiplier of zero closes the connection and leaves it no derivative at
    all.

    Both forms are the ratio of what MODFLOW ended up with to what it started
    with, so neither the flow area nor the screen geometry is formed again
    here, and a change to how MODFLOW computes them does not reach this.

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
    tuple[ndarray, int]
        The factor for every connection, one where no barrier sits, and the
        number of barriers whose effect MODFLOW is not carrying in the
        conductance and which are therefore not in the factor. The factor is
        below one for a barrier in series and takes the multiplier itself for
        the other form, which is not bounded by one.
    """

    def value(name, *components):
        return gwf.get_value(gwf.get_var_address(name, *components)).copy()

    factor = np.ones(int(nconn))
    nhfb = int(value("NHFB", gwf_name, MEMORY_PATH)[0])
    if nhfb == 0:
        return factor, 0

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
    # conductance alone, so there is nothing there to recover the factor from.
    folded = is_newton | ((icelltype[noden] == 0) & (icelltype[nodem] == 0))

    for ihfb in range(nhfb):
        if not folded[ihfb]:
            continue
        unbarriered = float(csatsav[ihfb])
        if unbarriered <= 0.0:
            continue
        iconn = int(jas[int(idxloc[ihfb])])
        ratio = float(condsat[iconn]) / unbarriered
        factor[iconn] = ratio * ratio if hydchr[ihfb] > 0.0 else ratio

    return factor, int(nhfb - np.count_nonzero(folded))
