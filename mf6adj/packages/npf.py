"""Node-property flow (NPF) terms for the adjoint solution.

The conductance between two cells follows the hydraulic conductivity of both,
and, where a cell is convertible, the saturated thickness the head leaves it.
These functions form the derivative of the flow residual with respect to the
horizontal and vertical conductivity, weighted by the adjoint state.
"""

import numpy as np


def cell_sat(top, bot, h) -> float:
    """Return cell saturation from cell top, bottom, and head.

    Parameters
    ----------
    top : float
        Cell top elevation.
    bot : float
        Cell bottom elevation.
    h : float
        Cell head.

    Returns
    -------
    float
        Cell saturation clipped to the interval ``[0, 1]``.
    """
    if h > top:
        sat = 1.0
    elif h < bot:
        sat = 0.0
    else:
        sat = (h - bot) / (top - bot)
    return sat


def smooth_sat_simple(sat) -> float:
    """Smooth saturation using the MODFLOW 6 sigmoid-style function.

    Parameters
    ----------
    sat : float
        Saturation value.

    Returns
    -------
    float
        Smoothed saturation bounded between 0 and 1.
    """
    satomega = 1.0e-6
    A_omega = 1 / (1 - satomega)
    s_sat = 1.0
    if sat < 0:
        s_sat = 0
    elif sat >= 0 and sat < satomega:
        s_sat = (A_omega / (2 * satomega)) * sat**2
    elif sat >= satomega and sat < 1 - satomega:
        s_sat = A_omega * sat + 0.5 * (1 - A_omega)
    elif sat >= 1 - satomega and sat < 1:
        s_sat = 1 - (A_omega / (2 * satomega)) * ((1 - sat) ** 2)
    return s_sat


def smooth_sat(ihighcellsat, top1, top2, bot1, bot2, h1, h2) -> float:
    """Return smoothed saturation for the upstream cell.

    Parameters
    ----------
    ihighcellsat : int
        Whether to use the highest cell bottom when calculating saturation.
    top1 : float
        Top elevation of node 1.
    top2 : float
        Top elevation of node 2.
    bot1 : float
        Bottom elevation of node 1.
    bot2 : float
        Bottom elevation of node 2.
    h1 : float
        Head of node 1.
    h2 : float
        Head of node 2.

    Returns
    -------
    float
        Smoothed saturation of the upstream node selected from the
        connection pair.
    """
    bot = None
    if ihighcellsat != 0:
        if (abs(bot1) - abs(bot2)) > 1e-2:
            bot = max(bot1, bot2)
    if h1 > h2:
        if bot is None:
            sat = cell_sat(top1, bot1, h1)
        else:
            sat = cell_sat(top1, bot, h1)
    else:
        if bot is None:
            sat = cell_sat(top2, bot2, h2)
        else:
            sat = cell_sat(top2, bot, h2)
    return smooth_sat_simple(sat)


def dconddhk(k1, k2, cl1, cl2, width, height1, height2) -> float:
    """Return the derivative of intercell conductance with respect to ``K``.

    The expression follows the MODFLOW-style conductance formulation for a
    two-cell connection and is used when building hydraulic-conductivity
    sensitivities.

    Parameters
    ----------
    k1 : float
        Hydraulic conductivity for connection 1.
    k2 : float
        Hydraulic conductivity for connection 2.
    cl1 : float
        Length of connection 1.
    cl2 : float
        Length of connection 2.
    width : float
        Connection width.
    height1 : float
        Saturated height of connection 1.
    height2 : float
        Saturated height of connection 2.

    Returns
    -------
    float
        Derivative of connection conductance with respect to hydraulic
        conductivity.
    """

    # todo: upstream weighting - could use height1 and height2 to check...
    # todo: vertically staggered
    d = (width * cl1 * height1 * (height2**2) * (k2**2)) / (
        ((cl2 * height1 * k1) + (cl1 * height2 * k2)) ** 2
    )
    return d


def lam_dresdk_h(
    is_newton,
    ihighcellsat,
    lamb,
    sat,
    head,
    ihc,
    ia,
    ja,
    jas,
    cl1,
    cl2,
    hwva,
    top,
    bot,
    icelltype,
    k11,
    k33,
    hfb_factor=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return adjoint-weighted residual derivatives.

    Derivatives are with respect to ``k11`` and ``k33``.

    Parameters
    ----------
    is_newton : bool
        Whether Newton terms are active.
    ihighcellsat : int
        Whether to use the highest cell bottom when calculating saturation.
    lamb : ndarray
        Adjoint state array.
    sat : ndarray
        Saturation array.
    head : ndarray
        Head array.
    ihc : ndarray
        Horizontal connection indicator array.
    ia : ndarray
        Connection index array in compressed sparse row format.
    ja : ndarray
        Connection array in compressed sparse row format.
    jas : ndarray
        Full connectivity array.
    cl1 : ndarray
        Connection length array for connection 1.
    cl2 : ndarray
        Connection length array for connection 2.
    hwva : ndarray
        Horizontal-width/vertical-area array.
    top : ndarray
        Top elevations.
    bot : ndarray
        Bottom elevations.
    icelltype : ndarray
        Convertible-cell type indicator array.
    k11 : ndarray
        Horizontal hydraulic conductivity array.
    k33 : ndarray
        Vertical hydraulic conductivity array.
    hfb_factor : ndarray, optional
        What a horizontal flow barrier does to the derivative of conductance,
        one entry per connection. Omitted where the model holds no barrier,
        which is the same as an array of ones.

    Returns
    -------
    tuple[ndarray, ndarray]
        Adjoint-weighted residual derivatives with respect to ``k11`` and
        ``k33``.
    """
    iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
    # array of number of connections per node (size nodes)

    # A horizontal flow barrier changes the conductance of the connections it
    # sits on, so it scales the derivative of that conductance too. A model
    # with no barrier scales by one, which is formed once here rather than
    # tested for on every connection below.
    if hfb_factor is None:
        hfb_factor = np.ones(hwva.shape[0])

    sat_mod = sat.copy()
    sat_mod[icelltype == 0] = 1.0

    height = top - bot

    result33 = np.zeros_like(head)
    result = np.zeros_like(head)

    for node, (offset, ncon) in enumerate(zip(ia, iac)):
        sum1 = 0.0
        sum2 = 0.0
        height1 = height[node]

        for ii in range(offset + 1, offset + ncon):
            mnode = ja[ii]
            height2 = height[mnode]

            jj = jas[ii]
            if jj < 0:
                raise Exception()
            iihc = ihc[jj]

            if iihc == 0:  # vertical con
                dconddk33 = dconddhk(
                    k33[node],
                    k33[mnode],
                    0.5 * height1,
                    0.5 * height2,
                    hwva[jj],
                    1.0,
                    1.0,
                )
                t2 = (
                    dconddk33
                    * hfb_factor[jj]
                    * (head[mnode] - head[node])
                    * (lamb[node] - lamb[mnode])
                )
                sum1 += t2

            else:
                # TODO: check if one cell is convertible (??is this required??)
                if is_newton:
                    dconddk = dconddhk(
                        k11[node],
                        k11[mnode],
                        cl1[jj],
                        cl2[jj],
                        hwva[jj],
                        height1,
                        height2,
                    )
                    SF = smooth_sat(
                        ihighcellsat,
                        top[node],
                        top[mnode],
                        bot[node],
                        bot[mnode],
                        head[node],
                        head[mnode],
                    )

                else:
                    dconddk = dconddhk(
                        k11[node],
                        k11[mnode],
                        cl1[jj],
                        cl2[jj],
                        hwva[jj],
                        height1 * sat_mod[node],
                        height2 * sat_mod[mnode],
                    )
                    SF = 1.0

                t1 = (
                    SF
                    * dconddk
                    * hfb_factor[jj]
                    * (head[mnode] - head[node])
                    * (lamb[node] - lamb[mnode])
                )
                sum2 += t1

        result33[node] = sum1
        result[node] = sum2
    return result, result33
