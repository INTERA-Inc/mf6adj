"""Cross-section rating for a streamflow-routing reach.

A reach given more than one station-height point takes its discharge from the
conveyance of the wetted section rather than from a power of the depth, so the
5/3 rating of a rectangular reach does not hold. The section is a chain of
straight segments between the points, and MODFLOW 6 sums the conveyance of the
wetted part of each::

    K = sum_n a_n ** (5 / 3) * p_n ** (-2 / 3) / r_n
    qman = unitconv * K * sqrt(slope)

where a_n and p_n are the wetted area and wetted perimeter of segment n and
r_n is the reach roughness times the roughness fraction of the segment. Both
a_n and p_n follow the depth, and the derivative of the rating is taken from
them rather than from a perturbation of the discharge.

The wetted perimeter of the section also sets the streambed conductance, so
unlike a rectangular reach, whose conductance is its width at any depth, the
leakage of a reach with a cross section follows the depth through the
conductance as well as through the stage.
"""

import numpy as np


def _wetted_station(x0, x1, d0, d1, d):
    """Return the wetted extent of one segment and the heights bounding it."""
    dmin = min(d0, d1)
    dmax = max(d0, d1)
    if d <= dmin:
        x1 = x0
    elif d < dmax:
        station = x0 + (x1 - x0) * (d - d0) / (d1 - d0)
        if d0 > d1:
            x0 = station
        else:
            x1 = station
    return x0, x1, dmin, dmax


def _vertical_faces(stations, heights):
    """Return whether each segment has a vertical face on its left and right."""
    npts = stations.shape[0]
    left = np.zeros(npts - 1, dtype=bool)
    right = np.zeros(npts - 1, dtype=bool)
    for n in range(npts - 1):
        if n > 0 and stations[n - 1] == stations[n] and heights[n - 1] > heights[n]:
            left[n] = True
        if (
            n < npts - 2
            and stations[n + 2] == stations[n + 1]
            and heights[n + 2] > heights[n + 1]
        ):
            right[n] = True
    return left, right


def _wet_vertical_face(heights, n, d, left):
    """Return the wetted length of a vertical face, and how it follows depth."""
    if left:
        above, below = heights[n - 1], heights[n]
    else:
        above, below = heights[n + 2], heights[n + 1]
    if above > d:
        # the face is wetted only to the depth, so it grows with it
        return d - below, 1.0
    return above - below, 0.0


def _segment_geometry(stations, heights, d):
    """Return the wetted area and perimeter of each segment, and their slopes."""
    npts = stations.shape[0]
    area = np.zeros(npts - 1)
    darea = np.zeros(npts - 1)
    perimeter = np.zeros(npts - 1)
    dperimeter = np.zeros(npts - 1)
    left, right = _vertical_faces(stations, heights)

    for n in range(npts - 1):
        x0, x1, dmin, dmax = _wetted_station(
            stations[n], stations[n + 1], heights[n], heights[n + 1], d
        )
        xlen = x1 - x0
        if xlen > 0.0:
            # the top width of the wetted segment is how its area grows
            if d > dmax:
                area[n] = xlen * (d - dmax)
            if dmax != dmin and d > dmin:
                if d < dmax:
                    area[n] += 0.5 * (d - dmin) * xlen
                else:
                    area[n] += 0.5 * (dmax - dmin) * xlen
            darea[n] = xlen
            dlen = dmax - dmin if d > dmax else d - dmin
        else:
            dlen = min(d, dmax) - dmin if d > dmin else 0.0

        perimeter[n] = np.sqrt(xlen**2 + dlen**2)
        if perimeter[n] > 0.0 and dmin < d < dmax:
            # the wetted length grows with the depth along both directions
            slope = (stations[n + 1] - stations[n]) / (dmax - dmin)
            dperimeter[n] = np.sqrt(slope**2 + 1.0) if xlen > 0.0 else 1.0

        if n > 0 and left[n]:
            face, dface = _wet_vertical_face(heights, n, d, True)
            perimeter[n] += face
            dperimeter[n] += dface
        if n < npts - 2 and right[n]:
            face, dface = _wet_vertical_face(heights, n, d, False)
            perimeter[n] += face
            dperimeter[n] += dface

    return area, darea, perimeter, dperimeter


def wetted_perimeter(stations, heights, d):
    """Return the wetted perimeter of a cross section and its slope in the depth.

    The streambed conductance of a reach follows its wetted perimeter, so for a
    reach with a cross section the conductance follows the depth as well.
    """
    _, _, perimeter, dperimeter = _segment_geometry(stations, heights, d)
    return float(perimeter.sum()), float(dperimeter.sum())


def mannings_section(stations, heights, roughfracs, roughness, unitconv, slope, d):
    """Return the discharge of a cross section and its derivative in the depth.

    Parameters
    ----------
    stations : numpy.ndarray
        Station of each cross-section point, already scaled by the reach width.
    heights : numpy.ndarray
        Height of each cross-section point above the streambed.
    roughfracs : numpy.ndarray
        Roughness of each point as a fraction of the reach roughness.
    roughness : float
        Reach roughness.
    unitconv : float
        Unit conversion of the package.
    slope : float
        Reach slope.
    d : float
        Depth over the lowest point of the section.

    Returns
    -------
    tuple
        The discharge and its derivative with respect to the depth.
    """
    area, darea, perimeter, dperimeter = _segment_geometry(stations, heights, d)
    if perimeter.sum() <= 0.0:
        return 0.0, 0.0

    conveyance = 0.0
    dconveyance = 0.0
    for n in range(perimeter.shape[0]):
        rough = roughness * roughfracs[n]
        if perimeter[n] * rough <= 0.0:
            continue
        a, p = area[n], perimeter[n]
        conveyance += a ** (5.0 / 3.0) * p ** (-2.0 / 3.0) / rough
        if a > 0.0:
            dconveyance += (
                (5.0 / 3.0) * a ** (2.0 / 3.0) * darea[n] * p ** (-2.0 / 3.0)
                - (2.0 / 3.0) * a ** (5.0 / 3.0) * p ** (-5.0 / 3.0) * dperimeter[n]
            ) / rough

    factor = unitconv * np.sqrt(slope)
    return factor * conveyance, factor * dconveyance
