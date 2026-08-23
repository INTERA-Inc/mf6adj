"""Streamflow routing (SFR) coupling for the adjoint solution.

A reach stage follows the flow through the reach, so it is a dependent variable
just as a lake stage is, and holding it fixed returns a partial derivative. The
adjoint system is bordered with one equation per reach, and those equations
carry the routing between reaches.

MODFLOW 6 sets a reach depth so that the Manning discharge equals the mean of
the flow entering and leaving the reach::

    qman(d) = (Q_in + Q_out) / 2,     Q_out = Q_in - leak

For a reach without a cross section the wetted perimeter is the reach width
with no depth dependence, so the discharge is a power of the depth::

    qman(d) = unitconv * width * d ** (5 / 3) * sqrt(slope) / roughness

and its derivative is that exponent times the discharge over the depth. Taking
the discharge MODFLOW reports rather than rebuilding the rating keeps this
correct whatever the unit conversion is.
"""

import numpy as np
import scipy.sparse as sparse

# The Manning discharge of a reach without a cross section is this power of the
# depth over the streambed, so its derivative is the exponent times the
# discharge over the depth.
MANNING_EXPONENT = 5.0 / 3.0

# MODFLOW 6 smooths the discharge to zero below this depth, and that smoothing
# is not reproduced here, so a shallower reach is treated as carrying no flow.
MIN_DEPTH = 1.0e-5


def _rating_discharge(width, depth, slope, rough, unitconv):
    """Return the Manning discharge of a reach without a cross section."""
    return unitconv * width * depth**MANNING_EXPONENT * np.sqrt(slope) / rough


def forward_terms(gwf, gwf_name: str, tag: str) -> dict:
    """Return the routing terms of a streamflow-routing package for one time step.

    Parameters
    ----------
    gwf : modflowapi.ModflowApi
        MODFLOW 6 groundwater-flow instance.
    gwf_name : str
        Name of the groundwater-flow model.
    tag : str
        Streamflow-routing package name from the GWF name file.

    Returns
    -------
    dict
        Depth, discharge, the derivative of discharge with respect to depth,
        and the reach network. ``reach_is_free`` marks a reach whose depth is
        solved, as opposed to one that is dry or holds a specified stage.
    """

    def value(name):
        return gwf.get_value(gwf.get_var_address(name, gwf_name, tag.upper())).copy()

    def optional(name, default, size):
        """Return a package quantity, or a default where MODFLOW omits it."""
        try:
            return value(name)
        except Exception:
            return np.full(size, default)

    depth = value("DEPTH")
    nreach = depth.shape[0]
    hcof = value("HCOF")
    dsflow = value("DSFLOW")
    width = value("WIDTH")
    slope = value("SLOPE")
    rough = value("ROUGH")
    unitconv = float(value("UNITCONV")[0])

    # a reach with a cross section follows a rating this does not reproduce
    ncrosspts = optional("NCROSSPTS", 1, nreach)
    has_section = (ncrosspts > 1).astype(int)

    discharge = np.zeros(nreach)
    ddischarge = np.zeros(nreach)
    for n in range(nreach):
        if depth[n] <= MIN_DEPTH or has_section[n]:
            continue
        discharge[n] = _rating_discharge(
            width[n], depth[n], slope[n], rough[n], unitconv
        )
        ddischarge[n] = MANNING_EXPONENT * discharge[n] / depth[n]

    # A reach that gives up every drop it carries leaks its own inflow rather
    # than a head-dependent amount, which MODFLOW marks by leaving hcof at zero
    # while the reach still holds water. That leakage follows the reaches above
    # it rather than its own stage, and that coupling is not formed here.
    flow_limited = ((hcof == 0.0) & (depth > MIN_DEPTH) & (dsflow <= 0.0)).astype(int)

    # a reach that carries no water, whose stage follows a cross section, or
    # whose leakage is limited by the flow it carries, has no depth for the
    # adjoint to solve for
    is_free = ((depth > MIN_DEPTH) & (has_section == 0) & (flow_limited == 0)).astype(
        int
    )

    # the reach network: idir is positive for a connection to an upstream reach
    ia = value("IA") - 1
    ja = value("JA") - 1
    idir = value("IDIR")
    ustrf = value("USTRF")

    # the share of an upstream reach's outflow this reach receives
    total = np.zeros(nreach)
    for n in range(nreach):
        for icon in range(ia[n], ia[n + 1]):
            if idir[icon] > 0:
                # ja[icon] is upstream of n, so n is one of its outflows
                total[ja[icon]] += ustrf[n]
    fraction = np.zeros(ja.shape[0])
    for n in range(nreach):
        for icon in range(ia[n], ia[n + 1]):
            if idir[icon] > 0:
                share = total[ja[icon]]
                fraction[icon] = ustrf[n] / share if share > 0.0 else 0.0

    ndiv = optional("NDIV", 0, nreach)

    return {
        "reach_ddischarge": ddischarge,
        "reach_depth": depth,
        "reach_discharge": discharge,
        "reach_fraction": fraction,
        "reach_has_section": has_section,
        "reach_flow_limited": flow_limited,
        "reach_ia": ia,
        "reach_idir": idir,
        "reach_is_free": is_free,
        "reach_ja": ja,
        "reach_ndiv": ndiv,
    }


class SfrCoupling:
    """The reach rows of the bordered adjoint system.

    One instance follows a performance measure through its backward sweep. A
    reach carries no storage, so unlike a lake nothing is carried between time
    steps; the routing is solved within each step.
    """

    def __init__(self, logger=None):
        """Initialize the coupling.

        Parameters
        ----------
        logger : object, optional
            Logger used to report where a reach is only partly differentiated.
        """
        self.logger = logger
        self.columns = {}
        self._warned = set()

    def reset_step(self) -> None:
        """Forget the previous time step's reach state."""
        self.columns = {}

    def _warn(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warning(message)

    def _warn_once(self, pname: str, grp) -> None:
        """Warn where a reach's routing is differentiated only in part."""
        if pname in self._warned:
            return
        self._warned.add(pname)

        if grp["reach_has_section"][:].max() > 0:
            self._warn(
                f"streamflow-routing package '{pname}' has reaches with a cross "
                "section. Their discharge follows the section rather than a "
                "power of the depth, and that derivative is not formed, so "
                "those reaches keep a fixed stage and the sensitivity is "
                "approximate."
            )
        if grp["reach_flow_limited"][:].max() > 0:
            self._warn(
                f"streamflow-routing package '{pname}' has reaches that give up "
                "all of the water they carry. Their leakage follows the reaches "
                "above them rather than their own stage, and that coupling is "
                "not formed, so the sensitivity is approximate."
            )
        if grp["reach_ndiv"][:].max() > 0:
            self._warn(
                f"streamflow-routing package '{pname}' has diversions. The flow "
                "they take is not differentiated, so the sensitivity is "
                "approximate."
            )

    def blocks(self, sol_dataset, gwf_package_dict) -> list:
        """Return the reaches and the terms coupling them to the aquifer.

        Returns
        -------
        list
            One entry per streamflow-routing package. A reach that is dry, or
            that follows a cross section, is left out.
        """
        found = []
        for ptype, pnames in gwf_package_dict.items():
            if ptype != "sfr6":
                continue
            for pname in pnames:
                if pname not in sol_dataset:
                    continue
                grp = sol_dataset[pname]
                if "reach_is_free" not in grp:
                    continue
                free = grp["reach_is_free"][:] == 1
                if not free.any():
                    continue

                self._warn_once(pname, grp)

                found.append(
                    {
                        "name": pname,
                        "node": grp["nodelist"][:] - 1,
                        "cond": grp["bound"][:, 1],
                        "hcof": grp["hcof"][:],
                        "ddischarge": grp["reach_ddischarge"][:],
                        "flow_limited": grp["reach_flow_limited"][:] == 1,
                        "fraction": grp["reach_fraction"][:],
                        "free": free,
                        "ia": grp["reach_ia"][:],
                        "idir": grp["reach_idir"][:],
                        "ja": grp["reach_ja"][:],
                    }
                )
        return found

    def dfds(self, entries, kk, blocks) -> dict:
        """Return the derivative of the measure with respect to each reach depth.

        A flux measure on a reach sums the exchange at the cells it names, and
        that exchange depends on the reach stage as well as on the head.
        """
        result = {}
        for block in blocks:
            weights = {
                pfr.inode: pfr.weight
                for pfr in entries
                if pfr.kperkstp == kk and pfr.pm_type == block["name"]
            }
            if not weights:
                continue
            for ireach in range(block["node"].shape[0]):
                node = int(block["node"][ireach])
                if node not in weights:
                    continue
                if block["flow_limited"][ireach]:
                    # this reach leaks the whole flow it receives, so the
                    # measure follows the reaches above it
                    for iup, share in self._upstream(block, ireach):
                        ddepth, _ = self._outflow_derivatives(block, iup)
                        key = (block["name"], iup)
                        result[key] = (
                            result.get(key, 0.0) + weights[node] * share * ddepth
                        )
                    continue
                key = (block["name"], ireach)
                result[key] = result.get(key, 0.0) + weights[node] * float(
                    block["cond"][ireach]
                )
        return result

    def _upstream(self, block, ireach):
        """Yield the free reaches feeding a reach, with the share each sends."""
        ia, ja, idir = block["ia"], block["ja"], block["idir"]
        for icon in range(ia[ireach], ia[ireach + 1]):
            if idir[icon] <= 0:
                continue
            iup = int(ja[icon])
            if not block["free"][iup]:
                # a reach that already gave up its whole flow sends nothing on
                continue
            yield iup, float(block["fraction"][icon])

    def _outflow_derivatives(self, block, iup):
        """Return how an upstream reach's outflow follows its depth and head.

        The flow leaving a reach is its Manning discharge less half of its
        leakage, so it follows the reach depth and the head beneath it.
        """
        ddepth = float(block["ddischarge"][iup]) - 0.5 * float(block["cond"][iup])
        dhead = 0.5 * float(block["cond"][iup])
        return ddepth, dhead

    def measure_dfdh(self, entries, kk, blocks, nnodes):
        """Return what a measure on a flow-limited reach adds to df/dh.

        Such a reach leaks the whole flow it receives, so the measure follows
        the head beneath the reaches above it rather than the head beneath the
        reach itself.
        """
        extra = np.zeros(nnodes)
        for block in blocks:
            weights = {
                pfr.inode: pfr.weight
                for pfr in entries
                if pfr.kperkstp == kk and pfr.pm_type == block["name"]
            }
            if not weights:
                continue
            for ireach in range(block["node"].shape[0]):
                if not block["flow_limited"][ireach]:
                    continue
                node = int(block["node"][ireach])
                if node not in weights:
                    continue
                for iup, share in self._upstream(block, ireach):
                    _, dhead = self._outflow_derivatives(block, iup)
                    extra[int(block["node"][iup])] += weights[node] * share * dhead
        return extra

    def augment(self, amat, rhs, blocks, dfds, nnodes):
        """Border the adjoint system with the reach routing equations.

        Returns
        -------
        tuple
            The bordered matrix and the bordered right-hand side. The column
            each free reach occupies is kept on the instance.
        """
        self.columns = {}
        ireach = 0
        for block in blocks:
            for n in range(block["free"].shape[0]):
                if block["free"][n]:
                    self.columns[(block["name"], n)] = ireach
                    ireach += 1
        nreach = ireach

        drds = sparse.lil_matrix((int(nnodes), int(nreach)))
        drdh = sparse.lil_matrix((int(nreach), int(nnodes)))
        drdr = sparse.lil_matrix((int(nreach), int(nreach)))
        # a reach that leaks the whole flow it receives ties one aquifer cell to
        # another, so it corrects the flow matrix rather than the border
        dgwf = sparse.lil_matrix((int(nnodes), int(nnodes)))

        for block in blocks:
            ia, ja, idir = block["ia"], block["ja"], block["idir"]

            # a reach that gives up all of the water it carries leaks its own
            # inflow, so the aquifer beneath it follows the reaches above
            for n in range(block["free"].shape[0]):
                if not block["flow_limited"][n]:
                    continue
                node = int(block["node"][n])
                for iup, share in self._upstream(block, n):
                    ddepth, dhead = self._outflow_derivatives(block, iup)
                    upstream = (block["name"], iup)
                    if upstream in self.columns:
                        drds[node, self.columns[upstream]] += share * ddepth
                    dgwf[node, int(block["node"][iup])] += share * dhead

            for n in range(block["free"].shape[0]):
                key = (block["name"], n)
                if key not in self.columns:
                    continue
                icol = self.columns[key]
                node = int(block["node"][n])
                cond = float(block["cond"][n])

                # the aquifer sees the reach through its stage
                drds[node, icol] += cond

                # the reach equation: the Manning discharge against the mean of
                # the flow in and out, so half of the leakage appears here
                drdr[icol, icol] += float(block["ddischarge"][n]) + 0.5 * cond
                drdh[icol, node] += 0.5 * float(block["hcof"][n])

                # what this reach receives from the reaches above it
                for icon in range(ia[n], ia[n + 1]):
                    if idir[icon] <= 0:
                        continue
                    upstream = (block["name"], int(ja[icon]))
                    if upstream not in self.columns:
                        continue
                    share = float(block["fraction"][icon])
                    jcol = self.columns[upstream]
                    iup = int(ja[icon])
                    drdr[icol, jcol] -= share * float(block["ddischarge"][iup])
                    drdr[icol, jcol] += 0.5 * share * float(block["cond"][iup])
                    # the flow leaving the upstream reach carries half of its
                    # leakage, so the head there enters with the same sign as
                    # this reach's own
                    upnode = int(block["node"][iup])
                    drdh[icol, upnode] += 0.5 * share * float(block["hcof"][iup])

        # amat is already transposed, so the reach blocks are transposed too
        bordered = sparse.bmat(
            [
                [amat + dgwf.transpose().tocsr(), drdh.transpose()],
                [drds.transpose(), drdr.transpose()],
            ],
            format="csr",
        )

        rhs_reach = np.zeros(nreach)
        for key, icol in self.columns.items():
            rhs_reach[icol] = -dfds.get(key, 0.0)
        return bordered, np.concatenate((rhs, rhs_reach))

    def split(self, lamb, nnodes):
        """Take the reach depths off the solution.

        A reach carries no storage, so nothing is carried to the previous step.
        """
        return lamb[:nnodes]
