"""Lake (LAK) coupling for the adjoint solution.

A lake stage is a dependent variable, so the adjoint system is bordered with
the lake water balance rather than holding the stage fixed. The lake specifics
live here so the adjoint solver stays free of package detail, and so the other
advanced packages can follow the same shape.
"""

import numpy as np
import scipy.sparse as sparse

# Depth over an outlet invert below this is treated as no flow, because
# MODFLOW 6 smooths the rating curve to zero there and the derivative of that
# smoothing is not reproduced here.
OUTLET_MIN_DEPTH = 1.0e-6

# An outlet discharge follows a power of the depth over its invert, so the
# derivative is that exponent times the discharge divided by the depth.
# 0 is a specified rate, 1 is Manning, 2 is a weir.
OUTLET_EXPONENT = {0: 0.0, 1: 5.0 / 3.0, 2: 1.5}


def table_slope(stage, stages, values) -> float:
    """Return the slope of a lake table at a stage.

    MODFLOW 6 interpolates between table rows, so the slope is that of the row
    interval holding the stage; outside the table the nearest interval is used.

    Parameters
    ----------
    stage : float
        Lake stage.
    stages : ndarray
        Table stages, increasing.
    values : ndarray
        Table values at those stages.

    Returns
    -------
    float
        Slope of ``values`` with respect to stage.
    """
    nrow = stages.shape[0]
    irow = 0
    for i in range(nrow - 1):
        if stage >= stages[i]:
            irow = i
    span = stages[irow + 1] - stages[irow]
    if span <= 0.0:
        return 0.0
    return float((values[irow + 1] - values[irow]) / span)


def forward_terms(gwf, gwf_name: str, tag: str) -> dict:
    """Return the water-balance terms of a lake package for one time step.

    Parameters
    ----------
    gwf : modflowapi.ModflowApi
        MODFLOW 6 groundwater-flow instance.
    gwf_name : str
        Name of the groundwater-flow model.
    tag : str
        Lake package name from the GWF name file.

    Returns
    -------
    dict
        Stage, storage and evaporating surface against stage, outlets, and the
        lake each connection belongs to. ``lake_is_free`` marks a lake whose
        stage is solved, as opposed to one held constant or inactive.
    """

    def value(name):
        return gwf.get_value(gwf.get_var_address(name, gwf_name, tag.upper())).copy()

    def optional(name, default):
        """Return a package quantity, or a default where MODFLOW omits it.

        A package exposes only what its options bring in, so a lake with no
        mover has no receiver count and one with no rainfall has no rate.
        """
        try:
            return value(name)
        except Exception:
            return np.full(nlakes, default)

    # IDXLAKECONN is a one-based pointer into the connection arrays
    idx = value("IDXLAKECONN") - 1
    nlakes = int(value("NLAKES")[0])
    sarea = value("SAREA")

    lake_of_conn = np.zeros(idx[-1], dtype=int)
    surface_area = np.zeros(nlakes)
    for ilak in range(nlakes):
        lake_of_conn[idx[ilak] : idx[ilak + 1]] = ilak
        surface_area[ilak] = sarea[idx[ilak] : idx[ilak + 1]].sum()

    # ibound is negative for a lake held at a constant stage and zero for an
    # inactive one; only a positive ibound has a water balance to solve
    is_free = (value("IBOUND") > 0).astype(int)

    # With a stage-volume-area table the lake's storage and its evaporating
    # surface both follow the table rather than the connection areas, so take
    # their slopes at the current stage.
    stage = value("XNEWPAK")
    stage_old = value("XOLDPAK")
    dvds = surface_area.copy()
    # the storage carried back to the previous step is the slope at that step's
    # stage, which differs once a step crosses a table interval
    dvds_old = surface_area.copy()
    dsads = np.zeros(nlakes)
    if int(value("NTABLES")[0]) > 0:
        itab = value("IALAKTAB") - 1
        tabstage = value("TABSTAGE")
        tabvolume = value("TABVOLUME")
        tabsarea = value("TABSAREA")
        for ilak in range(nlakes):
            rows = slice(itab[ilak], itab[ilak + 1])
            # a package can table some lakes and not others
            if tabstage[rows].shape[0] < 2:
                continue
            dvds[ilak] = table_slope(stage[ilak], tabstage[rows], tabvolume[rows])
            dvds_old[ilak] = table_slope(
                stage_old[ilak], tabstage[rows], tabvolume[rows]
            )
            dsads[ilak] = table_slope(stage[ilak], tabstage[rows], tabsarea[rows])

    # a horizontal connection's conductance and wetted area follow the stage
    # through the saturated fraction of the cell; that derivative is not formed
    nhorizontal = np.zeros(nlakes, dtype=int)
    ictype = value("ICTYPE")
    for iconn in range(lake_of_conn.shape[0]):
        if ictype[iconn] == 1:
            nhorizontal[lake_of_conn[iconn]] += 1

    nreceivers = int(optional("NRECEIVERS", 0)[0])
    noutlets = int(value("NOUTLETS")[0])
    if noutlets > 0:
        # LAKEIN and LAKEOUT are one-based; LAKEOUT of zero or less leaves the
        # model rather than feeding another lake
        outlet = {
            "outlet_lakein": value("LAKEIN") - 1,
            "outlet_lakeout": value("LAKEOUT") - 1,
            "outlet_type": value("IOUTTYPE"),
            "outlet_invert": value("OUTINVERT"),
            "outlet_dmax": value("OUTDMAX"),
            "outlet_rate": value("SIMOUTRATE"),
        }
    else:
        outlet = {}

    return {
        **outlet,
        "lake_dsads": dsads,
        "lake_dvds": dvds,
        "lake_dvds_old": dvds_old,
        "lake_evaporation": optional("EVAPORATION", 0.0),
        "lake_is_free": is_free,
        "lake_nhorizontal": nhorizontal,
        "lake_noutlets": np.full(nlakes, noutlets, dtype=int),
        "lake_nreceivers": np.full(nlakes, nreceivers, dtype=int),
        "lake_of_conn": lake_of_conn,
        "lake_rainfall": optional("RAINFALL", 0.0),
        "lake_stage": stage,
        "lake_stage_old": stage_old,
        "lake_surface_area": surface_area,
    }


class LakeCoupling:
    """The lake rows of the bordered adjoint system.

    One instance follows a performance measure through its backward sweep and
    holds the state that spans time steps: the column each free lake occupies
    and the storage each carries back to the previous step.
    """

    def __init__(self, logger=None):
        """Initialize the coupling.

        Parameters
        ----------
        logger : object, optional
            Logger used to report where a lake is only partly differentiated.
        """
        self.logger = logger
        self.carry = {}
        self.columns = {}
        self._warned = set()

    def reset_step(self) -> None:
        """Forget the previous time step's lake state."""
        self.columns = {}
        self.carry = {}

    def _warn(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warning(message)

    def _warn_once(self, pname: str, grp, free) -> None:
        """Warn where a free lake's water balance is differentiated only in part."""
        if pname in self._warned:
            return
        self._warned.add(pname)

        dvds = grp["lake_dvds"][:]
        for ilak in range(free.shape[0]):
            if free[ilak] and dvds[ilak] <= 0.0:
                self._warn(
                    f"lake {ilak + 1} of '{pname}' has a free stage and no "
                    "storage against stage, so its water balance cannot "
                    "constrain the adjoint. A lake with no vertical connection "
                    "and no table has no surface area to store against."
                )

        if "lake_nreceivers" in grp and grp["lake_nreceivers"][:].max() > 0:
            self._warn(
                f"lake package '{pname}' receives water through the mover. "
                "That inflow follows the state of the package providing it, "
                "and the derivative is not formed, so the sensitivity is "
                "approximate."
            )

        if "lake_nhorizontal" in grp and grp["lake_nhorizontal"][:].max() > 0:
            self._warn(
                f"lake package '{pname}' has horizontal connections. Their "
                "conductance and wetted area follow the stage through the "
                "saturated fraction of the cell, and that derivative is not "
                "formed, so the sensitivity is approximate."
            )

    def blocks(self, sol_dataset, gwf_package_dict) -> list:
        """Return the free-stage lakes and the terms coupling them to the aquifer.

        Parameters
        ----------
        sol_dataset : h5py.Group
            Forward-solution group for one time step.
        gwf_package_dict : dict
            Mapping of package types to package names.

        Returns
        -------
        list
            One entry per lake package. A lake held at a constant stage is left
            out, since its stage is not a dependent variable.
        """
        found = []
        for ptype, pnames in gwf_package_dict.items():
            if ptype != "lak6":
                continue
            for pname in pnames:
                if pname not in sol_dataset:
                    continue
                grp = sol_dataset[pname]
                if "lake_of_conn" not in grp:
                    continue
                free = grp["lake_is_free"][:] == 1
                if not free.any():
                    continue

                self._warn_once(pname, grp, free)

                outlets = {}
                if "outlet_lakein" in grp:
                    outlets = {
                        "lakein": grp["outlet_lakein"][:],
                        "lakeout": grp["outlet_lakeout"][:],
                        "type": grp["outlet_type"][:],
                        "invert": grp["outlet_invert"][:],
                        "dmax": grp["outlet_dmax"][:],
                        "rate": grp["outlet_rate"][:],
                    }

                found.append(
                    {
                        "name": pname,
                        "node": grp["nodelist"][:] - 1,
                        "cond": grp["bound"][:, 1],
                        "lake_of_conn": grp["lake_of_conn"][:],
                        "area": grp["lake_surface_area"][:],
                        "dvds": grp["lake_dvds"][:],
                        "dvds_old": grp["lake_dvds_old"][:],
                        "rainfall": grp["lake_rainfall"][:],
                        "dsads": grp["lake_dsads"][:],
                        "evaporation": grp["lake_evaporation"][:],
                        "stage": grp["lake_stage"][:],
                        "free": free,
                        "outlets": outlets,
                    }
                )
        return found

    def dfds(self, entries, kk, blocks) -> dict:
        """Return the derivative of the measure with respect to each lake stage.

        A flux measure on a lake sums the exchange at the cells it names, and
        that exchange depends on the lake stage as well as on the head.

        Returns
        -------
        dict
            ``(package name, lake)`` mapped to the derivative.
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
            for iconn in range(block["node"].shape[0]):
                node = int(block["node"][iconn])
                if node not in weights:
                    continue
                key = (block["name"], int(block["lake_of_conn"][iconn]))
                result[key] = result.get(key, 0.0) + weights[node] * float(
                    block["cond"][iconn]
                )
        return result

    def outlet_derivative(self, block, ioutlet) -> float:
        """Return the derivative of an outlet discharge with respect to stage."""
        outlets = block["outlets"]
        ilak = int(outlets["lakein"][ioutlet])
        exponent = OUTLET_EXPONENT.get(int(outlets["type"][ioutlet]), 0.0)
        if exponent == 0.0:
            return 0.0

        depth = float(block["stage"][ilak]) - float(outlets["invert"][ioutlet])
        dmax = float(outlets["dmax"][ioutlet])
        if depth < OUTLET_MIN_DEPTH:
            return 0.0
        if dmax > 0.0 and depth > dmax:
            # the rating is evaluated at the capped depth, so it stops
            # responding to the stage
            return 0.0

        # the rate is negative leaving the lake
        discharge = -float(outlets["rate"][ioutlet])
        return exponent * discharge / depth

    def augment(self, amat, rhs, blocks, dt, dfds, nnodes, transient=True):
        """Border the adjoint system with the lake water-balance equations.

        Returns
        -------
        tuple
            The bordered matrix and the bordered right-hand side. The column
            each free lake occupies is kept on the instance.
        """
        self.columns = {}
        ilake = 0
        for block in blocks:
            for ilak in range(block["free"].shape[0]):
                if block["free"][ilak]:
                    self.columns[(block["name"], ilak)] = ilake
                    ilake += 1
        nlake = ilake

        # dR/ds for the aquifer rows, dL/dh for the lake rows, and the lake's
        # own storage and conductance on the diagonal
        drds = sparse.lil_matrix((int(nnodes), int(nlake)))
        dldh = sparse.lil_matrix((int(nlake), int(nnodes)))
        diag = np.zeros(nlake)
        for block in blocks:
            for iconn in range(block["node"].shape[0]):
                key = (block["name"], int(block["lake_of_conn"][iconn]))
                if key not in self.columns:
                    continue
                node = int(block["node"][iconn])
                cond = float(block["cond"][iconn])
                icol = self.columns[key]
                drds[node, icol] += cond
                dldh[icol, node] -= cond
                diag[icol] += cond

        for key, icol in self.columns.items():
            for block in blocks:
                if block["name"] != key[0]:
                    continue
                ilak = key[1]
                # a steady-state step has no change in lake storage
                if transient:
                    diag[icol] += float(block["dvds"][ilak]) / dt
                # a lake whose surface grows with stage evaporates more as it
                # rises, and catches more rain, which act in opposite senses
                net_rate = float(block["evaporation"][ilak]) - float(
                    block["rainfall"][ilak]
                )
                diag[icol] += net_rate * float(block["dsads"][ilak])

        dlds = sparse.lil_matrix((int(nlake), int(nlake)))
        for icol in range(nlake):
            dlds[icol, icol] = diag[icol]

        # an outlet drains the lake it leaves, and fills the lake it enters
        for block in blocks:
            outlets = block["outlets"]
            if not outlets:
                continue
            for ioutlet in range(outlets["lakein"].shape[0]):
                source = (block["name"], int(outlets["lakein"][ioutlet]))
                if source not in self.columns:
                    continue
                dqds = self.outlet_derivative(block, ioutlet)
                if dqds == 0.0:
                    continue
                icol = self.columns[source]
                dlds[icol, icol] += dqds
                destination = (block["name"], int(outlets["lakeout"][ioutlet]))
                if destination in self.columns:
                    dlds[self.columns[destination], icol] -= dqds

        # amat is already transposed, so the lake blocks are transposed too
        bordered = sparse.bmat(
            [
                [amat, dldh.transpose()],
                [drds.transpose(), dlds.transpose()],
            ],
            format="csr",
        )

        # same convention as the aquifer rows: the carry-back from the later
        # time step minus the measure's own derivative
        rhs_lake = np.zeros(nlake)
        for key, icol in self.columns.items():
            rhs_lake[icol] = self.carry.get(key, 0.0) - dfds.get(key, 0.0)
        return bordered, np.concatenate((rhs, rhs_lake))

    def split(self, lamb, blocks, dt, nnodes, carry_back=True):
        """Take the lake stages off the solution and carry their storage back.

        Parameters
        ----------
        carry_back : bool
            False for a steady-state step, and for an instantaneous measure,
            where each time step is solved on its own.

        Returns
        -------
        ndarray
            The aquifer part of the adjoint state.
        """
        lake_lamb = lamb[nnodes:]
        self.carry = {}
        if carry_back:
            for block in blocks:
                for key, icol in self.columns.items():
                    if key[0] != block["name"]:
                        continue
                    # the storage that links the steps is the slope at the
                    # previous step's stage
                    dvds = float(block["dvds_old"][key[1]])
                    self.carry[key] = dvds / dt * lake_lamb[icol]
        return lamb[:nnodes]
