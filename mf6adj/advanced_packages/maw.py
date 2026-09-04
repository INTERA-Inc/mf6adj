"""Multi-aquifer well (MAW) coupling for the adjoint solution.

A multi-aquifer well head is a dependent variable, and MODFLOW solves it with
the flow equations rather than outside them, so the well's equation is already
one of the rows of the matrix the adjoint is taken from. Unlike a lake or a
reach, nothing here has to border the system or rebuild the well's equation.

What is left is the row each well occupies, the derivative of a performance
measure with respect to the well head, and the storage the well carries from
one time step to the next.
"""

import numpy as np

# MODFLOW 6 marks a well option that was not given with this value
NOT_SET = 1.0e20


def saturation_derivative(top, bot, x, eps):
    """Return the slope of the smoothed screen saturation with head.

    This is the derivative MODFLOW 6 forms for a multi-aquifer well connection
    under the Newton-Raphson formulation, over the interval of the screen that
    the connection covers. Away from the ends of that interval the saturation
    follows the head linearly; MODFLOW rounds the two corners over a fraction
    ``eps`` of the interval, and the slope follows that rounding.

    Parameters
    ----------
    top : ndarray
        Top of the screen interval.
    bot : ndarray
        Bottom of the screen interval.
    x : ndarray
        Upstream head.
    eps : float
        Smoothing fraction, the package's ``satomega``.

    Returns
    -------
    ndarray
        Slope of the saturation with respect to the upstream head.
    """
    thickness = np.asarray(top - bot, dtype=float)
    over = np.divide(
        x - bot, thickness, out=np.zeros_like(thickness), where=thickness > 0.0
    )
    fraction = np.clip(over, 0.0, 1.0)
    scale = 1.0 / (1.0 - eps) if eps < 1.0 else 1.0
    slope = np.full(fraction.shape, scale)
    if eps > 0.0:
        slope = np.where(fraction < eps, scale * fraction / eps, slope)
        slope = np.where(fraction > 1.0 - eps, scale * (1.0 - fraction) / eps, slope)
    slope = np.where(fraction >= 1.0, 0.0, slope)
    return np.divide(
        slope, thickness, out=np.zeros_like(thickness), where=thickness > 0.0
    )


def forward_terms(gwf, gwf_name: str, tag: str) -> dict:
    """Return the well terms of a multi-aquifer well package for one time step.

    Parameters
    ----------
    gwf : modflowapi.ModflowApi
        MODFLOW 6 groundwater-flow instance.
    gwf_name : str
        Name of the groundwater-flow model.
    tag : str
        Multi-aquifer well package name from the GWF name file.

    Returns
    -------
    dict
        The solution row of each well, the well each connection belongs to,
        the conductance of each connection, and the terms of the well's own
        storage. ``well_is_free`` marks a well whose head is solved, as opposed
        to one held constant or inactive. ``well_rate_limited`` marks a package
        whose rate follows the well head through a shutoff, a pumping-rate
        reduction, or a flowing well.
    """

    def value(name):
        return gwf.get_value(gwf.get_var_address(name, gwf_name, tag.upper())).copy()

    def optional(name, default, size):
        """Return a package quantity, or a default where MODFLOW omits it."""
        try:
            return value(name)
        except Exception:
            return np.full(size, default)

    # IDXLOCNODE is the one-based row the well's equation occupies in the
    # solution, which already counts the wells of any package before it
    row = value("IDXLOCNODE") - 1
    nwells = row.shape[0]
    # IMAP is the one-based well each connection belongs to
    well_of_conn = value("IMAP") - 1

    # MODFLOW keeps a well's status in the model ibound, in the entry for the
    # row its equation occupies: zero for an inactive well and negative for one
    # held at a constant head, neither of which has a water balance to solve
    ibound = gwf.get_value(gwf.get_var_address("IBOUND", gwf_name))
    is_free = (ibound[row] > 0).astype(int)

    # The well stores water over its own area, and MODFLOW clips the head it
    # stores against to the screen, so a well whose head left the screen
    # between the two steps stores nothing more as that head moves.
    head_old = value("XOLDPAK")
    head_sto_old = value("XOLDSTO")
    stores = is_free * (head_sto_old == head_old)

    # A connection's conductance follows the upstream head through the
    # saturated fraction of the screen it covers, so the exchange follows that
    # head twice over. MODFLOW carries the second part in the matrix under the
    # Newton-Raphson formulation, and a measure of the exchange needs it too.
    # Under the standard formulation the conductance is lagged instead, and the
    # matrix holds it fixed, so the measure holds it fixed as well.
    cond = value("SIMCOND")
    # MODFLOW scales the conductance a connection is given by the saturated
    # fraction of the screen, so a sensitivity to the value the user gives
    # carries that fraction
    satcond = value("SATCOND")
    sat = np.divide(cond, satcond, out=np.ones_like(cond), where=satcond > 0.0)
    head = value("HEAD")
    node = value("NODELIST") - 1
    hgwf = gwf.get_value(gwf.get_var_address("X", gwf_name))[node]
    hmaw = head[well_of_conn]
    # the derivative attaches to whichever side is upstream
    maw_upstream = hmaw > hgwf
    # MODFLOW pins the saturation at one in a cell it does not convert, where
    # the conductance is the full thickness and does not follow the head
    icelltype = gwf.get_value(gwf.get_var_address("ICELLTYPE", gwf_name, "NPF"))
    if int(value("INEWTON")[0]) != 0:
        dsatdh = saturation_derivative(
            value("TOPSCRN"),
            value("BOTSCRN"),
            np.maximum(hmaw, hgwf),
            float(value("SATOMEGA")[0]),
        )
        # the same term MODFLOW forms: the slope of the conductance times the
        # head difference that drives the exchange
        dterm = np.where(
            icelltype[node] != 0,
            dsatdh * value("SATCOND") * (hgwf - hmaw),
            0.0,
        )
    else:
        dterm = np.zeros_like(cond)

    # a rate that follows the well head is in the well's equation only where
    # MODFLOW assembled its derivative, which is the Newton-Raphson formulation
    limited = (
        (optional("SHUTOFFLEVEL", NOT_SET, nwells) < NOT_SET)
        | (optional("PUMPELEV", NOT_SET, nwells) < NOT_SET)
        | (optional("REDUCTION_LENGTH", NOT_SET, nwells) < NOT_SET)
        | (optional("FWCONDSIM", 0.0, nwells) > 0.0)
    ).astype(int)

    return {
        "well_area": value("AREA"),
        "well_cond": cond,
        "well_dterm": dterm,
        "well_head": head,
        "well_maw_upstream": maw_upstream.astype(int),
        "well_head_old": head_old,
        "well_is_free": is_free,
        "well_imover": np.full(nwells, int(optional("IMOVER", 0, 1)[0]), dtype=int),
        "well_iss": np.full(nwells, int(value("IMAWISS")[0]), dtype=int),
        "well_of_conn": well_of_conn,
        "well_rate_limited": limited,
        "well_row": row,
        "well_sat": sat,
        "well_stores": stores,
    }


class MawCoupling:
    """The multi-aquifer well rows of the adjoint system.

    One instance follows a performance measure through its backward sweep and
    holds the state that spans time steps: the row each well occupies and the
    storage each carries back to the previous step.
    """

    def __init__(self, logger=None):
        """Initialize the coupling.

        Parameters
        ----------
        logger : object, optional
            Logger used to report where a well is only partly differentiated.
        """
        self.logger = logger
        self.carry = {}
        self.rows = {}
        self._warned = set()

    def reset_step(self) -> None:
        """Forget the previous time step's well state."""
        self.rows = {}
        self.carry = {}

    def _warn(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warning(message)

    def _warn_once(self, pname: str, grp, is_newton: bool) -> None:
        """Warn where a well's equation is differentiated only in part."""
        if pname in self._warned:
            return
        self._warned.add(pname)

        if not is_newton and grp["well_rate_limited"][:].max() > 0:
            self._warn(
                f"multi-aquifer well package '{pname}' limits a rate on the "
                "well head, through a shutoff, a pumping-rate reduction, or a "
                "flowing well, and the flow model did not use the "
                "Newton-Raphson formulation, so the matrix does not carry how "
                "that limit follows the head and the sensitivity is "
                "approximate."
            )

        if grp["well_imover"][:].max() > 0:
            self._warn(
                f"multi-aquifer well package '{pname}' exchanges water through "
                "the mover. That flow follows the state of the package on the "
                "other side, and the derivative is not formed, so the "
                "sensitivity is approximate."
            )

    def blocks(self, sol_dataset, gwf_package_dict, is_newton=True) -> list:
        """Return the wells and the terms coupling them to the aquifer.

        Parameters
        ----------
        sol_dataset : h5py.Group
            Forward-solution group for one time step.
        gwf_package_dict : dict
            Mapping of package types to package names.
        is_newton : bool, optional
            Whether the flow model used the Newton-Raphson formulation.

        Returns
        -------
        list
            One entry per multi-aquifer well package.
        """
        found = []
        for ptype, pnames in gwf_package_dict.items():
            if ptype != "maw6":
                continue
            for pname in pnames:
                if pname not in sol_dataset:
                    continue
                grp = sol_dataset[pname]
                if "well_row" not in grp:
                    continue

                self._warn_once(pname, grp, is_newton)

                found.append(
                    {
                        "name": pname,
                        "node": grp["nodelist"][:] - 1,
                        "cond": grp["well_cond"][:],
                        "area": grp["well_area"][:],
                        "dterm": grp["well_dterm"][:],
                        "head": grp["well_head"][:],
                        "maw_upstream": grp["well_maw_upstream"][:] == 1,
                        "free": grp["well_is_free"][:] == 1,
                        "iss": grp["well_iss"][:],
                        "row": grp["well_row"][:],
                        "sat": grp["well_sat"][:],
                        "stores": grp["well_stores"][:],
                        "well_of_conn": grp["well_of_conn"][:],
                    }
                )
        return found

    def claimed_rows(self, blocks) -> set:
        """Return the solution rows the wells of these packages occupy."""
        claimed = set()
        for block in blocks:
            claimed.update(int(row) for row in block["row"])
        return claimed

    @staticmethod
    def _weights(entries, kk, block) -> dict:
        """Return the weight this measure gives each node of one package."""
        return {
            pfr.inode: pfr.weight
            for pfr in entries
            if pfr.kperkstp == kk and pfr.pm_type == block["name"]
        }

    def dfds(self, entries, kk, blocks) -> dict:
        """Return the derivative of the measure with respect to each well head.

        A flux measure on a multi-aquifer well sums the exchange at the cells
        it names, and that exchange depends on the well head as well as on the
        head in the cell. Where the well is the upstream side, it also sets the
        conductance, and that part is carried too.

        Returns
        -------
        dict
            ``(package name, well)`` mapped to the derivative.
        """
        result = {}
        for block in blocks:
            weights = self._weights(entries, kk, block)
            if not weights:
                continue
            for iconn in range(block["node"].shape[0]):
                node = int(block["node"][iconn])
                if node not in weights:
                    continue
                key = (block["name"], int(block["well_of_conn"][iconn]))
                value = float(block["cond"][iconn])
                if block["maw_upstream"][iconn]:
                    value -= float(block["dterm"][iconn])
                result[key] = result.get(key, 0.0) + weights[node] * value
        return result

    def measure_dfdh(self, entries, kk, blocks, nnodes) -> np.ndarray:
        """Return what a measure adds to the derivative with respect to head.

        The exchange follows the head in the cell twice over: through the head
        difference, which the package hcof already carries, and through the
        conductance, where the cell is the upstream side. Only the second is
        returned here.

        Returns
        -------
        ndarray
            Addition to the derivative of the measure with respect to head.
        """
        result = np.zeros(int(nnodes))
        for block in blocks:
            weights = self._weights(entries, kk, block)
            if not weights:
                continue
            for iconn in range(block["node"].shape[0]):
                node = int(block["node"][iconn])
                if node not in weights or block["maw_upstream"][iconn]:
                    continue
                result[node] -= weights[node] * float(block["dterm"][iconn])
        return result

    def fill(self, rhs, blocks, dfds) -> np.ndarray:
        """Put the well rows of the right-hand side in place.

        The rows the wells occupy are already in the matrix, so only their
        right-hand side is formed here: the carry-back from the later time step
        minus the measure's own derivative, the same convention the aquifer
        rows use.

        Returns
        -------
        ndarray
            The right-hand side, with the well rows filled in place.
        """
        self.rows = {}
        for block in blocks:
            for iwell in range(block["row"].shape[0]):
                key = (block["name"], iwell)
                irow = int(block["row"][iwell])
                self.rows[key] = irow
                rhs[irow] = self.carry.get(key, 0.0) - dfds.get(key, 0.0)
        return rhs

    def sensitivities(self, lamb, head, blocks, entries, kk) -> dict:
        """Return the sensitivity of the measure to the terms of each package.

        The rate a well is given enters only its own equation, so the
        sensitivity to it is the adjoint state of that equation. The
        conductance of a connection enters the equation of the cell and the
        equation of the well with opposite signs, and a measure of the exchange
        through that connection depends on it directly as well.

        Parameters
        ----------
        lamb : ndarray
            The solved adjoint state, aquifer rows first, before the well rows
            are taken off.
        head : ndarray
            Simulated head for every node.
        blocks : list
            The well packages of this time step.
        entries : list
            Entries of the performance measure.
        kk : tuple
            Zero-based stress period and time step.

        Returns
        -------
        dict
            Package name mapped to the well numbers, the nodes the connections
            reach, and the sensitivity to the rate of each well and to the
            conductance of each connection.
        """
        result = {}
        for block in blocks:
            weights = self._weights(entries, kk, block)
            rate = np.array([lamb[int(row)] for row in block["row"]], dtype=float)
            cond = np.zeros(block["node"].shape[0])
            for iconn in range(block["node"].shape[0]):
                node = int(block["node"][iconn])
                iwell = int(block["well_of_conn"][iconn])
                # the head difference that drives the exchange, which is what
                # a unit of conductance carries
                drop = float(block["head"][iwell]) - float(head[node])
                # a well whose head is held or is inactive has no equation for
                # the connection to enter, so only the cell it reaches responds
                well = rate[iwell] if block["free"][iwell] else 0.0
                # the conductance a user gives is scaled by the saturated
                # fraction of the screen before MODFLOW uses it
                cond[iconn] = float(block["sat"][iconn]) * (
                    (lamb[node] - well + weights.get(node, 0.0)) * drop
                )
            result[block["name"]] = {
                "well": np.arange(block["row"].shape[0]) + 1,
                "node": block["node"] + 1,
                "rate": rate,
                "cond": cond,
            }
        return result

    def split(self, lamb, blocks, dt, nnodes, carry_back=True) -> np.ndarray:
        """Take the well heads off the solution and carry their storage back.

        Parameters
        ----------
        lamb : ndarray
            The solved adjoint state, aquifer rows first.
        blocks : list
            The well packages of this time step.
        dt : float
            Length of this time step.
        nnodes : int
            Number of aquifer rows.
        carry_back : bool
            False for a steady-state step, and for an instantaneous measure,
            where each time step is solved on its own.

        Returns
        -------
        ndarray
            The aquifer part of the adjoint state.
        """
        self.carry = {}
        if carry_back and dt > 0.0:
            for block in blocks:
                for key, irow in self.rows.items():
                    if key[0] != block["name"]:
                        continue
                    iwell = key[1]
                    # a steady-state well has no change in storage, and a well
                    # whose head is held or is inactive has no equation to
                    # carry, which MODFLOW marks by clipping the head it
                    # stores against
                    if block["iss"][iwell] != 0 or not block["stores"][iwell]:
                        continue
                    # MODFLOW releases the storage of the previous step into
                    # the right-hand side of this one, so the derivative of
                    # that side with respect to the previous head is negative,
                    # as it is for the aquifer
                    area = float(block["area"][iwell])
                    self.carry[key] = -area / dt * lamb[irow]
        return lamb[:nnodes]
