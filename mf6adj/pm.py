import logging
import pathlib as pl
import types
from datetime import datetime
from typing import Any, List, Optional, Union

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import (
    LinearOperator,
    bicgstab,
    cg,
    gmres,
    lgmres,
    lsqr,
    spilu,
    spsolve,
    svds,
)

from .advanced_packages import LakeCoupling, SfrCoupling
from .boundary import DRAIN_CORNER_TOL, drop_corner_entries
from .utils.utils import SolverCallback
from .utils.utils_fileio import _write_group_to_hdf
from .utils.utils_logger import _LoggerUtil
from .utils.utils_modflow import (
    get_mf6_bound_dict,
)

PathLike = Union[str, pl.Path]


class PerfMeasRecord:
    """Single record from a performance-measure block.

    Parameters
    ----------
    kper : int
        Zero-based stress period.
    kstp : int
        Zero-based time step.
    inode : int
        Zero-based node number.
    pm_type : str
        Either ``head`` or a boundary package name from the GWF name file.
    pm_form : str
        Either ``direct`` or ``residual``.
    weight : float
        Weight value.
    obsval : float
        Observed counterpart; used only for ``residual`` form.
    k : int, optional
        Zero-based layer index for structured-grid reporting.
    i : int, optional
        Zero-based row index for structured-grid reporting.
    j : int, optional
        Zero-based column index for structured-grid reporting.
    """

    def __init__(
        self,
        kper: int,
        kstp: int,
        inode: int,
        pm_type: str,
        pm_form: str,
        weight: float,
        obsval: float,
        k: Optional[int] = None,
        i: Optional[int] = None,
        j: Optional[int] = None,
    ):
        """Initialize a performance-measure record.

        Parameters
        ----------
        kper : int
            Zero-based stress period.
        kstp : int
            Zero-based time step.
        inode : int
            Zero-based node number.
        pm_type : str
            Performance-measure type.
        pm_form : str
            Performance-measure form.
        weight : float
            Weight value.
        obsval : float
            Observed value used for residual measures.
        k : int, optional
            Zero-based layer index for structured-grid reporting.
        i : int, optional
            Zero-based row index for structured-grid reporting.
        j : int, optional
            Zero-based column index for structured-grid reporting.
        """
        self._kper = int(kper)
        self._kstp = int(kstp)
        self.kperkstp = (self._kper, self._kstp)
        if isinstance(inode, np.int64):
            inode = int(inode)
        elif isinstance(inode, np.ndarray):
            inode = int(inode[0])
        self.inode = int(inode)
        self._k = None
        if k is not None:
            self._k = int(k)
        self._i = None
        if i is not None:
            self._i = int(i)
        self._j = None
        if j is not None:
            self._j = int(j)
        self.weight = float(weight)
        self.obsval = float(obsval)
        self.pm_type = pm_type.lower().strip()

        self.pm_form = pm_form.lower().strip()
        if self.pm_form not in ["direct", "residual", "instantaneous"]:
            raise Exception(
                "PerfMeasRecord.pm_form must be 'direct', 'residual', or "
                + f"'instantaneous', not '{self.pm_form}'"
            )

    def __repr__(self) -> str:
        """Return a compact string representation of the record.

        Returns
        -------
        str
            Summary string containing time-step, node, type, and form fields.
        """
        s = (
            f"kperkstp:{self.kperkstp}, inode:{self.inode}, "
            + f"k:{self._k}, type:{self.pm_type}, form:{self.pm_form}"
        )
        if self._i is not None:
            s += f", i:{self._i}"
        if self._j is not None:
            s += f", j:{self._j}"
        return s


class PerfMeas:
    """Performance-measure container and adjoint-solve helper.

    Parameters
    ----------
    pm_name : str
        Name of the performance measure.
    pm_entries : list[PerfMeasRecord]
        Performance-measure entries.
    logging_level : str or int, optional
        Logging level.
    logger : logging.Logger, optional
        Logger instance. If omitted, a new logger is created.
    """

    def __init__(
        self,
        pm_name: str,
        pm_entries: List[PerfMeasRecord],
        logging_level: Union[int, str] = "INFO",
        logger: Optional[_LoggerUtil] = None,
    ):
        """Initialize a performance-measure container.

        Parameters
        ----------
        pm_name : str
            Name of the performance measure.
        pm_entries : list[PerfMeasRecord]
            Performance-measure entries.
        logging_level : int or str, optional
            Logging level to use when creating a new logger.
        logger : logging.Logger, optional
            Existing logger utility. If omitted, a new logger is created.
        """
        self._name = pm_name.lower().strip()
        self._entries = pm_entries
        self._lake = None
        self._sfr = None

        if logger is None:
            logger_name = f"{self.__class__.__name__}-{self.name}"
            logger_filename = f"{self.name}.log"
            self.logger = _LoggerUtil(
                logger_name,
                logging_level,
                logger_filename,
            )
        else:
            self.new_logger = False
            self.logger = logger

    @property
    def name(self) -> str:
        """Return the performance measure name.

        Returns
        -------
        str
            Performance-measure name.
        """
        return str(self._name)

    def _performance_measure_forward(
        self, head_dict, sp_package_dict, dt_dict=None
    ) -> float:
        """Calculate the forward value of the performance measure.

        This helper is used during perturbation testing to evaluate the
        forward-response value associated with the configured entries.

        Parameters
        ----------
        head_dict : dict
            Mapping from ``(kper, kstp)`` tuples to simulated head arrays.
        sp_package_dict : dict
            Nested mapping of stress-package results produced by
            :meth:`Mf6Adj.solve_forward_model` when ``pert_save=True``.
        dt_dict : dict, optional
            Mapping from ``(kper, kstp)`` tuples to time step lengths. Required
            for an instantaneous measure.

        Returns
        -------
        float
            Scalar forward value for the performance measure. Direct measures
            are accumulated linearly, residual measures are accumulated as
            weighted squared residuals, and instantaneous measures are the
            time-weighted mean of the per-time-step values.
        """
        is_instantaneous = self._entries[0].pm_form == "instantaneous"
        if is_instantaneous and dt_dict is None:
            raise Exception(
                "dt_dict is required to evaluate the instantaneous performance "
                + f"measure {self.name}"
            )

        # accumulate by time step so an instantaneous measure can be averaged
        # over time below. Every time step with an entry is represented, which
        # matches the time steps the adjoint solution accumulates.
        per_step = {pfr.kperkstp: 0.0 for pfr in self._entries}
        for pfr in self._entries:
            if pfr.pm_type == "head":
                if pfr.pm_form in ("direct", "instantaneous"):
                    per_step[pfr.kperkstp] += (
                        pfr.weight * head_dict[pfr.kperkstp][pfr.inode]
                    )
                elif pfr.pm_form == "residual":
                    per_step[pfr.kperkstp] += (
                        pfr.weight * (head_dict[pfr.kperkstp][pfr.inode] - pfr.obsval)
                    ) ** 2
                else:
                    raise Exception("something is wrong")
            else:
                for gwf_ptype, bnd_d in sp_package_dict.items():
                    for kk, kk_list in bnd_d.items():
                        if kk == pfr.kperkstp:
                            for kk_d in kk_list:
                                if (
                                    kk_d["node"] == pfr.inode + 1
                                    and kk_d["packagename"] == pfr.pm_type
                                ):
                                    if pfr.pm_form in ("direct", "instantaneous"):
                                        per_step[kk] += pfr.weight * kk_d["simval"]
                                    elif pfr.pm_form == "residual":
                                        per_step[kk] += (
                                            pfr.weight * (kk_d["simval"] - pfr.obsval)
                                        ) ** 2

        if not is_instantaneous:
            return float(sum(per_step.values()))

        # An instantaneous measure is the time-weighted mean of the per-step
        # values, so the result does not depend on the time discretization.
        # This matches the composite the adjoint solution reports.
        wsum = sum(dt_dict[kk] for kk in per_step)
        if wsum <= 0.0:
            return 0.0
        return float(sum(dt_dict[kk] * val for kk, val in per_step.items()) / wsum)

    def _setup_block_jacobi_preconditioner(self, amat, block_size=5000):
        """Setup a block Jacobi preconditioner

        Parameters
        ----------
        amat : scipy.sparse.spmatrix
            Sparse matrix for which to create the block Jacobi preconditioner.
        block_size : int
            Size of the blocks for the block Jacobi preconditioner.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
            Linear operator representing the block Jacobi preconditioner.

        """
        n = amat.shape[0]
        self.logger.logger.debug(
            f"Setup block Jacobi preconditioner with {block_size:,} block size "
            + f"for matrix with {n:,} rows"
        )

        # 1. Compute RCM permutation and reorder matrix
        # This clusters non-zeros to make the diagonal blocks more "meaningful"
        perm = reverse_cuthill_mckee(amat)
        inv_perm = np.argsort(perm)  # Used to "undo" the permutation after solving
        amat_rcm = amat[perm, :][:, perm].tocsc()

        # 2. Define block boundaries
        block_indices = []
        for i in range(0, n, block_size):
            end = min(i + block_size, n)
            block_indices.append((i, end))

        self.logger.logger.debug(
            "Number of blocks for block Jacobi preconditioner: "
            + f"{len(block_indices):,}"
        )

        # 3. Pre-factorize diagonal blocks using SPILU
        ilu_solvers = []
        for start, end in block_indices:
            block = amat_rcm[start:end, start:end]
            # spilu requires CSC format and usually benefits
            # from fill_factor adjustments
            try:
                ilu_solvers.append(spilu(block, fill_factor=10.0))
            except RuntimeError:
                # Fallback if a block is singular or problematic
                self.logger.logger.debug(
                    f"Warning: Block {start}:{end} is singular, using simpler solve."
                )
                ilu_solvers.append(None)

        # 4. Define M^-1 * v
        def block_jacobi_solve(v):
            # Reorder input vector to RCM space
            v_rcm = v[perm]
            res_rcm = np.zeros_like(v_rcm)

            for i, (start, end) in enumerate(block_indices):
                if ilu_solvers[i] is not None:
                    res_rcm[start:end] = ilu_solvers[i].solve(v_rcm[start:end])
                else:
                    # Basic identity fallback for failed blocks
                    res_rcm[start:end] = v_rcm[start:end]

            # Return to original ordering
            return res_rcm[inv_perm]

        return LinearOperator((n, n), matvec=block_jacobi_solve)

    def _setup_point_jacobi_preconditioner(self, amat):
        """Setup the point Jacobi preconditioner

        Parameters
        ----------
        amat : scipy.sparse.spmatrix
            Sparse matrix for which to create the Jacobi preconditioner.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
            Linear operator representing the Jacobi preconditioner.

        """
        self.logger.logger.debug("Setup Jacobi preconditioner")
        d = amat.diagonal()
        d_inv = 1.0 / d

        def point_jacobi_solve(v):
            return d_inv * v

        return LinearOperator(
            amat.shape,
            matvec=point_jacobi_solve,
        )

    def _setup_jacobi_preconditioner(
        self,
        amat,
        jacobi_type="point",
        precon_kwargs=None,
    ):
        """Setup a Jacobi preconditioner

        Parameters
        ----------
        amat : scipy.sparse.spmatrix
            Sparse matrix for which to create the Jacobi preconditioner.
        jacobi_type : str, optional
            Type of Jacobi preconditioner to use. Supported values are "point"
            and "block".
        precon_kwargs : dict, optional
            Keyword arguments passed to the Jacobi preconditioner setup method.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
            Linear operator representing the Jacobi preconditioner.

        """
        if jacobi_type == "point":
            return self._setup_point_jacobi_preconditioner(amat)
        elif jacobi_type == "block":
            block_size = (precon_kwargs or {}).get("block_size")
            precon_kwargs = (
                {"block_size": int(block_size)} if block_size is not None else {}
            )
            return self._setup_block_jacobi_preconditioner(
                amat,
                **precon_kwargs,
            )

    def solve_adjoint(
        self,
        hdf5_forward_solution_fname: PathLike,
        hdf5_adjoint_solution_fname: Optional[PathLike] = None,
        csv_summary: bool = False,
        linear_solver=None,
        linear_solver_kwargs: Optional[dict] = None,
        use_precon: bool = True,
        jacobi_preconditioner: Optional[str] = None,
        precon_kwargs: Optional[dict] = None,
        singular_test: bool = False,
        tikhonov: float = 0.0,
        dvclose: Optional[float] = 1e-6,
        rclose: Optional[float] = 1e-3,
        dvscale: bool = False,
        drain_corner_tol: float = DRAIN_CORNER_TOL,
    ) -> pd.DataFrame:
        """Solve for the adjoint state for the performance measure.

        This method writes an adjoint-solution HDF5 file and, when requested,
        writes a CSV summary file.

        Parameters
        ----------
        hdf5_forward_solution_fname : PathLike
            HDF5 file created by :meth:`Mf6Adj.solve_forward_model` containing the
            forward-solution
            information needed for the adjoint solve.
        hdf5_adjoint_solution_fname : PathLike, optional
            HDF5 file to write the adjoint solution. If omitted, a default
            name based on the performance-measure name is used. If the target
            file already exists, it is removed before writing.
        csv_summary : bool, optional
            Write a CSV summary of the sensitivity information beside the
            adjoint HDF5 output file.
        linear_solver : str or callable, optional
            Sparse linear solver to use. Supported string values are
            ``"direct"``, ``"bicgstab"``, ``"cg"``, ``"gmres"``,
            ``"lgmres"``, and ``"lsqr"``.
        linear_solver_kwargs : dict, optional
            Keyword arguments passed to the configured linear solver callable.
        use_precon : bool, optional
            Use a preconditioner with the iterative linear solver.
        jacobi_preconditioner : str, optional
            Use a Jacobi preconditioner with the iterative linear solver. If
            ``None``, the ILU preconditioner is used. Supported string values
            are ``"point"`` and ``"block"``.
        precon_kwargs : dict, optional
            Keyword arguments passed to the preconditioner setup. For the default
            ILU preconditioner, these are passed to ``spilu``. When
            ``jacobi_preconditioner="block"``, the ``block_size`` key sets the
            block size.
        singular_test : bool, optional
            Test for a singular matrix and apply Tikhonov regularization when
            needed.
        tikhonov : float, optional
            Tikhonov regularization value.
        dvclose : float, optional
            Custom convergence criterion based on the maximum absolute change
            between consecutive iterates.
        rclose : float, optional
            Custom convergence criterion based on the maximum absolute residual.
        dvscale : bool, optional
            Scale lambda and the right-hand side to improve iterative solver
            convergence for large lambda values.
        drain_corner_tol : float, optional
            Head above a drain elevation below which the entry is treated as
            sitting on the corner of the drain flow function and dropped from
            the performance-measure derivative. A value of zero disables the
            check.

        Returns
        -------
        DataFrame
            Summary of composite sensitivity information.
        """
        supported_iterative_solvers = (
            "bicgstab",
            "cg",
            "gmres",
            "lgmres",
            "lsqr",
        )
        supported_jacobi_preconditioners = (
            "point",
            "block",
        )
        if jacobi_preconditioner is not None:
            if jacobi_preconditioner not in supported_jacobi_preconditioners:
                raise Exception(
                    (
                        "Invalid 'jacobi_preconditioner' value: "
                        + f"{jacobi_preconditioner!r}. "
                        + "Supported values are None, "
                        + ", ".join(f"'{s}'" for s in supported_jacobi_preconditioners)
                    )
                )
            else:
                self.logger.logger.info(
                    f"Using '{jacobi_preconditioner}' Jacobi preconditioner"
                )
                use_precon = True
        linear_solver_kwargs = (
            {} if linear_solver_kwargs is None else dict(linear_solver_kwargs)
        )
        precon_kwargs = {} if precon_kwargs is None else dict(precon_kwargs)

        adj_start = datetime.now()
        self.logger.logger.info(f"Starting solve_adjoint at {adj_start}")
        hdf5_forward_solution_fname = pl.Path(hdf5_forward_solution_fname)
        try:
            hdf = h5py.File(hdf5_forward_solution_fname, "r")
        except Exception as e:
            raise Exception(
                (
                    f"error opening hdf5 file '{hdf5_forward_solution_fname}' "
                    + f"for PerfMeas {self._name}: {e!s}"
                )
            )
        if hdf5_adjoint_solution_fname is None:
            hdf5_adjoint_solution_fname = hdf5_forward_solution_fname.parent / (
                f"adjoint_solution_{self._name}_"
                + f"{hdf5_forward_solution_fname.name}"
            )
        else:
            hdf5_adjoint_solution_fname = pl.Path(hdf5_adjoint_solution_fname)

        if hdf5_adjoint_solution_fname.exists():
            self.logger.logger.warning(
                (
                    "Removing existing adjoint solution "
                    + f"file '{hdf5_adjoint_solution_fname}'"
                )
            )
            hdf5_adjoint_solution_fname.unlink()

        adf = h5py.File(hdf5_adjoint_solution_fname, "w")

        keys = list(hdf.keys())
        gwf_package_dict = dict(hdf["gwf_info"].attrs.items())

        sol_keys = [k for k in keys if k.startswith("solution")]
        sol_keys.sort()
        if len(sol_keys) == 0:
            raise Exception("no 'solution' keys found")
        kperkstp = [
            (kper, kstp) for kper, kstp in zip(hdf["aux"]["kper"], hdf["aux"]["kstp"])
        ]
        if len(kperkstp) != len(sol_keys):
            raise Exception(
                (
                    f"number of solution datasets ({len(sol_keys)}) != number "
                    + f"of kper,kstp entries ({len(kperkstp)})"
                )
            )
        kk_sol_map = {}
        for kk in kperkstp:
            sol = None
            for s in sol_keys:
                skper, skstp = hdf[s].attrs["kper"], hdf[s].attrs["kstp"]
                # print(skper, skstp)
                if skper == kk[0] and skstp == kk[1]:
                    sol = s
                    break
            if sol is None:
                raise Exception(f"no solution dataset found for kper,kstp:{kk!s}")
            kk_sol_map[kk] = sol

        nnodes = hdf["gwf_info"]["nnodes"][:]
        nodeuser = hdf["gwf_info"]["nodeuser"][:]
        nodereduced = hdf["gwf_info"]["nodereduced"][:]
        if len(nodeuser) == 1:
            nodeuser = np.arange(nnodes[0], dtype=int)
        if len(nodereduced) == 1:
            nodereduced = None
        lamb = np.zeros(nnodes)

        grid_shape = None
        if "nrow" in hdf["gwf_info"].keys():
            grid_shape = (
                int(hdf["gwf_info"]["nlay"][0]),
                int(hdf["gwf_info"]["nrow"][0]),
                int(hdf["gwf_info"]["ncol"][0]),
            )
            self.logger.logger.info(f"Structured grid found, shape: {grid_shape}")

        ia = hdf["gwf_info"]["ia"][:]
        ja = hdf["gwf_info"]["ja"][:]
        ihc = hdf["gwf_info"]["ihc"][:]
        jas = hdf["gwf_info"]["jas"][:]
        cl1 = hdf["gwf_info"]["cl1"][:]
        cl2 = hdf["gwf_info"]["cl2"][:]
        hwva = hdf["gwf_info"]["hwva"][:]
        top = hdf["gwf_info"]["top"][:]
        bot = hdf["gwf_info"]["bot"][:]
        icelltype = hdf["gwf_info"]["icelltype"][:]
        area = hdf["gwf_info"]["area"][:]
        ihighcellsat = hdf["gwf_info"]["ihighcellsat"][0]

        comp_k33_sens = np.zeros(nnodes)
        comp_k_sens = np.zeros(nnodes)
        comp_ss_sens = None
        comp_sy_sens = None

        has_sto = hdf[sol_keys[0]].attrs["has_sto"]
        if has_sto:
            comp_ss_sens = np.zeros(nnodes)
            comp_sy_sens = np.zeros(nnodes)

        # An "instantaneous" measure looks at each time step on its own and
        # ignores how later time steps would otherwise feed back into earlier
        # ones. The result is the sensitivity at each time step by itself. All
        # entries in a measure share the same form, so one flag covers the
        # whole measure.
        is_instantaneous = self._entries[0].pm_form == "instantaneous"
        # running total of the time-step weights, used below to average the
        # result of an instantaneous measure
        wsum = 0.0

        comp_welq_sens = np.zeros(nnodes)
        comp_rch_sens = np.zeros((nnodes))

        # A lake stage is a dependent variable, so the system is bordered with
        # the lake water balance. nnodes is a one-element array, so keep a
        # scalar for slicing the bordered solution.
        nnode = int(np.asarray(nnodes).ravel()[0])
        self._lake = LakeCoupling(logger=self.logger.logger)
        self._sfr = SfrCoupling(logger=self.logger.logger)

        bnd_dict = get_mf6_bound_dict()
        comp_bnd_results = {}
        for ptype, pnames in gwf_package_dict.items():
            if ptype in bnd_dict:
                for pname in pnames:
                    for idx, aname in bnd_dict[ptype].items():
                        comp_bnd_results[pname + "_" + aname] = np.zeros(nnodes)

        for itime, kk in enumerate(kperkstp[::-1]):
            # For an instantaneous measure, a time step with no observation
            # adds nothing, so skip it. The average below then uses only the
            # time steps that have an observation. Direct and residual measures
            # pass information from one time step to the next, so every step
            # must be solved and none are skipped.
            if is_instantaneous and not self.__pm_available(kk):
                continue

            data = {}
            kper_start = datetime.now()
            msg = (
                "Starting adjoint solve for PerfMeas: "
                + f"{self._name} (kper, kstp) ({int(kk[0] + 1)}, {int(kk[1] + 1)})"
            )
            self.logger.logger.info(msg)
            # if itime == 0:
            #     print(f"starting adjoint solve for PerfMeas: {self._name}")

            sol_key = kk_sol_map[kk]
            if sol_key in adf:
                raise Exception(
                    f"solution key '{sol_key}' already in adjoint hdf5 file"
                )

            start = datetime.now()
            self.logger.logger.debug("Forming rhs")

            self.logger.logger.debug("Calculate dfdh")
            dfdh = self.__dfdh(kk, hdf[sol_key], tol=drain_corner_tol)

            self.logger.logger.debug(
                "Calculating dfdh took: "
                + f"{(datetime.now() - start).total_seconds()} seconds"
            )

            data["dfdh"] = dfdh
            iss = hdf[sol_key]["iss"][0]

            # Weight for this time step, used when combining the per-step
            # results below. This does not change the old behavior: for direct
            # and residual measures w is 1.0, so "value * w" is exactly the same
            # as "value" and the divide by wsum at the end is skipped. Only an
            # instantaneous measure uses a real weight (the length of the time
            # step, dt) so its result can be averaged over time.
            w = float(hdf[sol_key].attrs["dt"]) if is_instantaneous else 1.0
            wsum += w

            if iss == 0 and not is_instantaneous:  # transient
                # drhsdh is the storage term that links this time step to the
                # next. Multiplying it by the later step's result (lamb) carries
                # that result backward in time into this step.
                drhsdh = hdf[sol_key]["drhsdh"][:]
                data["drhsdh"] = drhsdh
                rhs = (drhsdh * lamb) - dfdh
            else:
                # Either steady state or an instantaneous measure. In both
                # cases there is no carryover from the later time step, so each
                # time step is solved on its own.
                rhs = -dfdh

            self.logger.logger.debug(
                "Formulating rhs took: "
                + f"{(datetime.now() - start).total_seconds()} seconds"
            )

            start = datetime.now()

            self.logger.logger.debug("Taking transpose of amat")
            amat = hdf[sol_key]["amat"][:]
            head = hdf[sol_key]["head"][:]
            residual = hdf[sol_key]["residual"][:]
            amat = sparse.csr_matrix(
                (amat.copy()[: ja.shape[0]], ja.copy(), ia.copy()),
                shape=(len(ia) - 1, len(ia) - 1),
            )
            amat = amat.transpose().tocsr()

            # the lake state is per time step: a step with no free lake must
            # not inherit the previous step's columns, carry, or solution size
            # a steady-state step has no change in lake storage, so it has no
            # storage on the diagonal and nothing to carry. An instantaneous
            # measure keeps the storage, because that is part of the lake's
            # equation at this step, but is solved without the carry.
            lake_storage = hdf[sol_key]["iss"][0] == 0
            lake_carry_back = lake_storage and not is_instantaneous
            lake_blocks = self._lake.blocks(hdf[sol_key], gwf_package_dict)
            if not lake_blocks:
                self._lake.reset_step()
            else:
                dt = float(hdf[sol_key].attrs["dt"])
                amat, rhs = self._lake.augment(
                    amat,
                    rhs,
                    lake_blocks,
                    dt,
                    self._lake.dfds(self._entries, kk, lake_blocks),
                    nnode,
                    transient=lake_storage,
                )

            # the reach rows sit outside the lake rows, so they are added to
            # whatever the lake left, and index the aquifer nodes below it
            sfr_blocks = self._sfr.blocks(hdf[sol_key], gwf_package_dict)
            if not sfr_blocks:
                self._sfr.reset_step()
            else:
                # a measure on a reach that leaks its whole flow follows the
                # head beneath the reaches above it, which df/dh has to carry
                rhs = rhs - self._sfr.measure_dfdh(
                    self._entries, kk, sfr_blocks, rhs.shape[0]
                )
                amat, rhs = self._sfr.augment(
                    amat,
                    rhs,
                    sfr_blocks,
                    self._sfr.dfds(self._entries, kk, sfr_blocks),
                    amat.shape[0],
                )

            # the number of free lakes and reaches can change between steps, so
            # rebuild the initial guess from the aquifer part and zero the rest
            if lamb.shape[0] != amat.shape[0]:
                guess = np.zeros(amat.shape[0])
                guess[:nnode] = lamb[:nnode]
                lamb = guess
            self.logger.logger.debug(
                (
                    "Transpose of amat took: "
                    + f"{(datetime.now() - start).total_seconds()} seconds"
                )
            )

            start = datetime.now()
            self.logger.logger.info("Solving for lambda")

            is_newton = hdf[sol_key].attrs["is_newton"]
            self.logger.logger.debug(f"Newton-Raphson: {is_newton}")

            if linear_solver is None:
                if head.shape[0] < 50000:
                    linear_solver = "direct"
                else:
                    linear_solver = "bicgstab"
            self.logger.logger.debug(f"Linear solver: {linear_solver}")

            if linear_solver == "direct":
                _linear_solver = spsolve
                if not linear_solver_kwargs:
                    _linear_solver_kwargs = {"use_umfpack": True}
                else:
                    _linear_solver_kwargs = linear_solver_kwargs
            elif linear_solver in supported_iterative_solvers:
                if not linear_solver_kwargs:
                    if linear_solver == "lsqr":
                        _linear_solver_kwargs = {
                            "btol": 1e-6,
                            "atol": 1e-6,
                            "iter_lim": 5000,
                        }
                    else:
                        _linear_solver_kwargs = {
                            "rtol": 1e-6,
                            "atol": 1e-6,
                            "maxiter": 200,
                        }
                else:
                    _linear_solver_kwargs = linear_solver_kwargs

                if linear_solver == "bicgstab":
                    _linear_solver = bicgstab
                elif linear_solver == "cg":
                    if is_newton:
                        self.logger.logger.warning(
                            "GWF model is using the Newton-Raphson formulation. "
                            + f"Switching from '{linear_solver}' to 'bicgstab' "
                            + "linear solver."
                        )
                        linear_solver = "bicgstab"
                        _linear_solver = bicgstab
                    else:
                        _linear_solver = cg
                elif linear_solver == "lgmres":
                    _linear_solver = lgmres
                elif linear_solver == "gmres":
                    _linear_solver = gmres
                elif linear_solver == "lsqr":
                    _linear_solver = lsqr
                else:
                    raise Exception(
                        "unrecognized iterative 'linear_solver' value: "
                        + f"'{linear_solver}', "
                        + f"should be ({', '.join(supported_iterative_solvers)})."
                    )
            else:
                if not isinstance(linear_solver, types.FunctionType):
                    raise Exception(
                        f"specified linear_solver ('{linear_solver}') must be a "
                        + " function if it not a supported iterative solver type. "
                        + "Supported iterative solver types are "
                        + f"({', '.join(supported_iterative_solvers)})."
                    )

                _linear_solver = linear_solver
                _linear_solver_kwargs = linear_solver_kwargs

            # test if a singular matrix
            if singular_test:
                self.logger.logger.info("Testing if transpose of amat is singular")
                is_singular = self.__is_singular(amat)
                if is_singular:
                    self.logger.logger.info("Transpose of amat is singular")
                    if tikhonov <= 0.0:
                        tikhonov = 1e-6

            # add tikhonov regularization
            if tikhonov > 0.0:
                self.logger.logger.info(f"Adding Tikhonov regularization ({tikhonov})")
                sign = np.sign(amat.diagonal())
                mult = sign[0]
                I = sparse.eye(amat.shape[0], format=amat.format)
                amat = amat + mult * tikhonov * I

            # set up preconditioner
            m = None
            if linear_solver not in ("direct", "lsqr"):
                if use_precon:
                    self.logger.logger.debug("Setup preconditioner")
                    if jacobi_preconditioner is not None:
                        m = self._setup_jacobi_preconditioner(
                            amat,
                            jacobi_preconditioner,
                            precon_kwargs,
                        )
                    else:
                        if not precon_kwargs:
                            _precon_kwargs = {
                                "drop_tol": 1e-4,
                                "fill_factor": 10,
                                "drop_rule": "basic,area",
                            }
                        else:
                            _precon_kwargs = precon_kwargs
                        try:
                            amat_ilu = spilu(amat, **_precon_kwargs)
                            m = LinearOperator(
                                (amat.shape[0], amat.shape[0]),
                                amat_ilu.solve,
                            )
                        except Exception as e:
                            msg = (
                                "Failed to form preconditioner - "
                                + f"using Jacobi preconditioned {linear_solver} "
                                + "solver and reset maxiter to "
                                + f"{_linear_solver_kwargs['maxiter']}"
                            )
                            self.logger.logger.info(msg)

                            error_lines = str(e).splitlines()
                            msg = ""
                            for line in error_lines:
                                msg += f" {line}."
                            self.logger.logger.debug(msg)

                            # generate a simple Jacobi preconditioner
                            # and increase maxiter
                            if "maxiter" in _linear_solver_kwargs:
                                _linear_solver_kwargs["maxiter"] += 10000
                            else:
                                _linear_solver_kwargs["maxiter"] = 10000

                            m = self._setup_jacobi_preconditioner(
                                amat,
                                jacobi_type="block",
                                precon_kwargs=_precon_kwargs,
                            )

                    _linear_solver_kwargs["M"] = m

            if linear_solver != "direct" and (
                dvclose is not None or rclose is not None
            ):
                _linear_solver_kwargs["atol"] = 0.0
                _linear_solver_kwargs["rtol"] = 0.0

            self.logger.logger.info(
                f"Solving with {linear_solver} "
                + f"with solver options: {_linear_solver_kwargs}"
            )

            # solve the system of equations
            if linear_solver == "direct":
                lamb = _linear_solver(amat, rhs, **_linear_solver_kwargs)
            else:
                if dvscale:
                    idx_max = np.abs(lamb).argmax()
                    scale = float(np.abs(lamb[idx_max]))
                    if scale > 1e-30:
                        self.logger.logger.debug(f"Scaling lambda and rhs ({scale})")
                        lamb /= scale
                        rhs /= scale
                try:
                    solver_cb = SolverCallback(
                        logger=self.logger,
                        A=amat,
                        b=rhs,
                        dvclose=dvclose,
                        rclose=rclose,
                    )
                    if linear_solver == "gmres":
                        lamb, info = _linear_solver(
                            amat,
                            rhs,
                            x0=lamb,
                            callback=solver_cb,
                            callback_type="x",
                            **_linear_solver_kwargs,
                        )
                    elif linear_solver == "lsqr":
                        lamb, info = _linear_solver(
                            amat,
                            rhs,
                            damp=tikhonov,
                            x0=lamb,
                            **_linear_solver_kwargs,
                        )
                    else:
                        lamb, info = _linear_solver(
                            amat,
                            rhs,
                            x0=lamb,
                            callback=solver_cb,
                            **_linear_solver_kwargs,
                        )
                except StopIteration as e:
                    self.logger.logger.info(e)
                    lamb, info = solver_cb.xold, 0

            if linear_solver in supported_iterative_solvers:
                # info = lamb[1]
                # lamb = lamb[0]
                if dvscale:
                    if scale > 1e-30:
                        self.logger.logger.debug(f"Unscaling lambda and rhs ({scale})")
                        lamb *= scale
                        rhs *= scale

                residual = rhs - amat @ lamb
                residual_2norm = np.linalg.norm(residual)
                residual_infnorm = np.linalg.norm(residual, ord=np.inf)
                self.logger.logger.info(
                    (
                        f"Solver return code: {info} "
                        + f"iterations: {solver_cb.niter} "
                        + f"solver norms: L2 ({residual_2norm}) "
                        + f"infinity ({residual_infnorm})"
                    )
                )
                if info < 0:
                    self.logger.logger.error("Solver parameter breakdown")
                elif info > 0:
                    self.logger.logger.warning("Solver convergence not achieved")

            if self._sfr.columns:
                # the reach depths come off first, being the outer border
                lamb = self._sfr.split(lamb, lamb.shape[0] - len(self._sfr.columns))
            if self._lake.columns:
                # split the lake stages off and carry their storage back to the
                # previous time step, as drhsdh does for the aquifer
                lamb = self._lake.split(
                    lamb, lake_blocks, dt, nnode, carry_back=lake_carry_back
                )

            if np.any(np.isnan(lamb)):
                self.logger.logger.warning(
                    (
                        f"Adjoint states for pm {self.name} contain nans "
                        + f"at (kper,kstp) ({int(kk[0] + 1)}, {int(kk[1] + 1)})"
                    )
                )
            self.logger.logger.info(
                (
                    "Solving for lambda took: "
                    + f"{(datetime.now() - start).total_seconds()} seconds"
                )
            )
            is_newton = hdf[sol_key].attrs["is_newton"]

            # zero out the adj state for chd nodes
            chd_nodelist = []
            if "chd6" in gwf_package_dict:
                for pname in gwf_package_dict["chd6"]:
                    nodelist = list(hdf[sol_key][pname]["nodelist"][:] - 1)
                    chd_nodelist.extend(nodelist)
            chd_nodelist = np.array(chd_nodelist, dtype=int)
            lamb[chd_nodelist] = 0.0

            start = datetime.now()
            self.logger.logger.debug("Formulating lam_dresdk_h")
            k_sens, k33_sens = self.__lam_dresdk_h(
                is_newton,
                ihighcellsat,
                lamb,
                hdf[sol_key]["sat"][:],
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
                hdf[sol_key]["k11"][:],
                hdf[sol_key]["k33"][:],
            )

            data["k11"] = k_sens
            data["k33"] = k33_sens
            comp_k_sens += k_sens * w
            comp_k33_sens += k33_sens * w
            self.logger.logger.debug(
                (
                    "Formulating lam_dresdk_h took: "
                    + f"{(datetime.now() - start).total_seconds()} seconds"
                )
            )

            if has_sto:
                start = datetime.now()

                self.logger.logger.debug("Formulating storage")

                if iss == 0:
                    ss_sens = lamb * hdf[sol_key]["dresdss_h"][:]
                    # written by solve_forward_model; an older forward file
                    # carries only the specific-storage half
                    if "dresdsy_h" in hdf[sol_key]:
                        sy_sens = lamb * hdf[sol_key]["dresdsy_h"][:]
                    else:
                        sy_sens = np.zeros_like(lamb)
                else:
                    ss_sens = np.zeros_like(lamb)
                    sy_sens = np.zeros_like(lamb)
                data["ss"] = ss_sens
                data["sy"] = sy_sens
                comp_ss_sens += ss_sens * w
                comp_sy_sens += sy_sens * w
                self.logger.logger.debug(
                    (
                        "Formulating storage took: "
                        + f"{(datetime.now() - start).total_seconds()} seconds"
                    )
                )

            data["wel6_q"] = lamb
            comp_welq_sens += lamb * w
            # A well rate is already a flow, but recharge is a rate over the
            # cell area, and a package may scale it further by an auxiliary
            # multiplier. Both are carried by the right-hand side MODFLOW
            # formed, which is the rate times whatever it applies, so the flow
            # a recharge value produces is taken from there rather than from
            # the area alone. Where a package does not recharge a cell, or
            # recharges it at zero, the right-hand side says nothing about the
            # factor and the cell area stands.
            rch_factor = area.copy()
            for rch_name in gwf_package_dict.get("rch6", []):
                if rch_name not in hdf[sol_key]:
                    continue
                grp = hdf[sol_key][rch_name]
                if "recharge" not in grp:
                    continue
                rate = grp["recharge"][:]
                applied = -grp["rhs"][:]
                nodes = grp["nodelist"][:] - 1
                given = rate != 0.0
                rch_factor[nodes[given]] = applied[given] / rate[given]
            rch_sens = lamb * rch_factor
            comp_rch_sens += rch_sens * w
            data["rch6_recharge"] = rch_sens

            for ptype, pnames in gwf_package_dict.items():
                if ptype == "chd6":
                    continue
                if ptype in bnd_dict:
                    for pname in pnames:
                        if pname not in hdf[sol_key]:
                            continue
                        start = datetime.now()

                        self.logger.logger.debug(f"Formulating {ptype}, {pname}")

                        sp_bnd_dict = {
                            "bound": hdf[sol_key][pname]["bound"][:],
                            "node": hdf[sol_key][pname]["nodelist"][:],
                        }
                        direct_weights = {
                            pfr.inode: pfr.weight
                            for pfr in self._entries
                            if pfr.kperkstp == kk and pfr.pm_type == pname
                        }
                        sens_level, sens_cond = self.__lam_drhs_dbnd(
                            lamb, head, sp_bnd_dict, direct_weights
                        )
                        comp_bnd_results[pname + "_" + bnd_dict[ptype][0]] += (
                            sens_level * w
                        )
                        data[pname + "_" + bnd_dict[ptype][0]] = sens_level
                        if len(bnd_dict[ptype]) > 1:
                            comp_bnd_results[pname + "_" + bnd_dict[ptype][1]] += (
                                sens_cond * w
                            )
                            data[pname + "_" + bnd_dict[ptype][1]] = sens_cond
                        self.logger.logger.debug(
                            (
                                f"Formulating {ptype}, {pname} took: "
                                + f"{(datetime.now() - start).total_seconds()} "
                                + "seconds"
                            )
                        )

            data["lambda"] = lamb
            data["head"] = hdf[sol_key]["head"][:]
            msg = (
                "Adjoint solve took: "
                + f"{(datetime.now() - kper_start).total_seconds()!s} "
                + f"seconds to solve adjoint solution for PerfMeas: {self._name} "
                + f"(kper, kstp) ({int(kk[0] + 1)}, {int(kk[1] + 1)})"
            )
            self.logger.logger.info(msg)

            if self.logger.isDebugLogger:
                self.logger.logger.debug("Adding amat, rhs, and residual to hdf file")
                # the writer serializes csc, and maps vectors onto the grid,
                # so drop the lake rows the bordered system added
                data["amat"] = amat.tocsc()
                data["rhs"] = rhs[:nnode]
                data["residual"] = residual[:nnode]

            self.logger.logger.info("Write group to hdf file")
            _write_group_to_hdf(
                adf,
                sol_key,
                data,
                nodeuser=nodeuser,
                grid_shape=grid_shape,
                nodereduced=nodereduced,
                logger=self.logger.logger,
            )
        self.logger.logger.info("Formulate composite sensitivities")
        # An instantaneous measure handles each time step on its own, so its
        # overall result is the average of the per-step results (each weighted
        # by its time-step length) instead of a plain sum. Direct and residual
        # measures keep the plain sum.
        if is_instantaneous and wsum > 0.0:
            comp_k_sens /= wsum
            comp_k33_sens /= wsum
            comp_welq_sens /= wsum
            comp_rch_sens /= wsum
            if has_sto:
                comp_ss_sens /= wsum
                comp_sy_sens /= wsum
            for name in comp_bnd_results:
                comp_bnd_results[name] /= wsum
        data = {}
        data["k11"] = comp_k_sens
        data["k33"] = comp_k33_sens

        if has_sto:
            data["ss"] = comp_ss_sens
            data["sy"] = comp_sy_sens
        data["wel6_q"] = comp_welq_sens
        data["rch6_recharge"] = comp_rch_sens

        for name, vals in comp_bnd_results.items():
            data[name] = vals
        self.logger.logger.info("Writing composite sensitivities")
        _write_group_to_hdf(
            adf,
            "composite",
            data,
            nodeuser=nodeuser,
            grid_shape=grid_shape,
            nodereduced=nodereduced,
            logger=self.logger.logger,
        )
        adf.close()
        hdf.close()

        df = pd.DataFrame(
            {
                "k11": comp_k_sens,
                "k33": comp_k33_sens,
                "wel6_q": comp_welq_sens,
                "rch6_recharge": comp_rch_sens,
            },
            index=nodeuser + 1,
        )

        for name, vals in comp_bnd_results.items():
            df[name] = vals
        if has_sto:
            df["ss"] = comp_ss_sens
            df["sy"] = comp_sy_sens

        df.index.name = "node"
        if csv_summary:
            csv_path = hdf5_adjoint_solution_fname.with_suffix(".csv")
            df.to_csv(csv_path)

        kper, kstp = kk
        dtsec = (datetime.now() - adj_start).total_seconds()
        line = (
            "Adjoint solve took: "
            + f"{dtsec:10.5g} seconds "
            + f"for pm '{self._name}' at (kper,kstp) ({int(kper + 1)}, {int(kstp + 1)})"
        )
        self.logger.logger.info(line)

        return df

    def __dconddhk(self, k1, k2, cl1, cl2, width, height1, height2) -> float:
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

    def __smooth_sat_simple(self, sat) -> float:
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

    def __d_smooth_sat_dh_simple(self, sat, top, bot) -> float:
        """Return the derivative of smoothed saturation with respect to head.

        Parameters
        ----------
        sat : float
            Saturation.
        top : float
            Cell top elevation.
        bot : float
            Cell bottom elevation.

        Returns
        -------
        float
            Derivative of the smoothed saturation curve with respect to head.
        """
        satomega = 1.0e-6
        A_omega = 1 / (1 - satomega)
        d_s_sat_dh = 0.0
        if sat >= 0 and sat < satomega:
            d_s_sat_dh = (A_omega / satomega) * sat / (top - bot)
        elif sat >= satomega and sat < 1 - satomega:
            d_s_sat_dh = A_omega / (top - bot)
        elif sat >= 1 - satomega and sat < 1:
            d_s_sat_dh = (A_omega / satomega) * (1 - sat) / (top - bot)
        return d_s_sat_dh

    def __cell_sat(self, top, bot, h) -> float:
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

    def __smooth_sat(self, ihighcellsat, top1, top2, bot1, bot2, h1, h2) -> float:
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
                sat = self.__cell_sat(top1, bot1, h1)
            else:
                sat = self.__cell_sat(top1, bot, h1)
        else:
            if bot is None:
                sat = self.__cell_sat(top2, bot2, h2)
            else:
                sat = self.__cell_sat(top2, bot, h2)
        return self.__smooth_sat_simple(sat)

    def __d_smooth_sat_dh(self, sat, h1, h2, top, bot) -> float:
        """Return the derivative of smoothed saturation with respect to upstream head.

        Parameters
        ----------
        sat : float
            Saturation.
        h1 : float
            Head of node 1.
        h2 : float
            Head of node 2.
        top : float
            Upstream cell top elevation.
        bot : float
            Upstream cell bottom elevation.

        Returns
        -------
        float
            Derivative of smoothed saturation with respect to the upstream
            head. Returns zero when ``h1`` is not the upstream head.
        """
        value = 0.0
        if h1 >= h2:
            value = self.__d_smooth_sat_dh_simple(sat, top, bot)
        return value

    def __lam_dresdk_h(
        self,
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

        Returns
        -------
        tuple[ndarray, ndarray]
            Adjoint-weighted residual derivatives with respect to ``k11`` and
            ``k33``.
        """
        iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
        # array of number of connections per node (size nodes)

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
                    dconddk33 = self.__dconddhk(
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
                        * (head[mnode] - head[node])
                        * (lamb[node] - lamb[mnode])
                    )
                    sum1 += t2

                else:
                    # TODO: check if one cell is convertible (??is this required??)
                    if is_newton:
                        dconddk = self.__dconddhk(
                            k11[node],
                            k11[mnode],
                            cl1[jj],
                            cl2[jj],
                            hwva[jj],
                            height1,
                            height2,
                        )
                        SF = self.__smooth_sat(
                            ihighcellsat,
                            top[node],
                            top[mnode],
                            bot[node],
                            bot[mnode],
                            head[node],
                            head[mnode],
                        )

                    else:
                        dconddk = self.__dconddhk(
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
                        * (head[mnode] - head[node])
                        * (lamb[node] - lamb[mnode])
                    )
                    sum2 += t1

            result33[node] = sum1
            result[node] = sum2
        return result, result33

    def __lam_drhs_dbnd(
        self,
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
            ``node`` and ``bound`` arrays.
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
        for node, bound in zip(sp_dict["node"], sp_dict["bound"]):
            n = node - 1
            boundcond = 1e10
            if len(bound) > 1:
                boundcond = bound[1]
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
            result_cond[n] += lam_drhs_dcond + lam_dadcond_h
            if weight != 0.0:
                result_cond[n] += weight * (bound[0] - head[n])

        return result_head, result_cond

    def __pm_available(self, kk) -> bool:
        """Return whether a performance measure entry exists for a given time.

        Parameters
        ----------
        kk : tuple
            Zero-based ``(kper, kstp)`` tuple.

        Returns
        -------
        bool
            True if a performance measure entry exists, otherwise False.
        """

        relevant_entries = [p for p in self._entries if p.kperkstp == kk]
        return len(relevant_entries) > 0

    def __dfdh(self, kk, sol_dataset, tol=DRAIN_CORNER_TOL) -> np.ndarray:
        """Return the derivative of the performance measure with respect to head.

        Parameters
        ----------
        kk : tuple
            Zero-based stress period and time step.
        sol_dataset : h5py.Group
            Forward-solution HDF5 group for one time step. The group must
            contain a ``head`` dataset and, for flux performance measures,
            package subgroups with ``nodelist`` and ``hcof`` datasets.
        tol : float, optional
            Head above a drain elevation below which the entry is treated as
            sitting on the corner and dropped from the derivative.

        Returns
        -------
        ndarray
            Derivative of the performance measure with respect to head at the
            requested time step.
        """

        # 1. Load data once. Avoid [:] if sol_dataset is already in memory,
        # but for HDF5, reading once into a local variable is best.
        head = sol_dataset["head"][:]
        dfdh = np.zeros_like(head)

        # 2. Cache 'hcof' and 'nodelist' to avoid repeated disk I/O
        # Also pre-build a map for O(1) index lookups
        cached_maps = {}

        # 3. Filter entries first to reduce iterations
        relevant_entries = [p for p in self._entries if p.kperkstp == kk]

        for pfr in relevant_entries:
            if pfr.pm_type == "head":
                if pfr.pm_form in ("direct", "instantaneous"):
                    dfdh[pfr.inode] = pfr.weight
                elif pfr.pm_form == "residual":
                    # Scalar math is fine here, but make sure head is a numpy array
                    dfdh[pfr.inode] = 2.0 * pfr.weight * (head[pfr.inode] - pfr.obsval)
            else:
                # Handle non-head types with caching
                if pfr.pm_type not in cached_maps:
                    hcof = sol_dataset[pfr.pm_type]["hcof"][:]
                    nodes = sol_dataset[pfr.pm_type]["nodelist"][:] - 1
                    # a drain on its corner has no defensible derivative, so
                    # it contributes nothing to the measure
                    hcof, ncorner = drop_corner_entries(
                        sol_dataset[pfr.pm_type], hcof, head, tol
                    )
                    if ncorner > 0:
                        self.logger.logger.warning(
                            f"{pfr.pm_type}: dropped {ncorner} drain entries "
                            + f"within {tol:g} of their elevation from the "
                            + "performance measure derivative"
                        )
                    # A cell can carry more than one boundary from the same
                    # package - a lake connected both vertically and
                    # horizontally to it - and the measure sums the flow through
                    # all of them, so accumulate hcof by node.
                    node_hcof = np.zeros_like(dfdh)
                    np.add.at(node_hcof, nodes, hcof)
                    in_package = np.zeros(dfdh.shape, dtype=bool)
                    in_package[nodes] = True
                    cached_maps[pfr.pm_type] = (node_hcof, in_package)

                node_hcof, in_package = cached_maps[pfr.pm_type]

                if in_package[pfr.inode]:
                    dfdh[pfr.inode] = pfr.weight * node_hcof[pfr.inode]

        return dfdh

    def __get_value_from_gwf(self, gwf_name, pak_name, prop_name, gwf) -> Any:
        """Return a copy of a quantity from the MODFLOW 6 API.

        Parameters
        ----------
        gwf_name : str
            Name of the groundwater-flow instance.
        pak_name : str
            Package name from the GWF name file.
        prop_name : str
            Property name in ``pak_name``.
        gwf : modflowapi.ModflowApi
            MODFLOW 6 groundwater-flow API instance.

        Returns
        -------
        Any
            Requested quantity copied from the MODFLOW 6 API.
        """
        addr = gwf.get_var_address(prop_name, gwf_name, pak_name)
        return gwf.get_value(addr)

    def __is_singular(self, A, tol=1e-10) -> bool:
        """Return whether a matrix is singular or numerically ill-conditioned.

        Parameters
        ----------
        A : scipy.sparse.spmatrix or LinearOperator
            Matrix to evaluate.
        tol : float, optional
            Absolute tolerance used to test the smallest singular value.

        Returns
        -------
        bool
            True if the matrix is singular or effectively singular.
        """
        # svds computes the k largest or smallest singular values.
        # To check for singularity, we need the smallest ones.
        # 'SM' specifies "smallest magnitude".
        # We request only one singular value (k=1) for efficiency.
        try:
            # Note: svds works on square or rectangular matrices
            # We need to ensure k is less than min(M, N) for svds to work
            min_dim = min(A.shape)
            if min_dim == 0:
                return True  # Empty matrix is singular
            k = max(1, min(5, min_dim - 1))  # Get a few smallest, ensure k >= 1

            # Use 'SM' to find smallest magnitude singular values
            # svds returns U, s, Vh
            _, s, _ = svds(A, k=k, which="SM")

            # Check if the smallest singular value is close to zero
            smallest_s_value = np.min(s)
            return np.isclose(smallest_s_value, 0.0, atol=tol)
        except RuntimeError:
            # svds might raise a RuntimeError for very difficult
            # cases (e.g., convergence issues)
            return True  # Assume singular or ill-conditioned if solver fails
