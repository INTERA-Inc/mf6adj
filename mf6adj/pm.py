import logging
import pathlib as pl
import types
from datetime import datetime
from typing import Any, List, Optional, Union

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sparse
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

from .utils.utils_logger import LoggerUtil

PathLike = Union[str, pl.Path]


class SolverCallback:
    """Iterative-solver callback with optional custom convergence checks."""

    def __init__(self, logger, A, b, dvclose=None, rclose=None):
        """Initialize the solver callback.

        Parameters
        ----------
        logger : LoggerUtil
            Logger utility used to report iteration progress.
        A : scipy.sparse.spmatrix or LinearOperator
            Linear system matrix.
        b : ndarray
            Right-hand side vector.
        dvclose : float, optional
            Maximum allowed change in consecutive iterates.
        rclose : float, optional
            Maximum allowed residual.
        """
        self.logger = logger
        self.A = A
        self.b = b
        self.niter = 0
        self.dvclose = dvclose
        self.rclose = rclose
        self.dvmax = None
        self.rmax = None
        self.xold = None
        self.custom_convergence = False
        if self.dvclose is not None:
            self.custom_convergence = True
            self.logger.logger.info(f"Solver dvclose value: {self.dvclose}")
        if self.rclose is not None:
            self.custom_convergence = True
            self.logger.logger.info(f"Solver rclose value: {self.rclose}")

    def __call__(self, xk) -> None:
        """Evaluate custom convergence criteria for the current iterate.

        Parameters
        ----------
        xk : ndarray
            Current solver iterate.
        """
        self.niter += 1
        if self.dvclose is not None or self.rclose is not None:
            if self.dvclose is not None and self.niter == 1:
                self.xold = np.zeros_like(xk)

            debug_msg = f"Solver iteration: {self.niter}"

            # Calculate the maximum difference between current and
            # previous solutions
            if self.dvclose is not None:
                dv = xk - self.xold
                self.dvmax = np.abs(dv).max()
                self.xold = xk.copy()
                debug_msg += f" dvmax: {self.dvmax}"

            # calculate the maximum residual
            if self.rclose is not None:
                resid = self.b - self.A @ xk
                self.rmax = np.abs(resid).max()
                debug_msg += f" rmax: {self.rmax}"

            self.logger.logger.debug(debug_msg)

            if self.custom_convergence and self._is_converged:
                convergence_msg = "Custom convergence reached:"
                if self.dvclose is not None:
                    convergence_msg += f" dvmax = {self.dvmax} < {self.dvclose}"
                if self.rclose is not None:
                    convergence_msg += f" rmax = {self.rmax} < {self.rclose}"
                raise StopIteration(convergence_msg.strip())

    @property
    def _is_converged(self) -> bool:
        """Return whether the custom solver convergence criteria are met."""
        dv_converged = 1
        if self.dvmax is not None:
            if self.dvmax > self.dvclose:
                dv_converged = 0

        dr_converged = 1
        if self.rmax is not None:
            if self.rmax > self.rclose:
                dr_converged = 0

        return bool(int(dv_converged * dr_converged))


solver_callback = SolverCallback


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
        Either `head` or a boundary package name from the GWF name file.
    pm_form : str
        Either `direct` or `residual`.
    weight : float
        Weight value.
    obsval : float
        Observed counterpart; used only for `residual` form.
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
        if self.pm_form not in ["direct", "residual"]:
            raise Exception(
                "PerfMeasRecord.pm_form must be 'direct' or 'residual', "
                + f"not '{self.pm_form}'"
            )

    def __repr__(self) -> str:
        """Return a string representation of the performance-measure record."""
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
    logger : logging.Logger or LoggerUtil, optional
        Logger instance. If omitted, a new logger is created.
    """

    def __init__(
        self,
        pm_name: str,
        pm_entries: List[PerfMeasRecord],
        logging_level: Union[int, str] = "INFO",
        logger: Optional[LoggerUtil] = None,
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
        logger : LoggerUtil, optional
            Existing logger utility. If omitted, a new logger is created.
        """
        self._name = pm_name.lower().strip()
        self._entries = pm_entries

        if logger is None:
            logger_name = f"{self.__class__.__name__}-{self.name}"
            logger_filename = f"{self.name}.log"
            self.logger = LoggerUtil(
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

    @staticmethod
    def get_mf6_bound_dict() -> dict[str, dict[int, str]]:
        """Return MODFLOW 6 boundary-array index metadata.

        The returned mapping identifies which axes of a boundary `bound` array
        correspond to quantities used in sensitivity calculations.

        Returns
        -------
        dict[str, dict[int, str]]
            Boundary metadata used in sensitivity calculations.
        """
        d = {
            "ghb6": {0: "bhead", 1: "cond"},
            "drn6": {0: "elev", 1: "cond"},
            "riv6": {0: "stage", 1: "cond"},
            "sfr6": {0: "stage", 1: "cond"},
        }  # ,
        # "chd6":{0:"head"}}
        return d

    def solve_forward(self, head_dict, sp_package_dict) -> float:
        """Calculate the forward value of the performance measure.

        This method is used only during perturbation testing.
        """
        result = 0.0
        for pfr in self._entries:
            if pfr.pm_type == "head":
                if pfr.pm_form == "direct":
                    result += pfr.weight * head_dict[pfr.kperkstp][pfr.inode]
                elif pfr.pm_form == "residual":
                    result += (
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
                                    if pfr.pm_form == "direct":
                                        result += pfr.weight * kk_d["simval"]
                                    elif pfr.pm_form == "residual":
                                        result += (
                                            pfr.weight * (kk_d["simval"] - pfr.obsval)
                                        ) ** 2
        return result

    def solve_adjoint(
        self,
        hdf5_forward_solution_fname: PathLike,
        hdf5_adjoint_solution_fname: Optional[PathLike] = None,
        skip_solve: bool = False,
        csv_summary: bool = False,
        linear_solver=None,
        linear_solver_kwargs: Optional[dict] = None,
        use_precon: bool = True,
        precon_kwargs: Optional[dict] = None,
        singular_test: bool = False,
        tikhonov: float = 0.0,
        dvclose: Optional[float] = 1e-6,
        rclose: Optional[float] = 1e-3,
        dvscale: bool = False,
    ) -> pd.DataFrame:
        """Solve for the adjoint state for the performance measure.

        Parameters
        ----------
        hdf5_forward_solution_fname : PathLike
            HDF5 file created by `solve_gwf()` containing the forward-solution
            information needed for the adjoint solve.
        hdf5_adjoint_solution_fname : PathLike, optional
            HDF5 file to write the adjoint solution. If omitted, a default
            name based on the performance-measure name is used.
        skip_solve : bool, optional
            Skip the adjoint solve for time steps with no performance-measure
            entries.
        csv_summary : bool, optional
            Write a CSV summary of the sensitivity information.
        linear_solver : varies, optional
            Sparse linear solver to use. Supported string values are
            `"direct"`, `"bicgstab"`, `"cg"`, `"gmres"`, `"lgmres"`, and
            `"lsqr"`.
        linear_solver_kwargs : dict, optional
            Keyword arguments passed to `linear_solver`.
        use_precon : bool, optional
            Use an ILU preconditioner with an iterative linear solver.
        precon_kwargs : dict, optional
            Keyword arguments passed to the ILU preconditioner.
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
        ihighcellsat = hdf["gwf_info"]["ihighcellsat"][0]

        comp_k33_sens = np.zeros(nnodes)
        comp_k_sens = np.zeros(nnodes)
        comp_ss_sens = None

        has_sto = hdf[sol_keys[0]].attrs["has_sto"]
        if has_sto:
            comp_ss_sens = np.zeros(nnodes)

        has_flux_pm = False
        for entry in self._entries:
            if entry.pm_type != "head":
                has_flux_pm = True
                break

        comp_welq_sens = np.zeros(nnodes)
        comp_rch_sens = np.zeros((nnodes))

        bnd_dict = PerfMeas.get_mf6_bound_dict()
        comp_bnd_results = {}
        for ptype, pnames in gwf_package_dict.items():
            if ptype in bnd_dict:
                for pname in pnames:
                    for idx, aname in bnd_dict[ptype].items():
                        comp_bnd_results[pname + "_" + aname] = np.zeros(nnodes)

        for itime, kk in enumerate(kperkstp[::-1]):
            if skip_solve and not self._pm_available(kk):
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
            dfdh = self._dfdh(kk, hdf[sol_key])

            self.logger.logger.debug(
                "Calculating dfdh took: "
                + f"{(datetime.now() - start).total_seconds()} seconds"
            )

            data["dfdh"] = dfdh
            iss = hdf[sol_key]["iss"][0]

            if iss == 0:  # transient
                # get the derv of RHS WRT head
                drhsdh = hdf[sol_key]["drhsdh"][:]
                data["drhsdh"] = drhsdh
                rhs = (drhsdh * lamb) - dfdh
            else:
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
            amat = amat.transpose()
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
                is_singular = self._is_singular(amat)
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
                            (head.shape[0], head.shape[0]),
                            amat_ilu.solve,
                        )
                    except Exception as e:
                        if "maxiter" in _linear_solver_kwargs:
                            _linear_solver_kwargs["maxiter"] += 1000
                        else:
                            _linear_solver_kwargs["maxiter"] = 1000

                        msg = (
                            "Failed to form preconditioner - "
                            + f"using unpreconditioned {linear_solver} solver and "
                            + f"reset maxiter to {_linear_solver_kwargs['maxiter']}"
                        )
                        self.logger.logger.info(msg)

                        error_lines = str(e).splitlines()
                        msg = ""
                        for line in error_lines:
                            msg += f" {line}."
                        self.logger.logger.debug(msg)
                        m = None
                    _linear_solver_kwargs["M"] = m

            if linear_solver != "direct" and (
                dvclose is not None or rclose is not None
            ):
                _linear_solver_kwargs["atol"] = 0.0
                _linear_solver_kwargs["rtol"] = 0.0

            self.logger.logger.info(
                f"Solving with {linear_solver}"
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
            k_sens, k33_sens = PerfMeas.lam_dresdk_h(
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
            comp_k_sens += k_sens
            comp_k33_sens += k33_sens
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
                else:
                    ss_sens = np.zeros_like(lamb)
                data["ss"] = ss_sens
                comp_ss_sens += ss_sens
                self.logger.logger.debug(
                    (
                        "Formulating storage took: "
                        + f"{(datetime.now() - start).total_seconds()} seconds"
                    )
                )

            data["wel6_q"] = lamb
            comp_welq_sens += lamb
            comp_rch_sens += lamb
            data["rch6_recharge"] = lamb

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
                        sens_level, sens_cond = self.lam_drhs_dbnd(
                            lamb, head, sp_bnd_dict, has_flux_pm
                        )
                        comp_bnd_results[pname + "_" + bnd_dict[ptype][0]] += sens_level
                        data[pname + "_" + bnd_dict[ptype][0]] = sens_level
                        if len(bnd_dict[ptype]) > 1:
                            comp_bnd_results[pname + "_" + bnd_dict[ptype][1]] += (
                                sens_cond
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
                data["amat"] = amat
                data["rhs"] = rhs
                data["residual"] = residual

            self.logger.logger.info("Write group to hdf file")
            PerfMeas.write_group_to_hdf(
                adf,
                sol_key,
                data,
                nodeuser=nodeuser,
                grid_shape=grid_shape,
                nodereduced=nodereduced,
            )
        self.logger.logger.info("Formulate composite sensitivities")
        data = {}
        data["k11"] = comp_k_sens
        data["k33"] = comp_k33_sens

        if has_sto:
            data["ss"] = comp_ss_sens
        data["wel6_q"] = comp_welq_sens
        data["rch6_recharge"] = comp_rch_sens

        for name, vals in comp_bnd_results.items():
            data[name] = vals
        self.logger.logger.info("Writing composite sensitivities")
        PerfMeas.write_group_to_hdf(
            adf,
            "composite",
            data,
            nodeuser=nodeuser,
            grid_shape=grid_shape,
            nodereduced=nodereduced,
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

    @staticmethod
    def write_group_to_hdf(
        hdf: h5py.File,
        group_name: str,
        data_dict: dict,
        attr_dict: Optional[dict] = None,
        grid_shape: Optional[tuple[int, int, int]] = None,
        nodeuser: Optional[np.ndarray] = None,
        nodereduced: Optional[np.ndarray] = None,
    ) -> None:
        """Write a group of datasets to an open HDF5 file.

        Parameters
        ----------
        hdf : h5py.File
            Open HDF5 file handle.
        group_name : str
            Group name.
        data_dict : dict
            Datasets to write.
        attr_dict : dict, optional
            Group attributes to write.
        grid_shape : tuple[int, int, int], optional
            Structured-grid shape.
        nodeuser : ndarray, optional
            `nodeuser` array from MODFLOW 6.
        nodereduced : ndarray, optional
            `nodereduced` array from MODFLOW 6.
        """
        if attr_dict is None:
            attr_dict = {}
        if group_name in hdf:
            raise Exception(f"group_name {group_name} already in hdf file")
        grp = hdf.create_group(group_name)
        for name, val in attr_dict.items():
            grp.attrs[name] = val
        kijs = None
        if grid_shape is not None and nodeuser is not None:
            kijs = PerfMeas.get_lrc(grid_shape, list(nodeuser))
        for tag, item in data_dict.items():
            if isinstance(item, list):
                item = np.array(item)
            if isinstance(item, np.ndarray):
                if grid_shape is not None and nodeuser is not None:
                    if len(item) == len(nodeuser):  # 3D
                        arr = np.zeros(grid_shape, dtype=item.dtype)
                        for kij, v in zip(kijs, item):
                            arr[kij] = v

                        _ = grp.create_dataset(
                            tag, grid_shape, dtype=item.dtype, data=arr
                        )
                    else:
                        raise Exception("doh! " + str(tag))
                elif nodeuser is not None and nodereduced is not None:
                    arr = np.zeros_like(nodereduced, dtype=item.dtype)
                    for inode, v in zip(nodeuser, item):
                        arr[inode] = v
                    _ = grp.create_dataset(tag, arr.shape, dtype=item.dtype, data=arr)
                else:
                    _ = grp.create_dataset(tag, item.shape, dtype=item.dtype, data=item)
            elif isinstance(item, dict):
                subgrp = grp.create_group(tag)
                for k, v in item.items():
                    if isinstance(v, np.ndarray):
                        _ = subgrp.create_dataset(k, v.shape, dtype=v.dtype, data=v)
                    else:
                        subgrp.attrs[k] = v

            else:
                pass
                # raise Exception(
                #     "PerfMeas::write_group_to_hdf unrecognized data_dict entry: "
                #     + f"{tag},type:{type(item)}"
                # )
        if nodeuser is not None:
            _ = grp.create_dataset(
                "nodeuser", nodeuser.shape, dtype=nodeuser.dtype, data=nodeuser
            )
        if kijs is not None:
            for idx, name in enumerate(["k", "i", "j"]):
                arr = np.array([kij[idx] for kij in kijs], dtype=int)
                _ = grp.create_dataset(name, arr.shape, dtype=arr.dtype, data=arr)

    @staticmethod
    def _dconddhk(k1, k2, cl1, cl2, width, height1, height2) -> float:
        """Return the derivative of conductance with respect to `K`.

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
            Derivative of conductance with respect to `K`.
        """

        # todo: upstream weighting - could use height1 and height2 to check...
        # todo: vertically staggered
        d = (width * cl1 * height1 * (height2**2) * (k2**2)) / (
            ((cl2 * height1 * k1) + (cl1 * height2 * k2)) ** 2
        )
        return d

    @staticmethod
    def smooth_sat(sat) -> float:
        """Smooth saturation using the MODFLOW 6 sigmoid-style function.

        Parameters
        ----------
        sat : float
            Saturation value.

        Returns
        -------
        float
            Smoothed saturation.
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

    @staticmethod
    def d_smooth_sat_dh(sat, top, bot) -> float:
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
            Derivative of smoothed saturation with respect to head.
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

    @staticmethod
    def _cell_sat(top, bot, h) -> float:
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
            Cell saturation.
        """
        if h > top:
            sat = 1.0
        elif h < bot:
            sat = 0.0
        else:
            sat = (h - bot) / (top - bot)
        return sat

    @staticmethod
    def _smooth_sat(ihighcellsat, top1, top2, bot1, bot2, h1, h2) -> float:
        """Return smoothed saturation for the upstream cell.

        Parameters
        ----------
        ihighcellsat : int
            Flag for using the highest cell bottom when calculating saturation.
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
            Smoothed saturation of the upstream node.
        """
        bot = None
        if ihighcellsat != 0:
            if (abs(bot1) - abs(bot2)) > 1e-2:
                bot = max(bot1, bot2)
        if h1 > h2:
            if bot is None:
                sat = PerfMeas._cell_sat(top1, bot1, h1)
            else:
                sat = PerfMeas._cell_sat(top1, bot, h1)
        else:
            if bot is None:
                sat = PerfMeas._cell_sat(top2, bot2, h2)
            else:
                sat = PerfMeas._cell_sat(top2, bot, h2)
        return PerfMeas.smooth_sat(sat)

    @staticmethod
    def _d_smooth_sat_dh(sat, h1, h2, top, bot) -> float:
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
            Derivative of smoothed saturation with respect to upstream head.
        """
        value = 0.0
        if h1 >= h2:
            value = PerfMeas.d_smooth_sat_dh(sat, top, bot)
        return value

    @staticmethod
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return adjoint-weighted residual derivatives with respect to `k` and `k33`.

        Parameters
        ----------
        is_newton : bool
            Flag indicating whether Newton terms are active.
        ihighcellsat : int
            Flag for using the highest cell bottom when calculating saturation.
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
            Adjoint-weighted residual derivatives with respect to `k` and `k33`.
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
                    dconddk33 = PerfMeas._dconddhk(
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
                        dconddk = PerfMeas._dconddhk(
                            k11[node],
                            k11[mnode],
                            cl1[jj],
                            cl2[jj],
                            hwva[jj],
                            height1,
                            height2,
                        )
                        SF = PerfMeas._smooth_sat(
                            ihighcellsat,
                            top[node],
                            top[mnode],
                            bot[node],
                            bot[mnode],
                            head[node],
                            head[mnode],
                        )

                    else:
                        dconddk = PerfMeas._dconddhk(
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

    def lam_drhs_dbnd(
        self,
        lamb: np.ndarray,
        head: np.ndarray,
        sp_dict: dict,
        has_flux_pm: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return adjoint-weighted derivatives with respect to boundary values.

        Parameters
        ----------
        lamb : ndarray
            Adjoint state array.
        head : ndarray
            Head array.
        sp_dict : dict
            Stress-package data for a single time step.
        has_flux_pm : bool
            Whether a flux performance measure is active.

        Returns
        -------
        tuple[ndarray, ndarray]
            Adjoint-weighted derivatives with respect to boundary head-like
            terms and conductance terms.
        """
        result_head = np.zeros_like(lamb)
        result_cond = np.zeros_like(lamb)

        # for id in sp_dict:
        for node, bound in zip(sp_dict["node"], sp_dict["bound"]):
            n = node - 1
            boundcond = 1e10
            if len(bound) > 1:
                boundcond = bound[1]
            # the second item in bound should be cond
            result_head[n] = lamb[n] * boundcond
            # Add the direct effect
            if has_flux_pm:
                result_head[n] += boundcond
            # the first item in bound should be head
            lam_drhs_dcond = lamb[n] * bound[0]
            lam_dadcond_h = -1.0 * lamb[n] * head[n]
            result_cond[n] = lam_drhs_dcond + lam_dadcond_h
            # Add the direct effect
            if has_flux_pm:
                result_cond[n] += bound[0] - head[n]

        return result_head, result_cond

    def _pm_available(self, kk) -> bool:
        """Return whether a performance measure entry exists for a given time.

        Parameters
        ----------
        kk : tuple
            Zero-based stress period and time step.

        Returns
        -------
        bool
            True if a performance measure entry exists, otherwise False.
        """

        relevant_entries = [p for p in self._entries if p.kperkstp == kk]
        return len(relevant_entries) > 0

    def _dfdh(self, kk, sol_dataset) -> np.ndarray:
        """Return the derivative of the performance measure with respect to head.

        Parameters
        ----------
        kk : tuple
            Zero-based stress period and time step.
        sol_dataset : h5py.Dataset
            Forward-solution dataset.

        Returns
        -------
        ndarray
            Derivative of the performance measure with respect to head.
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
                if pfr.pm_form == "direct":
                    dfdh[pfr.inode] = pfr.weight
                elif pfr.pm_form == "residual":
                    # Scalar math is fine here, but make sure head is a numpy array
                    dfdh[pfr.inode] = 2.0 * pfr.weight * (head[pfr.inode] - pfr.obsval)
            else:
                # Handle non-head types with caching
                if pfr.pm_type not in cached_maps:
                    # Store hcof and a mapping of inode -> index
                    hcof = sol_dataset[pfr.pm_type]["hcof"][:]
                    nodes = sol_dataset[pfr.pm_type]["nodelist"][:] - 1
                    # Dict comprehension is much faster than np.where inside a loop
                    node_to_idx = {node: i for i, node in enumerate(nodes)}
                    cached_maps[pfr.pm_type] = (hcof, node_to_idx)

                hcof_arr, node_map = cached_maps[pfr.pm_type]

                # Fast O(1) lookup
                if pfr.inode in node_map:
                    idx = node_map[pfr.inode]
                    dfdh[pfr.inode] = hcof_arr[idx]

        return dfdh

    @staticmethod
    def get_value_from_gwf(gwf_name, pak_name, prop_name, gwf) -> Any:
        """Return a copy of a quantity from the MODFLOW 6 API.

        Parameters
        ----------
        gwf_name : str
            Name of the groundwater-flow instance.
        pak_name : str
            Package name from the GWF name file.
        prop_name : str
            Property name in `pak_name`.
        gwf : modflowapi.ModflowApi
            MODFLOW 6 groundwater-flow API instance.

        Returns
        -------
        Any
            Requested quantity.
        """
        addr = gwf.get_var_address(prop_name, gwf_name, pak_name)
        return gwf.get_value(addr)

    @staticmethod
    def get_ptr_from_gwf(gwf_name, pak_name, prop_name, gwf) -> Any:
        """Return a mutable reference to a quantity from the MODFLOW 6 API.

        Parameters
        ----------
        gwf_name : str
            Name of the groundwater-flow instance.
        pak_name : str
            Package name from the GWF name file.
        prop_name : str
            Property name in `pak_name`.
        gwf : modflowapi.ModflowApi
            MODFLOW 6 groundwater-flow API instance.

        Returns
        -------
        Any
            Mutable reference to the requested quantity.
        """
        addr = gwf.get_var_address(prop_name, gwf_name, pak_name)
        return gwf.get_value_ptr(addr)

    @staticmethod
    def get_node(shape, lrc_list) -> list[int]:
        """Return node numbers for a list of layer-row-column indices.

        Parameters
        ----------
        shape : tuple[int, ...]
            Model grid shape.
        lrc_list : list
            Layer-row-column indices.

        Returns
        -------
        list[int]
            Node numbers.
        """

        if not isinstance(lrc_list, list):
            lrc_list = [lrc_list]
        multi_index = tuple(np.array(lrc_list).T)
        return np.ravel_multi_index(multi_index, shape).tolist()

    @staticmethod
    def get_lrc(shape, nodes) -> list[tuple[int, ...]]:
        """Return layer-row-column indices for a list of node numbers.

        Parameters
        ----------
        shape : tuple[int, ...]
            Model grid shape.
        nodes : list
            Node numbers.

        Returns
        -------
        list[tuple[int, ...]]
            Layer-row-column indices.
        """
        if isinstance(nodes, int):
            nodes = [nodes]
        return list(zip(*np.unravel_index(nodes, shape)))

    @staticmethod
    def has_sto_iconvert(gwf) -> bool:
        """Return whether the forward model has an STO package with `ICONVERT`.

        Parameters
        ----------
        gwf : modflowapi.ModflowApi
            MODFLOW 6 groundwater-flow API instance.

        Returns
        -------
        bool
            True if the model has STO `ICONVERT`.
        """
        names = [
            n for n in list(gwf.get_input_var_names()) if "STO" in n and "ICONVERT" in n
        ]
        if len(names) == 0:
            return False
        return True

    @staticmethod
    def _is_singular(A, tol=1e-10) -> bool:
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
