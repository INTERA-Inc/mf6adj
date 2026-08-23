import logging
import pathlib as pl
import shutil
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Optional, Union

import flopy
import h5py
import modflowapi
import numpy as np
import pandas as pd
from xmipy.errors import XMIError

from .pm import PerfMeas, PerfMeasRecord
from .utils.utils import _utils_cd
from .utils.utils_fileio import _write_group_to_hdf
from .utils.utils_logger import _LoggerUtil
from .utils.utils_modflow import (
    get_lrc,
    get_mf6_bound_dict,
    get_model_names_from_mfsim,
    get_package_names_from_gwfname,
    get_ptr_from_gwf,
    has_sto_iconvert,
)
from .utils.utils_pm_read import (
    get_adj_parse_context,
    read_adj_file,
)

DT_FMT = "%Y-%m-%d %H:%M:%S"
PathLike = Union[str, pl.Path]


class Mf6Adj:
    """MODFLOW 6 adjoint solver interface.

    Parameters
    ----------
    adj_filename : str
        Adjoint input filename.
    lib_name : str
        MODFLOW 6 shared library path.
    logging_level : str or int, optional
        Logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, or ``CRITICAL``).
    logging_filename : str or pl.Path, optional
        Optional log filename. If omitted, logging is limited to the console.
    working_directory : str or pl.Path, optional
        Working directory. If omitted, the current directory is used.
    """

    def __init__(
        self,
        adj_filename: str,
        lib_name: str,
        logging_level: Union[int, str] = "INFO",
        logging_filename: Optional[PathLike] = None,
        working_directory: Optional[PathLike] = None,
    ):
        """Initialize the MODFLOW 6 adjoint helper."""

        if working_directory is None:
            working_directory = pl.Path(".").resolve()
        self.working_directory = pl.Path(working_directory).resolve()

        with _utils_cd(self.working_directory):
            adj_filename = pl.Path(adj_filename)
            if not adj_filename.is_file():
                raise Exception(f"adj_filename '{adj_filename}' not found")
            self.adj_filename = adj_filename

            # setup logger
            logger_name = f"{self.__class__.__name__}-{adj_filename.stem}"
            self.logger = _LoggerUtil(
                logger_name,
                logging_level,
                logging_filename,
            )

            # process the flow model
            # make sure the lib exists
            if not pl.Path(lib_name).exists():
                self.logger.logger.warning(
                    f"MODFLOW 6 library file '{lib_name}' not found...continuing..."
                )
            # find the model name
            self._gwf_model_dict, namfile_dict = get_model_names_from_mfsim(".")
            if len(self._gwf_model_dict) != 1:
                raise Exception("only one model is currently supported")
            self._gwf_name = next(iter(self._gwf_model_dict.keys()))
            self._gwf_namfile = namfile_dict[self._gwf_name]
            self._gwf_package_dict = get_package_names_from_gwfname(self._gwf_namfile)
            if self._gwf_model_dict[self._gwf_name] != "gwf6":
                raise Exception(
                    f"model is not a gwf6 type: {self._gwf_model_dict[self._gwf_name]}"
                )
            if "dis6" in self._gwf_package_dict:
                self.logger.logger.info("Structured grid found")
                is_structured = True
                unstructured_type = None
            elif "disv6" in self._gwf_package_dict:
                self.logger.logger.info("Unstructured disv grid found")
                is_structured = False
                unstructured_type = "disv"
            elif "disu6" in self._gwf_package_dict:
                self.logger.logger.info("Unstructured disu grid found")
                is_structured = False
                unstructured_type = "disu"
            else:
                raise Exception("gwf6 model discretization is not dis, disu, or disv.")

            self._gwf = None
            self._lib_name = lib_name
            self._flow_dir = "."
            self._gwf = self._initialize_gwf(lib_name, self._flow_dir)
            self._gwf_version = self._get_gwf_version()
            self._check_solution_coupling()
            self._hdf5_name = None
            self._dt_dict = None

            self._structured_mg = None
            self.is_structured = is_structured
            self.unstructured_type = unstructured_type
            self._shape = None
            if self.is_structured:
                nlay = self._gwf.get_value(
                    self._gwf.get_var_address("NLAY", self._gwf_name.upper(), "DIS")
                )[0]
                nrow = self._gwf.get_value(
                    self._gwf.get_var_address("NROW", self._gwf_name.upper(), "DIS")
                )[0]
                ncol = self._gwf.get_value(
                    self._gwf.get_var_address("NCOL", self._gwf_name.upper(), "DIS")
                )[0]
                self._structured_mg = flopy.discretization.StructuredGrid(
                    nrow=nrow, ncol=ncol, nlay=nlay
                )
                self._shape = (nlay, nrow, ncol)
            self._performance_measures = []
            self._read_adj_file()
            self._gwf_package_types = [
                "chd6",
                "wel6",
                "ghb6",
                "riv6",
                "drn6",
                "sfr6",
                "lak6",
                "rch6",
                "recha6",
                "evt6",
            ]
            self._gwf_boundary_attr_dict = {
                "chd6": ["head"],
                "ghb6": ["bhead", "cond"],
                "riv6": ["stage", "cond"],
                "drn6": ["elev", "cond"],
                "wel6": ["q"],
                "rch6": ["recharge"],
            }

    def _check_solution_coupling(self) -> None:
        """Raise if a package adds its own equations to the solution matrix.

        Raises
        ------
        Exception
            If any package contributes equations to the solution matrix.
        """
        coupled = []
        for package_type, tags in self._gwf_package_dict.items():
            for tag in tags:
                try:
                    npakeq = self._gwf.get_value(
                        self._gwf.get_var_address("NPAKEQ", self._gwf_name, tag.upper())
                    )[0]
                except Exception:
                    # only the advanced packages expose NPAKEQ
                    continue
                if npakeq > 0:
                    coupled.append(f"{tag} ({package_type}, {npakeq} equations)")

        if coupled:
            raise Exception(
                "the adjoint matrix is rebuilt from the groundwater flow grid "
                "connectivity, so packages that add their own equations to the "
                "solution matrix are not supported yet: "
                + f"{', '.join(coupled)}. An implicitly coupled lake (MODFLOW 6 "
                "6.8.0 and later) and maw6 both add equations; use an "
                "explicitly coupled lake and represent multi-aquifer wells "
                "with wel6."
            )

    def _lake_terms(self, tag: str) -> dict:
        """Water-balance terms for an explicitly coupled lake package.

        Parameters
        ----------
        tag : str
            Lake package name from the GWF name file.

        Returns
        -------
        dict
            Stage, previous stage, surface area, and the lake each connection
            belongs to. ``lake_is_constant`` marks a lake held at a fixed
            stage, whose water balance does not constrain the solution.
        """

        def value(name):
            return self._gwf.get_value(
                self._gwf.get_var_address(name, self._gwf_name, tag.upper())
            ).copy()

        # IDXLAKECONN is a one-based pointer into the connection arrays
        idx = value("IDXLAKECONN") - 1
        nlakes = int(value("NLAKES")[0])
        sarea = value("SAREA")

        lake_of_conn = np.zeros(idx[-1], dtype=int)
        surface_area = np.zeros(nlakes)
        for ilak in range(nlakes):
            lake_of_conn[idx[ilak] : idx[ilak + 1]] = ilak
            surface_area[ilak] = sarea[idx[ilak] : idx[ilak + 1]].sum()

        # a lake with ibound < 0 is held at a constant stage
        is_constant = (value("IBOUND") < 0).astype(int)

        noutlets = int(value("NOUTLETS")[0])
        if noutlets > 0:
            # LAKEIN and LAKEOUT are one-based; LAKEOUT of zero or less leaves
            # the model rather than feeding another lake
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
            "lake_noutlets": np.full(nlakes, noutlets, dtype=int),
            "lake_of_conn": lake_of_conn,
            "lake_stage": value("XNEWPAK"),
            "lake_stage_old": value("XOLDPAK"),
            "lake_surface_area": surface_area,
            "lake_is_constant": is_constant,
        }

    def _add_performance_measure(
        self, pm_name: str, pm_entries: list[PerfMeasRecord]
    ) -> None:
        """Validate and register a parsed performance measure.

        Parameters
        ----------
        pm_name : str
            Performance-measure name.
        pm_entries : list[PerfMeasRecord]
            Entries that make up the performance measure.

        Notes
        -----
        A performance measure must contain at least one entry and cannot mix
        ``pm_form`` values. ``head`` and flux-package entries may be combined,
        but a flux entry cannot be part of a ``residual`` measure.
        """
        if len(pm_entries) == 0:
            raise Exception(f"no entries found for PM {pm_name}")

        pm_forms = {entry.pm_form for entry in pm_entries}
        if len(pm_forms) > 1:
            raise Exception(
                f"performance measure {pm_name} has mixed 'pm_forms' "
                + f"({pm_forms}), this is not supported"
            )
        pm_form = next(iter(pm_forms))

        flux_types = {entry.pm_type for entry in pm_entries} - {"head"}
        if flux_types and pm_form == "residual":
            raise Exception(
                f"performance measure {pm_name} has flux 'pm_types' "
                + f"({flux_types}) and is a residual 'pm_form', "
                + "this is not supported"
            )
        if pm_name in [pm._name for pm in self._performance_measures]:
            raise Exception(f"PM {pm_name} multiply defined")

        self._performance_measures.append(
            PerfMeas(
                pm_name,
                pm_entries,
                self.logger.level,
                self.logger,
            )
        )

    def _read_adj_file(self) -> None:
        """Load performance-measure definitions from the adjoint input file.

        This method resets the in-memory performance-measure list, gathers the
        MODFLOW 6 parse context, and delegates the file parsing to
        ``utils_pm_read.read_adj_file``.

        Returns
        -------
        None
            Parsed performance measures are stored on ``self._performance_measures``.
        """
        # clear any existing PMs
        self._performance_measures = []
        self.logger.logger.info(f"Processing adjoint file: {self.adj_filename}")

        nuser, nstp, nper, ncpl = get_adj_parse_context(
            gwf=self._gwf,
            gwf_name=self._gwf_name,
            is_structured=self.is_structured,
            unstructured_type=self.unstructured_type,
        )

        self._hdf5_name = read_adj_file(
            adj_filename=self.adj_filename,
            nuser=nuser,
            nstp=nstp,
            nper=nper,
            ncpl=ncpl,
            is_structured=self.is_structured,
            unstructured_type=self.unstructured_type,
            shape=self._shape,
            gwf_package_dict=self._gwf_package_dict,
            record_factory=PerfMeasRecord,
            add_performance_measure=self._add_performance_measure,
            current_hdf5_name=self._hdf5_name,
            logger=self.logger.logger,
        )

    def _open_hdf(self, tag: Optional[PathLike]) -> h5py.File:
        """Open an HDF5 file for writing.

        Parameters
        ----------
        tag : str or PathLike, optional
            Filename or filename prefix.

        Returns
        -------
        h5py.File
            Open HDF5 file handle. The resolved filename is also stored on
            ``self._hdf5_name``.
        """
        if tag is None:
            fname = (
                self._gwf_name
                + "_"
                + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + ".hd5"
            )
        else:
            fname = tag
        fname = pl.Path(fname)
        self._hdf5_name = fname
        if fname.exists():
            fname.unlink()
        f = h5py.File(fname, "w")
        return f

    def _add_gwf_info_to_hdf(self, hdf: h5py.File) -> None:
        """Add model structure and metadata to an HDF5 file.

        Parameters
        ----------
        hdf : h5py.File
            Open HDF5 file handle.

        Notes
        -----
        The written group includes grid geometry, connectivity arrays,
        hydraulic-property arrays, and selected MODFLOW 6 metadata needed by
        later adjoint-processing steps.

        Returns
        -------
        None
            This method writes data in place to the supplied HDF5 handle.
        """
        gwf_name = self._gwf_name
        gwf = self._gwf
        has_sto = has_sto_iconvert(gwf)
        data_dict = {}

        dis_pak = "DIS"

        ihc = get_ptr_from_gwf(gwf_name, "CON", "IHC", gwf)
        data_dict["ihc"] = ihc
        ia = get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
        data_dict["ia"] = ia
        ja = get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1
        data_dict["ja"] = ja
        jas = get_ptr_from_gwf(gwf_name, "CON", "JAS", gwf) - 1
        data_dict["jas"] = jas
        cl1 = get_ptr_from_gwf(gwf_name, "CON", "CL1", gwf)
        data_dict["cl1"] = cl1
        cl2 = get_ptr_from_gwf(gwf_name, "CON", "CL2", gwf)
        data_dict["cl2"] = cl2
        hwva = get_ptr_from_gwf(gwf_name, "CON", "HWVA", gwf)
        data_dict["hwva"] = hwva
        top = get_ptr_from_gwf(gwf_name, dis_pak, "TOP", gwf)
        data_dict["top"] = top
        bot = get_ptr_from_gwf(gwf_name, dis_pak, "BOT", gwf)
        data_dict["bot"] = bot
        iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
        data_dict["iac"] = iac
        icelltype = get_ptr_from_gwf(gwf_name, "NPF", "ICELLTYPE", gwf)
        data_dict["icelltype"] = icelltype
        # IHIGHCELLSAT is not registered by MODFLOW 6 6.6.3 and earlier
        try:
            ihighcellsat = get_ptr_from_gwf(
                gwf_name,
                "NPF",
                "IHIGHCELLSAT",
                gwf,
            )
        except XMIError:
            ihighcellsat = np.array([0], dtype=int)
        ihighcellsat_value = int(np.asarray(ihighcellsat).ravel()[0])
        if ihighcellsat_value != 0:
            self.logger.logger.info("HIGHEST_CELL_SATURATION option specified")
        data_dict["ihighcellsat"] = ihighcellsat

        area = get_ptr_from_gwf(gwf_name, dis_pak, "AREA", gwf)
        data_dict["area"] = area
        if has_sto:
            iconvert = get_ptr_from_gwf(gwf_name, "STO", "ICONVERT", gwf)
            data_dict["iconvert"] = iconvert
            storage = get_ptr_from_gwf(gwf_name, "STO", "SS", gwf)
            data_dict["storage"] = storage
            sy = get_ptr_from_gwf(gwf_name, "STO", "SY", gwf)
            data_dict["sy"] = sy
        nodeuser = get_ptr_from_gwf(gwf_name, dis_pak, "NODEUSER", gwf) - 1
        data_dict["nodeuser"] = nodeuser
        nodereduced = get_ptr_from_gwf(gwf_name, dis_pak, "NODEREDUCED", gwf) - 1
        data_dict["nodereduced"] = nodereduced
        ndim = get_ptr_from_gwf(gwf_name, dis_pak, "NDIM", gwf)
        data_dict["ndim"] = ndim
        nnodes = get_ptr_from_gwf(gwf_name, "CON", "NODES", gwf)
        data_dict["nnodes"] = nnodes
        idomain = get_ptr_from_gwf(gwf_name, "DIS", "IDOMAIN", gwf)
        data_dict["idomain"] = idomain

        if self.is_structured:
            nlay = get_ptr_from_gwf(gwf_name, dis_pak, "NLAY", gwf)
            data_dict["nlay"] = nlay
            nrow = get_ptr_from_gwf(gwf_name, dis_pak, "NROW", gwf)
            data_dict["nrow"] = nrow
            ncol = get_ptr_from_gwf(gwf_name, dis_pak, "NCOL", gwf)
            data_dict["ncol"] = ncol

        _write_group_to_hdf(
            hdf,
            "gwf_info",
            data_dict,
            attr_dict=self._gwf_package_dict,
            logger=self.logger.logger,
        )

    def _dresdss_h(
        self,
        head: np.ndarray,
        head_old: np.ndarray,
        dt: float,
        sat: np.ndarray,
        sat_old: np.ndarray,
    ) -> np.ndarray:
        """Return the storage contribution to the residual derivative.

        This term represents the derivative of the transient groundwater-flow
        residual with respect to specific storage for the current time step. It
        accounts for partially saturated conditions by forcing confined cells
        (``ICONVERT == 0``) to remain fully saturated in the calculation.

        Parameters
        ----------
        head : ndarray
            Current heads.
        head_old : ndarray
            Heads from the previous solution step.
        dt : float
            Length of the current solution step in model time.
        sat : ndarray
            Current saturation.
        sat_old : ndarray
            Saturation from the previous solution step.

        Returns
        -------
        ndarray
            Per-cell derivative term used later in the adjoint sensitivity
            calculation for specific storage.
        """
        top = get_ptr_from_gwf(self._gwf_name, "DIS", "TOP", self._gwf)
        bot = get_ptr_from_gwf(self._gwf_name, "DIS", "BOT", self._gwf)
        area = get_ptr_from_gwf(self._gwf_name, "DIS", "AREA", self._gwf)
        iconvert = get_ptr_from_gwf(self._gwf_name, "STO", "ICONVERT", self._gwf)

        # handle iconvert
        sat_mod = sat.copy()
        sat_mod[iconvert == 0] = 1.0
        sat_old_mod = sat_old.copy()
        sat_old_mod[iconvert == 0] = 1.0

        height = top - bot

        # result = np.zeros_like(head)
        dSC1 = area * height
        result = (
            (dSC1 / dt) * (sat_old_mod * head_old - sat_mod * head)
            + (dSC1 / dt) * bot * (sat_mod - sat_old_mod)
            + (dSC1 / (2.0 * dt)) * height * (sat_mod**2 - sat_old_mod**2)
        )
        # zero out dry cells
        result[head <= bot] = 0.0
        result[head_old <= bot] = 0.0

        return result

    def _drhsdh(
        self,
        dt: float,
        sat_old: np.ndarray,
    ) -> np.ndarray:
        """Return the head derivative of the storage-related right-hand side.

        The derivative combines confined-storage and specific-yield terms for
        the previous time step. Specific-yield contributions are suppressed in
        fully saturated cells.

        Parameters
        ----------
        dt : float
            Length of the current solution step in model time.
        sat_old : ndarray
            Saturation from the previous solution step.

        Returns
        -------
        ndarray
            Per-cell derivative of the storage-related right-hand side with
            respect to head.
        """
        area = get_ptr_from_gwf(self._gwf_name, "DIS", "AREA", self._gwf)

        # specific storage
        top = get_ptr_from_gwf(self._gwf_name, "DIS", "TOP", self._gwf)
        bot = get_ptr_from_gwf(self._gwf_name, "DIS", "BOT", self._gwf)
        storage = get_ptr_from_gwf(self._gwf_name, "STO", "SS", self._gwf)

        # specific yield
        iconvert = get_ptr_from_gwf(self._gwf_name, "STO", "ICONVERT", self._gwf)
        sy = get_ptr_from_gwf(self._gwf_name, "STO", "SY", self._gwf)
        sat_old_mod = sat_old.copy()
        sat_old_mod[iconvert == 0] = 1.0
        sy_mod = sy.copy()
        sy_mod[sat_old_mod == 1.0] = 0.0

        # calculate drhsdh
        drhsdh = -1.0 * area * (storage * (top - bot) + sy_mod) / dt

        return drhsdh

    def solve_forward_model(
        self,
        verbose: bool = True,
        force_k_update: bool = False,
        sp_pert_dict: dict | None = None,
        pert_save: bool = False,
        hdf5_name: PathLike | None = None,
        solve_func_ptr: Callable[[modflowapi.ModflowApi], None] | None = None,
        presolve_func_ptr: Callable[[modflowapi.ModflowApi], None] | None = None,
        postsolve_func_ptr: Callable[[modflowapi.ModflowApi], None] | None = None,
    ) -> tuple[dict, dict] | None:
        """Solve the forward model and store adjoint inputs in HDF5.

        The forward model is run for all MODFLOW 6 time steps, and the
        solution components needed for the adjoint solve are harvested and
        written to the HDF5 file.

        Parameters
        ----------
        verbose : bool, optional
            Whether to report progress to stdout.
        force_k_update : bool, optional
            Force MODFLOW 6 to reprocess the ``K`` and ``K33`` arrays. Used only
            during perturbation testing.
        sp_pert_dict : dict, optional
            Perturbed boundary information used during perturbation testing.
        pert_save : bool, optional
            Save additional information for perturbation testing.
        hdf5_name : PathLike, optional
            Output HDF5 filename for forward-solution components. If omitted,
            a generic time-stamped filename is created.
        solve_func_ptr : callable, optional
            Callback invoked with the ``modflowapi.ModflowApi`` instance before
            each solver iteration. If omitted, no callback is run.
        presolve_func_ptr : callable, optional
            Callback invoked with the ``modflowapi.ModflowApi`` instance at the
            start of each time step, before the solve loop begins. If omitted,
            no callback is run.
        postsolve_func_ptr : callable, optional
            Callback invoked with the ``modflowapi.ModflowApi`` instance after
            each time step is finalized. If omitted, no callback is run.

        Returns
        -------
        tuple[dict, dict] or None
            Perturbation-testing data when ``pert_save`` is ``True``; otherwise
            ``None``.
        """
        with _utils_cd(self.working_directory):
            if self._gwf is None:
                raise Exception("gwf is None")
            if hdf5_name is not None:
                self._hdf5_name = hdf5_name
            fhd = self._open_hdf(self._hdf5_name)
            sim_start = datetime.now()

            self.logger.logger.info("Starting flow solution")

            # get current sim time
            ctime = self._gwf.get_current_time()
            # get ending sim time
            etime = self._gwf.get_end_time()
            # max number of iterations
            max_iter = self._gwf.get_value(self._gwf.get_var_address("MXITER", "SLN_1"))
            # let's do it!
            num_fails = 0

            sat_old = None
            visited = []
            ctimes = []
            dts = []
            kpers, kstps = [], []

            nnode = self._gwf.get_value(
                self._gwf.get_var_address("NODES", self._gwf_name, "DIS")
            )[0]

            is_newton = self._gwf.get_value(
                self._gwf.get_var_address("INEWTON", self._gwf_name)
            )[0]
            has_sto = False
            if has_sto_iconvert(self._gwf):
                has_sto = True

            sp_package_data = None
            head_dict = None
            dt_dict = None
            if pert_save:
                sp_package_data = {}
                head_dict = {}
                # time step lengths, used to weight an instantaneous measure
                dt_dict = {}

            while ctime < etime:
                sol_start = datetime.now()
                # the length of this sim time
                dt = self._gwf.get_time_step()
                # prep the current time step
                self._gwf.prepare_time_step(dt)

                kiter = 0
                # prep to solve
                stress_period = self._gwf.get_value(
                    self._gwf.get_var_address("KPER", "TDIS")
                )[0]
                time_step = self._gwf.get_value(
                    self._gwf.get_var_address("KSTP", "TDIS")
                )[0]
                kper, kstp = stress_period - 1, time_step - 1
                kperkstp = (kper, kstp)

                # this is to force mf6 to update cond sat using the k11 and k33 arrays
                # which is needed for the perturbation testing
                if kper == 0 and kstp == 0 and force_k_update:
                    kchangeper = self._gwf.get_value_ptr(
                        self._gwf.get_var_address("KCHANGEPER", self._gwf_name, "NPF")
                    )
                    kchangestp = self._gwf.get_value_ptr(
                        self._gwf.get_var_address("KCHANGESTP", self._gwf_name, "NPF")
                    )
                    kchangestp[0] = time_step
                    kchangeper[0] = stress_period
                    nodekchange = self._gwf.get_value_ptr(
                        self._gwf.get_var_address("NODEKCHANGE", self._gwf_name, "NPF")
                    )
                    nodekchange[:] = 1

                # apply any boundary condition perturbation info
                if sp_pert_dict is not None:
                    if sp_pert_dict["kperkstp"] == kperkstp:
                        for pert_item in self._gwf_boundary_attr_dict[
                            sp_pert_dict["packagetype"]
                        ]:
                            if pert_item not in sp_pert_dict:
                                self.logger.logger.info(
                                    f"pert_item '{pert_item}' not in sp_pert_dict"
                                )
                                continue
                            addr = [
                                pert_item.upper(),
                                self._gwf_name,
                                sp_pert_dict["packagename"].upper(),
                            ]
                            wbaddr = self._gwf.get_var_address(*addr)
                            bnd_ptr = self._gwf.get_value_ptr(wbaddr)
                            wbaddr = self._gwf.get_var_address(
                                "NODELIST",
                                self._gwf_name,
                                sp_pert_dict["packagename"].upper(),
                            )
                            nodelist = self._gwf.get_value_ptr(wbaddr)
                            idx = np.where(nodelist == sp_pert_dict["node"])[0]
                            if idx.shape[0] == 0:
                                print(nodelist)
                                raise Exception(
                                    "sp pert dict node not found :" + str(sp_pert_dict)
                                )
                            bnd_ptr[idx] = sp_pert_dict[pert_item]

                if presolve_func_ptr is not None:
                    presolve_func_ptr(self._gwf)

                self._gwf.prepare_solve(1)
                if sat_old is None:
                    sat_old = self._gwf.get_value(
                        self._gwf.get_var_address("SAT", self._gwf_name, "NPF")
                    )

                # solve until converged
                while kiter < max_iter:
                    if solve_func_ptr is not None:
                        solve_func_ptr(self._gwf)
                    convg = self._gwf.solve(1)
                    if convg:
                        td = (datetime.now() - sol_start).total_seconds() / 60.0
                        if verbose:
                            self.logger.logger.info(
                                f"Flow (stress period,time step) ({stress_period},"
                                + f"{time_step}) converged in {kiter} iters, took "
                                + f"{td:10.5G} mins"
                            )
                        break
                    kiter += 1

                if not convg:
                    td = (datetime.now() - sol_start).total_seconds() / 60.0
                    if verbose:
                        self.logger.logger.info(
                            f"Flow stress period,time step {stress_period},{time_step} "
                            + f"did not converge, {kiter} iters, took {td:10.5G} mins"
                        )
                    num_fails += 1
                try:
                    self._gwf.finalize_solve(1)
                except Exception as e:
                    print(f"{e}\n\nCould not execute finalize_solve()")

                self._gwf.finalize_time_step()
                if postsolve_func_ptr is not None:
                    postsolve_func_ptr(self._gwf)
                # update current sim time
                ctime = self._gwf.get_current_time()
                dt1 = self._gwf.get_time_step()

                ctimes.append(ctime)
                dts.append(dt1)
                kpers.append(kper)
                kstps.append(kstp)

                if kperkstp in visited:
                    raise Exception(f"{kperkstp} already visited")
                visited.append(kperkstp)

                amat = self._gwf.get_value(
                    self._gwf.get_var_address("AMAT", "SLN_1")
                ).copy()
                data_dict = {"amat": amat}

                residual = self._gwf.get_value(
                    self._gwf.get_var_address("D", "SLN_1", "IMSLINEAR")
                ).copy()
                data_dict["residual"] = residual

                head = self._gwf.get_value(
                    self._gwf.get_var_address("X", self._gwf_name.upper())
                )[:nnode]
                data_dict["head"] = head
                if pert_save:
                    head_dict[kperkstp] = head
                    dt_dict[kperkstp] = dt1

                head_old = self._gwf.get_value(
                    self._gwf.get_var_address("XOLD", self._gwf_name.upper())
                )[:nnode]
                data_dict["head_old"] = head_old

                k11 = self._gwf.get_value(
                    self._gwf.get_var_address("K11", self._gwf_name.upper(), "NPF")
                )
                data_dict["k11"] = k11
                k33 = self._gwf.get_value(
                    self._gwf.get_var_address("K33", self._gwf_name.upper(), "NPF")
                )
                data_dict["k33"] = k33
                condsat = self._gwf.get_value(
                    self._gwf.get_var_address("CONDSAT", self._gwf_name.upper(), "NPF")
                )
                data_dict["condsat"] = condsat

                iss = self._gwf.get_value(
                    self._gwf.get_var_address("ISS", self._gwf_name.upper())
                )
                data_dict["iss"] = iss

                sat = self._gwf.get_value(
                    self._gwf.get_var_address("SAT", self._gwf_name, "NPF")
                )
                data_dict["sat"] = sat
                data_dict["sat_old"] = sat_old

                sat_old = sat.copy()
                if has_sto:  # has storage
                    dresdss_h = self._dresdss_h(head, head_old, dt1, sat, sat_old)
                    data_dict["dresdss_h"] = dresdss_h
                    drhsdh = self._drhsdh(dt1, sat_old)
                    data_dict["drhsdh"] = drhsdh
                else:
                    data_dict["drhsdh"] = np.zeros_like(sat_old)

                for package_type in self._gwf_package_types:
                    if package_type in self._gwf_package_dict:
                        if pert_save and package_type not in sp_package_data:
                            sp_package_data[package_type] = {}
                        for tag in self._gwf_package_dict[package_type]:
                            nbound = self._gwf.get_value(
                                self._gwf.get_var_address(
                                    "NBOUND", self._gwf_name, tag.upper()
                                )
                            )[0]
                            if nbound > 0:
                                if (
                                    pert_save
                                    and kperkstp in sp_package_data[package_type]
                                ):
                                    if len(self._gwf_package_dict[package_type]) == 1:
                                        raise Exception(
                                            f"kperkstp '{kperkstp}' already in "
                                            + "sp_package_data"
                                        )
                                    else:
                                        pass
                                elif pert_save:
                                    sp_package_data[package_type][kperkstp] = []
                                nodelist = self._gwf.get_value(
                                    self._gwf.get_var_address(
                                        "NODELIST", self._gwf_name, tag.upper()
                                    )
                                )
                                bound = self._gwf.get_value(
                                    self._gwf.get_var_address(
                                        "BOUND", self._gwf_name, tag.upper()
                                    )
                                )
                                hcof = self._gwf.get_value(
                                    self._gwf.get_var_address(
                                        "HCOF", self._gwf_name, tag.upper()
                                    )
                                )
                                rhs = self._gwf.get_value(
                                    self._gwf.get_var_address(
                                        "RHS", self._gwf_name, tag.upper()
                                    )
                                )

                                simvals = self._gwf.get_value(
                                    self._gwf.get_var_address(
                                        "SIMVALS", self._gwf_name, tag.upper()
                                    )
                                )
                                bnd_attrs = {}
                                if package_type in self._gwf_boundary_attr_dict:
                                    attrs = self._gwf_boundary_attr_dict[package_type]
                                    # MODFLOW 6 keeps these packages' values in
                                    # their own arrays and leaves BOUND zeroed,
                                    # so always rebuild bound from the arrays
                                    bound = np.zeros((len(nodelist), len(attrs)))
                                    for i, attr in enumerate(attrs):
                                        vals = self._gwf.get_value(
                                            self._gwf.get_var_address(
                                                attr.upper(),
                                                self._gwf_name,
                                                tag.upper(),
                                            )
                                        )
                                        bnd_attrs[attr] = vals
                                        bound[:, i] = vals

                                if package_type == "sfr6":
                                    tag = self._gwf_package_dict[package_type][0]
                                    stage = self._gwf.get_value(
                                        self._gwf.get_var_address(
                                            "STAGE", self._gwf_name, tag.upper()
                                        )
                                    )
                                    bound[:, 0] = stage
                                    bound[:, 1] = -1.0 * hcof

                                if pert_save:
                                    for i in range(nbound):
                                        # note bound is an array!
                                        pak_data = {
                                            "node": nodelist[i],
                                            "bound": bound[i],
                                            "hcof": hcof[i],
                                            "rhs": rhs[i],
                                            "packagename": tag,
                                            "simval": simvals[i],
                                        }
                                        for key, val in bnd_attrs.items():
                                            pak_data[key] = val[i]
                                        sp_package_data[package_type][kperkstp].append(
                                            pak_data
                                        )
                                data_dict[tag] = {
                                    "ptype": package_type,
                                    "nodelist": nodelist,
                                    "bound": bound,
                                    "hcof": hcof,
                                    "rhs": rhs,
                                    "simvals": simvals,
                                }
                                for key, val in bnd_attrs.items():
                                    assert key not in data_dict[tag], (
                                        f"boundary attribute '{key}' already in "
                                        + f"data dict for {tag}"
                                    )
                                    data_dict[tag][key] = val
                                if package_type == "lak6":
                                    # the hdf writer nests one level, so these
                                    # sit alongside the package arrays
                                    data_dict[tag].update(self._lake_terms(tag))
                attr_dict = {
                    "ctime": ctime,
                    "dt": dt1,
                    "kper": kper,
                    "kstp": kstp,
                    "is_newton": is_newton,
                    "has_sto": has_sto,
                }
                _write_group_to_hdf(
                    fhd,
                    group_name=f"solution_kper:{kper:05d}_kstp:{kstp:05d}",
                    data_dict=data_dict,
                    attr_dict=attr_dict,
                    logger=self.logger.logger,
                )

            sim_end = datetime.now()
            td = (sim_end - sim_start).total_seconds() / 60.0
            if verbose:
                self.logger.logger.info(
                    f"Flow solution finished and took {td:10.5G} minutes"
                )
                if num_fails > 0:
                    self.logger.logger.info(
                        f"Flow solution failed to converge {num_fails} times"
                    )

            _write_group_to_hdf(
                fhd,
                "aux",
                {"totime": ctimes, "dt": dts, "kper": kpers, "kstp": kstps},
                logger=self.logger.logger,
            )
            self._add_gwf_info_to_hdf(fhd)
            fhd.close()
            if pert_save:
                self._dt_dict = dt_dict
                return head_dict, sp_package_data

    def solve_adjoint(
        self,
        hdf5_adjoint_solution_fname: Optional[PathLike] = None,
        csv_summary: bool = False,
        linear_solver=None,
        linear_solver_kwargs: dict = {},
        use_precon: bool = True,
        jacobi_preconditioner: Optional[str] = None,
        precon_kwargs: dict = {},
        singular_test: bool = False,
        tikhonov: float = 0.0,
        dvclose: Optional[float] = 1e-6,
        rclose: Optional[float] = 1e-3,
        dvscale: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Solve for the adjoint state, one performance measure at a time.

        Parameters
        ----------
        hdf5_adjoint_solution_fname : PathLike, optional
            HDF5 file to write the adjoint solution. If omitted, a default
            name based on the performance-measure name is used.
        csv_summary : bool, optional
            Write a CSV summary of the sensitivity information.
        linear_solver : str or callable, optional
            Sparse linear solver to use. If ``None``, either a direct solver or
            ``bicgstab`` is selected based on model size. If a string, supported
            values are ``"direct"`` and ``"bicgstab"``. A compatible solver
            callable may also be supplied.
        linear_solver_kwargs : dict, optional
            Keyword arguments passed to ``linear_solver``.
        use_precon : bool, optional
            Use a preconditioner with the iterative linear solver.
        jacobi_preconditioner : str, optional
            Use Jacobi preconditioner with the iterative linear solver. If ``None``, the
            ILU preconditioner will be used. If a string, supported values are
            ``"point"`` and ``"block"``.
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

        Returns
        -------
        dict[str, DataFrame]
            Composite sensitivity summaries keyed by performance-measure name.
        """
        generate_name = hdf5_adjoint_solution_fname is None

        dfs = {}
        with _utils_cd(self.working_directory):
            if self._hdf5_name is None or not pl.Path(self._hdf5_name).exists():
                raise Exception("need to call solve_forward_model() first")

            for pm in self._performance_measures:
                if generate_name:
                    hdf5_name = pl.Path(self._hdf5_name)
                    path = hdf5_name.parent
                    extension = hdf5_name.suffix
                    hdf5_adjoint_solution_fname = (
                        path / f"adjoint_solution_{pm.name}{extension}"
                    )

                df = pm.solve_adjoint(
                    hdf5_forward_solution_fname=self._hdf5_name,
                    hdf5_adjoint_solution_fname=hdf5_adjoint_solution_fname,
                    csv_summary=csv_summary,
                    linear_solver=linear_solver,
                    linear_solver_kwargs=linear_solver_kwargs,
                    jacobi_preconditioner=jacobi_preconditioner,
                    use_precon=use_precon,
                    precon_kwargs=precon_kwargs,
                    singular_test=singular_test,
                    tikhonov=tikhonov,
                    dvclose=dvclose,
                    rclose=rclose,
                    dvscale=dvscale,
                )
                dfs[pm.name] = df
        return dfs

    def _initialize_gwf(self, lib_name: str, sim_ws: PathLike) -> modflowapi.ModflowApi:
        """Initialize the MODFLOW 6 API.

        Parameters
        ----------
        lib_name : str
            MODFLOW 6 shared library filename.
        sim_ws : PathLike
            Simulation directory containing the shared library file.

        Returns
        -------
        modflowapi.ModflowApi
            Initialized MODFLOW 6 API instance for the requested simulation
            workspace.
        """
        # instantiate the flow model api
        if self._gwf is not None:
            self._gwf.finalize()
            self._gwf = None
        sim_ws = pl.Path(sim_ws)
        gwf = modflowapi.ModflowApi(
            str(sim_ws / lib_name), working_directory=str(sim_ws)
        )
        gwf.initialize()
        return gwf

    def _get_gwf_version(self) -> str:
        """Return the MODFLOW 6 version number.

        Returns
        -------
        str
            MODFLOW 6 version string reported by the API, or an empty string
            if the API call fails.
        """
        # get_version overruns the version buffer in MODFLOW 6 6.7.0 and
        # earlier, which can raise an access violation on Windows
        try:
            version = self._gwf.get_version()
        except OSError as e:
            self.logger.logger.warning(f"Could not get the MODFLOW 6 version: {e}")
            return ""
        self.logger.logger.info(f"MODFLOW 6 version: {version}")
        return version

    def finalize(self) -> None:
        """Close the API and file handles."""
        self.logger.logger.info(f"Finalizing {self.__class__.__name__}")
        try:
            self._gwf.finalize()
        except Exception as e:
            print(f"{e}\n\nCould not execute finalize()")
        self._gwf = None

    def perturbation_method(self, pert_mult: float = 1.01) -> pd.DataFrame:
        """Calculate finite-difference perturbation sensitivities.

        This utility perturbs active parameter entries one at a time, reruns the
        forward simulation, and estimates sensitivity using a forward-difference
        quotient. The resulting sensitivities can be compared directly with
        adjoint-based sensitivities for verification during development.

        Parameters
        ----------
        pert_mult : float, default 1.01
            Multiplicative perturbation factor applied to each parameter value.
            The perturbation increment is computed as ``epsilon = value *
            (pert_mult - 1.0)``.

        Returns
        -------
        DataFrame
            Dataframe with one row per perturbation and columns containing
            perturbation metadata (for example parameter, index, epsilon, and
            node/layer labels where applicable) plus finite-difference
            sensitivities for each configured performance measure.
        """

        working_directory = pl.Path(self.working_directory)
        self._gwf = self._initialize_gwf(self._lib_name, working_directory)
        self._gwf_version = self._get_gwf_version()

        gwf_name = self._gwf_name.upper()

        org_head, org_sp_package_data = self.solve_forward_model(pert_save=True)
        # tot = 0
        # for d in org_sp_package_data["ghb6"][(0, 0)]:
        #     # print(d)
        #     tot += d["simval"]
        dt_dict = self._dt_dict
        base_results = {
            pm.name: pm._performance_measure_forward(
                org_head, org_sp_package_data, dt_dict
            )
            for pm in self._performance_measures
        }
        assert len(base_results) == len(self._performance_measures)

        addr = ["NODEUSER", gwf_name, "DIS"]
        wbaddr = self._gwf.get_var_address(*addr)
        nuser = self._gwf.get_value(wbaddr) - 1
        if len(nuser) == 1:
            nuser = np.arange(org_head[next(iter(org_head.keys()))].shape[0], dtype=int)

        kijs = None
        nlay = 1
        if self.is_structured or self.unstructured_type == "disv":
            addr = ["NLAY", gwf_name, "DIS"]
            wbaddr = self._gwf.get_var_address(*addr)
            nlay = self._gwf.get_value(wbaddr)[0]

        if self.is_structured:
            kijs = get_lrc(self._shape, list(nuser))
            kijs = dict(zip(nuser, kijs))

        def _compute_perturbation_results(
            pert_head: dict,
            pert_sp_dict: dict,
            epsilon: float,
        ) -> dict[str, float]:
            """Return finite-difference perturbation responses for each measure.

            Parameters
            ----------
            pert_head : dict
                Perturbed head results by time step.
            pert_sp_dict : dict
                Perturbed stress-package data by time step.
            epsilon : float
                Perturbation magnitude.

            Returns
            -------
            dict[str, float]
                Perturbation responses indexed by performance-measure name.
            """
            return {
                pm.name: (
                    pm._performance_measure_forward(pert_head, pert_sp_dict, dt_dict)
                    - base_results[pm.name]
                )
                / epsilon
                for pm in self._performance_measures
            }

        def _add_spatial_labels(df: pd.DataFrame) -> pd.DataFrame:
            """Add structured-grid spatial labels to a perturbation summary.

            Parameters
            ----------
            df : DataFrame
                Perturbation summary indexed by node number.

            Returns
            -------
            DataFrame
                Input summary with optional ``k``, ``i``, and ``j`` columns.
            """
            if kijs is not None:
                for idx, lab in zip([0, 1, 2], ["k", "i", "j"]):
                    df.loc[:, lab] = df.index.map(lambda x: kijs[x][idx])
            return df

        dfs = []

        # boundary condition perturbations
        _ = get_mf6_bound_dict()

        for paktype, pdict in org_sp_package_data.items():
            if paktype == "chd6":
                continue
            pert_items = self._gwf_boundary_attr_dict[paktype]
            epsilons = []
            node_ids = []
            names = []
            pert_results_dict = {pm.name: [] for pm in self._performance_measures}
            self.logger.logger.info(f"Running perturbations for {paktype}")
            for kk, infolist in pdict.items():
                for ibnd, infodict in enumerate(infolist):
                    for pert_item in pert_items:
                        new_bound = infodict[pert_item].copy()
                        delt = new_bound * pert_mult
                        epsilon = delt - new_bound
                        epsilons.append(epsilon)
                        new_bound = delt
                        pakname = infodict["packagename"]
                        pert_dict = {
                            "kperkstp": kk,
                            "packagename": pakname,
                            "node": infodict["node"],
                            pert_item: new_bound,
                            "packagetype": paktype,
                        }
                        self._gwf = self._initialize_gwf(
                            self._lib_name,
                            working_directory,
                        )
                        pert_head, pert_sp_dict = self.solve_forward_model(
                            verbose=False, sp_pert_dict=pert_dict, pert_save=True
                        )
                        pert_results = _compute_perturbation_results(
                            pert_head,
                            pert_sp_dict,
                            epsilon,
                        )
                        for pm_name, result in pert_results.items():
                            pert_results_dict[pm_name].append(result)
                        node_ids.append(infodict["node"])
                        if paktype == "wel6":
                            names.append("wel6_q")
                        elif paktype == "rch6":
                            names.append("rch6_recharge")
                        else:
                            names.append(pakname + "_" + pert_item + f"_{ibnd}")

            if not epsilons:
                continue

            df = pd.DataFrame(pert_results_dict)
            df.loc[:, "node"] = node_ids
            df.loc[:, "epsilon"] = epsilons
            df.loc[:, "addr"] = names
            df.index = df.pop("node") - 1
            df = df.loc[df.index != -1, :]
            df.index = df.index.map(lambda x: nuser[x])
            df.index.name = "node"

            df = _add_spatial_labels(df)

            agg_map = dict.fromkeys(pert_results_dict, "sum")
            for col in ["epsilon", "k", "i", "j"]:
                if col in df.columns:
                    agg_map[col] = "first"

            gdf = (
                df.reset_index()
                .groupby(["node", "addr"], as_index=False)
                .agg(agg_map)
                .set_index("node")
            )
            dfs.append(gdf)

        # property perturbations
        addresses = [["K11", gwf_name, "NPF"]]
        if nlay > 1:
            addresses.append(["K33", gwf_name, "NPF"])

        has_sto = False
        if has_sto_iconvert(self._gwf):
            has_sto = True

        wbaddr = self._gwf.get_var_address(*addresses[0])
        inodes = self._gwf.get_value_ptr(wbaddr).shape[0]

        for addr in addresses:
            self.logger.logger.info(f"Running perturbations for {addr}")
            pert_results_dict = {pm.name: [] for pm in self._performance_measures}
            wbaddr = self._gwf.get_var_address(*addr)

            epsilons = []

            for inode in range(inodes):
                self._gwf = self._initialize_gwf(self._lib_name, self.working_directory)
                pert_arr = self._gwf.get_value_ptr(wbaddr)
                org = pert_arr[inode]
                delt = org * pert_mult
                epsilon = delt - pert_arr[inode]
                epsilons.append(epsilon)
                pert_arr[inode] = delt
                pert_head, pert_sp_dict = self.solve_forward_model(
                    verbose=False, force_k_update=True, pert_save=True
                )
                pert_results = _compute_perturbation_results(
                    pert_head,
                    pert_sp_dict,
                    epsilon,
                )
                for pm_name, result in pert_results.items():
                    pert_results_dict[pm_name].append(result)

            df = pd.DataFrame(pert_results_dict)
            df.index = [nuser[inode] for inode in range(inodes)]
            df.index.name = "node"
            df.loc[:, "epsilon"] = epsilons
            df = _add_spatial_labels(df)
            tag = "_".join(addr).lower()
            df.loc[:, "addr"] = tag
            dfs.append(df)

        if has_sto:
            test_dir = working_directory / "_pert_temp"
            if pl.Path(test_dir).exists():
                shutil.rmtree(test_dir)
            sim = flopy.mf6.MFSimulation.load(sim_ws=working_directory)
            gwf = sim.get_model()
            ss = gwf.sto.ss.array.copy().flatten()
            # this is an attempt to make sure we aren't using "layered"
            gwf.sto.ss = ss

            sim.set_sim_path(test_dir)
            sim.set_all_data_external()
            sim.write_simulation()
            ss_arr_name = pl.Path(test_dir) / f"{gwf.name}.sto_ss.txt"
            if not ss_arr_name.exists():
                raise Exception(
                    "couldn't find ss_arr_name '{0}' needed for BS super hack"
                )

            self.logger.logger.info(
                "Running manual flopy based perturbations for sto ss"
            )
            pert_results_dict = {pm.name: [] for pm in self._performance_measures}
            epsilons = []

            for inode in range(inodes):
                arr_node = nuser[inode]
                pert_arr = ss.copy()
                org = ss[arr_node]
                delt = org * pert_mult
                epsilon = delt - pert_arr[arr_node]
                epsilons.append(epsilon)
                pert_arr[arr_node] = delt

                # reset the ss property
                np.savetxt(ss_arr_name, pert_arr.flatten(), fmt="%15.6E")

                self._gwf = self._initialize_gwf(self._lib_name, test_dir)
                pert_head, pert_sp_dict = self.solve_forward_model(
                    verbose=False, force_k_update=True, pert_save=True
                )
                pert_results = _compute_perturbation_results(
                    pert_head,
                    pert_sp_dict,
                    epsilon,
                )
                for pm_name, result in pert_results.items():
                    pert_results_dict[pm_name].append(result)
            df = pd.DataFrame(pert_results_dict)
            df.index = [nuser[inode] for inode in range(inodes)]
            df.index.name = "node"
            df.loc[:, "epsilon"] = epsilons
            df = _add_spatial_labels(df)
            tag = "sto_ss"
            df.loc[:, "addr"] = tag
            dfs.append(df)

        df = pd.concat(dfs)
        df.index = df.index.values + 1
        df.index.name = "node"
        df.sort_index(inplace=True)
        df.to_csv(working_directory / "pert_results.csv")
        return df
