import logging
import os
from datetime import datetime
from typing import List

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu, spsolve


class PerfMeasRecord(object):
    """a performance measure record class - an instance for each row in the performance measure block

    Parameters
    ----------
    kper (int) : zero-based stress period
    kstp (int) : zero-based time step
    inode (int) : zero-based node number
    pm_type (str) : either 'head' or boundary package name as shown in the GWF nam file
    pm_form (str) : either 'direct' or 'residual'
    weight (float) : weight value
    obsval (float) : optional observed counterpart.  only used if 'pm_form' is residual
    k (int) : optional zero-based layer (for structured grids - only for reporting)
    i (int) : optional zero-based row (for structured grids - only for reporting)
    j (int) : optional zero-based column (for structured grids - only for reporting)


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
        k: int = None,
        i: int = None,
        j: int = None,
    ):
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
                f"PerfMeasRecord.pm_form must be 'direct' or 'residual', not '{self.pm_form}'"
            )

    def __repr__(self):
        s = f"kperkstp:{self.kperkstp}, inode:{self.inode}, k:{self._k}, type:{self.pm_type}, form:{self.pm_form}"
        if self._i is not None:
            s += f", i:{self._i}"
        if self._j is not None:
            s += f", j:{self._j}"
        return s


class PerfMeas(object):
    """Performance measures for adjoint solves

    Parameters
    ----------
    pm_name (str) : name of the performance measure
    pm_entries (list(PerfMeasRec)) : container of performance measure entries
    verbose_level (int) : how much stdout


    todo: preprocess all the connectivity in to faster look dict containers,
        including nnode to kij info for structured grids

        todo: convert several class methods to static methods - this might make testing easier

        todo: add a no-data value var to fill empty spots in output arrays.  currently using zero :(

        todo: check that each entry's kperkstp is in the dicts being passed to solve_adjoint()

    """

    def __init__(
        self, pm_name: str, pm_entries: List[PerfMeasRecord], verbose_level: int = 1
    ):
        self._name = pm_name.lower().strip()
        self._entries = pm_entries
        self.verbose_level = int(verbose_level)
        self.logger = logging.getLogger(logging.__name__ + self._name)
        logging.basicConfig(
            filename=self._name + ".log", format="%(asctime)s %(message)s"
        )

    @property
    def name(self):
        """get self._name

        Returns
        -------
        name (str) : performance measure name

        """
        return str(self._name)

    @staticmethod
    def get_mf6_bound_dict():
        """get a container of information about which axes of the 'bound'
        array from MODFLOW6 has the quantities for calculating the sensitivity to

        Returns
        -------
        d (dict) : dict of boundary info

        """
        d = {
            "ghb6": {0: "bhead", 1: "cond"},
            "drn6": {0: "elev", 1: "cond"},
            "riv6": {0: "stage", 1: "cond"},
            "sfr6": {0: "stage", 1: "cond"},
        }  # ,
        # "chd6":{0:"head"}}
        return d

    def solve_forward(self, head_dict, sp_package_dict):
        """calculate forward solution for the performance measure.
        Thjs is only for the perturbation testing process


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
        hdf5_forward_solution_fname: str,
        hdf5_adjoint_solution_fname: str = None,
        linear_solver=None,
        linear_solver_kwargs: dict = {},
        use_precon: bool = True,
    ):
        """Solve for the adjoint state for the performance measure.

        Parameters
        ----------
        hdf5_forward_solution_fname (str) : the HDF5 file written during the forward GWF solution
            that contains all the information needed to solve for the adjoint state
        hdf5_adjoint_solution_fname (str) : the HDF5 file to be created by the adjoint solution process.
            If None, use `"adjoint_solution_{0}_".format(self._name) + hdf5_forward_solution_fname`.
        linear_solver (varies) : the scipy sparse linear alg solver to use.  If None, a choice is made
            between direct and bicgstab, depending if the number of nodes is less than 50,000.  If `str`,
            can be "direct" or "bicgstab".  Otherwise, can be a function pointer to a solver function
            in which the first two args are the CSR amat matrix and the dense RHS vector, respectively
        linear_solver_kwargs (dict): dictionary of keyword args to pass to `linear_solver`.  Default is {}
        use_precon (bool): flag to use an ILU preconditioner with iterative linear solver.

        Returns
        -------
        dfs (DataFrame) : summary of composite sensitivity information

        """
        adj_start = datetime.now()
        try:
            hdf = h5py.File(hdf5_forward_solution_fname, "r")
        except Exception as e:
            raise Exception(
                f"error opening hdf5 file '{hdf5_forward_solution_fname}' for PerfMeas {self._name}: {e!s}"
            )
        if hdf5_adjoint_solution_fname is None:
            pth = os.path.split(hdf5_forward_solution_fname)[0]
            hdf5_adjoint_solution_fname = os.path.join(
                pth,
                f"adjoint_solution_{self._name}_" + hdf5_forward_solution_fname,
            )

        if os.path.exists(hdf5_adjoint_solution_fname):
            self.logger.warning(
                f"WARNING: removing existing adjoint solution file '{hdf5_adjoint_solution_fname}'"
            )
            os.remove(hdf5_adjoint_solution_fname)

        adf = h5py.File(hdf5_adjoint_solution_fname, "w")

        keys = list(hdf.keys())
        gwf_package_dict = {k: v for k, v in hdf["gwf_info"].attrs.items()}

        sol_keys = [k for k in keys if k.startswith("solution")]
        sol_keys.sort()
        if len(sol_keys) == 0:
            raise Exception("no 'solution' keys found")
        kperkstp = [
            (kper, kstp) for kper, kstp in zip(hdf["aux"]["kper"], hdf["aux"]["kstp"])
        ]
        if len(kperkstp) != len(sol_keys):
            raise Exception(
                f"number of solution datasets ({len(sol_keys)}) != number of kper,kstp entries ({len(kperkstp)})"
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
                hdf["gwf_info"]["nlay"][0],
                hdf["gwf_info"]["nrow"][0],
                hdf["gwf_info"]["ncol"][0],
            )
            self.logger.info("...structured grid found, shape:", grid_shape)

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
            data = {}
            kper_start = datetime.now()
            self.logger.info(
                kper_start,
                "-->starting adjoint solve for PerfMeas:",
                self._name,
                " (kper,kstp)",
                kk,
            )
            sol_key = kk_sol_map[kk]
            if sol_key in adf:
                raise Exception(
                    f"solution key '{sol_key}' already in adjoint hdf5 file"
                )

            start = datetime.now()
            self.logger.info("forming rhs")
            dfdh = self._dfdh(kk, hdf[sol_key])
            data["dfdh"] = dfdh
            iss = hdf[sol_key]["iss"][0]
            if itime != 0:  # transient
                # get the derv of RHS WRT head
                drhsdh = hdf[sol_key]["drhsdh"][:]
                data["drhsdh"] = drhsdh
                rhs = (drhsdh * lamb) - dfdh
            else:
                rhs = -dfdh
                self.logger.info(f"...took:{(datetime.now() - start).total_seconds()}")

            start = datetime.now()

            self.logger.info("forming amat")
            amat = hdf[sol_key]["amat"][:]
            head = hdf[sol_key]["head"][:]
            amat = sparse.csr_matrix(
                (amat.copy()[: ja.shape[0]], ja.copy(), ia.copy()),
                shape=(len(ia) - 1, len(ia) - 1),
            )
            amat = amat.transpose()
            self.logger.info(f"...took:{(datetime.now() - start).total_seconds()}")
            start = datetime.now()
            self.logger.info("lambda solve")
            m = None
            if linear_solver is None:
                if head.shape[0] < 50000:
                    _linear_solver = spsolve
                    _linear_solver_kwargs = {"use_umfpack": True}
                else:
                    _linear_solver = bicgstab
                    _linear_solver_kwargs = {"rtol": 1e-5, "atol": 1e-5, "maxiter": 200}
                    if use_precon:
                        amat_ilu = spilu(amat)
                        m = LinearOperator(
                            (head.shape[0], head.shape[0]), amat_ilu.solve
                        )

            elif isinstance(linear_solver, str):
                if linear_solver == "direct":
                    _linear_solver = spsolve
                    if len(linear_solver_kwargs) == 0:
                        _linear_solver_kwargs = {"use_umfpack": True}
                    else:
                        _linear_solver_kwargs = linear_solver_kwargs
                elif linear_solver == "bicgstab":
                    _linear_solver = bicgstab
                    if len(linear_solver_kwargs) == 0:
                        _linear_solver_kwargs = {
                            "rtol": 1e-5,
                            "atol": 1e-5,
                            "maxiter": 200,
                        }
                    else:
                        _linear_solver_kwargs = linear_solver_kwargs
                    if use_precon:
                        amat_ilu = spilu(amat)
                        m = LinearOperator(
                            (head.shape[0], head.shape[0]), amat_ilu.solve
                        )
                else:
                    raise Exception(
                        f"unrecognized 'linear_solver' value: '{linear_solver}', should be 'direct' or 'bicgstab'"
                    )
            else:
                _linear_solver = linear_solver
                _linear_solver_kwargs = linear_solver_kwargs
                if use_precon:
                    amat_ilu = spilu(amat)
                    m = LinearOperator((head.shape[0], head.shape[0]), amat_ilu.solve)

            if m is not None:
                _linear_solver_kwargs["M"] = m

            self.logger.info("...solving with " + str(_linear_solver))
            self.logger.info("...with options:" + str(_linear_solver_kwargs))

            # lamb = spsolve(amat, rhs,use_umfpack=True)
            lamb = _linear_solver(amat, rhs, **_linear_solver_kwargs)
            if isinstance(lamb, tuple):
                self.logger.info("solver returned:" + str(lamb[1]))
                lamb = lamb[0]
            if np.any(np.isnan(lamb)):
                self.logger.warning(
                    f"WARNING: nans in adjoint states for pm {self.name} at kperkstp {kk}"
                )
            self.logger.info(f"...took:{(datetime.now() - start).total_seconds()}")
            is_newton = hdf[sol_key].attrs["is_newton"]
            chd_nodelist = []
            if "chd6" in gwf_package_dict:
                for pname in gwf_package_dict["chd6"]:
                    nodelist = list(hdf[sol_key][pname]["nodelist"][:] - 1)
                    chd_nodelist.extend(nodelist)
            chd_nodelist = np.array(chd_nodelist, dtype=int)
            start = datetime.now()
            self.logger.info("lam_dresdk_h")
            k_sens, k33_sens = PerfMeas.lam_dresdk_h(
                is_newton,
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
                chd_nodelist,
            )

            data["k11"] = k_sens
            data["k33"] = k33_sens
            comp_k_sens += k_sens
            comp_k33_sens += k33_sens
            self.logger.info(f"...took:{(datetime.now() - start).total_seconds()}")

            if has_sto:
                start = datetime.now()

                self.logger.info("ss")

                if iss == 0:
                    ss_sens = lamb * hdf[sol_key]["dresdss_h"][:]
                else:
                    ss_sens = np.zeros_like(lamb)
                data["ss"] = ss_sens
                comp_ss_sens += ss_sens
                self.logger.info(f"...took:{(datetime.now() - start).total_seconds()}")

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

                        self.logger.info(f"{ptype},{pname}")

                        sp_bnd_dict = {
                            "bound": hdf[sol_key][pname]["bound"][:],
                            "node": hdf[sol_key][pname]["nodelist"][:],
                        }
                        # print(pname,ptype)
                        # print(hdf[sol_key][pname]["bound"][:])
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
                        self.logger.info(
                            f"...took:{(datetime.now() - start).total_seconds()}"
                        )

            data["lambda"] = lamb
            data["head"] = hdf[sol_key]["head"][:]
            self.logger.info(
                "-->took:" + str((datetime.now() - kper_start).total_seconds()),
                " seconds to solve adjoint solution for PerfMeas:"
                + self._name
                + " (kper,kstp)"
                + str(kk),
            )

            if self.verbose_level > 2:
                data["amat"] = amat
                data["rhs"] = rhs
            self.logger.info("...save")
            PerfMeas.write_group_to_hdf(
                adf,
                sol_key,
                data,
                nodeuser=nodeuser,
                grid_shape=grid_shape,
                nodereduced=nodereduced,
            )
        self.logger.info("...form composite sensitivities")
        data = {}
        data["k11"] = comp_k_sens
        data["k33"] = comp_k33_sens

        if has_sto:
            data["ss"] = comp_ss_sens
        data["wel6_q"] = comp_welq_sens
        data["rch6_recharge"] = comp_rch_sens

        for name, vals in comp_bnd_results.items():
            data[name] = vals
        self.logger.info("...save")
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
        df.to_csv(f"adjoint_summary_{self._name}.csv")

        print(
            datetime.now(),
            "adjoint solve took: "
            + str((datetime.now() - adj_start).total_seconds())
            + f" for pm {self._name} at kperkstp {kk}",
        )
        self.logger.info(
            "adjoint solve took: "
            + str((datetime.now() - adj_start).total_seconds())
            + f" for pm {self._name} at kperkstp {kk}"
        )
        return df

    @staticmethod
    def write_group_to_hdf(
        hdf,
        group_name,
        data_dict,
        attr_dict={},
        grid_shape=None,
        nodeuser=None,
        nodereduced=None,
    ):
        """write a group in data to an open HDF5 file

        Parameters
        ----------
        hdf (h5py.File) : an open HDF5 filehandle
        group_name (str) : the group name for the dataset
        data_dict (dict) : datasets to write to the HDF5 file
        attr_dict (dict) : optional dict of attributes to write for the group
        nodeuser (ndarray) : optional `nodeuser` array from MODFLOW6
        nodereduced (ndarray) : optional `nodereduced` array from MODFLOW6

        """
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
                raise Exception(
                    f"unrecognized data_dict entry: {tag},type:{type(item)}"
                )
        if nodeuser is not None:
            _ = grp.create_dataset(
                "nodeuser", nodeuser.shape, dtype=nodeuser.dtype, data=nodeuser
            )
        if kijs is not None:
            for idx, name in enumerate(["k", "i", "j"]):
                arr = np.array([kij[idx] for kij in kijs], dtype=int)
                _ = grp.create_dataset(name, arr.shape, dtype=arr.dtype, data=arr)

    @staticmethod
    def _dconddhk(k1, k2, cl1, cl2, width, height1, height2):
        """partial of conductance with respect to K

        Parameters
        ----------
        k1 (float) : K of connection 1
        k2 (float) : K of connection 2
        cl1 (float) : length of connection 1
        cl2 (float) : length of connection 2
        width (float) : connection width
        height1 (float) : height of connection 1
        height2 (float) : height of connmection 2

        Returns
        -------
        d (float) : partial of conductance WRT K

        """

        # todo: upstream weighting - could use height1 and height2 to check...
        # todo: vertically staggered
        d = (width * cl1 * height1 * (height2**2) * (k2**2)) / (
            ((cl2 * height1 * k1) + (cl1 * height2 * k2)) ** 2
        )
        return d

    @staticmethod
    def smooth_sat(sat):
        """smoother saturation using sigmoid function a la MODFLOW6

        Parameters
        ----------
        sat (ndarray) : saturation array

        Returns
        -------
        s_sat (ndarray) : smoothed saturation

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
    def d_smooth_sat_dh(sat, top, bot):
        """partial of smoother saturation with respect to head

        Parameters
        ----------
        sat (ndarray) : saturation
        top (ndarray) : cell top
        bot (ndarray) : cell bottom

        Returns
        -------
        d_s_sat_dh (ndarray) : partial of smoothed saturation WRT head

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
    def _smooth_sat(sat1, sat2, h1, h2):
        """private method for upstream smoothing

        Parameters
        ----------
        sat1 (float) : saturation of node 1
        sat2 (float) : saturation of node 2
        h1 (float) : head of node 1
        h2 (float) : head of node 2

        Returns
        -------
        value (float) smoothed saturation of the upstream node

        """
        if h1 >= h2:
            value = PerfMeas.smooth_sat(sat1)
        else:
            value = PerfMeas.smooth_sat(sat2)
        return value

    @staticmethod
    def _d_smooth_sat_dh(sat, h1, h2, top, bot):
        """private method of partial of smoothed saturation with respect to upstream head

        Parameters
        ----------
        sat (float) : saturation
        h1 (float) : head of node 1
        h2 (float) : head of node 2
        top (float): top

        Returns
        -------

        value (float) : partial of smoothed saturation WRT upstream head

        """
        value = 0.0
        if h1 >= h2:
            value = PerfMeas.d_smooth_sat_dh(sat, top, bot)
        return value

    @staticmethod
    def lam_dresdk_h(
        is_newton,
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
        chd_nodelist,
    ):
        """adjoint state times the partial of residual with respect to k times head

        Parameters
        ----------
        is_newton (bool) : flag for newton solution
        lamb (ndarray) : adjoint state array
        sat (ndarray) : saturation array
        head (ndarray) : head array
        ihc (ndarray) : horizontal connection indicator array
        ia (ndarray) : the index of connection array in the compressed sparse row format
        ja (ndarray) : the connection array in the compressed sparse row format
        jas (ndarray) : the full connectivity array
        cl1 (ndarray) : the connection length array for conn 1
        cl2 (ndarray) : the connection length array for conn 2
        hwva (ndarray) : the horizontal width vertical area array
        top (ndarray) : the top array
        bot (ndarray) : the bottom array
        icelltype (ndarray) : the convertible cell type indicator array
        k11 (ndarray) : the k11 array
        k33 (ndarray) : the k33 array
        chd_nodelist (ndarray) : zero-based index of chd nodes

        Returns
        -------
        result_k, result_k33 (ndarray) : the adjoint state times the partial of residual with respect to k and k33 times head
        """
        iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
        # array of number of connections per node (size nodes)

        sat_mod = sat.copy()
        sat_mod[icelltype == 0] = 1.0

        height = top - bot

        result33 = np.zeros_like(head)
        result = np.zeros_like(head)

        for node, (offset, ncon) in enumerate(zip(ia, iac)):
            if node in chd_nodelist:
                continue
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
                            sat_mod[node], sat_mod[mnode], head[node], head[mnode]
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

    def lam_drhs_dbnd(self, lamb, head, sp_dict, has_flux_pm):
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

    def _dfdh(self, kk, sol_dataset):
        """partial of the performance measure with respect to head

        Parameters
        ----------
        kk (tuple) : zero-based stress period and time step
        sol_dataset(h5py.Dataset): the forward solution dataset

        Returns
        -------
        result (ndarray) : partial of performance measure WRT head

        """
        head = sol_dataset["head"][:]
        dfdh = np.zeros_like(head)
        for pfr in self._entries:
            if pfr.kperkstp == kk:
                if pfr.pm_type == "head":
                    if pfr.pm_form == "direct":
                        dfdh[pfr.inode] = pfr.weight
                    elif pfr.pm_form == "residual":
                        dfdh[pfr.inode] = (
                            2.0 * pfr.weight * (head[pfr.inode] - pfr.obsval)
                        )
                else:
                    hcof = sol_dataset[pfr.pm_type]["hcof"][:]
                    inodelist = sol_dataset[pfr.pm_type]["nodelist"][:] - 1
                    idx = np.where(inodelist == pfr.inode)[0][0]
                    dfdh[pfr.inode] = hcof[idx]
        return dfdh

    @staticmethod
    def get_value_from_gwf(gwf_name, pak_name, prop_name, gwf):
        """get a copy of a quantity from the MODFLOW6 API

        Parameters
        ----------
        gwf_name (str): name of the GWF instance
        pak_name (str): name of the package in the GWF nam file
        prop_name (str): name of the property in the 'pak_name' package
        gwf (MODFLOW6 API): a MODFLOW6 GWF instance

        Returns
        -------
        value (varies): the quantity of interest

        """
        addr = gwf.get_var_address(prop_name, gwf_name, pak_name)
        return gwf.get_value(addr)

    @staticmethod
    def get_ptr_from_gwf(gwf_name, pak_name, prop_name, gwf):
        """get a pointer (well reference anyway) to a quantity from the MODFLOW6 API

        Parameters
        ----------
        gwf_name (str): name of the GWF instance
        pak_name (str): name of the package in the GWF nam file
        prop_name (str): name of the property in the 'pak_name' package
        gwf (MODFLOW6 API): a MODFLOW6 GWF instance

        Returns
        -------
        value (varies): a mutable reference to the quantity of interest

        """
        addr = gwf.get_var_address(prop_name, gwf_name, pak_name)
        return gwf.get_value_ptr(addr)

    @staticmethod
    def get_node(shape, lrc_list):
        """get the node numbers for a given list of lrc values. stolen from flopy

        Parameters
        ----------
        lrc_list (list): list of layer row column values

        Returns
        -------
        values (ndarray): node numbers

        """

        if not isinstance(lrc_list, list):
            lrc_list = [lrc_list]
        multi_index = tuple(np.array(lrc_list).T)
        return np.ravel_multi_index(multi_index, shape).tolist()

    @staticmethod
    def get_lrc(shape, nodes):
        """get layer row column values from node numbers.  Also stolen from flopy

        Parameters
        ----------
        nodes (list) : list of node numbers

        Returns
        -------
        values (list): list of layer-row-column values


        """
        if isinstance(nodes, int):
            nodes = [nodes]
        return list(zip(*np.unravel_index(nodes, shape)))

    @staticmethod
    def has_sto_iconvert(gwf):
        """does the forward model has an sto package with iconvert

        Parameters
        ----------
        gwf (MODFLOW6 API): a MODFLOW6 GWF API instance

        Returns
        -------
            flag (bool) : has sto iconvert?

        """
        names = [
            n for n in list(gwf.get_input_var_names()) if "STO" in n and "ICONVERT" in n
        ]
        if len(names) == 0:
            return False
        return True
