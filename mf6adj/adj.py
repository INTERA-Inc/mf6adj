import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import h5py
import modflowapi
import flopy

from .pm import PerfMeasRecord, PerfMeas

DT_FMT = "%Y-%m-%d %H:%M:%S"


class Mf6Adj(object):
    """The MODFLOW6 Adjoint solver

    Parameters
    ----------
    adj_filename (str): the adjoint input filename
    lib_name (str): the MODFLOW6 shared library file
    verbose_level (int): flag to control output.  Default is 1



    """
    def __init__(self, adj_filename: str, lib_name: str, verbose_level=1):

        """

        """
        self.verbose_level = int(verbose_level)
        if not os.path.exists(adj_filename):
            raise Exception("adj_filename '{0}' not found".format(adj_filename))
        self.adj_filename = adj_filename

        # process the flow model
        # make sure the lib exists
        if not os.path.exists(lib_name):
            raise Exception("MODFLOW-6 shared library  '{0}' not found".format(lib_name))
        # find the model name
        self._gwf_model_dict, namfile_dict = Mf6Adj.get_model_names_from_mfsim(".")
        if len(self._gwf_model_dict) != 1:
            raise Exception("only one model is current supported")
        self._gwf_name = list(self._gwf_model_dict.keys())[0]
        self._gwf_namfile = namfile_dict[self._gwf_name]
        self._gwf_package_dict = Mf6Adj.get_package_names_from_gwfname(self._gwf_namfile)
        if self._gwf_model_dict[self._gwf_name] != "gwf6":
            raise Exception("model is not a gwf6 type: {0}". \
                            format(self._gwf_model_dict[self._gwf_name]))
        if "dis6" in self._gwf_package_dict:
            print("...structured grid found")
            is_structured = True
        else:
            print("...unstructured grid found")
            is_structured = False
        self._gwf = None
        self._lib_name = lib_name
        self._flow_dir = "."
        self._gwf = self._initialize_gwf(lib_name, self._flow_dir)
        self._hdf5_name = None

        self._structured_mg = None
        self.is_structured = is_structured
        self._shape = None
        if self.is_structured:
            nlay = self._gwf.get_value(self._gwf.get_var_address("NLAY", self._gwf_name.upper(), "DIS"))[0]
            nrow = self._gwf.get_value(self._gwf.get_var_address("NROW", self._gwf_name.upper(), "DIS"))[0]
            ncol = self._gwf.get_value(self._gwf.get_var_address("NCOL", self._gwf_name.upper(), "DIS"))[0]
            self._structured_mg = flopy.discretization.StructuredGrid(nrow=nrow,
                                                                      ncol=ncol,
                                                                      nlay=nlay)
            self._shape = (nlay,nrow,ncol)
        self._performance_measures = []
        self._read_adj_file()
        self._gwf_package_types = ["wel6","ghb6","riv6","drn6","sfr6","rch6","recha6","sfr6"]

    def _read_adj_file(self):
        """private method to read the adj input file

        Note
        ----
        The input file structure is very similar to other MODFLOW6 input files.
        Each performance measure is defined in a 'performance_measure' block.  Each
        block contains one or more entries describing model output quantities that
        together define the performance measure.  Each performance measure entry must
        have information about the spatial and temporal location of the quantity such as
        node/lay-row-col information and stress period/time step information, as well as
        information about which output quantity to use (head or flux).  The entries must
        also include a weight and optionally an observed value (for residual type performance
        measures).

        For example, if the performance measure was for a head in a single cell located in
        layer 3, row 10, column 34 during the 4th timestep of the 25th stress period and
        it is a direct performance measure, the entry would be:
           25 3 3 10 34 head direct 1.0 -999 # the -999 is a null value for the unused obsval
        Alternatively, if the same spatial temporal location was used for a sum-of-squared
        residual performance measure and the observed value is 123.45, the entry would be:
           25 3 3 10 34 head residual 1.0 123.45

        If the performance measure is for the simulated flux exchanged with a GHB boundary
        in model layer 10, row 2, column 3 for stress periods 1 and 2 (assuming 1 timestep
        per stress period and assuming the GHB package is named 'ghb_1' in the GWF nam file):
           1 1 10 2 3 ghb_1 direct 1.0 -999
           2 1 10 2 3 ghb_1 direct 1.0 -999
        The resulting adjoint sensitivities will be with respect to the ghb flux in model cell
        (10,2,3) for both stress periods 1 and 2

        As presently coded, performance measure forms (i.e. 'direct' or 'residual') cannot be mixed
        for a given performance measure and performance type (i.e. 'head' or flux) cannot be mixed
        for a given performance measure.





        """
        # clear any existing PMs
        self._performance_measures = []

        addr = ["NODEUSER", self._gwf_name.upper(), "DIS"]
        wbaddr = self._gwf.get_var_address(*addr)
        nuser = self._gwf.get_value(wbaddr) - 1

        addr = ["NODEREDUCED", self._gwf_name.upper(), "DIS"]
        wbaddr = self._gwf.get_var_address(*addr)

        ncpl = None
        if not self.is_structured:
            addr = ["NCPL", self._gwf_name.upper(), "DIS"]
            wbaddr = self._gwf.get_var_address(*addr)
            ncpl = self._gwf.get_value(wbaddr)
            addr = ["NLAY", self._gwf_name.upper(), "DIS"]
            wbaddr = self._gwf.get_var_address(*addr)

        with open(self.adj_filename, 'r') as f:
            count = 0
            while True:
                line = f.readline()
                count += 1
                # eof
                if line == "":
                    break

                # skip empty lines or comment lines
                if len(line.strip()) == 0 or line.strip()[0] == "#":
                    continue

                # read the options block
                if line.lower().strip().startswith("begin options"):
                    while True:
                        line2 = f.readline()
                        count += 1

                        if line2 == "":
                            raise EOFError("EOF while reading options")
                        elif len(line2.strip()) == 0 or line2.strip()[0] == "#":
                            continue
                        elif line2.lower().strip().startswith("begin"):
                            raise Exception("a new begin block found while parsing options")
                        elif line2.lower().strip().startswith("end options"):
                            break
                        elif line2.lower().strip().split()[0] == "hdf5_name":
                            self._hdf5_name = line2.strip().split()[1]
                        else:
                            raise Exception("unrecognized option line:"+line2.strip())

                # parse a new performance measure block

                elif line.lower().strip().startswith("begin performance_measure"):

                    raw = line.lower().strip().split()

                    if len(raw) != 3:
                        raise Exception("'begin' line {0} has wrong number of items, should be 3, not {1}".format(count,len(raw)))

                    pm_name = raw[2].strip().lower()

                    pm_entries = []
                    pm_line = 0
                    while True:
                        line2 = f.readline()
                        pm_line += 1
                        count += 1
                        if line2 == "":
                            raise EOFError("EOF while reading performance_measure block '{0}'".format(line))
                        elif len(line.strip()) == 0 or line.strip()[0] == "#":
                            continue
                        elif line2.lower().strip().startswith("begin"):
                            raise Exception(
                                "a new begin block found while parsing performance_measure block '{0}'".format(line))
                        elif line2.lower().strip().startswith("end performance_measure"):
                            break
                        elif line2.lower().strip().startswith("open"):
                            fname = line2.split()[1]
                            if not os.path.exists(fname):
                                raise Exception("external file '{0}' found".format(fname))
                            #df = pd.read_csv()
                            raise NotImplementedError()

                        raw = line2.lower().strip().split()
                        if self.is_structured and len(raw) != 9:
                            print("parsed line",raw)
                            raise Exception("performance measure entry on line {0} has the wrong number of items, found {1}, should have 8".format(count,len(raw)))
                        elif not self.is_structured and len(raw) != 8:
                            print("parsed line", raw)
                            raise Exception(
                                "performance measure entry on line {0} has the wrong number of items, found {1}, should have 9".format(
                                    count, len(raw)))
                        kper = int(raw[0]) - 1
                        kstp = int(raw[1]) - 1
                        # todo: check limits of kper, kstp
                        i,j = None,None
                        if self.is_structured:
                            kij = []
                            for i in range(3):
                                try:
                                    kij.append(int(raw[i + 2]) - 1)
                                except:
                                    raise Exception("error casting k-i-j info on line {0}: '{1}'".format(count, line2))
                            k,i,j = kij[0],kij[1],kij[2]
                            # convert to node number
                            inode = PerfMeas.get_node(self._shape,[kij])[0]
                            # if there is a reduced node scheme
                            if len(nuser) > 1:
                                nn = np.where(nuser == inode)[0]
                                if nn.shape[0] != 1:
                                    print(n, nn)
                                    if self.is_structured:
                                        print(kij)
                                    raise Exception("node num {0} not in reduced node num".format(n))

                                inode = nn[0]

                        else:
                            try:
                                lay = int(raw[2])
                            except:
                                raise Exception("error casting layer info info on line {0}: '{1}'".format(count, line2))
                            k = lay - 1
                            try:
                                node = int(raw[3])
                            except:
                                raise Exception("error casting layer info info on line {0}: '{1}'".format(count, line2))

                            inode = ((ncpl * (lay-1)) + node) - 1

                            # if there is a reduced node scheme
                            if len(nuser) > 1:
                                nn = np.where(nuser == inode)[0]
                                if nn.shape[0] != 1:
                                    raise Exception("node num {0} not in reduced node num".format(n))
                                inode = nn[0]

                        obsval = float(raw[-1])
                        weight = float(raw[-2])
                        pm_form = raw[-3].strip().lower()
                        pm_type = raw[-4].strip().lower()
                        pm_entries.append(PerfMeasRecord(kper,kstp,inode,pm_type,pm_form,weight,obsval,k,i,j))
                    if len(pm_entries) == 0:
                        raise Exception("no entries found for PM {0}".format(pm_name))
                    pm_types = set([entry.pm_type for entry in pm_entries])
                    if len(pm_types) > 1:
                        raise Exception("performance measure"+\
                                        "{0} has mixed 'pm_types' ({1}), this is not supported".\
                                        format(pm_name,str(pm_types)))
                    pm_forms = set([entry.pm_form for entry in pm_entries])
                    if len(pm_forms) > 1:
                        raise Exception("performance measure" + \
                                        "{0} has mixed 'pm_forms' ({1}), this is not supported". \
                                        format(pm_name, str(pm_forms)))
                    if list(pm_types)[0] != "head" and list(pm_forms)[0] != "direct":
                        raise Exception("performance measure" + pm_name +\
                                        " has a flux 'pm_form' and is a " +\
                                        "residual 'pm_type', this is not supported")
                    if pm_name in [pm._name for pm in self._performance_measures]:
                        raise Exception("PM {0} multiply defined".format(pm_name))
                    self._performance_measures.append(
                        PerfMeas(pm_name, pm_entries, self.verbose_level))

                else:
                    raise Exception("unrecognized adj file input on line {0}: '{1}'".format(count, line))
        if len(self._performance_measures) == 0:
            raise Exception("no PMs found in adj file")

    @staticmethod
    def get_model_names_from_mfsim(sim_ws):
        """return the model names from an mfsim.nam file

        Parameters
        ----------
            sim_ws (str): the simulation path

        Returns
        -------
            dict,dict: a pair of dicts, first is model-name:model-type (e.g. {"gwf-1":"gwf"},
                the second is model namfile: model-type (e.g. {"gwf-1":"gwf_1.nam"})

        """
        sim_nam = os.path.join(sim_ws, "mfsim.nam")
        if not os.path.exists(sim_nam):
            raise Exception("simulation nam file '{0}' not found".format(sim_nam))
        model_dict = {}
        namfile_dict = {}
        with open(sim_nam, 'r') as f:
            while True:
                line = f.readline()
                if line == "":
                    raise EOFError("EOF when looking for 'models' block")
                if line.strip().lower().startswith("begin") and "models" in line.lower():
                    while True:
                        line2 = f.readline()
                        if line2 == "":
                            raise EOFError("EOF when reading 'models' block")
                        elif line2.strip().lower().startswith("end") and "models" in line2.lower():
                            break
                        raw = line2.strip().lower().split()
                        if raw[-1] in model_dict:
                            raise Exception("duplicate model name found: '{0}'".format(raw[-1]))
                        model_dict[raw[2]] = raw[0]
                        namfile_dict[raw[2]] = raw[1]
                    break
        return model_dict, namfile_dict

    @staticmethod
    def get_package_names_from_gwfname(gwf_nam_file):
        """return the package names from a GWF nam file

        Parameters
        ----------
            gwf_nam_file (str): GWF nam file

        Returns
        -------
            dict: package types as keys and list of package names as values

        """
        if not os.path.exists(gwf_nam_file):
            raise Exception("gwf nam file '{0}' not found".format(gwf_nam_file))
        package_dict = {}
        count_dict = {}
        with open(gwf_nam_file, 'r') as f:
            while True:
                line = f.readline()
                if line == "":
                    raise EOFError("EOF when looking for 'packages' block")
                if line.strip().lower().startswith("begin") and "packages" in line.lower():
                    while True:
                        line2 = f.readline()
                        if line2 == "":
                            raise EOFError("EOF when reading 'packages' block")
                        elif line2.strip().lower().startswith("end") and "packages" in line2.lower():
                            break
                        raw = line2.strip().lower().split()
                        if raw[0].startswith("#"):
                            continue
                        if "#" in line2:
                            raw = line2.split("#")[0].lower().split()
                        if len(raw) < 2:
                            raise Exception("wrong number of items on line: {0}".format(line2))
                        tag_name = None
                        if len(raw) > 2:
                            tag_name = raw[2]
                        package_type = raw[0]
                        if package_type not in count_dict:
                            count_dict[package_type] = 1

                        if package_type not in package_dict:
                            package_dict[package_type] = []
                        filename = raw[1]
                        if tag_name is None:
                            tag_name = package_type.replace("6", "") + "-{0}".format(count_dict[package_type])
                        package_dict[package_type].append(tag_name)
                        count_dict[package_type] += 1

                    break
        return package_dict

    @staticmethod
    def write_group_to_hdf(hdf, group_name, data_dict, attr_dict={}):
        """write information to an open HDF5 file

        Parameters
        ----------
            hdf (h5py.File) : an open HDF5 filehandle
            group_name (str) : name of the group to create
            data_dict (dict) : dict of info to write as the group.  If key
                is a list, its cast to an ndarray.  If key is a dict itself,
                only 'nodelist' and 'bound' are stored.
            attr_dict (dict) : an optional dict of attributes to store with the
                group


        """
        if group_name in hdf:
            raise Exception("group_name {0} already in hdf file".format(group_name))
        grp = hdf.create_group(group_name)
        for name, val in attr_dict.items():
            grp.attrs[name] = val
        for tag, item in data_dict.items():
            if isinstance(item, list):
                item = np.array(item)
            if isinstance(item, np.ndarray):
                dset = grp.create_dataset(tag, item.shape, dtype=item.dtype, data=item)
            elif isinstance(item, dict):
                if "nodelist" in item:
                    iitem = item["nodelist"]
                    dset = grp.create_dataset(tag, iitem.shape, dtype=iitem.dtype, data=iitem)
                elif "bound" in item:
                    iitem = item["bound"]
                    dset = grp.create_dataset(tag, iitem.shape, dtype=iitem.dtype, data=iitem)
                else:
                    print("Mf6Adj._write_group_to_hdf(): unused data_dict item {0}".format(tag))
            else:
                raise Exception("unrecognized data_dict entry: {0},type:{1}".format(tag, type(item)))

    def _open_hdf(self, tag):
        """private method to open an HDF5 filehandle for writing

        Parameters
        ----------
        tag (str) : a prefix tag for the file

        Returns
        -------
        f (h5py.File) : filehandle

        """
        if tag is None:
            fname = self._gwf_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".hd5"
        else:
            fname = tag
        self._hdf5_name = fname
        if os.path.exists(fname):
            os.remove(fname)
        f = h5py.File(fname, 'w')
        return f

    def _add_gwf_info_to_hdf(self, hdf):
        """add model structure and metadata to an HDF5 file

        Parameters
        ----------
        hdf (h5py.File) : an HDF5 filehandle



        """
        gwf_name = self._gwf_name
        gwf = self._gwf
        data_dict = {}

        #todo: work out what dis type we have
        dis_pak = "DIS"

        ihc = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IHC", gwf)
        data_dict["ihc"] = ihc
        ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
        data_dict["ia"] = ia
        ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1
        data_dict["ja"] = ja
        jas = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JAS", gwf) - 1
        data_dict["jas"] = jas
        cl1 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL1", gwf)
        data_dict["cl1"] = cl1
        cl2 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL2", gwf)
        data_dict["cl2"] = cl2
        hwva = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "HWVA", gwf)
        data_dict["hwva"] = hwva
        top = PerfMeas.get_ptr_from_gwf(gwf_name, dis_pak, "TOP", gwf)
        data_dict["top"] = top
        bot = PerfMeas.get_ptr_from_gwf(gwf_name, dis_pak, "BOT", gwf)
        data_dict["bot"] = bot
        iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
        data_dict["iac"] = iac
        icelltype = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "ICELLTYPE", gwf)
        data_dict["icelltype"] = icelltype

        area = PerfMeas.get_ptr_from_gwf(gwf_name, dis_pak, "AREA", gwf)
        data_dict["area"] = area
        iconvert = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "ICONVERT", gwf)
        data_dict["iconvert"] = iconvert
        storage = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "SS", gwf)
        data_dict["storage"] = storage
        nodeuser = PerfMeas.get_ptr_from_gwf(gwf_name, dis_pak, "NODEUSER", gwf) - 1
        data_dict["nodeuser"] = nodeuser
        nodereduced = PerfMeas.get_ptr_from_gwf(gwf_name, dis_pak, "NODEREDUCED", gwf) - 1
        data_dict["nodereduced"] = nodereduced
        ndim = PerfMeas.get_ptr_from_gwf(gwf_name, dis_pak, "NDIM", gwf)
        data_dict["ndim"] = ndim
        nnodes = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "NODES", gwf)
        data_dict["nnodes"] = nnodes
        ndim = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "NDIM", gwf)
        data_dict["ndim"] = ndim
        idomain = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "IDOMAIN", gwf)
        data_dict["idomain"] = idomain

        if self.is_structured:
            nlay = PerfMeas.get_ptr_from_gwf(gwf_name, dis_pak, "NLAY", gwf)
            data_dict["nlay"] = nlay
            nrow = PerfMeas.get_ptr_from_gwf(gwf_name, dis_pak, "NROW", gwf)
            data_dict["nrow"] = nrow
            ncol = PerfMeas.get_ptr_from_gwf(gwf_name, dis_pak, "NCOL", gwf)
            data_dict["ncol"] = ncol

        PerfMeas.write_group_to_hdf(hdf, "gwf_info", data_dict,attr_dict=self._gwf_package_dict)


    @staticmethod
    def dresdss_h(gwf_name, gwf, head, head_old, dt, sat, sat_old):
        """partial of residual wrt ss times h.  Just need to mult
        times lambda in the PerfMeas.solve_adjoint()

        Parameters
        ----------
        gwf_name (str) : name of the GWF model
        gwf (MODFLOW6 API) : the API instance
        head (ndarray) : current heads
        head_old (ndarray) : heads from the last solve
        dt (float) : length of the current solution step in model time
        sat (ndarray) : current saturation
        sat_old (ndarray) : saturation from the last solve

        Returns
        -------
        result (ndarray) : dresdss_h

        """
        top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
        bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
        area = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
        iconvert = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "ICONVERT", gwf)

        # handle iconvert
        sat_mod = sat.copy()
        sat_mod[iconvert == 0] = 1.0
        sat_old_mod = sat_old.copy()
        sat_old_mod[iconvert == 0] = 1.0

        height = top - bot

        #result = np.zeros_like(head)
        dSC1 = area * height
        result = ((dSC1 / dt) * (sat_old_mod * head_old - sat_mod * head) +
                (dSC1 / dt) * bot * (sat_mod - sat_old_mod) +
                (dSC1 / (2.0 * dt)) * height * (
                        sat_mod**2 - sat_old_mod**2))
        # zero out dry cells
        result[head<=bot] = 0.0
        result[head_old<=bot] = 0.0

        return result


    @staticmethod
    def drhsdh(gwf_name, gwf, dt):
        """partial of the RHS WRT H

        Parameters
        ----------
        gwf_name (str) : name of the GWF model
        gwf (MODFLOW6 API) : the API instance
        dt (float) : length of the current solution step in model time

        Returns
        -------
        drhsdh (ndarray) : drhsdh

        """
        top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
        bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
        area = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
        storage = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "SS", gwf)
        drhsdh = -1. * storage * area * (top - bot) / dt
        return drhsdh


    def solve_gwf(self, verbose=True, _force_k_update=False, _sp_pert_dict=None,pert_save=False,
                  hdf5_name=None):
        """solve the flow across the modflow sim times and harvest the solution
        components needed for the adjoint solution and store them in the HDF5 file

        Parameters
        ----------
        verbose (bool) : flag to control stdout reporting
        _force_k_update (bool) : flag to force MODFLOW6 to re-process the K and K33 arrays.
            This is used in the perturbation testing
        _sp_pert_dict (dict) : a dictionary of perturbed boundary information.
            This is used in the perturbation testing
        pert_save (bool) : flag to save more information for the perturbation testing
        hdf5_name (str) : optional hdf5 filename to store forward solution components in.
            If None, a generic time-stamped filename is created

        Returns
        -------
        pert_results (dict) : information for the perturbation testing.

        """
        if self._gwf is None:
            raise Exception("gwf is None")
            self._gwf = self._initialize_gwf(self._lib_name, self._flow_dir)
        if hdf5_name is not None:
            self._hdf5_name = hdf5_name
        fhd = self._open_hdf(self._hdf5_name)
        sim_start = datetime.now()
        if verbose:
            print("...starting flow solution at {0}".format(sim_start.strftime(DT_FMT)))
        # get current sim time
        ctime = self._gwf.get_current_time()
        # get ending sim time
        etime = self._gwf.get_end_time()
        # max number of iterations
        max_iter = self._gwf.get_value(self._gwf.get_var_address("MXITER", "SLN_1"))
        # let's do it!
        num_fails = 0

        sat_old = None
        visited = list()
        ctimes = []
        dts = []
        kpers, kstps = [], []

        is_newton = self._gwf.get_value(self._gwf.get_var_address("INEWTON", self._gwf_name))[0]
        has_sto = False
        if PerfMeas.has_sto_iconvert(self._gwf):
            has_sto = True

        sp_package_data = None
        head_dict = None
        if pert_save:
            sp_package_data = {}
            head_dict = {}

        while ctime < etime:
            sol_start = datetime.now()
            # the length of this sim time
            dt = self._gwf.get_time_step()
            # prep the current time step
            self._gwf.prepare_time_step(dt)
            kiter = 0
            # prep to solve
            stress_period = self._gwf.get_value(self._gwf.get_var_address("KPER", "TDIS"))[0]
            time_step = self._gwf.get_value(self._gwf.get_var_address("KSTP", "TDIS"))[0]
            kper, kstp = stress_period - 1, time_step - 1
            kperkstp = (kper, kstp)

            # this is to force mf6 to update cond sat using the k11 and k33 arrays
            # which is needed for the perturbation testing
            if kper == 0 and kstp == 0 and _force_k_update:
                kchangeper = self._gwf.get_value_ptr(self._gwf.get_var_address("KCHANGEPER", self._gwf_name, "NPF"))
                kchangestp = self._gwf.get_value_ptr(self._gwf.get_var_address("KCHANGESTP", self._gwf_name, "NPF"))
                kchangestp[0] = time_step
                kchangeper[0] = stress_period
                nodekchange = self._gwf.get_value_ptr(self._gwf.get_var_address("NODEKCHANGE", self._gwf_name, "NPF"))
                nodekchange[:] = 1

            # apply any boundary condition perturbation info
            if _sp_pert_dict is not None:
                if _sp_pert_dict["kperkstp"] == kperkstp:
                    addr = ["BOUND", self._gwf_name, _sp_pert_dict["packagename"].upper()]
                    wbaddr = self._gwf.get_var_address(*addr)
                    bnd_ptr = self._gwf.get_value_ptr(wbaddr)
                    wbaddr = self._gwf.get_var_address("NODELIST", self._gwf_name, _sp_pert_dict["packagename"].upper())
                    nodelist = self._gwf.get_value_ptr(wbaddr)
                    idx = np.where(nodelist == _sp_pert_dict["node"])[0]
                    if idx.shape[0] == 0:
                        print(nodelist)
                        raise Exception("sp pert dict node not found :"+str(_sp_pert_dict))
                    bnd_ptr[idx] = _sp_pert_dict["bound"]

            self._gwf.prepare_solve(1)
            if sat_old is None:
                sat_old = self._gwf.get_value(self._gwf.get_var_address("SAT", self._gwf_name, "NPF"))

            # solve until converged
            while kiter < max_iter:
                convg = self._gwf.solve(1)
                if convg:
                    td = (datetime.now() - sol_start).total_seconds() / 60.0
                    if verbose:
                        print(
                            "flow stress period,time step {0},{1} converged with {2} iters, took {3:10.5G} mins".format(
                                stress_period, time_step, kiter, td))
                    break
                kiter += 1

            if not convg:
                td = (datetime.now() - sol_start).total_seconds() / 60.0
                if verbose:
                    print(
                        "flow stress period,time step {0},{1} did not converge, {2} iters, took {3:10.5G} mins".format(
                            stress_period, time_step, kiter, td))
                num_fails += 1
            try:
                self._gwf.finalize_solve(1)
            except:
                pass

            self._gwf.finalize_time_step()
            # update current sim time
            ctime = self._gwf.get_current_time()
            dt1 = self._gwf.get_time_step()

            ctimes.append(ctime)
            dts.append(dt1)
            kpers.append(kper)
            kstps.append(kstp)

            if kperkstp in visited:
                raise Exception("{0} already visited".format(kperkstp))
            visited.append(kperkstp)

            amat = self._gwf.get_value(self._gwf.get_var_address("AMAT", "SLN_1")).copy()
            data_dict = {"amat": amat}

            head = self._gwf.get_value(self._gwf.get_var_address("X", self._gwf_name.upper()))
            data_dict["head"] = head
            if pert_save:
                head_dict[kperkstp] = head

            head_old = self._gwf.get_value(self._gwf.get_var_address("XOLD", self._gwf_name.upper()))
            data_dict["head_old"] = head_old

            k11 = self._gwf.get_value(self._gwf.get_var_address("K11", self._gwf_name.upper(), "NPF"))
            data_dict["k11"] = k11
            k33 = self._gwf.get_value(self._gwf.get_var_address("K33", self._gwf_name.upper(), "NPF"))
            data_dict["k33"] = k33
            condsat = self._gwf.get_value(self._gwf.get_var_address("CONDSAT", self._gwf_name.upper(), "NPF"))
            data_dict["condsat"] = condsat


            iss = self._gwf.get_value(self._gwf.get_var_address("ISS", self._gwf_name.upper()))
            data_dict["iss"] = iss

            sat = self._gwf.get_value(self._gwf.get_var_address("SAT", self._gwf_name, "NPF"))
            data_dict["sat"] = sat
            data_dict["sat_old"] = sat_old

            sat_old = sat.copy()
            if has_sto: #has storage
                dresdss_h = Mf6Adj.dresdss_h(self._gwf_name,self._gwf,head,head_old,dt1,sat,sat_old)
                data_dict["dresdss_h"] = dresdss_h
                drhsdh = Mf6Adj.drhsdh(self._gwf_name,self._gwf,dt1)
                data_dict["drhsdh"] = drhsdh
            else:
                data_dict["drhsdh"] = np.zeros_like(sat_old)

            for package_type in self._gwf_package_types:

                if package_type in self._gwf_package_dict:
                    if pert_save and package_type not in sp_package_data:
                        sp_package_data[package_type] = {}
                    for tag in self._gwf_package_dict[package_type]:
                        nbound = self._gwf.get_value(self._gwf.get_var_address("NBOUND", self._gwf_name, tag.upper()))[
                            0]
                        if nbound > 0:
                            if pert_save and kperkstp in sp_package_data[package_type]:
                                if len(self._gwf_package_dict[package_type]) == 1:
                                    raise Exception("kperkstp '{0}' already in sp_package_data".format(str(kperkstp)))
                                else:
                                    pass
                            elif pert_save:
                                sp_package_data[package_type][kperkstp] = []
                            nodelist = self._gwf.get_value(
                                self._gwf.get_var_address("NODELIST", self._gwf_name, tag.upper()))
                            bound = self._gwf.get_value(
                                self._gwf.get_var_address("BOUND", self._gwf_name, tag.upper()))
                            hcof = self._gwf.get_value(
                                self._gwf.get_var_address("HCOF", self._gwf_name, tag.upper()))
                            rhs = self._gwf.get_value(self._gwf.get_var_address("RHS", self._gwf_name, tag.upper()))

                            simvals = self._gwf.get_value(self._gwf.get_var_address("SIMVALS", self._gwf_name, tag.upper()))

                            if package_type == "sfr6":
                                tag = self._gwf_package_dict[package_type][0]
                                stage = self._gwf.get_value(
                                    self._gwf.get_var_address("STAGE", self._gwf_name, tag.upper()))
                                bound[:,0] = stage
                                bound[:,1] = -1. * hcof

                            if pert_save:
                                for i in range(nbound):
                                    # note bound is an array!

                                    sp_package_data[package_type][kperkstp].append(
                                        {"node": nodelist[i], "bound": bound[i],
                                         "hcof": hcof[i], "rhs": rhs[i], "packagename": tag,"simval":simvals[i]})
                            data_dict[tag] = {"ptype": package_type, "nodelist": nodelist, "bound": bound,"hcof":hcof,"rhs":rhs,"simvals":simvals}
            attr_dict = {"ctime": ctime, "dt": dt1, "kper": kper, "kstp": kstp,"is_newton":is_newton,"has_sto":has_sto}
            PerfMeas.write_group_to_hdf(fhd, group_name="solution_kper:{0:05d}_kstp:{1:05d}".format(kper, kstp), data_dict=data_dict,
                                     attr_dict=attr_dict)


        sim_end = datetime.now()
        td = (sim_end - sim_start).total_seconds() / 60.0
        if verbose:
            print("\n...flow solution finished at {0}, took: {1:10.5G} mins".format(sim_end.strftime(DT_FMT), td))
            if num_fails > 0:
                print("...failed to converge {0} times".format(num_fails))
            print("\n")

        PerfMeas.write_group_to_hdf(fhd, "aux", {"totime": ctimes, "dt": dts, "kper": kpers, "kstp": kstps})
        self._add_gwf_info_to_hdf(fhd)
        fhd.close()
        if pert_save:
            return head_dict, sp_package_data

    def solve_adjoint(self):
        """solve for the adjoint state, one performance measure at at time

        Parameters
        ----------
        None

        Returns
        -------

        dfs (dict) : dictionary of dataframes (one per performance measure) summarizing
            the composite sensitivity information.  More granular information can be
            found in the corresponding HDF5 file that is created by the adjoint
            solve


        """
        if self._hdf5_name is None or not os.path.exists(self._hdf5_name):
            raise Exception("need to call solve_gwf() first")

        dfs = {}
        for pm in self._performance_measures:
            df = pm.solve_adjoint(self._hdf5_name)
            dfs[pm.name] = df
        return dfs

    def _initialize_gwf(self, lib_name, sim_ws):
        """initialze the MODFLOW6 API

        Parameters
        ----------
        lib_name (str) : MODFLOW6 shared library file
        sim_ws (str) : directory of the simulation.  This dir
            is assumed to contain the shared library file

        """
        # instantiate the flow model api
        if self._gwf is not None:
            self._gwf.finalize()
            self._gwf = None
        gwf = modflowapi.ModflowApi(os.path.join(sim_ws, lib_name), working_directory=sim_ws)
        gwf.initialize()
        return gwf

    def finalize(self):
        """close the api and file handles

        """
        try:
            self._gwf.finalize()
        except:
            pass
        self._gwf = None

    def _perturbation_test(self, pert_mult=1.01):
        """run the perturbation testing - this is for dev and testing only"""

        self._gwf = self._initialize_gwf(self._lib_name, self._flow_dir)
        gwf_name = self._gwf_name.upper()

        org_head, org_sp_package_data = self.solve_gwf(pert_save=True)
        tot = 0
        for d in org_sp_package_data["ghb6"][(0,0)]:
            print(d)
            tot += d["simval"]
        base_results = {pm.name: pm.solve_forward(org_head,org_sp_package_data) for pm in self._performance_measures}
        assert len(base_results) == len(self._performance_measures)

        addr = ["NODEUSER", gwf_name, "DIS"]
        wbaddr = self._gwf.get_var_address(*addr)
        nuser = self._gwf.get_value(wbaddr) - 1
        if len(nuser) == 1:
            nuser = np.arange(org_head[list(org_head.keys())[0]].shape[0], dtype=int)

        addr = ["NODES", gwf_name, "DIS"]
        wbaddr = self._gwf.get_var_address(*addr)
        nodes = self._gwf.get_value(wbaddr)[0]

        kijs = None
        if self.is_structured:
            kijs = PerfMeas.get_lrc(self._shape,list(nuser))
            kijs = {n:kij for n,kij in zip(nuser,kijs)}
        addr = ["NLAY", gwf_name, "DIS"]
        wbaddr = self._gwf.get_var_address(*addr)
        nlay = self._gwf.get_value(wbaddr)[0]

        dfs = []

        # boundary condition perturbations
        bnd_dict = PerfMeas.get_mf6_bound_dict()

        for paktype, pdict in org_sp_package_data.items():
            epsilons = []
            bound_idx = []
            nodes = []
            names = []
            pert_results_dict = {pm.name: [] for pm in self._performance_measures}
            print("running perturbations for ", paktype)
            for kk, infolist in pdict.items():
                for infodict in infolist:
                    bnd_items = infodict["bound"].shape[0]
                    for ibnd in range(min(bnd_items,2)):
                        new_bound = infodict["bound"].copy()
                        org = new_bound[ibnd]
                        delt = org * pert_mult
                        epsilons.append(delt - new_bound[ibnd])
                        new_bound[ibnd] = delt
                        pakname = infodict["packagename"]
                        pert_dict = {"kperkstp": kk, "packagename": pakname, "node": infodict["node"],
                                     "bound": new_bound}
                        print("...",pakname,ibnd,kk,org,delt,infodict["node"])

                        self._gwf = self._initialize_gwf(self._lib_name, self._flow_dir)
                        pert_head, pert_sp_dict = self.solve_gwf(verbose=False, _sp_pert_dict=pert_dict,pert_save=True)
                        pert_results = {pm.name: (pm.solve_forward(pert_head,pert_sp_dict) - base_results[pm.name]) / epsilons[-1]
                                        for pm in self._performance_measures}
                        for pm, result in pert_results.items():
                            pert_results_dict[pm].append(result)
                        bound_idx.append(ibnd)
                        nodes.append(infodict["node"])
                        if paktype == "wel6":
                            names.append("wel6_q")
                        elif paktype == "rch6":
                            names.append("rch6_recharge")
                        else:
                            names.append(pakname+"_"+bnd_dict[paktype][ibnd].format(ibnd))
            df = pd.DataFrame(pert_results_dict)
            df.loc[:, "node"] = nodes
            df.loc[:,"epsilon"] = epsilons
            df.loc[:,"addr"] = names
            df.index = df.pop("node") - 1
            df = df.loc[df.index != -1,:]
            df.index = df.index.map(lambda x: nuser[x])

            if kijs is not None:
                for idx, lab in zip([0, 1, 2], ["k", "i", "j"]):
                    df.loc[:, lab] = df.index.map(lambda x: kijs[x][idx])
            col_dict = {col:df.loc[:,col].to_dict() for col in df.columns}
            gdf = df.groupby(["node","addr"]).sum()
            gdf["node"] = gdf.index.get_level_values(0)
            gdf["addr"] = gdf.index.get_level_values(1)
            gdf.index = gdf.pop("node")
            for col in df.columns:
                if col in pert_results_dict:
                    continue
                if col in ["addr","node"]:
                    continue
                gdf.loc[:,col] = gdf.index.map(lambda x: col_dict[col][x])
            dfs.append(gdf)
        # property perturbations
        address = [["K11", gwf_name, "NPF"]]
        if nlay > 1:
            address.append(["K33", gwf_name, "NPF"])

        has_sto = False
        if PerfMeas.has_sto_iconvert(self._gwf):
            has_sto = True

        wbaddr = self._gwf.get_var_address(*address[0])
        inodes = self._gwf.get_value_ptr(wbaddr).shape[0]

        for addr in address:
            print("running perturbations for ", addr)
            pert_results_dict = {pm.name: [] for pm in self._performance_measures}
            wbaddr = self._gwf.get_var_address(*addr)

            epsilons = []

            for inode in range(inodes):
                self._gwf = self._initialize_gwf(self._lib_name, self._flow_dir)
                pert_arr = self._gwf.get_value_ptr(wbaddr)
                org = pert_arr[inode]
                delt = org * pert_mult
                epsilons.append(delt - pert_arr[inode])
                pert_arr[inode] = delt
                print("...",addr,inode,org,delt)
                pert_head, pert_sp_dict = self.solve_gwf(verbose=False, _force_k_update=True,pert_save=True)
                pert_results = {pm.name: (pm.solve_forward(pert_head,pert_sp_dict) - base_results[pm.name]) / epsilons[-1] for pm in
                                self._performance_measures}
                for pm, result in pert_results.items():
                    pert_results_dict[pm].append(result)

            df = pd.DataFrame(pert_results_dict)
            df.index = [nuser[inode] for inode in range(inodes)]
            df.index.name = "node"
            df.loc[:, "epsilon"] = epsilons
            if kijs is not None:
                for idx, lab in zip([0, 1, 2], ["k", "i", "j"]):
                    df.loc[:, lab] = df.index.map(lambda x: kijs[x][idx])
            tag = '_'.join(addr).lower()
            df.loc[:, "addr"] = tag
            dfs.append(df)

        if has_sto:
            import flopy
            if self._flow_dir != ".":
                test_dir = self._flow_dir + "_pert_temp"
            else:
                test_dir = "pert_temp"
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            sim = flopy.mf6.MFSimulation.load(sim_ws=self._flow_dir)
            gwf = sim.get_model()
            ss = gwf.sto.ss.array.copy().flatten()
            # this is an attempt to make sure we arent using "layered"
            gwf.sto.ss = ss

            sim.set_sim_path(test_dir)
            sim.set_all_data_external()
            sim.write_simulation()
            ss_arr_name = os.path.join(test_dir,"{0}.sto_ss.txt".format(gwf.name))
            if not os.path.exists(ss_arr_name):
                raise Exception("couldnt find ss_arr_name '{0}' needed for BS super hack")
            if os.path.exists(os.path.join(self._flow_dir,self._lib_name)):
                shutil.copy2(os.path.join(self._flow_dir,self._lib_name),os.path.join(test_dir,self._lib_name))

            print("running manual flopy based perturbations for sto ss")
            pert_results_dict = {pm.name: [] for pm in self._performance_measures}
            epsilons = []

            for inode in range(inodes):
                arr_node = nuser[inode]
                pert_arr = ss.copy()
                org = ss[arr_node]
                delt = org * pert_mult
                epsilons.append(delt - pert_arr[arr_node])
                print("...ss", inode, arr_node, org, delt)
                pert_arr[arr_node] = delt

                # reset the ss property
                np.savetxt(ss_arr_name,pert_arr.flatten(),fmt="%15.6E")

                self._gwf = self._initialize_gwf(self._lib_name, test_dir)
                pert_head, pert_sp_dict = self.solve_gwf(verbose=False, _force_k_update=True,pert_save=True)
                pert_results = {pm.name: (pm.solve_forward(pert_head,pert_sp_dict) - base_results[pm.name]) / epsilons[-1] for pm in
                                self._performance_measures}
                for pm, result in pert_results.items():
                    pert_results_dict[pm].append(result)
            df = pd.DataFrame(pert_results_dict)
            df.index = [nuser[inode] for inode in range(inodes)]
            df.index.name = "node"
            df.loc[:, "epsilon"] = epsilons
            if kijs is not None:
                for idx, lab in zip([0, 1, 2], ["k", "i", "j"]):
                    df.loc[:, lab] = df.index.map(lambda x: kijs[x][idx])
            tag = 'sto_ss'
            df.loc[:, "addr"] = tag
            dfs.append(df)

        df = pd.concat(dfs)
        df.index = df.index.values + 1
        df.index.name = "node"
        df.sort_index(inplace=True)
        df.to_csv("pert_results.csv")
