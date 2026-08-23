import pathlib as pl
from typing import Any, Union

import numpy as np

PathLike = Union[str, pl.Path]


def parse_models_block(f) -> tuple[dict[str, str], dict[str, str]]:
    """Parse the `MODELS` block from an `mfsim.nam` file.

    Parameters
    ----------
    f : file-like
        Open file handle positioned immediately after the `BEGIN MODELS`
        line.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Pair of dictionaries. The first maps model names to model types,
        and the second maps model names to model name files.
    """
    model_dict: dict[str, str] = {}
    namfile_dict: dict[str, str] = {}
    while True:
        line = f.readline()
        if line == "":
            raise EOFError("EOF when reading 'models' block")

        line_strip = line.strip().lower()
        if line_strip.startswith("end") and "models" in line_strip:
            break
        if line_strip == "" or line_strip.startswith("#"):
            continue

        raw = line_strip.split()
        if len(raw) < 3:
            raise Exception(f"wrong number of items on line: {line.strip()}")
        if raw[2] in model_dict:
            raise Exception(f"duplicate model name found: '{raw[2]}'")
        model_dict[raw[2]] = raw[0]
        namfile_dict[raw[2]] = raw[1]
    return model_dict, namfile_dict


def get_model_names_from_mfsim(
    sim_ws: PathLike,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return model names from an `mfsim.nam` file.

    Parameters
    ----------
    sim_ws : PathLike
        Simulation workspace containing `mfsim.nam`.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Pair of dictionaries. The first maps model names to model types,
        and the second maps model names to model name files.
    """
    sim_nam = pl.Path(sim_ws) / "mfsim.nam"
    if not sim_nam.exists():
        raise Exception(f"simulation nam file '{sim_nam}' not found")

    with sim_nam.open("r") as f:
        for line in f:
            line_strip = line.strip().lower()
            if line_strip.startswith("begin") and "models" in line_strip:
                return parse_models_block(f)

    raise EOFError("EOF when looking for 'models' block")


def parse_packages_block(f) -> dict[str, list[str]]:
    """Parse the `PACKAGES` block from a GWF name file.

    Parameters
    ----------
    f : file-like
        Open file handle positioned immediately after the `BEGIN PACKAGES`
        line.

    Returns
    -------
    dict[str, list[str]]
        Mapping of package types to package names.
    """
    package_dict: dict[str, list[str]] = {}
    count_dict: dict[str, int] = {}

    while True:
        line = f.readline()
        if line == "":
            raise EOFError("EOF when reading 'packages' block")

        line_strip = line.strip().lower()
        if line_strip.startswith("end") and "packages" in line_strip:
            break
        if line_strip == "" or line_strip.startswith("#"):
            continue

        data = line.split("#", 1)[0].strip().lower()
        if data == "":
            continue

        raw = data.split()
        if len(raw) < 2:
            raise Exception(f"wrong number of items on line: {line}")

        package_type = raw[0]
        tag_name = raw[2] if len(raw) > 2 else None

        if package_type not in count_dict:
            count_dict[package_type] = 1
        if package_type not in package_dict:
            package_dict[package_type] = []

        if tag_name is None:
            tag_name = package_type.replace("6", "") + f"-{count_dict[package_type]}"

        package_dict[package_type].append(tag_name)
        count_dict[package_type] += 1

    return package_dict


def get_package_names_from_gwfname(
    gwf_nam_file: PathLike,
) -> dict[str, list[str]]:
    """Return package names from a GWF name file.

    Parameters
    ----------
    gwf_nam_file : PathLike
        Groundwater-flow name file.

    Returns
    -------
    dict[str, list[str]]
        Mapping of package types to package names.
    """
    gwf_nam_file = pl.Path(gwf_nam_file)
    if not gwf_nam_file.exists():
        raise Exception(f"gwf nam file '{gwf_nam_file}' not found")

    with gwf_nam_file.open("r") as f:
        for line in f:
            line_strip = line.strip().lower()
            if line_strip.startswith("begin") and "packages" in line_strip:
                return parse_packages_block(f)

    raise EOFError("EOF when looking for 'packages' block")


def get_mf6_bound_dict() -> dict[str, dict[int, str]]:
    """Return MODFLOW 6 boundary-array index metadata.

    Returns
    -------
    dict[str, dict[int, str]]
        Mapping from package type to the column indices and names used in
        boundary `bound` arrays.
    """
    return {
        "ghb6": {0: "bhead", 1: "cond"},
        "drn6": {0: "elev", 1: "cond"},
        "riv6": {0: "stage", 1: "cond"},
        "sfr6": {0: "stage", 1: "cond"},
        "lak6": {0: "stage", 1: "cond"},
    }


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


def get_ptr_from_gwf(gwf_name, pak_name, prop_name, gwf) -> Any:
    """Return a mutable reference to a quantity from the MODFLOW 6 API.

    Unlike :func:`get_value_from_gwf`, this function returns the live MODFLOW 6
    array reference so callers can update values in place.

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
        Node numbers corresponding to the supplied indices.
    """
    if not isinstance(lrc_list, list):
        lrc_list = [lrc_list]
    multi_index = tuple(np.array(lrc_list).T)
    return np.ravel_multi_index(multi_index, shape).tolist()


def get_lrc(shape, nodes) -> list[tuple[int, ...]]:
    """Return layer-row-column indices for a list of node numbers.

    Parameters
    ----------
    shape : tuple[int, ...]
        Model grid shape.
    nodes : list or int
        Node numbers.

    Returns
    -------
    list[tuple[int, ...]]
        Layer-row-column indices for the supplied node numbers.
    """
    if isinstance(nodes, int):
        nodes = [nodes]
    return list(zip(*np.unravel_index(nodes, shape)))


def has_sto_iconvert(gwf) -> bool:
    """Return whether the forward model has an STO package with `ICONVERT`.

    Parameters
    ----------
    gwf : modflowapi.ModflowApi
        MODFLOW 6 groundwater-flow API instance.

    Returns
    -------
    bool
        True if the model has STO `ICONVERT`, otherwise False.
    """
    names = [
        n for n in list(gwf.get_input_var_names()) if "STO" in n and "ICONVERT" in n
    ]
    return len(names) > 0
