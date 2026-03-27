from typing import Optional

import h5py
import numpy as np


def write_group_to_hdf(
    hdf: h5py.File,
    group_name: str,
    data_dict: dict,
    attr_dict: Optional[dict] = None,
    grid_shape: Optional[tuple] = None,
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
        Datasets to write. Lists are converted to ndarrays. Nested
        dictionaries are written as sub-groups (ndarray values become
        datasets; scalar values become group attributes).
    attr_dict : dict, optional
        Group attributes to write.
    grid_shape : tuple, optional
        Structured-grid shape ``(nlay, nrow, ncol)``. When supplied together
        with *nodeuser* the node-ordered arrays are mapped onto the full grid.
    nodeuser : ndarray, optional
        ``nodeuser`` array from MODFLOW 6 (zero-based).
    nodereduced : ndarray, optional
        ``nodereduced`` array from MODFLOW 6 (zero-based).
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
        kijs = list(zip(*np.unravel_index(list(nodeuser), grid_shape)))

    for tag, item in data_dict.items():
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            if grid_shape is not None and nodeuser is not None:
                if len(item) == len(nodeuser):
                    arr = np.zeros(grid_shape, dtype=item.dtype)
                    for kij, v in zip(kijs, item):
                        arr[kij] = v
                    _ = grp.create_dataset(tag, grid_shape, dtype=item.dtype, data=arr)
                else:
                    raise Exception(
                        f"write_group_to_hdf: array length mismatch for '{tag}'"
                    )
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
                f"write_group_to_hdf: unrecognized data_dict entry '{tag}', "
                f"type: {type(item)}"
            )

    if nodeuser is not None:
        _ = grp.create_dataset(
            "nodeuser", nodeuser.shape, dtype=nodeuser.dtype, data=nodeuser
        )
    if kijs is not None:
        for idx, name in enumerate(["k", "i", "j"]):
            arr = np.array([kij[idx] for kij in kijs], dtype=int)
            _ = grp.create_dataset(name, arr.shape, dtype=arr.dtype, data=arr)
