from typing import Optional

import h5py
import numpy as np


def _write_group_to_hdf(
    hdf: h5py.File,
    group_name: str,
    data_dict: dict,
    attr_dict: Optional[dict] = None,
    grid_shape: Optional[tuple] = None,
    nodeuser: Optional[np.ndarray] = None,
    nodereduced: Optional[np.ndarray] = None,
    logger: Optional[object] = None,
) -> None:
    """Write a dictionary of arrays and metadata to a new group in an open HDF5 file.

    Arrays in *data_dict* are assumed to be in reduced-node order (one value
    per active cell). If *grid_shape* and *nodeuser* are given, each array is
    mapped onto a full ``(nlay, nrow, ncol)`` grid and the ``nodeuser``, ``k``,
    ``i``, and ``j`` index arrays are also saved. If only *nodeuser* and
    *nodereduced* are given, each array is scattered into a full user-node-
    length array using *nodeuser* as the index map and ``nodeuser`` is saved.
    If no mapping arrays are given, arrays are written as-is.

    Nested dicts in *data_dict* become HDF5 sub-groups (ndarray values as
    datasets, scalars as attributes). A CSR sparse matrix keyed ``"amat"`` is
    stored as a sub-group containing its ``data``, ``indices``, and ``indptr``
    datasets plus ``shape`` and ``format`` attributes.

    Parameters
    ----------
    hdf : h5py.File
        Open, writable HDF5 file handle.
    group_name : str
        Name of the new HDF5 group to create.
    data_dict : dict
        Data to write. Keys become dataset or sub-group names.
    attr_dict : dict, optional
        Metadata to attach as attributes on the top-level group.
    grid_shape : tuple of int, optional
        Structured-grid shape ``(nlay, nrow, ncol)`` for full-grid mapping.
    nodeuser : ndarray of int, optional
        Zero-based MODFLOW 6 ``nodeuser`` array (reduced node → user node).
    nodereduced : ndarray of int, optional
        Zero-based MODFLOW 6 ``nodereduced`` array (user node → reduced node).
        Its length sets the output array size in unstructured mapping mode.
    logger : object, optional
        Logger for warnings about unrecognised data types; falls back to
        ``print`` if not provided.

    Raises
    ------
    Exception
        If *group_name* already exists in *hdf*, or if an array length does
        not match *nodeuser* during structured-grid mapping.
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
                        f"_write_group_to_hdf: array length mismatch for '{tag}'"
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
        elif tag == "amat":
            subgrp = grp.create_group(tag)
            _ = subgrp.create_dataset("data", data=item.data)
            _ = subgrp.create_dataset("indices", data=item.indices)
            _ = subgrp.create_dataset("indptr", data=item.indptr)

            # Save shape and format as attributes for reconstruction
            subgrp.attrs["shape"] = item.shape
            subgrp.attrs["format"] = "csr"
        else:
            msg = (
                f"_write_group_to_hdf: unrecognized data_dict entry '{tag}', "
                + f"type: {type(item)}"
            )
            if logger is not None:
                logger.warning(msg)
            else:
                print(msg)

    if nodeuser is not None:
        _ = grp.create_dataset(
            "nodeuser", nodeuser.shape, dtype=nodeuser.dtype, data=nodeuser
        )
    if kijs is not None:
        for idx, name in enumerate(["k", "i", "j"]):
            arr = np.array([kij[idx] for kij in kijs], dtype=int)
            _ = grp.create_dataset(name, arr.shape, dtype=arr.dtype, data=arr)
