import pathlib as pl
from typing import Callable, Optional

import numpy as np

from .utils_modflow import SUPPORTED_PACKAGE_TYPES, get_node

PathLike = str | pl.Path


def get_adj_parse_context(
    gwf,
    gwf_name: str,
    is_structured: bool,
    unstructured_type: Optional[str],
) -> tuple[np.ndarray, np.ndarray, int, Optional[int]]:
    """Return context arrays and dimensions needed to parse an adjoint file.

    Parameters
    ----------
    gwf : modflowapi.ModflowApi
        Initialized MODFLOW 6 API instance.
    gwf_name : str
        Groundwater-flow model name.
    is_structured : bool
        Whether the model discretization is structured.
    unstructured_type : str, optional
        Unstructured discretization type (`disv` or `disu`).

    Returns
    -------
    tuple[ndarray, ndarray, int, int | None]
        Tuple of ``(nuser, nstp, nper, ncpl)`` where ``nuser`` is zero-based
        and ``ncpl`` is populated for unstructured models when applicable.
    """
    addr = ["NODEUSER", gwf_name.upper(), "DIS"]
    wbaddr = gwf.get_var_address(*addr)
    nuser = gwf.get_value(wbaddr) - 1

    nstp = gwf.get_value(gwf.get_var_address("NSTP", "TDIS"))
    nper = nstp.shape[0]

    ncpl = None
    if not is_structured:
        if unstructured_type == "disv":
            addr = ["NCPL", gwf_name.upper(), "DIS"]
            wbaddr = gwf.get_var_address(*addr)
            ncpl = int(np.asarray(gwf.get_value(wbaddr)).ravel()[0])
        elif unstructured_type == "disu":
            addr = ["NODES", gwf_name.upper(), "DIS"]
            wbaddr = gwf.get_var_address(*addr)
            ncpl = int(np.asarray(gwf.get_value(wbaddr)).ravel()[0])

    return nuser, nstp, nper, ncpl


def read_adj_file(
    adj_filename: PathLike,
    nuser: np.ndarray,
    nstp: np.ndarray,
    nper: int,
    ncpl: Optional[int],
    is_structured: bool,
    unstructured_type: Optional[str],
    shape: Optional[tuple[int, ...]],
    gwf_package_dict: dict[str, list[str]],
    record_factory: Callable[..., object],
    add_performance_measure: Callable[[str, list], None],
    current_hdf5_name: Optional[PathLike] = None,
    logger=None,
) -> Optional[pl.Path]:
    """Read and parse an adjoint input file.

    Parameters
    ----------
    adj_filename : PathLike
        Adjoint input filename.
    nuser : ndarray
        Reduced-node user mapping.
    nstp : ndarray
        Number of time steps per stress period.
    nper : int
        Number of stress periods.
    ncpl : int, optional
        Cells per layer for `disv`.
    is_structured : bool
        Whether the model discretization is structured.
    unstructured_type : str, optional
        Unstructured discretization type (`disv` or `disu`).
    shape : tuple[int, ...], optional
        Structured-grid shape used to convert layer-row-column locations to
        node numbers.
    gwf_package_dict : dict[str, list[str]]
        Mapping of package types to package names.
    record_factory : callable
        Constructor/callable used to create PM record objects.
    add_performance_measure : callable
        Callback used to store each parsed performance measure.
    current_hdf5_name : PathLike, optional
        Existing hdf5 name to preserve unless overridden by options.
    logger : logging.Logger, optional
        Optional logger for progress and debug context.

    Returns
    -------
    pathlib.Path or None
        Parsed hdf5 output filename from the options block, if present.

        Notes
        -----
        The input file structure closely follows MODFLOW 6 input files. Each
        performance measure is defined in a ``performance_measure`` block. Entries
        provide time location, spatial location, output type (``head`` or a flux
        package name), form (``direct``, ``residual``, or ``instantaneous``),
        weight, and optionally an observed value.

        Example entries for a structured model:

        - Direct head measure:
            ``25 3 3 10 34 head direct 1.0 -999``
        - Residual head measure:
            ``25 3 3 10 34 head residual 1.0 123.45``
        - Instantaneous head measure:
            ``25 3 3 10 34 head instantaneous 1.0 -999``

        An ``instantaneous`` measure uses the same value as a ``direct``
        measure, but it looks at each time step on its own instead of letting
        later time steps feed back into earlier ones. The result is the
        sensitivity at each time step by itself, rather than the total
        sensitivity over the whole run that ``direct`` gives. Time steps with no
        entry are skipped, and the overall (composite) result is the average
        over the observation times (each weighted by its time-step length)
        instead of the plain sum used for ``direct`` and ``residual`` measures.
        An ``instantaneous`` form works for both ``head`` and flux measures.

        At present, forms (``direct``, ``residual``, ``instantaneous``) cannot be
        mixed within a single performance measure, and types (``head`` vs flux
        package) also cannot be mixed within a single performance measure.
    """
    hdf5_name = None if current_hdf5_name is None else pl.Path(current_hdf5_name)

    with pl.Path(adj_filename).open("r") as f:
        count = 0
        parsed_pm_count = 0
        while True:
            line = f.readline()
            count += 1
            if line == "":
                break

            if is_empty_or_comment(line):
                continue

            line_strip = line.lower().strip()
            if line_strip.startswith("begin options"):
                count, hdf5_name = parse_adj_options_block(
                    f,
                    count,
                    current_hdf5_name=hdf5_name,
                )
            elif line_strip.startswith("begin performance_measure"):
                count, pm_name, pm_entries = parse_performance_measure_block(
                    f,
                    line,
                    count,
                    nuser,
                    nstp,
                    nper,
                    ncpl,
                    is_structured=is_structured,
                    unstructured_type=unstructured_type,
                    shape=shape,
                    gwf_package_dict=gwf_package_dict,
                    record_factory=record_factory,
                    logger=logger,
                )
                add_performance_measure(pm_name, pm_entries)
                parsed_pm_count += 1
            else:
                raise Exception(
                    f"unrecognized adj file input on line {count}: '{line}'"
                )

    if parsed_pm_count == 0:
        raise Exception("no PMs found in adj file")

    return hdf5_name


def is_empty_or_comment(line: str) -> bool:
    """Return whether a line is empty or a comment.

    Parameters
    ----------
    line : str
        Input text line.

    Returns
    -------
    bool
        True when the line is blank or starts with `#`.
    """
    stripped = line.strip()
    return len(stripped) == 0 or stripped[0] == "#"


def validate_pm_entry_length(
    raw: list[str],
    count: int,
    is_structured: bool,
    unstructured_type: Optional[str],
    logger=None,
) -> None:
    """Validate the number of tokens in a PM entry line.

    Parameters
    ----------
    raw : list[str]
        Tokenized PM entry line.
    count : int
        Source-line counter in the adjoint input file.
    is_structured : bool
        Whether the model discretization is structured.
    unstructured_type : str, optional
        Unstructured discretization type (`disv` or `disu`).
    logger : logging.Logger, optional
        Optional logger for debug context.
    """
    if is_structured and len(raw) != 9:
        if logger is not None:
            logger.info(f"Parsed line: {raw}")
        raise Exception(
            (
                f"performance measure entry on line {count} has "
                + f"the wrong number of items, found {len(raw)}, "
                + "should have 9"
            )
        )

    if not is_structured:
        if unstructured_type == "disv" and len(raw) != 8:
            if logger is not None:
                logger.info(f"Parsed line: {raw}")
            raise Exception(
                (
                    f"performance measure entry on line {count} has "
                    + f"the wrong number of items, found {len(raw)}, "
                    + "should have 8"
                )
            )
        if unstructured_type == "disu" and len(raw) != 7:
            if logger is not None:
                logger.info(f"Parsed line: {raw}")
            raise Exception(
                (
                    f"performance measure entry on line {count} has "
                    + f"the wrong number of items, found {len(raw)}, "
                    + "should have 7"
                )
            )


def validate_pm_time_indices(
    kper: int,
    kstp: int,
    nper: int,
    nstp: np.ndarray,
    count: int,
) -> None:
    """Validate PM stress-period and time-step indices.

    Parameters
    ----------
    kper : int
        Zero-based stress period index.
    kstp : int
        Zero-based time step index.
    nper : int
        Number of stress periods.
    nstp : ndarray
        Number of time steps per stress period.
    count : int
        Source-line counter in the adjoint input file.
    """
    if kper > nper - 1:
        raise Exception(f"kper > nper -1 on line number {count}")
    if kstp > nstp[kper] - 1:
        raise Exception(f"kstp > nstp[kper] -1 on line number {count}")


def validate_pm_type(
    pm_type: str,
    gwf_package_dict: dict[str, list[str]],
    logger=None,
) -> None:
    """Validate a PM type against available package names.

    Parameters
    ----------
    pm_type : str
        PM type token (`head` or a package instance name).
    gwf_package_dict : dict[str, list[str]]
        Mapping of package types to package names.
    logger : logging.Logger, optional
        Optional logger for debug context.
    """
    if pm_type == "head":
        return

    found = False
    ppnames = []
    for package_type, pnames in gwf_package_dict.items():
        if pm_type in pnames:
            found = True
            # the package is in the model, but its terms may not be formed, and
            # a measure of it would otherwise fail much later with no
            # explanation of why
            if package_type not in SUPPORTED_PACKAGE_TYPES:
                raise Exception(
                    f"`pm_type` {pm_type} is a {package_type} package, which "
                    "the adjoint does not form terms for, so a measure of it "
                    "has no sensitivity. The supported package types are "
                    + f"{', '.join(SUPPORTED_PACKAGE_TYPES)}."
                )
            break
        ppnames.extend(pnames)
    if not found:
        if logger is not None:
            logger.info(f"{ppnames}")
        raise Exception(
            f"`pm_type` {pm_type} names a GWF package instance that was not found"
        )


def map_reduced_node(
    inode: int,
    nuser: np.ndarray,
    kij: Optional[list[int]] = None,
    logger=None,
) -> int:
    """Map a full-grid node number to the reduced MODFLOW 6 node index.

    MODFLOW 6 may omit inactive cells from internal node numbering. This
    helper converts a full-grid node number into the corresponding index in
    the reduced node list used by the API.

    Parameters
    ----------
    inode : int
        Zero-based full-grid node number.
    nuser : ndarray
        Zero-based ``NODEUSER`` array from MODFLOW 6.
    kij : list[int], optional
        Structured-grid layer-row-column indices used only for additional
        debug context when a mapping fails.
    logger : logging.Logger, optional
        Optional logger used to emit debug context before raising.

    Returns
    -------
    int
        Zero-based reduced node index.
    """
    if len(nuser) <= 1:
        return inode

    nn = np.where(nuser == inode)[0]
    if nn.shape[0] != 1:
        if logger is not None:
            logger.info(f"{nuser} {nn}")
            if kij is not None:
                logger.info(f"{kij}")
        raise Exception(f"node num {inode} not in reduced node num")
    return int(nn[0])


def parse_pm_location(
    raw: list[str],
    count: int,
    line2: str,
    nuser: np.ndarray,
    ncpl: Optional[int],
    is_structured: bool,
    unstructured_type: Optional[str],
    shape: Optional[tuple[int, ...]],
    logger=None,
) -> tuple[int, Optional[int], Optional[int], Optional[int]]:
    """Parse one performance-measure location specification.

    The location syntax depends on the model discretization:

    - structured ``DIS`` input provides ``layer row column``
    - ``DISV`` input provides ``layer cell2d``
    - ``DISU`` input provides ``node``

    Parameters
    ----------
    raw : list[str]
        Tokenized performance-measure entry.
    count : int
        Source line number used in error messages.
    line2 : str
        Raw source line used in error messages.
    nuser : ndarray
        Zero-based ``NODEUSER`` array from MODFLOW 6.
    ncpl : int, optional
        Number of cells per layer for ``DISV`` and ``DISU`` parsing.
    is_structured : bool
        Whether the model discretization is structured.
    unstructured_type : str, optional
        Unstructured discretization type.
    shape : tuple[int, ...], optional
        Structured-grid shape used for ``DIS`` node conversion.
    logger : logging.Logger, optional
        Optional logger used to emit debug context before raising.

    Returns
    -------
    tuple[int, int | None, int | None, int | None]
        Tuple of ``(inode, k, i, j)`` where ``inode`` is always the zero-based
        reduced node index and the spatial indices are included when available
        for the active discretization.
    """
    i, j, k = None, None, None
    if is_structured:
        if shape is None:
            raise Exception("shape is None for structured PM location parsing")
        kij = []
        for idx in range(3):
            try:
                kij.append(int(raw[idx + 2]) - 1)
            except Exception as e:
                print(
                    f"{e}\n\nerror casting k-i-j info on " + f"line {count}: '{line2}'"
                )
        k, i, j = kij[0], kij[1], kij[2]
        inode = get_node(shape, [kij])[0]
        inode = map_reduced_node(inode, nuser, kij=kij, logger=logger)
        return inode, k, i, j

    if unstructured_type == "disv":
        try:
            lay = int(raw[2])
        except Exception as e:
            print(
                f"{e}\n\nerror casting layer info info on " + f"line {count}: '{line2}'"
            )
        k = lay - 1
        try:
            node = int(raw[3])
        except Exception as e:
            print(
                f"{e}\n\nerror casting layer info info on " + f"line {count}: '{line2}'"
            )

        if ncpl is None:
            raise Exception("ncpl is None for disv parsing")
        inode = ((ncpl * (lay - 1)) + node) - 1
        inode = map_reduced_node(inode, nuser, kij=None, logger=logger)
        return inode, k, i, j

    if unstructured_type == "disu":
        try:
            inode = int(raw[2]) - 1
        except Exception as e:
            print(f"{e}\n\nerror casting node info on line {count}: '{line2}'")
        inode = map_reduced_node(inode, nuser, kij=None, logger=logger)
        return inode, k, i, j

    raise Exception(
        "gwf6 model discretization is not dis, disu, or disv "
        + f"({unstructured_type})"
    )


def parse_adj_options_block(
    f,
    count: int,
    current_hdf5_name: Optional[PathLike] = None,
) -> tuple[int, Optional[pl.Path]]:
    """Parse an `options` block from an adjoint input file.

    Parameters
    ----------
    f : file-like
        Open adjoint-file handle positioned after `begin options`.
    count : int
        Current source-line counter.
    current_hdf5_name : PathLike, optional
        Existing hdf5 name to preserve unless overridden.

    Notes
    -----
    The only currently supported option is ``hdf5_name``. Any other option
    line raises an exception so that unsupported input is not silently ignored.

    Returns
    -------
    tuple[int, pathlib.Path | None]
        Updated line counter and the resolved HDF5 filename, if one is set.
    """
    hdf5_name = None if current_hdf5_name is None else pl.Path(current_hdf5_name)
    while True:
        line2 = f.readline()
        count += 1
        if line2 == "":
            raise EOFError("EOF while reading options")
        if is_empty_or_comment(line2):
            continue

        line2s = line2.lower().strip()
        if line2s.startswith("begin"):
            raise Exception("a new begin block found while parsing options")
        if line2s.startswith("end options"):
            break

        parts = line2s.split()
        if parts[0] == "hdf5_name":
            hdf5_name = pl.Path(line2.strip().split()[1])
        else:
            raise Exception("unrecognized option line:" + line2.strip())

    return count, hdf5_name


def parse_performance_measure_block(
    f,
    line: str,
    count: int,
    nuser: np.ndarray,
    nstp: np.ndarray,
    nper: int,
    ncpl: Optional[int],
    is_structured: bool,
    unstructured_type: Optional[str],
    shape: Optional[tuple[int, ...]],
    gwf_package_dict: dict[str, list[str]],
    record_factory: Callable[..., object],
    logger=None,
) -> tuple[int, str, list]:
    """Parse a `performance_measure` block from an adjoint input file.

    Parameters
    ----------
    f : file-like
        Open adjoint-file handle positioned on the line after `begin`.
    line : str
        The `begin performance_measure ...` line.
    count : int
        Current source-line counter.
    nuser : ndarray
        Reduced-node user mapping.
    nstp : ndarray
        Number of time steps per stress period.
    nper : int
        Number of stress periods.
    ncpl : int, optional
        Cells per layer for `disv`.
    is_structured : bool
        Whether the model discretization is structured.
    unstructured_type : str, optional
        Unstructured discretization type (`disv` or `disu`).
    shape : tuple[int, ...], optional
        Structured-grid shape used for ``DIS`` node conversion.
    gwf_package_dict : dict[str, list[str]]
        Mapping of package types to package names.
    record_factory : callable
        Constructor/callable used to create PM record objects.
    logger : logging.Logger, optional
        Optional logger for debug context.

    Returns
    -------
    tuple[int, str, list]
        Updated line count, PM name, and list of PM records.
    """
    raw = line.lower().strip().split()
    if len(raw) != 3:
        raise Exception(
            f"'begin' line {count} has wrong number of items, should be 3, "
            f"not {len(raw)}"
        )

    pm_name = raw[2].strip().lower()
    pm_entries = []

    while True:
        line2 = f.readline()
        count += 1
        if line2 == "":
            raise EOFError(f"EOF while reading performance_measure block '{line}'")
        if is_empty_or_comment(line2):
            continue

        line2s = line2.lower().strip()
        if line2s.startswith("begin"):
            raise Exception(
                "a new begin block found while parsing "
                + f"performance_measure block '{line}'"
            )
        if line2s.startswith("end performance_measure"):
            break
        if line2s.startswith("open"):
            fname = line2.split()[1]
            if not pl.Path(fname).exists():
                raise Exception(f"External file '{fname}' not found")
            raise NotImplementedError()

        raw = line2s.split()
        validate_pm_entry_length(
            raw,
            count,
            is_structured=is_structured,
            unstructured_type=unstructured_type,
            logger=logger,
        )

        kper = int(raw[0]) - 1
        kstp = int(raw[1]) - 1
        validate_pm_time_indices(kper, kstp, nper, nstp, count)

        inode, k, i, j = parse_pm_location(
            raw,
            count,
            line2,
            nuser,
            ncpl,
            is_structured,
            unstructured_type,
            shape,
            logger,
        )

        obsval = float(raw[-1])
        weight = float(raw[-2])
        pm_form = raw[-3].strip().lower()
        pm_type = raw[-4].strip().lower()
        validate_pm_type(pm_type, gwf_package_dict=gwf_package_dict, logger=logger)

        pm_entries.append(
            record_factory(
                kper,
                kstp,
                inode,
                pm_type,
                pm_form,
                weight,
                obsval,
                k,
                i,
                j,
            )
        )

    return count, pm_name, pm_entries
