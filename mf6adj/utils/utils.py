import os
import pathlib as pl
import platform
from contextlib import contextmanager
from typing import Generator

import numpy as np


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


@contextmanager
def utils_cd(newdir: pl.Path) -> Generator[None, None, None]:
    """Temporarily change the current working directory.

    Parameters
    ----------
    newdir : pl.Path
        Directory to enter for the duration of the context.

    Yields
    ------
    None
        Context manager that restores the original directory on exit.
    """
    prevdir = pl.Path().cwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)


def get_conda_mf6_paths(
    bin_path="bin", lib_path="lib"
) -> tuple[pl.Path | None, pl.Path | None]:
    """
    Locate the MODFLOW 6 executable and shared library in the active conda environment.

    Resolves platform-specific paths for the ``mf6`` executable and
    ``libmf6`` shared library using the ``CONDA_PREFIX`` environment variable.
    On Windows, ``bin_path`` is overridden to ``"Scripts"`` and the
    appropriate extensions (``.exe``, ``.dll``) are applied automatically.

    Parameters
    ----------
    bin_path : str, optional
        Subdirectory of the conda environment containing executables.
        Default is ``"bin"``. Overridden to ``"Scripts"`` on Windows.
    lib_path : str, optional
        Subdirectory of the conda environment to search first for the
        shared library. Default is ``"lib"``. Falls back to *bin_path*
        if the library is not found there.

    Returns
    -------
    mf6_path : pl.Path or None
        Absolute path to the ``mf6`` executable, or ``None`` if not found.
    libmf6_path : pl.Path or None
        Absolute path to the ``libmf6`` shared library, or ``None`` if not found.

    Raises
    ------
    RuntimeError
        If the ``CONDA_PREFIX`` environment variable is not set.
    """
    env_path_str = os.environ.get("CONDA_PREFIX")
    if env_path_str is None:
        raise RuntimeError("CONDA_PREFIX environment variable not set.")
    env_path = pl.Path(env_path_str).expanduser().resolve()

    exe_ext = ""
    if "linux" in platform.platform().lower():
        lib_ext = ".so"
    elif (
        "darwin" in platform.platform().lower()
        or "macos" in platform.platform().lower()
    ):
        lib_ext = ".dylib"
    else:
        bin_path = "Scripts"
        lib_ext = ".dll"
        exe_ext = ".exe"
    mf6_path = env_path / f"{bin_path}/mf6{exe_ext}"

    if not mf6_path.is_file():
        print(
            f" MODFLOW 6 executable not found at {mf6_path.parent}. "
            + "Please ensure MODFLOW 6 is installed in the current conda "
            + "environment. FloPy get-modflow can be used to install MODFLOW 6 "
            + "if it is not present."
        )
        mf6_path = None

    libmf6_path = env_path / f"{lib_path}/libmf6{lib_ext}"
    if not libmf6_path.is_file():
        libmf6_path = env_path / f"{bin_path}/libmf6{lib_ext}"

    if not libmf6_path.is_file():
        print(
            f" libmf6{lib_ext} not found in {env_path}. "
            + "Please ensure MODFLOW 6 shared library is installed in "
            + "the current conda environment. FloPy get-modflow can be used to "
            + "install the shared library if it is not present."
        )
        libmf6_path = None

    return mf6_path, libmf6_path
