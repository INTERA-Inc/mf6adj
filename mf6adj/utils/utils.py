import os
import pathlib as pl
from contextlib import contextmanager
from typing import Generator


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
