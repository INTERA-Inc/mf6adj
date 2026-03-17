import os
import pathlib as pl
from contextlib import contextmanager
from typing import Generator


@contextmanager
def context_cd(newdir: pl.Path) -> Generator[None, None, None]:
    prevdir = pl.Path().cwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)
