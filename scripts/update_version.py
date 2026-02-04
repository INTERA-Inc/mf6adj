import argparse
import re
import textwrap
from datetime import datetime
from pathlib import Path

from filelock import FileLock
from packaging.version import Version

_epilog = """\
Update version information stored in version.txt in the project root,
as well as several other files in the repository. If --version is not
provided, the version number will not be changed. A file lock is held
to synchronize file access. The version tag must comply with standard
'<major>.<minor>.<patch>' format conventions for semantic versioning.
To show the version without changing anything, use --get (short -g).
"""
_project_name = "mf6adj"
_project_root_path = Path(__file__).parent.parent
_version_py_path = _project_root_path / _project_name / "version.py"

# file names and the path to the file relative to the repo root directory
file_paths_list = [
    _project_root_path / "CITATION.cff",
    _project_root_path / "README.md",
    _project_root_path / "docs" / "PyPI_release.md",
    _project_root_path / "mf6adj" / "version.py",
]
file_paths = {pth.name: pth for pth in file_paths_list}  # keys for each file


def split_nonnumeric(s):
    match = re.compile(r"[^0-9]").search(s)
    return [s[: match.start()], s[match.start() :]] if match else s


def read_version(filename):
    """Reads the value of __version__ from a file without importing it."""
    # Open in Latin-1 encoding to avoid errors with non-ASCII characters
    with open(filename, "r", encoding="latin-1") as f:
        content = f.read()

    # Use a regex to find the __version__ assignment
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", content, re.MULTILINE)
    if match:
        return match.group(1)
    else:
        raise RuntimeError(f"Unable to find version string in {filename}.")


_current_version = Version(read_version(_version_py_path))


def update_version_py(timestamp: datetime, version: Version):
    with open(_version_py_path, "w") as f:
        f.write(
            f"# {_project_name} version file automatically created using\n"
            f"# {Path(__file__).name} on {timestamp:%B %d, %Y %H:%M:%S}\n\n"
        )
        f.write(f'__version__ = "{version}"\n')
        f.close()
    print(f"Updated {_version_py_path} to version {version}")


def update_version(
    timestamp: datetime = datetime.now(),
    version: Version = None,
):
    lock_path = Path(_version_py_path.name + ".lock")
    try:
        lock = FileLock(lock_path)
        previous = Version(read_version(_version_py_path))
        version = (
            version
            if version
            else Version(previous.major, previous.minor, previous.micro)
        )

        with lock:
            update_version_py(timestamp, version)
    finally:
        try:
            lock_path.unlink()
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"Update {_project_name} version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(_epilog),
    )
    parser.add_argument(
        "-v",
        "--version",
        required=False,
        help="Specify the release version",
    )
    parser.add_argument(
        "-g",
        "--get",
        required=False,
        action="store_true",
        help="Just get the current version number, no updates (defaults false)",
    )
    args = parser.parse_args()

    if args.get:
        print(_current_version)
    else:
        update_version(
            timestamp=datetime.now(),
            version=(Version(args.version) if args.version else _current_version),
        )
