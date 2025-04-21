import os
import pathlib as pl


def find_files_with_extension(directory, extension):
    """
    Finds all files with the specified extension in the given directory and its subdirectories.

    Args:
        directory: The path to the directory to search.
        extension: The file extension to search for (e.g., ".txt", ".pdf").

    Returns:
        A list of Path objects representing the files found.
    """
    path = pl.Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a valid directory.")

    return list(path.rglob(f"*{extension}"))

directory_path = ".."
file_extension = ".ipynb"
notebook_count = 0
try:
    files = find_files_with_extension(directory_path, file_extension)
    if files:
        print(f"Files with extension '{file_extension}' found in '{directory_path}':")
        for file in files:
            print(f"clearing...{file}")
            os.system(f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {file}")
            notebook_count += 1
    else:
        print(f"No files with extension '{file_extension}' found in '{directory_path}'.")
    print(f"notebooks cleared...{notebook_count}")

except NotADirectoryError as e:
    print(e)
