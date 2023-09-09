__all__ = ["zip_results"]

from typing import List

import glob
import logging
import os
import pathlib
import shutil
import zipfile

import click


def is_path_valid(
    path: pathlib.Path, ignore_directory: List[str], ignore_extension: List[str]
):
    if path.is_file():
        if ignore_extension and path.suffix in ignore_extension:
            return False
    else:
        if not ignore_directory:
            return True

    if ignore_directory:
        for directory in path.as_posix().split("/"):
            if directory in ignore_directory:
                return False

    return True


def zip_directory_recursively(
    path: pathlib.Path,
    root_dir: pathlib.Path,
    zipfile_handle=None,
    ignore_directory=None,
    ignore_extension=None,
    verbose: bool = False,
):
    ignore_directory = ignore_directory or []
    ignore_extension = ignore_extension or []

    path = pathlib.Path(path)
    root_dir = pathlib.Path(root_dir)

    if path.is_file():
        if is_path_valid(path, ignore_directory, ignore_extension):
            if zipfile_handle is not None:  # dry run
                zipfile_handle.write(path, path.relative_to(root_dir))
        return

    if verbose:
        print(f"    {path.as_posix()}")
    for sub_dir in path.iterdir():
        if not is_path_valid(sub_dir, ignore_directory, ignore_extension):
            if verbose:
                print(f"    ignored: {sub_dir.as_posix()}")
            continue

        zip_directory_recursively(
            sub_dir,
            root_dir,
            zipfile_handle,
            ignore_directory,
            ignore_extension,
            verbose,
        )


@click.command()
@click.option(
    "--path",
    "-p",
    default=".",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Result path to compress",
)
@click.option(
    "--output-file",
    "-o",
    default="results.zip",
    type=click.Path(),
    help="Output file name",
)
@click.option(
    "--include-cache",
    is_flag=True,
    show_default=True,
    default=False,
    help="Include .cache folders.",
)
@click.option(
    "--dry",
    "-d",
    is_flag=True,
    show_default=True,
    default=False,
    help="Dry run. No files will be created.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    show_default=True,
    default=False,
    help="Verbose output.",
)
def zip_results(path, output_file, include_cache, dry, verbose):
    """
    Zip the result path.
    """
    if dry:
        print("Dry run. No files will be created.")
    if verbose:
        print(f"Compressing: {path}")
        print(f"Output file: {output_file}")
        if include_cache:
            print("Including .cache folders.")

    ignore_dir = [".cache"]
    ignore_ext = []
    if dry:
        zip_handle = None
    else:
        zip_handle = zipfile.ZipFile(output_file, "w")

    try:
        zip_directory_recursively(
            path, path, zip_handle, ignore_dir, ignore_ext, verbose
        )
    finally:
        if not dry:
            zip_handle.close()

    if verbose:
        print("Compression done.")
