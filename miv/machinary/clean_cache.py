import glob
import logging
import pathlib
import shutil

import click


@click.command()
@click.option(
    "--path",
    "-p",
    default=".",
    type=click.Path(exists=True),
    help="Path to clean cache files in.",
)
@click.option(
    "--dry",
    "-d",
    is_flag=True,
    show_default=True,
    default=False,
    help="Dry run. No files will be deleted.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    show_default=True,
    default=False,
    help="Verbose output.",
)
def clean_cache(path, dry, verbose):
    """
    Clean all cache files, recursively, in the given path.
    """
    if dry:
        print("Dry run. No folders will be actually deleted.")
    pattern = pathlib.Path(path).glob("**/.cache")
    for path in pattern:
        if verbose:
            print(f"Removing: {path}", flush=True)
        if not dry:
            shutil.rmtree(path)
