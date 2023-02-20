__all__ = ["get_file", "check_file_hash"]

import typing
from typing import Any, Callable, Optional

import hashlib
import logging
import os
import pathlib
import shutil
import tarfile
import urllib
import zipfile
from urllib.request import urlopen

import numpy as np
from tqdm import tqdm


def get_file(
    file_url: str,
    directory: str,
    fname: str,
    file_hash: Optional[str] = None,
    archive_format: Optional[str] = "zip",
    cache_dir: str = "datasets",
    progbar_disable: bool = False,
):
    """
    Downloads a file from a URL if it not already in the directory.
    If the archive_format is provided, the downloaded file will be extracted.

    By default the file at the url `file_url` is downloaded to the
    `datasets/<directory>` with the filename `fname`.

    If the downloaded file is compressed or archived file, the parameter
    `archive_format` can be specified to extract the file after download.
    Currently supports tar (tar.gz, tar.bz) and zip archive format.

    The hash of the downloaded file can be checked by passing `file_hash`
    parameter. We use sha256 hashing algorithm. If `None`, validation process is
    skipped. Note, the hash validation is the only way to skip the download phase;
    if the file is expected to be large and take time to download, make sure
    you add file_hash.

    Parameters
    ----------
    file_url : str
        URL of the file.
    directory : str
        Subdirectory under the cache dir to save the file.
    fname : str
        Name of the downloaded file.
    file_hash : Optional[str]
        The expected hash string of the file (sha256) after download.
        Note, the only way to validate the file is by comparing hash.
        If hash is not give or not valid, the file will be re-downloaded.
        (default=None)
    archive_format :
        Archived format of the downloaded file.  Currently supports `tar`,
        `zip`, and `None`. None will skip the extraction step. (default="zip")
    cache_dir : str
        Location to store files. If not given, file will be installed in "datasets".
        (default="datasets")
    progbar_disable : bool
        Disable progress bar. (default=False)

    Returns
    -------
    path: str
        Path to the downloaded folder. If archived, path returns the
        unpacked directory.
    """

    # Prepare directory
    data_dir = os.path.join(cache_dir, directory)
    filepath = os.path.join(data_dir, fname)
    os.makedirs(data_dir, exist_ok=True)

    # Check hash
    download = True
    if os.path.exists(filepath) and file_hash is not None:
        # File found; verify integrity if a hash was provided.
        hash_check = check_file_hash(filepath, file_hash)
        if not hash_check:
            logging.info("Re-download the data: hash is invalid")
            download = True  # re-download if hash is different
        else:
            download = False

    # Download
    if download:
        logging.info(f"Downloading: {file_url}")

        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                _url_retrieve(file_url, filepath, progbar_disable=progbar_disable)
            except urllib.error.HTTPError as e:  # pragma: no cover
                raise Exception(error_msg.format(file_url, e.code, e.msg))
            except urllib.error.URLError as e:  # pragma: no cover
                raise Exception(error_msg.format(file_url, e.errno, e.reason))
        except (Exception, RuntimeError, KeyboardInterrupt):  # pragma: no cover
            # Remove residue upon error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise RuntimeError("Interrupted during downloading the file.")

        # Validate download
        if os.path.exists(filepath) and file_hash is not None:
            hash_check = check_file_hash(filepath, file_hash)
            if not hash_check:
                raise ValueError(
                    f"The sha256 file hash does not match "
                    f"the provided value of {file_hash}. If the issue "
                    f"remains, please report this error on GitHub issue."
                )
    else:
        logging.info("Downloading skipped: valid file already exists.")

    if archive_format:
        flag = _extract_archive(filepath, data_dir, archive_format)
        # remove file extension
        if archive_format.endswith((".gz", ".bz")):  # Corner case
            filepath = filepath[:-7]
        else:
            filepath = filepath[:-4]
        assert flag, (
            "File extraction is not completed properly."
            "Please report this error on GitHub issue."
        )

    return filepath


def check_file_hash(filepath: str, file_hash: str, packet_size: int = 65535) -> bool:
    """
    Validates a file using sha256 hash.

    Example
    -------
        >>> check_file_hash(
        ...     filepath='/path/to/file.zip',
        ...     file_hash='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        ... )
        True

    Parameters
    ----------
    filepath : str
        Path of the file to check hash
    file_hash : str
        The expected hash string
    packet_size : int
        Bytes to read at a time for large files hashing.

    Returns
    -------
    validation : bool
    """
    hasher = hashlib.sha256()  # consider adding md5 option for legacy archives
    with open(filepath, "rb") as file:
        for packet in iter(lambda: file.read(packet_size), b""):
            hasher.update(packet)
    hashed = hasher.hexdigest()
    return str(hashed) == file_hash


def _url_retrieve(
    url: str, filename: str, url_data: Optional[Any] = None, progbar_disable=False
):
    """
    Download file given URL

    Parameters
    ----------
    url : str
        Url to retrieve.
    filename : str
        Local storage filename
    url_data : Optional[Any]
        Data argument passed to `urlopen`.
    progbar_disable : bool
        Disable progress bar. (default=False)
    """

    def packet_read(response, packet_size=8192):
        content_type = response.info().get("Content-Length")
        if content_type is not None:
            total_size = int(content_type.strip())
        else:  # pragma: no cover
            raise FileNotFoundError("Invalid content. Please check your URL and file")
            total_size = -1
        progbar = tqdm(total=total_size // 1024**1, disable=progbar_disable)
        while True:
            packet = response.read(packet_size)
            progbar.update(packet_size)
            if packet:
                yield packet
            else:
                progbar.update(total_size - progbar.n)
                progbar.close()
                break

    response = urlopen(url, url_data)
    with open(filename, "wb") as fd:
        for packet in packet_read(response):
            fd.write(packet)


def _extract_archive(
    file_path, output_path: str = ".", archive_format: str = "zip"
) -> bool:
    """
    Extracts an archive.
    Supports .tar (tar, tar.gz, tar.bz) or .zip format.

    If any error is encountered, the downloaded file is deleted.

    Parameters
    ----------
    file_path : str
        path to the archive file
    output_path : str
        path to extract the archive file
    archive_format : str
        Format of archived file. (default='zip')
        Options are ['tar','tar.gz','tar.bz','zip'] files.

    Returns
    -------
    Completion : bool
        return true if an archive extraction was completed,
    """
    assert archive_format in [
        "zip",
        "tar",
        "tar.gz",
        "tar.bz",
    ], "Provided archive format is not yet supported."
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archive file ({file_path}) does not exist.")

    if archive_format.startswith("tar"):
        open_fn = tarfile.open
        is_match_fn = tarfile.is_tarfile
    elif archive_format == "zip":
        open_fn = zipfile.ZipFile
        is_match_fn = zipfile.is_zipfile

    if is_match_fn(file_path):
        with open_fn(file_path, "r") as archive:
            try:
                # Extraction attempts
                archive.extractall(output_path)
            except (
                zipfile.BadZipFile,
                tarfile.TarError,
                RuntimeError,
                KeyboardInterrupt,
            ):  # pragma: no cover
                # On failure, remove any residue packets
                if os.path.exists(output_path):
                    if os.path.isfile(output_path):
                        os.remove(output_path)
                    else:
                        shutil.rmtree(output_path)
                raise RuntimeError(
                    "Failure/Interrupted during extracting archive-file."
                )
        return True
    return False
