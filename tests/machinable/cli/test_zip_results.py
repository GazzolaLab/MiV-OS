import os
import pathlib
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

import miv.machinary.zip_results as module

_FILENAME1 = "SAFE_TO_REMOVE.zip"


@pytest.fixture(scope="session")
def result_directory(tmp_path_factory):
    """
    Create structure like:
    tmp_path
    └── results
        ├── .cache
        │   └── cache.txt
        └── hello.txt

    """
    content = "hello"

    # Create ".cache" directory in tmp_path
    result_dir = tmp_path_factory.mktemp("results")
    cache_dir = result_dir / ".cache"
    cache_dir.mkdir(parents=True)
    c = cache_dir / "cache.txt"
    c.write_text(content)
    p = result_dir / "hello.txt"
    p.write_text(content)

    # Make sure contents are created
    assert c.exists()
    assert c.read_text() == content
    assert p.exists()
    assert p.read_text() == content

    return result_dir


def test_zip_folder_running(tmp_path, result_directory):
    import zipfile

    # Run the clean_cache command
    runner = CliRunner()
    result = runner.invoke(
        module.zip_results,
        ["--path", result_directory, "--output-file", tmp_path / _FILENAME1],
    )

    # Check that the cache directory is gone
    assert result.exit_code == 0
    assert (tmp_path / _FILENAME1).exists()

    # Check file structure
    with zipfile.ZipFile(tmp_path / _FILENAME1, "r") as zip_ref:
        zip_ref.extractall(tmp_path / "extracted")
    assert (tmp_path / "extracted" / "hello.txt").exists()
    assert not (tmp_path / "extracted" / ".cache").exists()


def test_zip_folder_dry_running(tmp_path, result_directory):
    # Run the clean_cache command
    runner = CliRunner()
    result = runner.invoke(
        module.zip_results,
        ["--path", result_directory, "--output-file", tmp_path / _FILENAME1, "--dry"],
    )

    # Check that the cache directory is gone
    assert result.exit_code == 0
    assert not (tmp_path / _FILENAME1).exists()


def test_is_path_valid(tmp_path):
    file = tmp_path / "hello.txt"
    file.write_text("hello")

    assert module.is_path_valid(tmp_path, [], [])
    assert not module.is_path_valid(
        tmp_path, ignore_directory=[tmp_path.stem], ignore_extension=[]
    )
    assert module.is_path_valid(file, [], [])
    assert not module.is_path_valid(
        file, ignore_extension=[".txt"], ignore_directory=[]
    )


def test_zip_directory_recursively_single_file(tmp_path, capfd):
    file = tmp_path / "hello.txt"
    file.write_text("hello")

    with patch(
        f"{__name__}.module.is_path_valid", return_value=True
    ) as mock_is_path_valid:
        arg1, arg2 = ["test1"], ["test2"]
        mock_zipfile_handle = MagicMock()
        module.zip_directory_recursively(
            file, tmp_path, mock_zipfile_handle, arg1, arg2, verbose=True
        )
        mock_is_path_valid.assert_called_once_with(file, arg1, arg2)
        mock_zipfile_handle.write.assert_called_once_with(
            file, file.relative_to(tmp_path)
        )

        output, error = capfd.readouterr()
        assert tmp_path.as_posix() in output


def test_zip_directory_recursively_single_file_negative(tmp_path):
    file = tmp_path / "hello.txt"
    file.write_text("hello")

    with patch(
        f"{__name__}.module.is_path_valid", return_value=False
    ) as mock_is_path_valid:
        mock_zipfile_handle = MagicMock()
        module.zip_directory_recursively(file, tmp_path, mock_zipfile_handle)
        mock_is_path_valid.assert_called_once_with(file, [], [])
        mock_zipfile_handle.write.assert_not_called()
