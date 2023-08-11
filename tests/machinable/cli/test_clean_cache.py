import os
import pathlib

import pytest
from click.testing import CliRunner

from miv.machinary.clean_cache import clean_cache


@pytest.mark.parametrize("path", [".", "test", "test/test2"])
def test_clean_cache_folders(tmp_path, path):
    content = "hello"

    # Create ".cache" directory in tmp_path
    cache_dir = tmp_path / path / ".cache"
    cache_dir.mkdir(parents=True)
    p = cache_dir / "hello.txt"
    p.write_text(content)

    assert p.exists()
    assert p.read_text() == content

    # Run the clean_cache command
    runner = CliRunner()
    result = runner.invoke(clean_cache, ["--path", str(tmp_path)])

    # Check that the cache directory is gone
    assert result.exit_code == 0
    assert not cache_dir.exists()


def test_clean_cache_multiple_folders(tmp_path):
    content = "hello"
    paths = [".", "test", "test/test2"]

    # Create ".cache" directory in tmp_path
    for path in paths:
        cache_dir = tmp_path / path / ".cache"
        cache_dir.mkdir(parents=True)
        p = cache_dir / "hello.txt"
        p.write_text(content)

        assert p.exists()
        assert p.read_text() == content

    # Run the clean_cache command
    runner = CliRunner()
    result = runner.invoke(clean_cache, ["--path", str(tmp_path)])

    # Check that the cache directory is gone
    assert result.exit_code == 0
    for path in paths:
        cache_dir = tmp_path / path / ".cache"
        assert not cache_dir.exists()


@pytest.mark.parametrize("path", [".", "test", "test/test2"])
def test_clean_cache_dryrun(tmp_path, path):
    content = "hello"

    # Create ".cache" directory in tmp_path
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True)
    p = cache_dir / "hello.txt"
    p.write_text(content)

    assert p.exists()
    assert p.read_text() == content

    # Run the clean_cache command
    runner = CliRunner()
    result = runner.invoke(clean_cache, ["--path", str(tmp_path), "--dry"])

    # Check that the cache directory is gone
    assert result.exit_code == 0
    assert cache_dir.exists()


def test_clean_cache_verbose(tmp_path):
    content = "hello"

    # Create ".cache" directory in tmp_path
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True)
    p = cache_dir / "hello.txt"
    p.write_text(content)

    assert p.exists()
    assert p.read_text() == content

    # Run the clean_cache command
    runner = CliRunner()
    result = runner.invoke(clean_cache, ["--path", str(tmp_path), "-v"])

    # Check that the cache directory is gone
    assert result.exit_code == 0
    assert cache_dir.as_posix() in result.output
    assert not cache_dir.exists()
