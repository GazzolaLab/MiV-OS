import pathlib

import pytest

from miv.core.pipeline import Pipeline
from tests.core.mock_chain import MockChainRunnable, MockChainRunnableWithCache


def test_pipeline():
    pipeline = Pipeline(MockChainRunnable(1))
    assert pipeline is not None


@pytest.fixture
def pipeline():
    a = MockChainRunnable(1)
    b = MockChainRunnable(2)
    c = MockChainRunnable(3)
    d = MockChainRunnable(4)
    e = MockChainRunnable(5)
    a >> b >> d
    a >> c >> e >> d
    return Pipeline(e)


def test_pipeline_run(tmp_path, pipeline):
    pipeline.run(tmp_path / "results", verbose=True)


def test_pipeline_summarize(pipeline):
    pipeline.summarize()


def test_pipeline_execution_count(tmp_path):
    a = MockChainRunnable(1)
    b = MockChainRunnable(2)
    c = MockChainRunnable(3)

    a >> b >> c

    # Note, Pipeline-run itself should not invoke chain
    Pipeline(c).run(tmp_path / "results")
    assert c.run_counter == 1

    Pipeline([a, c]).run(tmp_path / "results")
    assert a.run_counter == 1
    assert c.run_counter == 2


def test_pipeline_temporary_directory_copies_files(tmp_path):
    """Test that files in temporary_directory are copied to working_directory."""
    a = MockChainRunnable(1)
    pipeline = Pipeline(a)

    # Create temporary directory with files
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    (temp_dir / "file1.txt").write_text("content1")
    (temp_dir / "file2.txt").write_text("content2")

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    pipeline.run(working_directory=work_dir, temporary_directory=temp_dir)

    # Verify files were copied
    assert (work_dir / "file1.txt").exists()
    assert (work_dir / "file2.txt").exists()
    assert (work_dir / "file1.txt").read_text() == "content1"
    assert (work_dir / "file2.txt").read_text() == "content2"


def test_pipeline_temporary_directory_preserves_subdirectories(tmp_path):
    """Test that subdirectories in temporary_directory are recursively copied."""
    a = MockChainRunnable(1)
    pipeline = Pipeline(a)

    # Create temporary directory with subdirectories
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    (subdir / "file.txt").write_text("subcontent")
    (temp_dir / "root_file.txt").write_text("rootcontent")

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    pipeline.run(working_directory=work_dir, temporary_directory=temp_dir)

    # Verify directory structure is preserved
    assert (work_dir / "subdir").exists()
    assert (work_dir / "subdir" / "file.txt").exists()
    assert (work_dir / "subdir" / "file.txt").read_text() == "subcontent"
    assert (work_dir / "root_file.txt").exists()
    assert (work_dir / "root_file.txt").read_text() == "rootcontent"


def test_pipeline_temporary_directory_works_with_string_paths(tmp_path):
    """Test that temporary_directory works with string path inputs."""
    a = MockChainRunnable(1)
    pipeline = Pipeline(a)

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    (temp_dir / "file.txt").write_text("content")

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    # Use string paths
    pipeline.run(working_directory=str(work_dir), temporary_directory=str(temp_dir))

    assert (work_dir / "file.txt").exists()
    assert (work_dir / "file.txt").read_text() == "content"


def test_pipeline_temporary_directory_works_with_pathlib_paths(tmp_path):
    """Test that temporary_directory works with pathlib.Path inputs."""
    a = MockChainRunnable(1)
    pipeline = Pipeline(a)

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    (temp_dir / "file.txt").write_text("content")

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    # Use pathlib.Path objects
    pipeline.run(working_directory=work_dir, temporary_directory=temp_dir)

    assert (work_dir / "file.txt").exists()
    assert (work_dir / "file.txt").read_text() == "content"


def test_pipeline_temporary_directory_creates_working_directory(tmp_path):
    """Test that working_directory is created if it doesn't exist."""
    a = MockChainRunnable(1)
    pipeline = Pipeline(a)

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    (temp_dir / "file.txt").write_text("content")

    work_dir = tmp_path / "work"
    # Don't create work_dir - it should be created by the pipeline

    pipeline.run(working_directory=work_dir, temporary_directory=temp_dir)

    assert work_dir.exists()
    assert (work_dir / "file.txt").exists()


def test_pipeline_temporary_directory_overwrites_existing_files(tmp_path):
    """Test that existing files in working_directory are overwritten."""
    a = MockChainRunnable(1)
    pipeline = Pipeline(a)

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    (temp_dir / "file.txt").write_text("new_content")

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "file.txt").write_text("old_content")

    pipeline.run(working_directory=work_dir, temporary_directory=temp_dir)

    # Verify file was overwritten
    assert (work_dir / "file.txt").read_text() == "new_content"


def test_pipeline_temporary_directory_handles_empty_directory(tmp_path):
    """Test that empty temporary_directory is handled gracefully."""
    a = MockChainRunnable(1)
    pipeline = Pipeline(a)

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()  # Empty directory

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    # Should not raise an error
    pipeline.run(working_directory=work_dir, temporary_directory=temp_dir)

    # Working directory should still exist
    assert work_dir.exists()


def test_pipeline_temporary_directory_raises_error_when_temp_not_exists(tmp_path):
    """Test that appropriate error is raised when temporary_directory doesn't exist."""
    a = MockChainRunnable(1)
    pipeline = Pipeline(a)

    temp_dir = tmp_path / "nonexistent"
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    # Should raise FileNotFoundError when trying to iterate non-existent directory
    with pytest.raises(FileNotFoundError):
        pipeline.run(working_directory=work_dir, temporary_directory=temp_dir)


def test_pipeline_temporary_directory_merges_existing_directories(tmp_path):
    """Test that existing directories in working_directory are merged correctly."""
    a = MockChainRunnable(1)
    pipeline = Pipeline(a)

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    (subdir / "new_file.txt").write_text("new")
    (temp_dir / "new_root.txt").write_text("root")

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    existing_subdir = work_dir / "subdir"
    existing_subdir.mkdir()
    (existing_subdir / "old_file.txt").write_text("old")

    pipeline.run(working_directory=work_dir, temporary_directory=temp_dir)

    # Verify new files were added
    assert (work_dir / "new_root.txt").exists()
    assert (work_dir / "subdir" / "new_file.txt").exists()
    # Verify existing files are still there (merged, not replaced)
    assert (work_dir / "subdir" / "old_file.txt").exists()
