import logging
import os
import pathlib
import tempfile
import urllib

import pytest


class TestDownloadUtilities:
    # shasum -a 256
    hash_check_data = [
        ("123", "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"),
        (
            "1\n2\n3\n4",
            "67497b776854008d38c2340e14925a64b36686230bccaa777db68f644196015f",
        ),
        ("abcd", "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589"),
    ]
    supported_archive_format = ["zip", "tar", "tar.gz", "tar.bz"]

    @pytest.fixture(params=hash_check_data)
    def make_mockfile(self, request, tmp_path):
        datastring, hashstr = request.param
        filepath = os.path.join(tmp_path, f"data{hash(datastring)}.txt")
        with open(filepath, "wb") as file:
            file.write(bytearray(datastring, "utf-8"))
        return filepath, hashstr

    def test_check_file_hash(self, make_mockfile):
        from miv.datasets.utils import check_file_hash

        filepath, hashstr = make_mockfile
        assert check_file_hash(filepath, hashstr), "hash is broken"

    def test_url_retrieve(self, make_mockfile):
        from miv.datasets.utils import _url_retrieve, check_file_hash

        filepath, hashstr = make_mockfile
        file_url = "file:///" + filepath
        download_path = filepath + ".downloaded"
        _url_retrieve(file_url, download_path, progbar_disable=True)
        assert check_file_hash(download_path, hashstr), "Hash changed during download."

    def test_url_retrieve_false_url(self, tmp_path):
        from miv.datasets.utils import _url_retrieve

        file_url = "file:///_"  # False URL
        with pytest.raises(urllib.error.URLError):
            _url_retrieve(file_url, tmp_path, progbar_disable=True)

    @pytest.mark.parametrize("archive_format", supported_archive_format)
    def test_file_get_file_and_extract_archives(
        self, make_mockfile, archive_format, tmp_path
    ):
        from miv.datasets.utils import _extract_archive, check_file_hash, get_file

        filepath, hashstr = make_mockfile
        zippath = filepath + "." + archive_format
        arcname = os.path.basename(filepath)

        if archive_format.startswith("tar"):
            import tarfile

            open_fn = tarfile.open
        elif archive_format == "zip":
            import zipfile

            open_fn = zipfile.ZipFile

        # Create zip files
        assert os.path.exists(
            filepath
        ), "File is not created. Check 'make_mockfile' fixture"
        with open_fn(zippath, "w") as arcfile:
            if archive_format.startswith("tar"):
                arcfile.add(filepath, arcname=arcname)
            elif archive_format == "zip":
                arcfile.write(filepath, arcname=arcname)
        assert os.path.exists(zippath), f"Zip file not created for {archive_format}."

        # Test extraction
        unzippath = filepath + "." + archive_format + ".unziped"
        unzipfile = os.path.join(unzippath, os.path.basename(filepath))
        assert _extract_archive(zippath, unzippath, archive_format)
        assert check_file_hash(
            unzipfile, hashstr
        ), "Hash changed during archive extraction."

        # Test download and extraction
        file_url = "file:///" + zippath
        fname = "test." + archive_format
        downloadpath = os.path.join(tmp_path, "datasets", "testfiles")
        downloadfile = os.path.join(downloadpath, "test")
        downloadedpath = get_file(
            file_url,
            "testfiles",
            fname,
            archive_format=archive_format,
            cache_dir=os.path.join(tmp_path, "datasets"),
        )
        assert downloadfile == downloadedpath, f"{downloadpath} vs {downloadedpath}"
        assert check_file_hash(
            os.path.join(downloadpath, arcname), hashstr
        ), "Hash changed during download and extraction."

    def test_get_file_without_extraction(self, make_mockfile, tmp_path):
        from miv.datasets.utils import check_file_hash, get_file

        filepath, hashstr = make_mockfile
        file_url = "file:///" + filepath
        fname = filepath + ".from_download"
        downloaded_path = get_file(
            file_url,
            "testfiles",
            fname,
            file_hash=hashstr,
            archive_format=None,
            cache_dir=os.path.join(tmp_path, "datasets"),
        )
        assert check_file_hash(
            downloaded_path, hashstr
        ), "Hash changed during URL download."

    def test_get_file_wrong_hash(self, make_mockfile, tmp_path):
        from miv.datasets.utils import check_file_hash, get_file

        filepath, hashstr = make_mockfile
        file_url = "file:///" + filepath
        fname = filepath + ".from_download"
        with pytest.raises(ValueError) as e:
            get_file(
                file_url,
                "testfiles",
                fname,
                file_hash="_",
                archive_format=None,
                cache_dir=os.path.join(tmp_path, "datasets"),
            )
        assert "sha256 file hash does not match" in str(e)

    def test_get_file_redownload_due_to_invalid_hash(self, make_mockfile, tmp_path):
        from miv.datasets.utils import check_file_hash, get_file

        filepath, hashstr = make_mockfile
        file_url = "file:///" + filepath
        fname = filepath + ".from_download"
        downloaded_path = os.path.join(tmp_path, "datasets", "testfiles", fname)
        pathlib.Path(downloaded_path).touch()  # create arbitrary file
        assert not check_file_hash(
            downloaded_path, hashstr
        ), "Hash changed during URL download."
        get_file(
            file_url,
            "testfiles",
            fname,
            file_hash=hashstr,
            archive_format=None,
            cache_dir=os.path.join(tmp_path, "datasets"),
        )
        assert check_file_hash(
            downloaded_path, hashstr
        ), "Hash changed during URL download."

    @pytest.mark.parametrize("expected_repeat", [2, 5])
    def test_get_file_redownload_log(
        self, expected_repeat, make_mockfile, tmp_path, caplog
    ):
        from miv.datasets.utils import check_file_hash, get_file

        filepath, hashstr = make_mockfile
        caplog.set_level(logging.INFO)
        file_url = "file:///" + filepath
        fname = filepath + ".from_download"
        get_file(
            file_url,
            "testfiles",
            fname,
            file_hash=hashstr,
            archive_format=None,
            cache_dir=os.path.join(tmp_path, "datasets"),
        )
        for _ in range(expected_repeat):
            get_file(
                file_url,
                "testfiles",
                fname,
                file_hash=hashstr,
                archive_format=None,
                cache_dir=os.path.join(tmp_path, "datasets"),
            )
        n_skipped_log = 0
        for record in caplog.records:
            if "skipped" in record.message:
                n_skipped_log += 1
        assert n_skipped_log == expected_repeat, "Wrong number of log generated."

    @pytest.mark.parametrize("archive_format", supported_archive_format)
    def test_file_extract_archives_no_archive_file(self, archive_format, tmp_path):
        from miv.datasets.utils import _extract_archive

        filename = os.path.join(tmp_path, "test.failformat")
        with open(filename, "w") as file:
            file.write("It is not to late!!")
        result = _extract_archive(filename, tmp_path, archive_format)
        assert not result, "Extract failure is not working correctly"

    @pytest.mark.parametrize("archive_format", supported_archive_format)
    def test_file_extract_archives_no_file_error(self, archive_format, tmp_path):
        from miv.datasets.utils import _extract_archive

        with pytest.raises(FileNotFoundError):
            _extract_archive(
                "must_not_exist." + archive_format, tmp_path, archive_format
            )
