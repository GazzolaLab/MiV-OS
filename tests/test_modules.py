import re


def test_version_regex_match():
    from miv import version

    re_version = re.search(r"([\d.]+)", version).group(1)
    assert re_version == version, "Version numbers should be integer."
    lead_zero_stripped_version = ".".join([str(int(v)) for v in re_version.split(".")])
    assert (
        lead_zero_stripped_version == version
    ), "Version numbers should not have leading zeros."
