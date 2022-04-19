import pytest
import re

import numpy as np


def test_version_regex_match():
    import miv.version
    from miv.version import VERSION

    re_version = re.search(r"([\d.]+)", VERSION).group(1)
    assert re_version == VERSION, "Version numbers should be integer."
    lead_zero_stripped_version = ".".join([str(int(v)) for v in re_version.split(".")])
    assert (
        lead_zero_stripped_version == VERSION
    ), "Version numbers should not have leading zeros."
