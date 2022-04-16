from typing import Any, Iterable

import numpy as np
from miv.signal.filter import FilterProtocol


class Filter1:
    def __init__(self):
        self.tag = "test tag"

    def __call__(self, a):
        return a


class Filter2:
    tag = "test tag"

    def __call__(self, a, b):
        return a


class Filter3:
    def __call__(self, a, b):
        return a

    @property
    def tag(self) -> str:
        return "test tag"


mock_filter_list: Iterable[FilterProtocol] = [Filter1, Filter2, Filter3]


class NonFilter1:
    # Without __call__
    tag = "test tag"


class NonFilter2:
    # Without tag
    def __call__(self, a, b):
        return a


mock_nonfilter_list: Iterable[Any] = [NonFilter1, NonFilter2]
