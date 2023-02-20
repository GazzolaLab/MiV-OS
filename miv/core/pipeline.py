from typing import List

from miv.core.operator.chainable import _Chainable
from miv.core.policy import _Runnable


class Pipeline:
    def __init__(self, node: _Chainable):
        self.execution_order: List[_Runnable] = node.topological_sort()

    def run(self, dry_run: bool = False):
        for node in self.execution_order:
            node.run(dry_run=dry_run)
