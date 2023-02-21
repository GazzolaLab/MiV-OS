from typing import List, Optional, Union

import pathlib

from miv.core.operator.chainable import _Chainable
from miv.core.policy import _Runnable


class Pipeline:
    def __init__(self, node: _Chainable):
        self.execution_order: List[_Runnable] = node.topological_sort()

    def run(
        self,
        save_path: Optional[Union[str, pathlib.Path]] = None,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        # TODO: implement save_path
        for node in self.execution_order:
            if verbose:
                print("Running: ", node)
            node.run(dry_run=dry_run)

    def summarize(self):
        strs = []
        strs.append("Execution order:")
        for i, op in enumerate(self.execution_order):
            strs.append(f"{i}: {op}")
        return "\n".join(strs)
