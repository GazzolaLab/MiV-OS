__doc__ = """

.. autoclass:: miv.core.pipeline.Pipeline
   :members:

"""
__all__ = ["Pipeline"]

from typing import List, Optional, Union

import pathlib

from miv.core.operator.chainable import _Chainable
from miv.core.policy import _Runnable


class Pipeline:
    """
    A pipeline is a collection of operators that are executed in a specific order.

    If operator structure is a tree like

    .. mermaid::

        flowchart LR
            A --> B --> D --> F
            A --> C --> E --> F
            B --> E

    then the execution order of `Pipeline(F)` is A->B->D->C->E->F.
    If nodes already have cached results, they will be loaded instead of being executed.
    For example, if E is already cached, then the execution order of `Pipeline(F)` is A->B->D->F. (C is skipped, E is loaded from cache)
    """

    def __init__(self, node: _Chainable):
        self.execution_order: List[_Runnable] = node.topological_sort()

    def run(
        self,
        working_directory: Optional[Union[str, pathlib.Path]] = "./results",
        no_cache: bool = False,
        dry_run: bool = False,
        verbose: bool = False,  # Use logging
    ):
        """
        Run the pipeline.

        Parameters
        ----------
        working_directory : Optional[Union[str, pathlib.Path]], optional
            The working directory where the pipeline will be executed. By default "./results"
        no_cache : bool, optional
            If True, the cache will be disabled. By default False
        dry_run : bool, optional
            If True, the pipeline will not be executed. By default False
        verbose : bool, optional
            If True, the pipeline will log debugging informations. By default False
        """
        for node in self.execution_order:
            if verbose:
                print("Running: ", node)
            if hasattr(node, "cacher"):
                node.cacher.cache_policy = "OFF" if no_cache else "AUTO"
            node.run(dry_run=dry_run, save_path=working_directory)
        if verbose:
            print("Pipeline done:")
            self.summarize()
            print("-" * 46)

    def summarize(self):
        strs = []
        strs.append("Execution order:")
        for i, op in enumerate(self.execution_order):
            strs.append(f"{i}: {op}")
        return "\n".join(strs)

    def export(self, working_directory: Optional[Union[str, pathlib.Path]]):
        # TODO
        pass
