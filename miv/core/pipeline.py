__doc__ = """

.. autoclass:: miv.core.pipeline.Pipeline
   :members:

"""
__all__ = ["Pipeline"]

from typing import List, Optional, Union

import pathlib
import time

from miv.core.operator.chainable import _Chainable


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

    def __init__(self, node: _Chainable | list[_Chainable]):
        if not isinstance(node, list):
            # FIXME: check if the node is standalone operator
            self.nodes_to_run = [node]
        else:
            self.nodes_to_run = node

    def run(
        self,
        working_directory: Union[str, pathlib.Path] = "./results",
        cache_directory: Optional[Union[str, pathlib.Path]] = None,
        skip_plot: bool = False,
        verbose: bool = False,  # Use logging
    ):
        """
        Run the pipeline.

        Parameters
        ----------
        working_directory : Optional[Union[str, pathlib.Path]], optional
            The working directory where the pipeline will be executed. By default "./results"
        cache_directory : Optional[Union[str, pathlib.Path]], optional
            The cache directory where the pipeline will be executed. By default None
            If None, the cache directory will be the same as the working directory.
        verbose : bool, optional
            If True, the pipeline will log debugging informations. By default False
        """
        # Set working directory
        if cache_directory is None:
            cache_directory = working_directory

        # Reset all callbacks
        for last_node in self.nodes_to_run:
            for node in last_node.topological_sort():
                if hasattr(node, "set_save_path"):
                    node.set_save_path(working_directory, cache_directory)
                    node._reset_callbacks(plot=skip_plot)

        # Execute
        for node in self.nodes_to_run:
            if verbose:
                stime = time.time()
                print("Running: ", node, flush=True)

            try:
                node.run()
            except Exception as e:
                print("  Exception raised: ", node, flush=True)
                raise e

            if verbose:
                etime = time.time()
                print(f"  Finished: {etime - stime:.03f} sec", flush=True)
                print(f"Pipeline done.")

    def summarize(self):
        for node in self.nodes_to_run:
            execution_order = node.topological_sort()

            strs = []
            strs.append(f"Execution order for {node}:")
            for i, op in enumerate(execution_order):
                strs.append(f"{i}: {op}")
            return "\n".join(strs)
