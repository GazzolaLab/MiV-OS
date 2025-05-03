__doc__ = """

.. autoclass:: miv.core.pipeline.Pipeline
   :members:

"""
__all__ = ["Pipeline"]

from typing import cast
from collections.abc import Sequence

import os
import pathlib
import time

from miv.core.operator.protocol import _Node
from miv.core.utils.graph_sorting import topological_sort


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

    def __init__(self, node: _Node | Sequence[_Node]) -> None:
        self.nodes_to_run: list[_Node]
        if not isinstance(node, list):
            # FIXME: check if the node is standalone operator
            node = cast(_Node, node)
            self.nodes_to_run = [node]
        else:
            node = cast(Sequence[_Node], node)
            self.nodes_to_run = list(node)

    def run(
        self,
        working_directory: str | pathlib.Path = "./results",
        cache_directory: str | pathlib.Path | None = None,
        temporary_directory: str | pathlib.Path | None = None,
        skip_plot: bool = False,
        verbose: bool = False,  # Use logging
    ) -> None:
        """
        Run the pipeline.

        Parameters
        ----------
        working_directory : Optional[Union[str, pathlib.Path]], optional
            The working directory where the pipeline will be executed. By default "./results"
        cache_directory : Optional[Union[str, pathlib.Path]], optional
            The cache directory where the pipeline will be executed. By default None
            If None, the cache directory will be the same as the working directory.
        temporary_directory : Optional[Union[str, pathlib.Path]], optional
            If given, files will be saved in temporary directory, and will be moved to
            working directory after. Cache directory is not altered.
        verbose : bool, optional
            If True, the pipeline will log debugging informations. By default False
        """
        # Set working directory
        if cache_directory is None:
            cache_directory = working_directory

        # Reset all callbacks
        for last_node in self.nodes_to_run:
            for node in topological_sort(last_node):
                if hasattr(node, "reset_callbacks"):
                    node.reset_callbacks(plot=skip_plot)
                if hasattr(node, "set_save_path"):
                    if temporary_directory is not None:
                        node.set_save_path(temporary_directory, cache_directory)
                    else:
                        node.set_save_path(working_directory, cache_directory)

        # Execute
        for node in self.nodes_to_run:
            if verbose:
                stime = time.time()
                print("Running: ", node, flush=True)

            try:
                node.output()
            except Exception as e:
                print("  Exception raised: ", node, flush=True)
                raise e

            if verbose:
                etime = time.time()
                print(f"  Finished: {etime - stime:.03f} sec", flush=True)
                print("Pipeline done.")

        if temporary_directory is not None:
            os.system(f"cp -rf {temporary_directory}/* {working_directory}/")
            # import shutil
            # shutil.move(temporary_directory, working_directory)

    def summarize(self) -> str:
        strs = []
        for node in self.nodes_to_run:
            execution_order = topological_sort(node)

            strs.append(f"Execution order for {node}:")
            for i, op in enumerate(execution_order):
                strs.append(f"{i}: {op}")
        return "\n".join(strs)
