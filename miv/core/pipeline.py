__doc__ = """

.. autoclass:: miv.core.pipeline.Pipeline
   :members:

"""
__all__ = ["Pipeline"]

from typing import List, Optional, Union

import os
import pathlib
import time

from miv.core.operator.operator import Operator


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

    def __init__(self, node: Operator):
        self._start_node = node
        self.execution_order = None

    def run(
        self,
        working_directory: Union[str, pathlib.Path] = "./results",
        cache_directory: Optional[Union[str, pathlib.Path]] = None,
        temporary_directory: Optional[Union[str, pathlib.Path]] = None,
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
        temporary_directory : Optional[Union[str, pathlib.Path]], optional
            If given, files will be saved in temporary directory, and will be moved to
            working directory after. Cache directory is not altered.
        verbose : bool, optional
            If True, the pipeline will log debugging informations. By default False
        """
        # Set working directory
        if cache_directory is None:
            cache_directory = working_directory
        for node in self._start_node.topological_sort():
            if hasattr(node, "set_save_path"):
                if temporary_directory is not None:
                    node.set_save_path(temporary_directory, cache_directory)
                else:
                    node.set_save_path(working_directory, cache_directory)

        self.execution_order = [
            self._start_node
        ]  # TODO: allow running multiple operation
        if verbose:
            stime = time.time()
            print("Execution order = ", self.execution_order, flush=True)
        for node in self.execution_order:
            if verbose:
                stime = time.time()
                print("  Running: ", node, flush=True)

            try:
                node.run(skip_plot=skip_plot)
            except Exception as e:
                print("  Exception raised: ", node, flush=True)
                raise e

            if verbose:
                print(f"  Finished: {time.time() - stime:.03f} sec", flush=True)
        if verbose:
            print(f"Pipeline done: computing {self._start_node}", flush=True)

        if temporary_directory is not None:
            os.system(f"cp -rf {temporary_directory}/* {working_directory}/")
            # import shutil
            # shutil.move(temporary_directory, working_directory)

    def summarize(self):
        if self.execution_order is None:
            self.execution_order = self._start_node.topological_sort()

        strs = []
        strs.append("Execution order:")
        for i, op in enumerate(self.execution_order):
            strs.append(f"{i}: {op}")
        return "\n".join(strs)
