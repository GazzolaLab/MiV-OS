__doc__ = """

.. autoclass:: miv.core.pipeline.Pipeline
   :members:

"""

from typing import cast
from collections.abc import Sequence

import pathlib
import shutil
import time
import sys

from loguru import logger

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
        verbose: int = 1,
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
            This feature is useful when the pipeline is running with MPI but I/O bandwidth
            is limited. Each node can save results to local temporary directory, and then
            collectively moved to working directory. This is only suppored when mpi4py is
            available.
        verbose : int, optional
            Verbosity level. 0: quiet, 1: info, 2: debug.
            If True, the pipeline will log debugging informations. By default 1
        """
        # Configure logger
        logger.remove()
        if verbose == 0:
            logger.add(sys.stderr, level="WARNING")
        elif verbose == 1:
            logger.add(sys.stderr, level="INFO")
        else:  # verbose >= 2
            logger.add(sys.stderr, level="DEBUG")

        # Set working directory
        if cache_directory is None:
            cache_directory = working_directory

        # Setup nodes
        all_nodes = list(dict.fromkeys(topological_sort(*self.nodes_to_run)))
        for node in all_nodes:
            if hasattr(node, "reset_callbacks"):
                node.reset_callbacks(plot=skip_plot)
            if hasattr(node, "set_save_path"):
                if temporary_directory is not None:
                    node.set_save_path(temporary_directory, cache_directory)
                else:
                    node.set_save_path(working_directory, cache_directory)

        # Execute
        logger.info(f"Total {len(all_nodes)} operators to run.")
        for node in all_nodes:
            stime = time.time()
            logger.info(f"Running: {node}")

            is_cached = hasattr(node, "check_cached") and node.check_cached()
            logger.debug(f"  Cache exists: {is_cached}")

            try:
                node.output()
            except Exception as e:
                logger.exception(f"  Exception raised while running {node}: {e}")
                raise e

            etime = time.time()
            logger.info(f"  Finished: {etime - stime:.03f} sec")

            if is_cached and verbose >= 2 and hasattr(node, "output_path"):
                output_path = pathlib.Path(node.output_path)
                if output_path.exists():
                    size = output_path.stat().st_size
                    logger.debug(f"  Cache size: {size / 1024**2:.3f} MB")

        logger.info("Pipeline done.")

        if temporary_directory is not None:
            temp_dir = pathlib.Path(temporary_directory)
            work_dir = pathlib.Path(working_directory)
            work_dir.mkdir(parents=True, exist_ok=True)

            # Copy each item from temp_dir to work_dir
            for item in temp_dir.iterdir():
                dest = work_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

    def summarize(self) -> str:
        strs = []
        for node in self.nodes_to_run:
            execution_order = topological_sort(node)

            strs.append(f"Execution order for {node}:")
            for i, op in enumerate(execution_order):
                strs.append(f"{i}: {op}")
        return "\n".join(strs)
