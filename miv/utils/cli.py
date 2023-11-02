__all__ = ["command_run"]

import subprocess


def command_run(cmd, logger=None, exception="skip"):  # pragma: no cover
    args = cmd
    output = subprocess.run(args, capture_output=True)
    if logger is not None:
        logger.info(f"Running: {cmd=}")
        logger.info(f"{output.returncode=}")
        if output.returncode == 0:
            logger.info(f"{output.stdout.decode('utf-8')=}")
        else:
            logger.info(f"{output.stderr.decode('utf-8')=}")
        logger.info("Done")
    if output.returncode != 0:
        if exception == "raise":
            raise RuntimeError(
                f"Error running {cmd=}\n" f"{output.stderr.decode('utf-8')}"
            )
        elif exception == "skip":
            if logger is not None:
                logger.warning(
                    f"CLI error running {cmd=}\n" f"{output.stderr.decode('utf-8')}"
                )
    return output.returncode
