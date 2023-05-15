__all__ = ["interp_2d", "command_run"]

import subprocess

import numpy as np
import scipy.signal as sps
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter


def interp_2d(data, ratio=10):
    # scipy interp. cubic
    X = np.arange(0, data.shape[1], 1)
    Y = np.arange(0, data.shape[0], 1)
    f = interp2d(X, Y, data, kind="cubic")
    xnew = np.arange(0, data.shape[1], 1.0 / ratio)
    ynew = np.arange(0, data.shape[0], 1.0 / ratio)
    data1 = f(xnew, ynew)
    # data1 = gaussian_filter(data1, sigma=3)
    Xn, Yn = np.meshgrid(xnew, ynew)

    return Xn[:-ratio, :-ratio], Yn[:-ratio, :-ratio], data1[:-ratio, :-ratio]


def command_run(
    cmd, verbose=False, exception="skip"
):  # TODO: Probably should be in other file
    args = cmd
    output = subprocess.run(args, capture_output=True)
    if verbose:
        print(f"Running: {cmd=}")
        print(f"{output.returncode=}")
        if output.returncode == 0:
            print(f"{output.stdout.decode('utf-8')=}")
        else:
            print(f"{output.stderr.decode('utf-8')=}")
        print("Done")
    if output.returncode != 0:
        if exception == "raise":
            raise RuntimeError(
                f"Error running {cmd=}\n" f"{output.stderr.decode('utf-8')}"
            )
    return output.returncode
