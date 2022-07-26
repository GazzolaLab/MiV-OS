# Pairwise Granger Causality
import os

import numpy as np
from elephant.causality.granger import pairwise_granger

from miv.typing import SignalType


def pairwise_causality(signal: SignalType, start: int, end: int):
    """
    Estimates pairwise Granger Causality between all channels.

    Parameters
    ----------
    signal : SignalType
       Input signal.
    start : int
       starting point from signal
    end : int
       End point from signal

    Returns
    -------
    C : np.ndarray
        Causality Matrix (shape=2x2) containing directional causalities for X -> Y and Y -> X,
        instantaneous causality between X,Y, and total causality. X and Y represents electrodes

    See Also
    --------
    miv.visualization.causality.pairwise_causality_plot

    """

    p = len(signal[0])  # Number of channels
    C = np.zeros((4, p, p))  # Causality Matrix

    for j in range(p):
        for k in range(j + 1, p):
            C[:, j, k] = pairwise_granger(
                np.transpose([signal[start:end, j], signal[start:end, k]]),
                max_order=1,
            )
            C[:, k, j] = pairwise_granger(
                np.transpose([signal[start:end, k], signal[start:end, j]]),
                max_order=1,
            )
    # for i in range(4):
    #    np.fill_diagonal(C[i], 0.0)

    # C or causality matrix contains four p X p matrices. These are directional causalities for X -> Y and Y -> X,
    # instantaneous causality between X,Y, and total causality. X and Y represents electrodes
    return C
