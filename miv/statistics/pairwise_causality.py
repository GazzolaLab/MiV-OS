# Pairwise Granger Causality
import os

import numpy as np
from elephant.causality.granger import pairwise_granger

from miv.typing import SignalType


def pairwise_causality(signal: SignalType, start: float, end: float):

    # Estimates pairwise Granger Causality

    # Parameters
    # ----------
    # signal : SignalType
    #    Input signal
    # start : float
    #    starting point from signal
    # end : float
    #    End point from signal

    # Returns
    # -------
    # C : Causality Matrix containing directional causalities for X -> Y and Y -> X,
    # instantaneous causality between X,Y, and total causality. X and Y represents electrodes

    p = len(signal[0])
    C = np.zeros((4, p, p))  # Causality Matrix
    q = np.arange(0, p)

    for j in q:
        for k in q:
            if j == k:
                C[:, j, k] = 0
            else:
                C[:, j, k] = pairwise_granger(
                    np.transpose([signal[start:end, j], signal[start:end, k]]),
                    max_order=1,
                )

    # C or causality matrix contains four p X p matrices. These are directional causalities for X -> Y and Y -> X,
    # instantaneous causality between X,Y, and total causality. X and Y represents electrodes
    return C
