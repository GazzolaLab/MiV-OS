# Pairwise Granger Causality Plot

import os

import matplotlib.pyplot as plt
import numpy as np
from elephant.causality.granger import pairwise_granger
from viziphant.spike_train_correlation import plot_corrcoef

from miv.statistics import pairwise_causality
from miv.typing import SignalType


def pairwise_causality_plot(signal: SignalType, start: float, end: float):

    # Plots pairwise Granger Causality

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
    # figure, axes
    # Contains subplots for directional causalities for X -> Y and Y -> X,
    # instantaneous causality between X,Y, and total causality. X and Y represents electrodes

    C_M = pairwise_causality(signal, start, end)

    # Plotting

    fig, axes = plt.subplots(2, 2)
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6
    )
    plot_corrcoef(C[0], axes=axes[0, 0])
    plot_corrcoef(C[1], axes=axes[0, 1])
    plot_corrcoef(C[2], axes=axes[1, 0])
    plot_corrcoef(C[3], axes=axes[1, 1])
    # axes[:,:].set_xlabel('Electrode')
    # axes.set_ylabel('Electrode')
    axes[0, 0].set_xlabel("Electrode")
    axes[0, 1].set_xlabel("Electrode")
    axes[1, 0].set_xlabel("Electrode")
    axes[1, 1].set_xlabel("Electrode")
    axes[0, 0].set_ylabel("Electrode")
    axes[1, 0].set_ylabel("Electrode")
    axes[0, 1].set_ylabel("Electrode")
    axes[1, 1].set_ylabel("Electrode")
    axes[0, 0].set_title("Directional causality X => Y")
    axes[0, 1].set_title("Directional causality Y => X")
    axes[1, 0].set_title("Instantaneous causality of X,Y")
    axes[1, 1].set_title("Total interdependence of X,Y")
    return fig, axes
