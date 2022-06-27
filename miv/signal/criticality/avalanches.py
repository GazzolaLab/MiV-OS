__all__ = ["detect_avalanche"]


import numpy as np

from miv.core.spiketrain import SpikeTrain


def detect_avalanche(
    signal: SpikeTrain,
):
    """Finds neuronal avalanches present in a spike train signal

    Parameters
    ----------
    signal : miv.core.Spiketrain

    Returns
    -------

    """
    # todo: parse the arguments

    # operations are optimized when the raster is a sparse matrix such that
    # each column represents a time bin since str_avas requires us to take sub
    # matrices of the raster from one time index to another. See CSC sparse
    # data format for why this is the case...

    # todo:
    # asdf to raster conversion
    raster = 1

    # find number of spikes in each time bin
    # from sparse to full
    # pop_fir raster.sum()

    if custom_threshold:
        threshold = np.mean(np.nonzero(pop_fir)) / 2

    # put 1s where a time bin is within a valid avalache, 0 otherwise
    evts = pop_fir > thresh

    # pad and find where avalanches start or end based on where 0s change to
    # 1s and vice versa
    act_change = diff(concat([0, evts, 0]))

    # 1s indicate an avalanche began in the given time bin
    starts = np.flatnonzero(act_change == 1).reshape(1, -1) + 1
    # -1s indicate an avalanche ended in the previous time bin
    ends = np.flatnonzero(act_change == -1).reshape(1, -1)

    if td != ts:
        ed2 = ends(np.arange(1, end(), 1).reshape(1, -1))
        t2 = starts(arange(2, end()))
        q = st2 - ed2
        inds = arange(1, length(st2))
        inds = inds(q <= td)
        starts = starts(concat([1, inds + 1]))
        ends = ends(concat([inds, length(ends)]))

    lens = ends - starts + 1
    str_avas = cell(length(lens), 1)
    szes = zeros(size(lens))

    for ii in arange(1, length(lens)).reshape(-1):
        # select sub arrays of valid avalanches
        str_avas[ii] = rast(arange(), arange(starts(ii), ends(ii)))
        szes[ii] = nnz(str_avas[ii])

    return str_avas, szes, lens, rast, starts
