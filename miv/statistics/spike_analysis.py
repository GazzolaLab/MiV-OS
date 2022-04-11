from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy
import scipy.signal


def get_spike_stats(spikes, channels_list=[8, 25], exclude=True, t_i=0, t_f=-1):
    tot_num_channels = np.array(spikes, dtype="object").shape[0]
    channels = []
    channels_list = list(np.array(channels_list) - 1)

    if exclude:
        for i in range(0, tot_num_channels):
            if channels_list.count(i) < 1:
                channels.append(i)
    else:
        channels = channels_list

    num_channels = len(channels)

    rates = np.zeros(num_channels)

    for i in range(0, num_channels):
        rates[i] = len(spikes[channels[i]]) / (t_f - t_i)

    ind = np.argmax(rates)

    return [rates[ind], channels[ind] + 1], np.mean(rates), np.var(rates)
