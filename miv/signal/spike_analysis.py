from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy
import scipy.signal


def load_data(
    data_file: str,
    channels: int,
    timestamps_npy: Optional[str] = "",
    sampling_rate=30000,
):
    """
    Describe function

    Parameters:
        data_file: continuous.dat file from Open_Ethys recording
        channels: number of recording channels recorded from

    Returns:
        raw_data:
        timestamps:


    """

    raw_data = np.memmap(data_file, dtype="int16")
    length = raw_data.size // channels
    raw_data = np.reshape(raw_data, (length, channels))

    timestamps_zeroed = np.array(range(0, length)) / sampling_rate
    if timestamps_npy == "":
        timestamps = timestamps_zeroed
    else:
        timestamps = np.load(timestamps_npy) / sampling_rate

    # only take first 32 channels
    raw_data = raw_data[:, 0:32]

    return np.array(timestamps), np.array(raw_data)


def total_recording_time(timestamps):
    timestamps_zeroed = timestamps - timestamps[0]
    total_time = datetime.timedelta(seconds=timestamps_zeroed[-1])
    return total_time


def filter_data(
    raw_data,
    low_freq=600,
    high_freq=3000,
    non_causal=True,
    order=4,
    analog_filter=False,
    sampling_rate=30000,
    filter_type="butter",
):
    """
    raw_data: np array

    """

    b, a = scipy.signal.butter(
        order,
        [low_freq, high_freq],
        btype="bandpass",
        analog=analog_filter,
        output="ba",
        fs=sampling_rate,
    )

    num_channels = raw_data.shape[1]

    filtered_data = np.copy(raw_data)

    for i in range(0, num_channels):
        current_channel = raw_data[:, i]
        filtered_data[:, i] = scipy.signal.filtfilt(b, a, current_channel)

    return filtered_data


def plot_channels(
    timestamps,
    data,
    save_file,
    title,
    channels_list,
    exclude=True,
    k=0,
    raster_on=False,
    spikes=None,
    y_lim=(-300, 300),
    t_i=0,
    t_f=-1,
    sampling_rate=30000,
):
    num_channels = data.shape[1]
    channels = []
    channels_list = list(np.array(channels_list) - 1)

    if exclude:
        for i in range(0, num_channels):
            if channels_list.count(i) < 1:
                channels.append(i)
    else:
        channels = channels_list

    t_i = int(t_i * sampling_rate)
    if t_f == -1:
        t_f = len(timestamps)
    else:
        t_f = int(t_f * sampling_rate)

    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.95  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for blank space between subplots
    hspace = 0.001  # the amount of height reserved for white space between subplots

    num_channels = len(channels)
    sig_n = np.zeros(num_channels)
    if k != 0:
        for i in range(0, num_channels):
            sig_n[i] = np.median(np.abs(data[t_i:t_f, channels[i]])) / 0.6745

    fig, chs = plt.subplots(nrows=num_channels, ncols=1, sharex=True, sharey=True)
    plt.ylim(y_lim[0], y_lim[1])
    plt.xlim(timestamps[t_i], timestamps[t_f])
    plt.xlabel("Time (s)")
    plt.suptitle(title)
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    plt.yticks(np.array([-100, 0, 100]))
    plt.ticklabel_format(useOffset=False, style="plain")

    i = 0
    for j in channels:
        chs[i].set_yticklabels(["", "", ""])
        chs[i].plot(timestamps[t_i:t_f], data[t_i:t_f, j])
        chs[i].set_ylabel(str(j + 1))
        if k != 0:
            thresh = k * sig_n[i]
            chs[i].plot(timestamps[t_i:t_f], thresh * np.ones(t_f - t_i), "green")
            chs[i].plot(timestamps[t_i:t_f], -thresh * np.ones(t_f - t_i), "green")
        i = i + 1

    if raster_on:
        for i in range(0, num_channels):
            chs[i].eventplot(
                spikes[channels[i]],
                lineoffsets=0,
                linelengths=np.abs(y_lim[1] - y_lim[0]),
                colors="r",
            )

    # plt.savefig(save_file + '.png')
    plt.show()


def rasterize(
    spikes,
    save_file,
    title="Raster Plot",
    channels_list=[8, 25, 40, 56, 57],
    exclude=True,
):
    num_channels = np.array(spikes, dtype="object").shape[0]
    channels = []
    channels_list = list(np.array(channels_list) - 1)

    if exclude:
        for i in range(0, num_channels):
            if channels_list.count(i) < 1:
                channels.append(i)
    else:
        channels = channels_list
    num_channels = len(channels)

    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.95  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for blank space between subplots
    hspace = 0.001  # the amount of height reserved for white space between subplots

    fig, chs = plt.subplots(nrows=num_channels, ncols=1, sharex=True, sharey=True)
    plt.ticklabel_format(useOffset=False, style="plain")
    plt.xlabel("Time (s)")
    plt.suptitle(title)

    plt.ylim(-1, 1)
    plt.xlabel("Time (s)")
    plt.suptitle(title)
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    plt.yticks(np.array([-1, 1]))
    plt.ticklabel_format(useOffset=False, style="plain")

    for i in range(0, num_channels):
        chs[i].set_yticklabels(["", ""])
        chs[i].set_ylabel(str(channels[i] + 1))
        chs[i].eventplot(
            spikes[channels[i]],
            lineoffsets="False",
            linelengths=2,
            linewidths=1,
            colors="k",
        )

    plt.savefig(save_file + ".png")
    plt.show()


def get_spikes(
    data, k, channels_list=[8, 25], exclude=True, sampling_rate=30000, t_i=0, t_f=-1
):
    num_channels = data.shape[1]
    channels = []
    channels_list = list(np.array(channels_list) - 1)

    spikes = [[]]
    for i in range(0, num_channels - 1):
        spikes.append([])

    if exclude:
        for i in range(0, num_channels):
            if channels_list.count(i) < 1:
                channels.append(i)
    else:
        channels = channels_list

    num_channels = len(channels)

    t_i = int(t_i * sampling_rate)
    if t_f == -1:
        t_f = len(data[:, 0])
    else:
        t_f = int(t_f * sampling_rate)

    sig_n = np.zeros(num_channels)
    for i in range(0, num_channels):
        sig_n[i] = np.median(np.abs(data[t_i:t_f, channels[i]])) / 0.6745

    # return all nans for excluded channels

    hw = int(0.0005 * sampling_rate)
    skip = int(0.001 * sampling_rate)

    n = 0
    for j in channels:
        i = t_i
        thresh = k * sig_n[n]
        while i < t_f:
            y = data[i, j]
            if y > thresh:
                lb = int(np.maximum(t_i, i - hw / 2))
                hb = int(np.minimum(t_f, i + hw / 2))
                # find argument of this and then choose that for where to place spike
                spikes[j].append((lb + np.argmax(data[lb:hb, j])) / sampling_rate)
                i = i + int(skip)
            elif y < -thresh:
                lb = int(np.maximum(t_i, i - hw / 2))
                hb = int(np.minimum(t_f, i + hw / 2))
                # find argument of this and then choose that for where to place spike
                spikes[j].append((lb + np.argmin(data[lb:hb, j])) / sampling_rate)
                i = i + int(skip)
            i = i + 1
        n = n + 1
    return spikes


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
