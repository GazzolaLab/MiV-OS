__doc__ = """
Export prb files, which is inherited from phy.
"""

import os
import pathlib

import click
import numpy as np
from miv_os_contrib.impedance_check import (
    get_channel_in_impedance_range,
    get_impedances_from_csv,
)

from miv.mea import mea_map
from miv.mea.channel_mapping import MEA64, MEA128


@click.command()
@click.option("--mea-name", "-m", help="Name of the mea.")
@click.option("--spacing", "-s", type=int, help="Path to the impedance file.")
@click.option(
    "--output", "-o", type=click.Path(exists=False), help="Path to the output file."
)
@click.option(
    "--impedance-path",
    "-i",
    default=None,
    type=click.Path(),
    help="Path to the impedance file.",
)
@click.option(
    "--is-intan",
    "-intan",
    type=bool,
    default=False,
    is_flag=True,
    help="Whether the file is recorded from intan.",
)
@click.option(
    "--total-nb-channels",
    "-t",
    type=int,
    default=64,
    help="Total number of channels. Default is 64.",
)
def export(
    mea_name, spacing, output, impedance_path, is_intan: bool, total_nb_channels: int
):
    probes_path = pathlib.Path(os.environ["HOME"]) / "spyking-circus" / "probes"

    if is_intan:
        if mea_name == "64_intanRHD":
            grid = MEA64(mea_name).mea_intan
        elif mea_name == "128_dual_connector_two_64_rhd":
            grid = MEA128(mea_name).mea_intan
    else:
        grid = mea_map[mea_name]

    if impedance_path is not None:
        range_impedance = (1e5, 1.3e6)
        low_impedance, high_impedance = range_impedance
        if impedance_path.endswith(".xml"):
            # Impedances
            impedances = get_channel_in_impedance_range(impedance_path, range_impedance)
            channels_with_impedances_in_range = list(impedances.keys())
        elif impedance_path.endswith(".csv"):
            # Impedances
            impedances = get_impedances_from_csv(impedance_path)
            channels_with_impedances_in_range = np.where(
                np.logical_and(
                    impedances > low_impedance,
                    impedances < high_impedance,
                )
            )[0]
        print("Channels with impedances in range:")
        for ch in channels_with_impedances_in_range:
            print(f"Channel {ch} has impedance {impedances[ch]:.2f}")
    else:
        channels_with_impedances_in_range = np.arange(total_nb_channels)

    # Locations
    locs = {}
    for ch in channels_with_impedances_in_range:
        where = np.where(grid == ch)
        y, x = where[0][0], where[1][0]
        locs[ch] = (x * spacing, y * spacing)

    text = configuration_to_string(
        locs, channels_with_impedances_in_range, total_nb_channels=total_nb_channels
    )

    with open(probes_path / output, "w") as f:
        f.write(text)


def configuration_to_string(locs, channels, radius=250, total_nb_channels=64):
    geometry = {ch: list(locs[ch]) for ch in channels}
    channel_groups = {
        1: {"channels": list(channels), "graph": [], "geometry": geometry}
    }

    return f"""total_nb_channels = {total_nb_channels}
radius            = {radius}

channel_groups = {channel_groups}
"""


if __name__ == "__main__":
    export()
