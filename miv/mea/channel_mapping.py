__all__ = ["MEA128"]

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from miv.mea import mea_map, rhd_64, rhs_32


class MEA128:
    def __init__(
        self, map_key="128_dual_connector_two_64_rhd", reverse=False, num_electrodes=128
    ):
        """
        [[ MEA side
        OE Arangement:
        (1)rhd_64 ->, (2)rhd_64 <-
        Intan Arangement
        (D)rhs_32 ->, (C)rhs_32 ->, (B)rhs_32 <-, (A)rhs_32 <-
        ]] Out-side

        -> means chip-side on bottom

        Parameters
        ----------
        reverse : bool
            If false, mapping is from OE RHD
            If true, mapping is from Intan RHS
        """
        rhd_64_1 = rhd_64[::-1, ::-1].copy()
        rhd_64_2 = rhd_64.copy()
        rhd_64_1[rhd_64_1 != -1] += 64 * 0
        rhd_64_2[rhd_64_2 != -1] += 64 * 1
        self.oe_map = np.concatenate([rhd_64_1, rhd_64_2], axis=0)

        rhs_32_1 = rhs_32[::-1, ::-1].copy()
        rhs_32_2 = rhs_32[::-1, ::-1].copy()
        rhs_32_3 = rhs_32.copy()
        rhs_32_4 = rhs_32.copy()
        rhs_32_1[rhs_32_1 != -1] += 32 * 3
        rhs_32_2[rhs_32_2 != -1] += 32 * 2
        rhs_32_3[rhs_32_3 != -1] += 32 * 1
        rhs_32_4[rhs_32_4 != -1] += 32 * 0
        self.intan_map = np.concatenate(
            [rhs_32_1, rhs_32_2, rhs_32_3, rhs_32_4], axis=0
        )

        if not reverse:  # Given map is in RHD
            self.mea = mea_map[map_key]
            self.mea_intan = np.zeros_like(self.mea, dtype=np.int_) - 1
            for channel in range(num_electrodes):
                in_channel = self.channel_mapping(channel, reverse=reverse)
                self.mea_intan[self.mea == channel] = in_channel
        else:  # Given map is in RHS
            self.mea_intan = mea_map[map_key]
            self.mea = np.zeros_like(self.mea_intan, dtype=np.int_) - 1
            for channel in range(num_electrodes):
                oe_channel = self.channel_mapping(channel, reverse=reverse)
                self.mea[self.mea_intan == channel] = oe_channel

    def channel_mapping(self, channel, reverse=False):
        """

        Parameters
        ----------
        channel : int or str
        reverse : bool
            If false, mapping is from "given OE channel, output Intan channel"
            If true, mapping is from "given Intan channel, output OE channel"
        """
        if reverse:
            oe = self.oe_map[self.intan_map == channel][0]
            return oe
        else:
            intan = self.intan_map[self.oe_map == channel][0]
            return intan

    def intan_channel_int_to_str(self, intan_channel):
        return chr((intan_channel // 32) + ord("A")) + "-" + str(intan_channel % 32)

    def intan_channel_str_to_int(self, intan_channel):
        group, channel = intan_channel.split("-")
        return (ord(group) - ord("A")) * 32 + int(channel)

    def plot_network(self, channels, show=False, save_path=None, reverse=False):
        """
        OE Arangement:
        (2)rhd_64 ->, (1)rhd_64 <-
        Intan Arangement
        (D)rhs_32 ->, (C)rhs_32 ->, (B)rhs_32 <-, (A)rhs_32 <-

        -> means chip-side on bottom

        Parameters
        ----------
        channel : int or str
        reverse : bool
            If false, mapping is from "given OE channel, output Intan channel"
            If true, mapping is from "given Intan channel, output OE channel"
        """

        # Network Mapping
        oe_arrangement = np.rot90(self.oe_map)
        intan_arrangement = np.rot90(self.intan_map)
        mea = self.mea

        # DEBUG
        # print(oe_arrangement)
        # print(intan_arrangement)
        # print(mea)

        if reverse:
            channels = [self.intan_channel_str_to_int(channel) for channel in channels]

        G = nx.Graph()

        for idl_channel in range(128):
            # if idl_channel in channels:
            #    continue
            idl_intan_channel = self.channel_mapping(idl_channel)

            oe_x, oe_y = np.where(oe_arrangement == idl_channel)
            oe_x, oe_y = oe_x[0], oe_y[0]
            intan_x, intan_y = np.where(intan_arrangement == idl_intan_channel)
            intan_x, intan_y = intan_x[0], intan_y[0]
            mea_x, mea_y = np.where(mea == idl_channel)
            mea_x, mea_y = mea_x[0], mea_y[0]

            G.add_node(
                ("Idel_oe", idl_channel),
                name=str(idl_channel),
                pos=(oe_x, oe_y),
                color="lightgrey",
                alpha=0.5,
            )
            G.add_node(
                ("Idel_intan", idl_intan_channel),
                name=self.intan_channel_int_to_str(idl_intan_channel),
                pos=(intan_x + oe_arrangement.shape[0], intan_y + 0.3),
                color="lightgrey",
                alpha=0.5,
            )
            G.add_node(
                ("Idel_mea", idl_channel),
                name=str(idl_channel),
                pos=(
                    mea_x + oe_arrangement.shape[0] + intan_arrangement.shape[0],
                    mea_y + 0.6,
                ),
                color="lightgrey",
                alpha=0.5,
            )

        for channel in channels:
            if reverse:
                intan_channel = channel
                channel = self.channel_mapping(intan_channel, reverse=True)
            else:
                intan_channel = self.channel_mapping(channel)

            oe_x, oe_y = np.where(oe_arrangement == channel)
            oe_x, oe_y = oe_x[0], oe_y[0]
            intan_x, intan_y = np.where(intan_arrangement == intan_channel)
            intan_x, intan_y = intan_x[0], intan_y[0]
            mea_x, mea_y = np.where(mea == channel)
            mea_x, mea_y = mea_x[0], mea_y[0]
            G.add_node(
                ("OE", channel),
                name=str(channel),
                pos=(oe_x, oe_y),
                color="blue",
                alpha=1.0,
            )
            G.add_node(
                ("Intan", intan_channel),
                name=self.intan_channel_int_to_str(intan_channel),
                pos=(intan_x + oe_arrangement.shape[0], intan_y + 0.3),
                color="red",
                alpha=1.0,
            )
            G.add_node(
                ("MEA", channel),
                name=str(channel),
                pos=(
                    mea_x + oe_arrangement.shape[0] + intan_arrangement.shape[0],
                    mea_y + 0.6,
                ),
                color="grey",
                alpha=1.0,
            )
            G.add_edge(("OE", channel), ("Intan", intan_channel))
            G.add_edge(("Intan", intan_channel), ("MEA", channel))
        pos = {node: (data[1], -data[0]) for node, data in G.nodes(data="pos")}
        labels = {node: data for node, data in G.nodes(data="name")}
        colors = [data for node, data in G.nodes(data="color")]
        alphas = [data for node, data in G.nodes(data="alpha")]

        plt.figure(figsize=(8, 16))
        nx.draw(G, pos=pos, node_size=500, node_color=colors, alpha=alphas)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="white")
        nx.draw_networkx_edges(G, pos, width=2, edge_color="black", alpha=0.8)

        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()

        return


if __name__ == "__main__":
    mea_128 = MEA128()

    # targets = [0,11,22,33,44,55,66,77,88,99,101,111]
    # mea_128.plot_network(targets)

    targets = [
        "B-21",
        "A-25",
        "A-29",
        "A-27",
        "A-31",
        "A-26",
        "A-4",
        "C-24",
        "C-9",
        "C-6",
        "C-4",
        "C-16",
        "C-14",
        "D-28",
        "D-31",
        "D-20",
        "D-5",
        "D-14",
        "D-4",
    ]
    mea_128.plot_network(targets, reverse=True, show=True)
