import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# fmt: off
mg_cdict = {
    "red": (
        (0.0    , 0.231373 , 0.231373) ,
        (0.4    , 0.535058 , 0.535058) ,
        (0.4875 , 1.0      , 1.0)      ,
        (0.5    , 1.0      , 1.0)      ,
        (0.5125 , 1.0      , 1.0)      ,
        (0.6    , 1.0      , 1.0)      ,
        (1.0    , 0.733898 , 0.733898) ,
    ),
    "green": (
        (0.0    , 0.298039 , 0.298039) ,
        (0.4    , 0.723751 , 0.723751) ,
        (0.4875 , 1.0      , 1.0)      ,
        (0.5    , 1.0      , 1.0)      ,
        (0.5125 , 1.0      , 1.0)      ,
        (0.6    , 0.713756 , 0.713756) ,
        (1.0    , 0.0134737, 0.0134737),
    ),
    "blue": (
        (0.0    , 0.752941 , 0.752941) ,
        (0.4    , 0.85594  , 0.85594)  ,
        (0.4875 , 1.0      , 1.0)      ,
        (0.5    , 1.0      , 1.0)      ,
        (0.5125 , 1.0      , 1.0)      ,
        (0.6    , 0.294972 , 0.294972) ,
        (1.0    , 0.150759 , 0.150759) ,
    ),
}

mg_alpha_cdict = {
    **mg_cdict,
    "alpha": (
        (0.0    , 1.0 , 1.0) ,
        (0.4    , 1.0 , 1.0) ,
        (0.4875 , 0.5 , 0.5) ,
        (0.5    , 0.3 , 0.3) ,
        (0.5125 , 0.5 , 0.5) ,
        (0.6    , 1.0 , 1.0) ,
        (1.0    , 1.0 , 1.0) ,
    ),
}
# fmt: on

mpl.colormaps.register(LinearSegmentedColormap("MGBlueOrange", mg_cdict))
mpl.colormaps.register(LinearSegmentedColormap("MGBlueOrangeAlpha", mg_alpha_cdict))

if __name__ == "__main__":
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(category, cmap_list):
        # Create figure and adjust figure height to number of colormaps
        nrows = len(cmap_list)
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
        fig.subplots_adjust(
            top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99
        )
        axs[0].set_title(f"{category} colormaps", fontsize=14)

        for ax, name in zip(axs, cmap_list):
            ax.imshow(gradient, aspect="auto", cmap=mpl.colormaps[name])
            ax.text(
                -0.01,
                0.5,
                name,
                va="center",
                ha="right",
                fontsize=10,
                transform=ax.transAxes,
            )

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axs:
            ax.set_axis_off()

        plt.show()

    plot_color_gradients("MiV-Special", ["MGBlueOrange", "MGBlueOrangeAlpha"])
