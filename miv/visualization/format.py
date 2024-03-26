import matplotlib.pyplot


def format_plot2(plt: matplotlib.pyplot):

    plt.style.use("default")
    plt.rcParams["axes.linewidth"] = 2.0
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["xtick.minor.width"] = 1.2
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["ytick.minor.width"] = 1.2

    # FONT
    SMALL_SIZE = 18
    # MEDIUM_SIZE = 21
    BIGGER_SIZE = 24
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 24
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
