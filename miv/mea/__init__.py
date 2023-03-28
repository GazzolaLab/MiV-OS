import numpy as np

from miv.mea.grid import *
from miv.mea.protocol import *
from miv.mea.unstructured import *

mea_map = {
    "64_upper_half_intanRHD_stim": np.array(
        [
            [-1, 2, 4, 6, 23, 20, 18],
            [10, 9, 3, 5, 24, 19, -1],
            [12, 11, 1, 7, 22, -1, -1],
            [15, 14, 13, 8, 21, -1, -1],
            [-1, 25, 26, -1, -1, -1, -1],
            [27, 28, 31, -1, -1, -1, -1],
            [29, 30, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.int_,
    )
    - 1,
    "64_intanRHD": np.array(
        [
            [-1, 10, 12, 14, 31, 28, 26, -1],
            [18, 17, 11, 13, 32, 27, 38, 37],
            [20, 19, 9, 15, 30, 39, 36, 35],
            [23, 22, 21, 16, 29, 34, 33, 56],
            [-1, 1, 2, 61, 44, 53, 54, 55],
            [3, 4, 7, 62, 43, 48, 51, 52],
            [5, 6, 59, 64, 41, 46, 49, 50],
            [-1, 58, 60, 63, 42, 45, 47, -1],
        ],
        dtype=np.int_,
    )
    - 1,
    "128_first_32_rhs": np.array(
        [
            [10, 23, 9, 24],
            [12, 21, 11, 22],
            [14, 19, 13, 20],
            [16, 17, 15, 18],
            [8, 25, 7, 26],
            [6, 27, 5, 28],
            [4, 29, 3, 30],
            [2, 31, 1, 32],
        ],
        dtype=np.int_,
    )
    - 1,
    "128_dual_connector_two_64_rhd": np.array(
        [
            (np.array([25, 24, 19, 18]) + 64).tolist()
            + (np.array([112, 109, 106, 103]) - 64).tolist(),
            (np.array([27, 26, 22, 17]) + 64).tolist()
            + (np.array([111, 108, 104, 101]) - 64).tolist(),
            (np.array([29, 28, 23, 20]) + 64).tolist()
            + (np.array([110, 105, 102, 99]) - 64).tolist(),
            (np.array([31, 32, 30, 21]) + 64).tolist()
            + (np.array([107, 100, 98, 97]) - 64).tolist(),
            (np.array([3, 4, 1, 2]) + 64).tolist()
            + (np.array([128, 127, 126, 125]) - 64).tolist(),
            (np.array([7, 8, 5, 6]) + 64).tolist()
            + (np.array([124, 123, 122, 121]) - 64).tolist(),
            (np.array([11, 12, 9, 10]) + 64).tolist()
            + (np.array([120, 119, 118, 117]) - 64).tolist(),
            (np.array([15, 16, 13, 14]) + 64).tolist()
            + (np.array([116, 115, 114, 113]) - 64).tolist(),
            (np.array([63, 64, 61, 62]) + 64).tolist()
            + (np.array([68, 67, 66, 65]) - 64).tolist(),
            (np.array([59, 60, 57, 58]) + 64).tolist()
            + (np.array([72, 71, 70, 69]) - 64).tolist(),
            (np.array([55, 56, 53, 54]) + 64).tolist()
            + (np.array([76, 75, 74, 73]) - 64).tolist(),
            (np.array([51, 52, 49, 50]) + 64).tolist()
            + (np.array([80, 79, 78, 77]) - 64).tolist(),
            (np.array([34, 33, 35, 44]) + 64).tolist()
            + (np.array([86, 93, 95, 96]) - 64).tolist(),
            (np.array([36, 37, 39, 45]) + 64).tolist()
            + (np.array([83, 89, 91, 94]) - 64).tolist(),
            (np.array([38, 42, 43, 48]) + 64).tolist()
            + (np.array([82, 85, 88, 92]) - 64).tolist(),
            (np.array([40, 41, 46, 47]) + 64).tolist()
            + (np.array([81, 84, 87, 90]) - 64).tolist(),
        ],
        dtype=np.int_,
    )
    - 1,
}

# Note: [:,::-1] to mirror the provided image
rhd_64 = np.array(
    [  # https://intantech.com/RHD_headstages.html?tabSelect=RHD64ch&yPos=0
        # Chip side
        [-1, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, -1],
        [-1, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17, -1],
        [-1, 49, 51, 53, 55, 57, 59, 61, 63, 1, 3, 5, 7, 9, 11, 13, 15, -1],
        [-1, 48, 50, 52, 54, 56, 58, 60, 62, 0, 2, 4, 6, 8, 10, 12, 14, -1],
    ]
)[:, ::-1]

rhs_32 = np.array(
    [  # https://intantech.com/RHS_headstages.html?tabSelect=RHS32ch&yPos=0
        # Chip side
        [-1, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1],
        [-1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, -1],
    ]
)[:, ::-1]
