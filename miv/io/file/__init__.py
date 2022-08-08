# HDF5-based file format for heterogeneous numerical data.
# Based on code from and inspired by
#
# HEPfile: https://github.com/mattbellis/hepfile
# NeuroH5: https://github.com/iraikov/neuroh5
#

__version__ = "0.0.1"

__all__ = ("__version__",)

from miv_file.read import *
from miv_file.write import *
