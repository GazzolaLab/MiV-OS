__all__ = ["MEA"]

from typing import List

import os

import MEAutility
import yaml


class _MEA:
    """
    Motivated module as MEAutility.core
    """

    _reg_electrode_paths = [
        os.path.join(os.path.split(MEAutility.__file__)[0], "electrodes"),  # default
        os.path.join(os.path.split(__file__)[0], "electrodes"),  # for development
    ]

    # Make it singleton class
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Make it singleton class.
        """
        if not cls._instance:
            cls._instance = super(MEA, cls).__new__(cls)
        return cls._instance

    def __call__(self):
        """
        Show all available MEA models.
        """
        self.return_mea_list(verbose=True)

    def register_electrode_path(self, path):
        """
        Register path for electrode definition files.

        Parameters
        ----------
        path: str
            Path to folder with electrode definition files.
        """
        self._reg_electrode_paths.append(path)

    def get_electrode_paths(self):
        """
        Get paths for electrode definition files in yaml.
        """
        paths = []
        for folder in self._reg_electrode_paths:
            if not os.path.exists(folder):
                continue
            for f in os.listdir(folder):
                # check stem
                if not f.endswith(".yml") and not f.endswith(".yaml"):
                    continue
                paths.append(os.path.join(folder, f))
        return paths

    def return_mea(self, electrode_name: str):
        """
        Returns MEA object.

        Parameters
        ----------
        electrode_name: str
            Probe/MEA name

        Returns
        -------
        mea: MEA
            The MEA object
        """
        mea = None
        counter = 0

        # load MEA info
        for fname in self.get_electrode_paths():
            with open(fname) as meafile:
                if MEAutility.core.use_loader:
                    elinfo = yaml.load(meafile, Loader=yaml.FullLoader)
                else:
                    elinfo = yaml.load(meafile)

            if elinfo["electrode_name"] == electrode_name:
                if counter > 1:
                    self.logger.warning(
                        f"Found multiple MEA models named {electrode_name}."
                        " Returning the last one."
                    )
                pos = MEAutility.core.get_positions(elinfo)
                # create MEA object
                if MEAutility.core.check_if_rect(elinfo):
                    mea = MEAutility.core.RectMEA(positions=pos, info=elinfo)
                else:
                    mea = MEAutility.core.MEA(positions=pos, info=elinfo)
                counter += 1

        if mea is None:
            self.logger.error(
                f"MEA model named {electrode_name} not found\n"
                f"Available models: {self.__call__()}"
            )
        return mea

    def return_mea_from_dict(self, info: dict):
        """
        Build MEA object from info dictionary.

        Parameters
        ----------
        info: dict
            Dictionary with electrode info

        Returns
        -------
        mea
        """
        elinfo = info
        if "center" in elinfo.keys():
            center = elinfo["center"]
        else:
            center = True
        pos = MEAutility.core.get_positions(elinfo, center=center)
        # create MEA object
        if MEAutility.core.check_if_rect(elinfo):
            mea = MEAutility.core.RectMEA(positions=pos, info=elinfo)
        else:
            mea = MEAutility.core.MEA(positions=pos, info=elinfo)

        return mea

    def return_mea_info(self, electrode_name=None):
        """
        Returns probe information.

        Parameters
        ----------
        electrode_name: str
            Probe name

        Returns
        -------
        info: dict
            Dictionary with electrode info
        """
        info = None

        # load MEA info
        for fname in self.get_electrode_paths():
            with open(fname) as meafile:
                if MEAutility.core.use_loader:
                    elinfo = yaml.load(meafile, Loader=yaml.FullLoader)
                else:
                    elinfo = yaml.load(meafile)

            if elinfo["electrode_name"] == electrode_name:
                info = elinfo

        return info

    def return_mea_list(self, verbose=False):
        """
        Returns available probe models.

        Returns
        -------
        probes: List[str]
            List of available probe_names
        """
        electrode_names = []
        for fname in self.get_electrode_paths():
            with open(fname) as meafile:
                if MEAutility.core.use_loader:
                    elinfo = yaml.load(meafile, Loader=yaml.FullLoader)
                else:
                    elinfo = yaml.load(meafile)
                electrode_names.append(elinfo["electrode_name"])

        if verbose:
            self.logger.info("Available MEA models:")
            for probe in electrode_names:
                self.logger.info(probe)

        return sorted(electrode_names)


MEA = _MEA()
