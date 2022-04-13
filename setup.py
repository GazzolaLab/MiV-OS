#!/usr/bin/env python
# Thanks and credits to https://github.com/navdeep-G/setup.py for setup.py format
import io
import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup

from miv.version import VERSION

# Package meta-data.
NAME = "MiV-OS"
DESCRIPTION = (
    "Python software for analysis and computing framework used in MiV project."
)
URL = "https://github.com/GazzolaLab/MiV-OS"
EMAIL = "skim449@illinois.edu"
AUTHOR = "GazzolaLab"
REQUIRES_PYTHON = ">=3.8.0"

# What packages are required for this module to be executed?
REQUIRED = [
    "matplotlib",
    "matplotlib>=3.3.2",
    "McsPyDataTools",
    "numba>=0.51.0",
    "numpy>=1.19.2",
    "omegaconf",
    "pandas",
    "Pillow",
    "quantities",
    "scikit-learn",
    "scipy>=1.5.2",
    "seaborn",
    "sklearn",
    "tqdm>=4.61.1",
]

# What packages are optional?
EXTRAS = {
    "dev": ["black", "pre-commit", "pytest", "flake8", "mypy"],
    "build": ["twine"],
    "docs": [
        "sphinx==4.4.0",
        "sphinx_rtd_theme==1.0.0",
        "sphinx-book-theme",
        "readthedocs-sphinx-search==0.1.1",
        "sphinx-autodoc-typehints",
        "myst-parser",
        "numpydoc",
        "docutils",
    ],
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    # packages=["miv"],
    package_dir={"miv": "./miv"},
    packages=find_packages(),
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
    ],
    download_url=f"https://github.com/GazzolaLab/MiV-OS/archive/refs/tags/{VERSION}.tar.gz",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    # $ setup.py publish support: We don't know how to use this part. We already know how to publish.!
    cmdclass={
        "upload": UploadCommand,
    },
)
