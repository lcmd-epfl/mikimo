from glob import glob
from os import path

import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

script_files = []
for fname in glob("navicat_mikimo/**/*", recursive=True):
    if path.isfile(fname):
        script_files += [fname]

setup(
    name="navicat_mikimo",
    version="1.0.1",
    description="microkinetic modeling code for homogeneous catalytic reactions",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="pregabalin_hoshiyomi",
    author_email="thanapat.worakul@epfl.ch",
    url="https://github.com/lcmd-epfl/mikimo",
    packages=["navicat_mikimo"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "autograd",
        "matplotlib",
        "pandas",
        "h5py",
        "fire",
        "navicat_volcanic",
        "openpyxl",
    ],
    keywords="computational chemistry utility",
    entry_points={"console_scripts": ["navicat_mikimo=navicat_mikimo.__main__:main"]},
    include_package_data=True,
)
