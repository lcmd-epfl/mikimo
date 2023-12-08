from glob import glob
from os import path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

script_files = []
for fname in glob("navicat_mikimo/**/*", recursive=True):
    if path.isfile(fname):
        script_files += [fname]

# List all Cython files
cython_modules = ["navicat_mikimo/*.pyx"]

# Convert Cython files to extensions
extensions = [
    Extension(name=mod.replace(".pyx", "").replace("/", "."), sources=[mod])
    for mod in cython_modules
]

setup(
    name="navicat_mikimo",
    version="2.0.1",
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
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)
