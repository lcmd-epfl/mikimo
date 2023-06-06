from glob import glob
from os import path

import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
     long_description = fh.read()

script_files = []
for fname in glob("spectre/**/*", recursive=True):
    if path.isfile(fname):
        script_files += [fname]
        
setup(
    name='spectre',
    version='0.0',
    description='microkinetic modeling code for homogeneous catalytic reactions',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author='pregabalin_Hoshiyomi',
    author_email='thanapat.worakul@epfl.ch',
    url="https://github.com/PregY/spectre",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        'numpy',
        'scipy',
        'autograd',
        'matplotlib',
        'pandas',
        'h5py',
        'fire',
        'navicat_volcanic'
    ],
    keywords="computational chemistry utility",
    entry_points={'console_scripts': ['spectre=spectre.__main__:main']},
    include_package_data=True,
)
