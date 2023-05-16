from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
     long_description = fh.read()
     
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
        'navicat_volcanic'
    ],
    include_package_data=True,
)
