import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import glob
with open("README.rst", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    return config

setup(
    name='PySCC',
    version='1.0',
    packages=['.'],
    url='https://github.com/fabienlacasa/PySSC',
    author='PySSC',
    author_email='to.fabien.lacasa@unige.ch',
    description='A Python implementation of the fast super-sample covariance from arXiv:1809.05437 ',
    install_requires=['numpy', 'scipy', 'classy'],
    long_description=long_description,
    configuration=configuration)

