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
    name='PySSC',
    version='3.1',
    packages=['.'],
    url='https://github.com/fabienlacasa/PySSC',
    author='PySSC team',
    author_email='to.fabien.lacasa@unige.ch',
    description='A Python implementation of the fast Sij approach to Super-Sample Covariance (SSC). You can find the full documentation at https://pyssc.readthedocs.io',
    install_requires=['numpy', 'scipy', 'classy'],
    long_description=long_description,
    configuration=configuration)

