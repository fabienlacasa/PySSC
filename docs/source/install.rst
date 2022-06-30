=======================
Installation
=======================

Installation requirements
-------------------------

Basic requirements
..................
- Basic python modules ``math``, ``numpy`` (>=1.15), ``scipy`` and ``copy`` .

- ``classy``, the python-wrapper of CLASS. See `CLASS wiki <https://github.com/lesgourg/class_public/wiki/Installation>`_.

Optional requirements
.....................
- To run the notebooks: ``jupyter``.
- To make the plots: ``matplotlib``.
- To run the partial sky implementation: ``healpy`` and ``astropy``.
- To use the ``Angpow`` method:

 * Python modules: ``time``, ``os``, ``shutil``, ``mpi4py``
 * Linux Ubuntu packages: ``g++``, ``libfftw3-dev``, ``mpich``
 * Go to the :doc:`angpow` section for further detailed instructions


Installing
----------
Clone the repository

   .. code-block::
    
    git clone --recurse-submodules https://github.com/fabienlacasa/PySSC.git
    
| Note that the --recurse-submodules argument also installs the `AngPow <https://gitlab.in2p3.fr/campagne/AngPow>`_ submodule
| Then either put the path to PySSC.py in your $PYTHONPATH, or install the package with pip

   .. code-block::
    
    pip install -e .
    
Note that the pip method does not (yet) work for routines using AngPow.
