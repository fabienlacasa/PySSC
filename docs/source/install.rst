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
Simply clone the repository, then put the PySSC.py module in your Python path
   .. code-block::
    
    git clone https://github.com/fabienlacasa/PySSC.git
    
