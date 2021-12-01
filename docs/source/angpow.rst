=======================
AngPow method
=======================

| You can run PySSC with the AngPow method.
| To do so requires steps outlined below.
| Note that the computation with AngPow is currently slower than with the classic method and you thus need access to a computer with several cores to parallelise the computation.


Installation
------------

Requirements
..................

- Python modules: ``time``, ``os``, ``shutil``, ``mpi4py``
- Linux Ubuntu packages: ``g++``, ``libfftw3-dev``, ``mpich``


Installing
..........

After cloning the PySSC git repository, open a terminal and move to the PySSC directory. Then run in the terminal

.. code-block:: bash
    
    $ cd AngPow ; git clone https://gitlab.in2p3.fr/campagne/AngPow.git ; cd AngPow ; make
    

Notes
.....
- By default, the code runs locally and uses all available
- You need to have a python executable called ``python``, or make a symbolic link to your python executable
- You need to tell Class to go to high wavenumbers: add ``'P_k_max_h/Mpc':20`` to the dictionnary fed as ``cosmo_Class`` input

Use
---
TBD
