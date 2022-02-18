=======================
AngPow method
=======================

| You can run PySSC with the `AngPow <https://gitlab.in2p3.fr/campagne/AngPow>`_ method.
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

After cloning the PySSC git repository using the --recurse-submodules argument, move to the AngPow directory and compile the AngPow method:
.. code-block::
    git clone --recurse-submodules https://github.com/fabienlacasa/PySSC.git
    cd PySSC/AngPow
    make
    

Notes
.....
- By default, the code runs locally and uses the number of threads defined in the environment variable `OMP_NUM_THREADS`
- You need to have a python executable called ``python``, or make a symbolic link to your python executable
- You need to tell Class to go to high wavenumbers: add ``'P_k_max_h/Mpc':20`` to the dictionnary fed as ``cosmo_Class`` input

Use
---

The following example computes the full sky Sij matrix for top-hat redshift bins, using the AngPow method

.. code-block:: python
    # Define window functions (redshift bins)
    nz       = 500
    z_arr    = np.linspace(0,2,num=nz+1)[1:]
    nbins_T   = len(zstakes)-1
    windows_T = np.zeros((nbins_T,nz))
    for i in range(nbins_T):
        zminbin = zstakes[i] ; zmaxbin = zstakes[i+1] ; Dz = zmaxbin-zminbin
        for iz in range(nz):
            z = z_arr[iz]
            if ((z>zminbin) and (z<=zmaxbin)):
                windows_T[i,iz] = 1/Dz
    # Defining an arbitrary classy dict with high value for the 'P_k_max_h/Mpc' parameter
    cosmo_par = {'P_k_max_h/Mpc' :20 , 'h': 0.67, 'Omega_b': 0.05, 'Omega_cdm':0.27, 'n_s':0.96, 'A_s':2.1265e-9,'output':'mPk'}
    
    # Compute the matrix
    PySSC.Sij(z_arr, windows_T, sky='full', method='AngPow', cosmo_params=cosmo_par)

By default AngPow uses a number of threads defined by the OMP_NUM_THREADS environment variable. You can localy set this variable by passing the Np parameter to PySSC.Sij:

.. code-block:: python
    PySSC.Sij(z_arr, windows_T, sky='full', method='AngPow', cosmo_params=cosmo_par, Np=10)
    
For a parallel computation of PySSC with the AngPow method (using mpi4py), you must provide a path to a text file storing the IP addresses of all the nodes in the cluster network, and associated number of threads. An example is given in AngPow_tools/machinefile_example. On top of this machinefile path, the Number of threads Nn on which you want the AngPow routine to be run in mpi must be passed to PySSC.Sij:

.. code-block:: python
    PySSC.Sij(z_arr, windows_T, sky='full', method='AngPow', cosmo_params=cosmo_par, Np='Default',machinefile='path/to/the/machinefile',Nn=28)
    
Concerning the partial sky coverage, the same arguments as in the classical method can be used.
    