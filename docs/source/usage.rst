Usage
=====

After installing all the required dependencies, import the module with ``import PySSC``.


Computing Sij matrices
----------------------

The module contains functions to compute the Sij matrix (defined in the article) that allows to easily build the SSC covariance matrix.

.. autosummary:: PySSC.Sij

computes the Sij matrix for given redshift bins ``windows`` defined on redshift array ``z_arr``. 
The ``sky`` parameter is either ``"full"`` or ``"partial"`` for full-sky or partial-sky computation, respectively.
The ``method`` parameter can be ``"classic"``, ``"alternative"`` or ``"angpow"``.


Example
-------

The following example computes the full sky Sij matrix for top-hat redshift bins

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

  # Compute the matrix
  Sij = PySSC.Sij(z_arr,windows_T,sky='full',method='classic') 
  
  # Plot the matrix
  # Sij can be negative (anti-correlation between bins), and varies by some order of magnitude due to redshift evolution.
  fig = plt.figure(figsize=(5.5,5))
  P = plt.imshow(np.log(abs(Sij)),interpolation='none',cmap='bwr',extent=[zmin,zmax,zmax,zmin])
  plt.xticks([]) ; plt.yticks([])
  ax1 = fig.add_axes([0.89, 0.1, 0.035, 0.8])
  cbar = plt.colorbar(P,ax1)
  cbar.ax.tick_params(labelsize=15)
  plt.show()
  
For more extensive examples, see the :doc:`Example` section or download the corresponding `notebook <https://github.com/fabienlacasa/PySSC/blob/docu/examples.ipynb>`_

It is possible to compute the Sijkl matrix, i.e. the most general case with cross-spectra, or the simplified Sij matrix with the two following functions:

.. autosummary:: PySSC.Sij
.. autosummary:: PySSC.Sijkl
