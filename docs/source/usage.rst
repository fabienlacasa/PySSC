Usage
=====

After installing all the required dependencies, import the module with ``import PySSC``.


Creating recipes
----------------

The module contains functions to compute the Sij matrix (defined in the article) that allows to easily build the SSC covariance matrix.

.. autofunction:: PySSC.Sij

computes the Sij matrix for given redshift bins ``windows`` defined on redshift array ``z_arr``. 
The ``sky`` parameter is either ``"full"`` or ``"partial"`` for full-sky or partial-sky computation, respectively.
The ``method`` parameter can be ``"classic"``, ``"alternative"`` or ``"angpow"``.
