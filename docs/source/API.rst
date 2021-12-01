API reference of PySSC
======================

Wrapper routines
----------------

All of the necessary routines are encapsulated in the PySSC module and the main wrapper function.

.. autofunction:: PySSC.Sij
.. autofunction:: PySSC.Sijkl


Full-sky routines
-----------------

Several full-sky implementation are available to compute the Sij matrix. 

.. autofunction:: PySSC.Sij_fullsky
.. autofunction:: PySSC.Sij_alt_fullsky
.. autofunction:: PySSC.Sij_AngPow_fullsky

And only one is implemented for the Sijkl matrix:

.. autofunction:: PySSC.Sijkl_fullsky

Partial-sky routines
--------------------

Two partial-sky implemetentationas are available to compute the Sij matrix:

.. autofunction:: PySSC.Sij_psky
.. autofunction:: PySSC.Sij_AngPow

And only one is implemented for the Sijkl matrix:

.. autofunction:: PySSC.Sijkl_psky


Flat-sky routine
----------------

An additional Sij implementation is provided in the flat-sky limit.

.. autofunction:: PySSC.Sij_flatsky


