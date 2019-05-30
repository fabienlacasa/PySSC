# PySSC
A Python implementation of the fast Super-Sample Covariance (SSC) from Lacasa & Grain 2018 arXiv:1809.05437

Dependencies: math, numpy, scipy, classy (Python wrapper of CLASS, http://class-code.net)

Contains:
- PySSC.py : the module to compute the SSC
- plots-article.ipynb : a jupyter notebook showcasing how the computation works and reproducing the plots of the article 
- examples.ipynb : a jupyter notebook with example applications using the module

The module is PySSC.py
You can place it in your Python path and import it with
$ import PySSC

The module contains functions to compute the Sij matrix (defined in the article) that allows to easily build the SSC covariance matrix.
- PySSC.turboSij() : computes Sij with sharp disjoint redshift bins
- PySSC.Sij() : computes Sij with more general window functions
- PySSC.Sij_alt() : alternative to PySSC.Sij(), computation through a different route for comparison. Generally slower for high number of redshift integration points.
