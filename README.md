# pyssc
A Python implementation of the fast Super-Sample Covariance (SSC) from Lacasa & Grain 2018 arXiv:1809.05437

Dependencies: math, numpy, scipy, classy (python wrapper of CLASS, http://class-code.net)

Contains:
- pyssc.py : the module to compute the SSC
- examples.ipynb : a jupyter notebook with example applications using the module
- plots-article.ipynb : another jupyter notebook showcasing how the computation works and reproducing the plots of the article 

The module is pyssc.py
It can be placed in your Python Path and imported with
$ import pyssc
It contains a function pyssc.turboSij() which computes the Sij matrix (defined in the article) with sharp disjoint redshift bins. That allows to easily build the SSC covariance matrix.
In progress : new pyssc.Sij() function that allows to account for more general window functions.
