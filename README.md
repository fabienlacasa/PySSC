# pyssc
A Python implementation of the fast Super-Sample Covariance (SSC) from Lacasa & Grain 2018 arXiv:1809.05437

Dependencies: math, numpy, scipy, classy (python wrapper of CLASS, http://class-code.net)

The module is pyssc.py.
It can be placed in your Python Path and imported with
$ import pyssc
It contains a function pyssc.Sij which computes the Sij matrix (defined in the article), that allows to easily build the SSC covariance matrix.

A jupyter notebook shows hows the Sij matrix is computed and reproduces the plots of the article. It requires matplotlib and can be launched with 
$ jupyter-notebook plots-article.ipynb
Self-explanatory comments are included.
