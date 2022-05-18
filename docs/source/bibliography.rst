=======================
Bibliography
=======================
Bibtex entries for each reference are given in footnote, ready for ctrl+c.

What to cite
-------------
- Minimal: Lacasa & Grain 2018, Numpy and Scipy.
- If you use this documentation (for instance if you are reading this right now): Gouyou Beauchamps et al.
- Unless you provide your own cosmological quantities as input: CLASS
- If you use the partial sky implementation: Healpix, Astropy
- If you use the AngPow method: AngPow

Base articles
-------------

- Original: `Lacasa & Grain 2018 <https://ui.adsabs.harvard.edu/abs/2019A%26A...624A..61L>`_ [1]_, `arXiv:1809.05437 <https://arxiv.org/abs/1809.05437>`_

- Partial sky implementation and documentation: `Gouyou Beauchamps et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022A%2526A...659A.128G>`_ [2]_, `arXiv:2109.02308 <https://arxiv.org/abs/2109.02308>`_

Cosmology dependencies
----------------------
- Healpix: `Gorski et al. 2005 <https://ui.adsabs.harvard.edu/abs/2005ApJ...622..759G>`_ [3]_, `arXiv:astro-ph/0409513 <https://arxiv.org/abs/astro-ph/0409513>`_
- CLASS: `Blas et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011JCAP...07..034B>`_ [4]_, `arXiv:1104.2933 <https://arxiv.org/abs/1104.2933>`_
- AngPow: `Campagne et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017A%26A...602A..72C/abstract>`_ [5]_, `arXiv:1701.03592 <https://arxiv.org/abs/1701.03592>`_

Python dependencies
-------------------
- Numpy: `<www.numpy.org>`_ [6]_
- Scipy: `<www.scipy.org>`_ [7]_
- Matplotlib: `Hunter 2007 <https://ui.adsabs.harvard.edu/abs/2007CSE.....9...90H>`_ [8]_
- Astropy: `Astropy Collaboration 2013 <https://ui.adsabs.harvard.edu/abs/2013A%26A...558A..33A>`_ [9]_ `arXiv:1307.6212 <https://arxiv.org/abs/1307.6212>`_, `Astropy Collaboration 2018 <https://ui.adsabs.harvard.edu/abs/2018AJ....156..123A>`_ [10]_ `arXiv:1801.02634 <https://arxiv.org/abs/1801.02634>`_


Bibtex entries
--------------
   .. [1]
   .. code-block::
    
    @ARTICLE{Lacasa2019,
        author = {{Lacasa}, Fabien and {Grain}, Julien},
        title = "{Fast and easy super-sample covariance of large-scale structure observables}",
        journal = {\aap},
        keywords = {large-scale structure of Universe, galaxies: statistics, methods: data analysis, methods: analytical, Astrophysics - Cosmology and Nongalactic Astrophysics},
        year = 2019,
        month = apr,
        volume = {624},
        eid = {A61},
        pages = {A61},
        doi = {10.1051/0004-6361/201834343},
        archivePrefix = {arXiv},
        eprint = {1809.05437},
        primaryClass = {astro-ph.CO},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2019A&A...624A..61L},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    
   .. [2]
   .. code-block::
    
    @ARTICLE{GouyouBeauchamps2022,
        author = {{Gouyou Beauchamps}, S. and {Lacasa}, F. and {Tutusaus}, I. and {Aubert}, M. and {Baratta}, P. and {Gorce}, A. and {Sakr}, Z.},
        title = "{Impact of survey geometry and super-sample covariance on future photometric galaxy surveys}",
        journal = {\aap},
        keywords = {large-scale structure of Universe, Astrophysics - Cosmology and Nongalactic Astrophysics},
        year = 2022,
        month = mar,
        volume = {659},
        eid = {A128},
        pages = {A128},
        doi = {10.1051/0004-6361/202142052},
        archivePrefix = {arXiv},
        eprint = {2109.02308},
        primaryClass = {astro-ph.CO},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2022A&A...659A.128G},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    

   .. [3]
   .. code-block::
    
    @ARTICLE{Gorski2005,
        author = {{G{\'o}rski}, K.~M. and {Hivon}, E. and {Banday}, A.~J. and {Wandelt}, B.~D. and {Hansen}, F.~K. and {Reinecke}, M. and {Bartelmann}, M.},
        title = "{HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere}",
        journal = {\apj},
        eprint = {astro-ph/0409513},
        keywords = {Cosmology: Cosmic Microwave Background, Cosmology: Observations, Methods: Statistical},
        year = 2005,
        month = apr,
        volume = 622,
        pages = {759-771},
        doi = {10.1086/427976},
        adsurl = {http://adsabs.harvard.edu/abs/2005ApJ...622..759G},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    
   .. [4]
   .. code-block::
    
    @ARTICLE{Blas2011,
        author = {{Blas}, Diego and {Lesgourgues}, Julien and {Tram}, Thomas},
        title = "{The Cosmic Linear Anisotropy Solving System (CLASS). Part II: Approximation schemes}",
        journal = {\jcap},
        keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
        year = 2011,
        month = jul,
        volume = {2011},
        number = {7},
        eid = {034},
        pages = {034},
        doi = {10.1088/1475-7516/2011/07/034},
        archivePrefix = {arXiv},
        eprint = {1104.2933},
        primaryClass = {astro-ph.CO},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2011JCAP...07..034B},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    
   .. [5]
   .. code-block::
    
    @ARTICLE{Campagne2017,
        author = {{Campagne}, J. -E. and {Neveu}, J. and {Plaszczynski}, S.},
        title = "{Angpow: a software for the fast computation of accurate tomographic power spectra}",
        journal = {\aap},
        keywords = {large-scale structure of Universe, methods: numerical, Astrophysics - Cosmology and Nongalactic Astrophysics},
        year = 2017,
        month = jun,
        volume = {602},
        eid = {A72},
        pages = {A72},
        doi = {10.1051/0004-6361/201730399},
        archivePrefix = {arXiv},
        eprint = {1701.03592},
        primaryClass = {astro-ph.CO},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2017A&A...602A..72C},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    
   .. [6]
   .. code-block::
    
    @Misc{numpy,
        author =    {Travis Oliphant},
        title =     {{NumPy}: A guide to {NumPy}},
        year =      {2006},
        howpublished = {USA: Trelgol Publishing},
        url = "https://www.numpy.org"
    }
    
   .. [7]
   .. code-block::
    
    @Misc{scipy,
        author =    {{Jones}, E. and {Oliphant}, T. and {Peterson}, P. and others},
        title =     {{SciPy}: Open source scientific tools for {Python}},
        year =      {2001},
        url = "https://www.scipy.org"
    }
    
   .. [8]
   .. code-block::
    
    @ARTICLE{Matplotlib,
        author = {{Hunter}, John D.},
        title = "{Matplotlib: A 2D Graphics Environment}",
        journal = {Computing in Science and Engineering},
        keywords = {Python, Scripting languages, Application development, Scientific programming},
        year = 2007,
        month = may,
        volume = {9},
        number = {3},
        pages = {90-95},
        doi = {10.1109/MCSE.2007.55},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2007CSE.....9...90H},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    
   .. [9]
   .. code-block::
    
    @ARTICLE{astropy,
        author = {{Astropy Collaboration: Robitaille}, T.~P. and {Tollerud}, E.~J. and {Greenfield}, P. and {Droettboom}, M. and {Bray}, E. and {Aldcroft}, T. and {Davis}, M. and {Ginsburg}, A. and {Price-Whelan}, A.~M. and {Kerzendorf}, W.~E. and {Conley}, A. and {Crighton}, N. and {Barbary}, K. and {Muna}, D. and {Ferguson}, H. and {Grollier}, F. and {Parikh}, M.~M. and {Nair}, P.~H. and {Unther}, H.~M. and {Deil}, C. and {Woillez}, J. and {Conseil}, S. and {Kramer}, R. and {Turner}, J.~E.~H. and {Singer}, L. and {Fox}, R. and {Weaver}, B.~A. and {Zabalza}, V. and {Edwards}, Z.~I. and {Azalee Bostroem}, K. and {Burke}, D.~J. and	{Casey}, A.~R. and {Crawford}, S.~M. and {Dencheva}, N. and {Ely}, J. and {Jenness}, T. and {Labrie}, K. and {Lim}, P.~L. and {Pierfederici}, F. and {Pontzen}, A. and {Ptak}, A. and {Refsdal}, B. and {Servillat}, M. and {Streicher}, O.},
        title = "{Astropy: A community Python package for astronomy}",
        journal = {\aap},
        archivePrefix = "arXiv",
        eprint = {1307.6212},
        primaryClass = "astro-ph.IM",
        keywords = {methods: data analysis, methods: miscellaneous, virtual observatory tools},
        year = 2013,
        month = oct,
        volume = 558,
        eid = {A33},
        pages = {A33},
        doi = {10.1051/0004-6361/201322068},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2013A%26A...558A..33A},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    
   .. [10]
   .. code-block::
    
    @ARTICLE{astropy2,
        author = {{Astropy Collaboration: Price-Whelan}, A.~M. and {Sip{\H{o}}cz}, B.~M. and {G{\"u}nther}, H.~M. and {Lim}, P.~L. and {Crawford}, S.~M. and {Conseil}, S. and {Shupe}, D.~L. and {Craig}, M.~W. and {Dencheva}, N. and {Ginsburg}, A. and {VanderPlas}, J.~T. and {Bradley}, L.~D. and {P{\'e}rez-Su{\'a}rez}, D. and {de Val-Borro}, M. and {Aldcroft}, T.~L. and {Cruz}, K.~L. and {Robitaille}, T.~P. and {Tollerud}, E.~J. and {Ardelean}, C. and {Babej}, T. and {Bach}, Y.~P. and {Bachetti}, M. and {Bakanov}, A.~V. and {Bamford}, S.~P. and {Barentsen}, G. and {Barmby}, P. and {Baumbach}, A. and {Berry}, K.~L. and {Biscani}, F. and {Boquien}, M. and {Bostroem}, K.~A. and {Bouma}, L.~G. and {Brammer}, G.~B. and {Bray}, E.~M. and {Breytenbach}, H. and {Buddelmeijer}, H. and {Burke}, D.~J. and {Calderone}, G. and {Cano Rodr{\'\i}guez}, J.~L. and {Cara}, M. and {Cardoso}, J.~V.~M. and {Cheedella}, S. and {Copin}, Y. and {Corrales}, L. and {Crichton}, D. and {D'Avella}, D. and {Deil}, C. and {Depagne}, {\'E}. and {Dietrich}, J.~P. and {Donath}, A. and {Droettboom}, M. and {Earl}, N. and {Erben}, T. and {Fabbro}, S. and {Ferreira}, L.~A. and {Finethy}, T. and {Fox}, R.~T. and {Garrison}, L.~H. and {Gibbons}, S.~L.~J. and {Goldstein}, D.~A. and {Gommers}, R. and {Greco}, J.~P. and {Greenfield}, P. and {Groener}, A.~M. and {Grollier}, F. and {Hagen}, A. and {Hirst}, P. and {Homeier}, D. and {Horton}, A.~J. and {Hosseinzadeh}, G. and {Hu}, L. and {Hunkeler}, J.~S. and {Ivezi{\'c}}, {\v{Z}}. and {Jain}, A. and {Jenness}, T. and {Kanarek}, G. and {Kendrew}, S. and {Kern}, N.~S. and {Kerzendorf}, W.~E. and {Khvalko}, A. and {King}, J. and {Kirkby}, D. and {Kulkarni}, A.~M. and {Kumar}, A. and {Lee}, A. and {Lenz}, D. and {Littlefair}, S.~P. and {Ma}, Z. and {Macleod}, D.~M. and {Mastropietro}, M. and {McCully}, C. and {Montagnac}, S. and {Morris}, B.~M. and {Mueller}, M. and {Mumford}, S.~J. and {Muna}, D. and {Murphy}, N.~A. and {Nelson}, S. and {Nguyen}, G.~H. and {Ninan}, J.~P. and {N{\"o}the}, M. and {Ogaz}, S. and {Oh}, S. and {Parejko}, J.~K. and {Parley}, N. and {Pascual}, S. and {Patil}, R. and {Patil}, A.~A. and {Plunkett}, A.~L. and {Prochaska}, J.~X. and {Rastogi}, T. and {Reddy Janga}, V. and {Sabater}, J. and {Sakurikar}, P. and {Seifert}, M. and {Sherbert}, L.~E. and {Sherwood-Taylor}, H. and {Shih}, A.~Y. and {Sick}, J. and {Silbiger}, M.~T. and {Singanamalla}, S. and {Singer}, L.~P. and {Sladen}, P.~H. and {Sooley}, K.~A. and {Sornarajah}, S. and {Streicher}, O. and {Teuben}, P. and {Thomas}, S.~W. and {Tremblay}, G.~R. and {Turner}, J.~E.~H. and {Terr{\'o}n}, V. and {van Kerkwijk}, M.~H. and {de la Vega}, A. and {Watkins}, L.~L. and {Weaver}, B.~A. and {Whitmore}, J.~B. and {Woillez}, J. and {Zabalza}, V. and {Astropy Contributors}},
        title = "{The Astropy Project: Building an Open-science Project and Status of the v2.0 Core Package}",
        journal = {\aj},
        keywords = {methods: data analysis, methods: miscellaneous, methods: statistical, reference systems, Astrophysics - Instrumentation and Methods for Astrophysics},
        year = 2018,
        month = sep,
        volume = {156},
        number = {3},
        eid = {123},
        pages = {123},
        doi = {10.3847/1538-3881/aabc4f},
        archivePrefix = {arXiv},
        eprint = {1801.02634},
        primaryClass = {astro-ph.IM},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2018AJ....156..123A},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    