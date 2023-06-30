#!/usr/bin/python
# Filename: PySSC.py

####################################################################################################
#################################     IMPORT NECESSARY MODULES    #################################
####################################################################################################

import math ; pi=math.pi
import numpy as np
import warnings
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import copy

##################################################

# Default values for redshift bin, cosmo parameters etc
default_zstakes = [0.9,1]
default_cosmo_params = {'omega_b':0.022, 'omega_cdm':0.12, 'H0':67., 'n_s':0.96, 'A_s' : 2.035e-9, 'output' : 'mPk'}
AngPow_cosmo_params = copy.deepcopy(default_cosmo_params)
AngPow_cosmo_params['z_max_pk']=0
AngPow_cosmo_params['P_k_max_h/Mpc']=20

####################################################################################################
#################################           MAIN WRAPPER           #################################
####################################################################################################

def Sij(z_arr, kernels, order=2, sky='full', method='classic', cosmo_params=default_cosmo_params,
        cosmo_Class=None, convention=0, precision=10, clmask=None, mask=None, mask2=None, 
        var_tol=0.05, machinefile=None, Nn=None, Np='default', AngPow_path=None, verbose=False, debug=False):
    """Wrapper routine to compute the Sij matrix.
    It calls different routines depending on the inputs : fullsky or partial sky, computation method.    

    Parameters
    ----------
    z_arr : array_like
        Input array of redshifts of size nz.

    kernels : array_like
        2d array for the collection of kernels, shape (nbins, nz).

    order : int, default 2
        The passed kernels will be multiplied to that power, e.g. if two the kernels will be squared.
        You should normally set it to one and feed the product of kernels that you want.
        Examples: for cluster counts, set to one and feed the redshift selection functions. For cross-spectra Cl(i,j), set to one and feed the products Wi(z)*Wj(z) for all pairs of bins (i,j).
        The default is two for backward compatibility to times where the function was intended only for auto-spectra, so kernels needed to be squared internally. That default may change in future.

    sky : str, default ``'full'``
        Choice of survey geometry, given as a case-insensitive string.
        Valid choices: \n
        (i) ``'full'``/``'fullsky'``/``'full sky'``/``'full-sky'``,
        (ii) ``'psky'``/``'partial sky'``/``'partial-sky'``/``'partial'``/``'masked'``.

    method : str, default ``'classic'``
        Choice of computational method, given as a case-insensitive string.
        Valid choices: \n
        (i) ``'classic'``/``'standard'``/``'default'``/``'std'``,
        (ii) ``'alternative'``/``'alt'`` (only available for full sky),
        (iii) ``'AngPow'``/``'AP'``.

        ``'classic'`` calls to `PySSC.Sij_fullsky` or `PySSC.Sij_psky` routine. \n
        ``'alternative'`` calls to `PySSC.Sij_alt_fullsky` only in the case of `sky` set to ``'full'``. \n
        ``'AngPow'`` calls to `PySSC.Sij_Angpow` or `PySSC.Sij_AngPow_fullsky`.

    cosmo_params : dict, default `default_cosmo_params`
        Dictionary of cosmology or cosmological parameters that can be accepted by ``classy``

    cosmo_Class : classy.Class object, default None
        classy.Class object containing precomputed cosmology, if you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    convention : int, default 0
        Integer to dictate the convention used in the definition of the kernels.
        0 = Lacasa & Grain 2019.
        1 = Cosmosis, Euclid Forecasts (Blanchard et al 2020).

    precision : int, default 10
        Integer which drives the number of Fourier wavenumbers in internal integrals such as : Nk = 2**precision.

    clmask : str or numpy.ndarray, default None
        Array or path to fits file containing the angular power spectrum of the mask.
        Only implemented if `sky` is set to psky.

    mask : str or numpy.ndarray, default None
        Array or path to fits file containing the mask in healpix form.
        In that case PySSC will use healpy to compute the mask power spectrum.
        Thus it is faster to directly give clmask if you have it (or if you compute several Sij matrices for some reason).
        Only implemented if `sky` is set to psky

    mask2 : str or numpy.ndarray, default None
        Array or path to fits file containing a potential second mask in healpix form.
        In the case where you want the covariance between observables measured on different areas of the sky.
        PySSC will use healpy to compute the mask power spectrum.
        Again, it is faster to directly give clmask if you have it.
        Only implemented if `sky` is set to psky.
        If mask is set and mask2 is None, PySSC assumes that all observables share the same mask.
        
    multimask [To be implemented] : list of dictionnaries, default None
        list where each element is a dictionnary of the form {'mask':'mask.fits', 'kernels':kernels_array}.
        That is, it gives an observable and the corresponding mask.
        To be used to compute the SSC of different observables with different sky coverage.
        If multimask is set, it overrides mask and mask2.

    var_tol : float, default 0.05
         Float that drives the target precision for the sum over angular multipoles.
         Default is 5%. Lowering it means increasing the number of multipoles thus increasing computational time.
         Only implemented if `sky`  is set to psky.

    machinefile : str, default None
        Path to text file storing the IP addresses of all the nodes in the cluster network, and associated number of threads.
        machinefile is used for parallel computing in mpi.
        Default is None (running in local). If not None, the `Nn` variable must be set by the user.
        Only implemented if `method` is set to AngPow.

    Nn : int, default None
        Number of threads on which the user wants the AngPow routine to be run in mpi.
        This number should not exceed the maximum number of threads provided in machinefile.
        Default is None. If not None, the machinefile variable must be set by the user.
        Only implemented if `method` is set to AngPow.

    Np : str, default 'default'
     Equivalent to set the local environment variable OMP_NUM_THREADS to Np.
     It represents the number of processes AngPow is allowed to use on each machine.
     Default is 'default' : AngPow uses the pre-existing `OMP_NUM_THREADS` value.

    AngPow_path : str, default None
        path to the Angpow binary repertory (finishing by '/').
        Default is None and assumes that AngPow is installed at ``'./AngPow/'``.

    verbose : bool, default False
        Verbosity of the routine.
        Defaults to False.

    debug : bool, default False
        Debuging options to look for incoherence in the routine.
        Defaults to False.

    Returns
    -------

    Array_like
        Sij matrix of shape (nbins, nbins).

    """

    test_zw(z_arr,kernels)
    
    # Full sky
    if sky.casefold() in ['full','fullsky','full sky','full-sky']:
        if method.casefold() in ['classic','standard','default','std']:
            Sij=Sij_fullsky(z_arr, kernels, order=order, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, convention=convention, precision=precision)
        elif method.casefold() in ['alternative','alt']:
            Sij=Sij_alt_fullsky(z_arr, kernels, order=order, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, convention=convention)
        elif method.casefold() in ['angpow','ap']:
            test_inputs_angpow(cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, order=order, convention=convention, machinefile=machinefile, Nn=Nn, Np=Np, AngPow_path=AngPow_path)
            Sij=Sij_AngPow_fullsky(z_arr, kernels, cosmo_params=cosmo_params, machinefile=machinefile, Nn=Nn, Np=Np, AngPow_path=AngPow_path, verbose=verbose, debug=debug)
        else:
            raise Exception('Invalid string given for method parameter. Main possibilities: classic, alternative or AngPow (or variants, see code for details).')
    # Partial sky
    elif sky.casefold() in ['psky','partial sky','partial-sky','partial','masked']:
        test_mask(mask, clmask, mask2=mask2)
        if method.casefold() in ['classic','standard','default']:
            Sij=Sij_psky(z_arr, kernels, order=order, clmask=clmask, mask=mask, mask2=mask2, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, convention=convention, precision=precision, var_tol=var_tol, verbose=verbose, debug=debug)
        elif method.casefold() in ['alt','alternative']:
            raise Exception('No implementation of the alternative method for partial sky. Use classic instead.')
        elif method.casefold() in ['angpow','ap']:
            test_inputs_angpow(cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, order=order, convention=convention, machinefile=machinefile, Nn=Nn, Np=Np, AngPow_path=AngPow_path)
            Sij=Sij_AngPow(z_arr, kernels, clmask=clmask, mask=mask, mask2=mask2, cosmo_params=cosmo_params, var_tol=var_tol, machinefile=machinefile, Nn=Nn, Np=Np, AngPow_path=AngPow_path, verbose=verbose, debug=debug)
        else:
            raise Exception('Invalid string given for method parameter. Main possibilities: classic, alternative or AngPow (or variants, see code for details).')
    # Wrong geometry
    else:
        raise Exception('Invalid string given for sky geometry parameter. Main possibilities : full sky or partial sky (or abbreviations, see code for details).')

    return Sij

####################################################################################################
#################################        FULL SKY ROUTINES         #################################
####################################################################################################

##### Sij_fullsky #####
def Sij_fullsky(z_arr, kernels, order=2, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0, precision=10):
    """ Routine to compute the Sij matrix in full sky. Standard computation method.

    Parameters
    ----------
    z_arr : array_like
        Input array of redshifts of size nz.

    kernels : array_like
        2d array for the collection of kernels, shape (nbins, nz).

    order : int, default 2
        The passed kernels will be multiplied to that power, e.g. if two the kernels will be squared.
        You should normally set it to one and feed the product of kernels that you want.
        Examples: for cluster counts, set to one and feed the redshift selection functions. For cross-spectra Cl(i,j), set to one and feed the products Wi(z)*Wj(z) for all pairs of bins (i,j).
        The default is two for backward compatibility to times where the function was intended only for auto-spectra, so kernels needed to be squared internally. That default may change in future.

    cosmo_params : dict, default `default_cosmo_params`
        Dictionary of cosmology or cosmological parameters that can be accepted by ``classy``

    cosmo_Class : classy.Class object, default None
        classy.Class object containing precomputed cosmology.
        If you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    convention : int, default 0
        Integer to dictate the convention used in the definition of the kernels.
        0 = Lacasa & Grain 2019.
        1 = Cosmosic , Euclid Forecasts.
        Defaults to 0.

    precision : int, default 10
        Integer which drives the number of Fourier wavenumbers in internal integrals such as : Nk = 2**precision.

    Returns
    -------

    Array_like
        Sij matrix of shape (nbins, nbins).

    Notes
    -----
    Equation used

    .. math::
        S_{ij} = \\frac{1}{2\pi^2} \int k^2 dk \ P(k)  \\frac{U(i,k)}{I_\mathrm{norm}(i)} \\frac{U(j,k)}{I_\mathrm{norm}(j)}
    with :math:`I_\mathrm{norm}(i) = \int dX \ W(i,z)^\mathrm{order}` and :math:`U(i,k) = \int dX \ W(i,z)^\mathrm{order} \ G(z) \ j_0(kr)`.
    
    This can also be seen as an angular power spectrum:

    .. math::
       S_{ij} = \\frac{C_S(\ell=0,i,j)}{4 \pi}

    with :math:`C_S(\ell=0,i,j) = \\frac{2}{\pi} \int k^2 dk \ P(k) \\frac{U(i,k)}{I_\mathrm{norm}(i)} \\frac{U(j,k)}{I_\mathrm{norm}(j)}`.

    :math:`dX` depends on the convention used to define the observable's kernel:

    .. math::
            C_\mathrm{observable}(\ell,i,j) = \int dX \ W(i,z) \, W(j,z) \ P(k=(\ell+1/2)/r,z)

    0. :math:`dX = dV = \\frac{dV}{dz} dz = r^2(z) \\frac{dr}{dz} dz`. Used in Lacasa & Grain 2019. \n
    1. :math:`dX = d\chi/\chi^2 = \\frac{dr/dz}{r^2(z)} dz`. Used in cosmosis. \n
    The convention of the Euclid Forecasts is nearly the same as 1 \
    up to a factor :math:`c^2` (or :math:`\\frac{c^2}{H_0^2}` depending on the probe), \
    which is a constant so does not matter in the ratio here.
    """
    
    # Find number of redshifts and bins    
    nz    = z_arr.size
    nbins = kernels.shape[0]
    
    #Get cosmology, comoving distances etc from dedicated auxiliary routine
    cosmo, h, comov_dist, dcomov_dist, growth = get_cosmo(z_arr, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class)

    #Get element of z integration, depending on kernel convention
    dX_dz = get_dX_dz(comov_dist, dcomov_dist, convention=convention)
    
    # Compute normalisations
    Inorm       = np.zeros(nbins)
    for i1 in range(nbins):
        integrand = dX_dz * kernels[i1,:]**order
        Inorm[i1] = integrate.simps(integrand,z_arr)
    
    # Compute U(i,k), numerator of Sij (integral of kernels**2 * matter )
    keq         = 0.02/h                                          #Equality matter radiation in 1/Mpc (more or less)
    klogwidth   = 10                                              #Factor of width of the integration range. 10 seems ok
    kmin        = min(keq,1./comov_dist.max())/klogwidth #1e-4
    kmax        = max(keq,1./comov_dist.min())*klogwidth #1
    nk          = 2**precision #1000                                  #10 seems to be enough. Increase to test precision, reduce to speed up.
    #kk          = np.linspace(kmin,kmax,num=nk)                   #linear grid on k
    logkmin     = np.log(kmin) ; logkmax   = np.log(kmax)
    logk        = np.linspace(logkmin,logkmax,num=nk)
    kk          = np.exp(logk)                                     #logarithmic grid on k
    Pk          = np.zeros(nk)
    for ik in range(nk):
        Pk[ik] = cosmo.pk(kk[ik],0.)                              #In Mpc^3
    Uarr        = np.zeros((nbins,nk))
    for ibin in range(nbins):
        for ik in range(nk):
            kr            = kk[ik]*comov_dist
            integrand     = dX_dz * kernels[ibin,:]**order * growth * np.sin(kr)/kr
            Uarr[ibin,ik] = integrate.simps(integrand,z_arr)
    
    # Compute Sij finally
    Cl_zero     = np.zeros((nbins,nbins))
    #For i<=j
    for ibin in range(nbins):
        U1 = Uarr[ibin,:]/Inorm[ibin]
        for jbin in range(ibin,nbins):
            U2 = Uarr[jbin,:]/Inorm[jbin]
            integrand = kk**2 * Pk * U1 * U2
            #Cl_zero[ibin,jbin] = 2/pi * integrate.simps(integrand,kk)     #linear integration
            Cl_zero[ibin,jbin] = (2/pi) * integrate.simps(integrand*kk,logk) #log integration
    #Fill by symmetry   
    for ibin in range(nbins):
        for jbin in range(nbins):
            Cl_zero[ibin,jbin] = Cl_zero[min(ibin,jbin),max(ibin,jbin)]
    Sij = Cl_zero / (4*pi)
    
    return Sij

##### Sij_alt_fullsky #####
def Sij_alt_fullsky(z_arr, kernels, order=2, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0):
    """Alternative routine to compute the Sij matrix in full sky.

    Parameters
    ----------
    z_arr : array_like
       Input array of redshifts of size nz.

    kernels : array_like
       2d array for the collection of kernels, shape (nbins, nz).

    order : int, default 2
        The passed kernels will be multiplied to that power, e.g. if two the kernels will be squared.
        You should normally set it to one and feed the product of kernels that you want.
        Examples: for cluster counts, set to one and feed the redshift selection functions. For cross-spectra Cl(i,j), set to one and feed the products Wi(z)*Wj(z) for all pairs of bins (i,j).
        The default is two for backward compatibility to times where the function was intended only for auto-spectra, so kernels needed to be squared internally. That default may change in future.

    cosmo_params : dict, default `default_cosmo_params`
       Dictionary of cosmology or cosmological parameters that can be accepted by ``classy``

    cosmo_Class : classy.Class object, default None
       classy.Class object containing precomputed cosmology.
       If you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    convention : int, default 0
       Integer to dictate the convention used in the definition of the kernels.
       0 = Lacasa & Grain 2019.
       1 = Cosmosic , Euclid Forecasts
       Defaults to 0.

    Returns
    -------

    array_like
       Sij matrix of shape (nbins, nbins).

    Notes
    -----
    Equation used

    .. math::
        S_{ij} = \int dX_1 \, dX_2 \\frac{W(i,z_1)^\mathrm{order}}{I_\mathrm{norm}(i)} \\frac{W(j,z_2)^\mathrm{order}}{I_\mathrm{norm}(j)} \ \sigma^2(z_1,z_2)

    with :math:`I_\mathrm{norm}(i) = \int dX \ W(i,z)^\mathrm{order}` and \
    :math:`\sigma^2(z_1,z_2) = \\frac{1}{2\pi^2} \int k^2 dk \ P(k|z_1,z_2) \ j_0(kr_1) j_0(kr_2)`.
    The latter is computed with the auxiliary function sigma2_fullsky.

    :math:`dX` depends on the convention used to define the observable's kernel:

    .. math::
            C_\mathrm{observable}(\ell,i,j) = \int dX \ W(i,z) \, W(j,z) \ P(k=(\ell+1/2)/r,z)

    0. :math:`dX = dV = \\frac{dV}{dz} dz = r^2(z) \\frac{dr}{dz} dz`. Used in Lacasa & Grain 2019. \n
    1. :math:`dX = d\chi/\chi^2 = \\frac{dr/dz}{r^2(z)} dz`. Used in cosmosis. \n
    The convention of the Euclid Forecasts is nearly the same as 1 \
    up to a factor :math:`c^2` (or :math:`\\frac{c^2}{H_0^2}` depending on the probe), \
    which is a constant so does not matter in the ratio here.
    """

    # Find number of bins
    nbins = kernels.shape[0]
    
    #Get cosmology, comoving distances etc from dedicated auxiliary routine
    cosmo, h, comov_dist, dcomov_dist, growth = get_cosmo(z_arr, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class)

    #Get element of z integration, depending on kernel convention
    dX_dz = get_dX_dz(comov_dist, dcomov_dist, convention=convention)   

    sigma2 = sigma2_fullsky(z_arr, cosmo_params=default_cosmo_params, cosmo_Class=None)         

    # Compute normalisations
    Inorm       = np.zeros(nbins)
    for i1 in range(nbins):
        integrand = dX_dz * kernels[i1,:]**order
        Inorm[i1] = integrate.simps(integrand,z_arr)
    
    
    # Compute Sij finally
    prefactor  = sigma2 * (dX_dz * dX_dz[:,None])
    Sij        = np.zeros((nbins,nbins))
    #For i<=j
    for i1 in range(nbins):
        for i2 in range(i1,nbins):
            integrand  = prefactor * (kernels[i1,:]**order * kernels[i2,:,None]**order)
            Sij[i1,i2] = integrate.simps(integrate.simps(integrand,z_arr),z_arr)/(Inorm[i1]*Inorm[i2])
    #Fill by symmetry   
    for i1 in range(nbins):
        for i2 in range(nbins):
            Sij[i1,i2] = Sij[min(i1,i2),max(i1,i2)]
    
    return Sij

##### sigma2_fullsky #####
def sigma2_fullsky(z_arr, cosmo_params=default_cosmo_params, cosmo_Class=None):
    """Routine to compute sigma^2(z1,z2) in full sky.

    Parameters
    ----------
    z_arr : array_like
       Input array of redshifts of size nz.

    cosmo_params : dict, default `default_cosmo_params`
       Dictionary of cosmology or cosmological parameters that can be accepted by ``classy``

    cosmo_Class : classy.Class object, default None
       classy.Class object containing precomputed cosmology.
       If you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    Returns
    -------

    array_like
       sigma2 array of shape (nz, nz).

    Notes
    -----
    Equation used

    :math:`\sigma^2(z_1,z_2) = \\frac{1}{2\pi^2} \int k^2 dk \ P(k|z_1,z_2) \ j_0(kr_1) j_0(kr_2)`.
    The latter can be rewritten as \
    :math:`\sigma^2(z_1,z_2) = \\frac{1}{2 \pi^2 r_1 r_2} G(z_1) G(z_2) \int dk \ P(k,z=0) \left[\cos\left(k(r_1-r_2)\\right)-\cos\left(k(r_1+r_2)\\right)\\right]/2` \
    and computed with an FFT.

    """

    # Find number of redshifts    
    nz    = z_arr.size
    
    #Get cosmology, comoving distances etc from dedicated auxiliary routine
    cosmo, h, comov_dist, dcomov_dist, growth = get_cosmo(z_arr, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class)
    
    keq         = 0.02/h                                    #Equality matter radiation in 1/Mpc (more or less)
    klogwidth   = 10                                        #Factor of width of the integration range.
    #10 seems ok ; going higher needs to increase nk_fft to reach convergence (fine cancellation issue noted in Lacasa & Grain)
    kmin        = min(keq,1./comov_dist.max())/klogwidth
    kmax        = max(keq,1./comov_dist.min())*klogwidth
    nk_fft      = 2**11                                     #seems to be enough. Increase to test precision, reduce to speed up.
    k_4fft      = np.linspace(kmin,kmax,nk_fft)             #linear grid on k, as we need to use an FFT
    Deltak      = kmax - kmin
    Dk          = Deltak/nk_fft
    Pk_4fft     = np.zeros(nk_fft)
    for ik in range(nk_fft):
        Pk_4fft[ik] = cosmo.pk(k_4fft[ik],0.)               #In Mpc^3
    dr_fft      = np.linspace(0,nk_fft//2,nk_fft//2+1)*2*pi/Deltak
    
    # Compute necessary FFTs and make interpolation functions
    fft0        = np.fft.rfft(Pk_4fft)*Dk
    dct0        = fft0.real ; dst0 = -fft0.imag
    Pk_dct      = interp1d(dr_fft,dct0,kind='cubic')
    
    # Compute sigma^2(z1,z2)
    sigma2_nog = np.zeros((nz,nz))
    #First without growth functions (i.e. with P(k,z=0)) and z1<=z2
    for iz in range(nz):
        r1 = comov_dist[iz]
        for jz in range(iz,nz):
            r2                = comov_dist[jz]
            rsum              = r1+r2
            rdiff             = abs(r1-r2)
            Icp0              = Pk_dct(rsum) ; Icm0 = Pk_dct(rdiff)
            sigma2_nog[iz,jz] = (Icm0-Icp0)/(4*pi**2 * r1 * r2)
    #Now fill by symmetry and put back growth functions
    sigma2      = np.zeros((nz,nz))
    for iz in range(nz):
        growth1 = growth[iz]
        for jz in range(nz):
            growth2       = growth[jz]
            sigma2[iz,jz] = sigma2_nog[min(iz,jz),max(iz,jz)]*growth1*growth2            
    
    return sigma2

####################################################################################################
#################################       PARTIAL SKY ROUTINES       #################################
####################################################################################################

##### Sij_psky #####
def Sij_psky(z_arr, kernels, order=2, clmask=None, mask=None, mask2=None, multimask=None, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0, precision=10, var_tol=0.05, verbose=False, debug=False):
    """Routine to compute the Sij matrix in partial sky.

    Parameters
    ----------
    z_arr : array_like
       Input array of redshifts of size nz.

    kernels : array_like
       2d array for the collection of kernels, shape (nbins, nz).

    order : int, default 2
        The passed kernels will be multiplied to that power, e.g. if two the kernels will be squared.
        You should normally set it to one and feed the product of kernels that you want.
        Examples: for cluster counts, set to one and feed the redshift selection functions. For cross-spectra Cl(i,j), set to one and feed the products Wi(z)*Wj(z) for all pairs of bins (i,j).
        The default is two for backward compatibility to times where the function was intended only for auto-spectra, so kernels needed to be squared internally. That default may change in future.

    cosmo_params : dict, default `default_cosmo_params`
       Dictionary of cosmology or cosmological parameters that can be accepted by ``classy``

    cosmo_Class : classy.Class object, default None
       classy.Class object containing precomputed cosmology.
       If you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    convention : int, default 0
       Integer to dictate the convention used in the definition of the kernels.
       0 = Lacasa & Grain 2019.
       1 = Cosmosic , Euclid Forecasts
       Defaults to 0.

    precision : int, default 10
        Integer which drives the number of Fourier wavenumbers in internal integrals.
        Nk = 2**precision.

    clmask : str or numpy.ndarray, default None
        Array or path to fits file containing the angular power spectrum of the mask.
        To be used when the observable(s) have a single mask.

    mask : str or numpy.ndarray, default None
        Array or path to fits file containing the mask in healpix form.
        PySSC will use healpy to compute the mask power spectrum.
        It is faster to directly give clmask if you have it (particularly when calling PySSC several times).
        To be used when the observable(s) have a single mask.

    mask2 : str or numpy.ndarray, default None
        Array or path to fits file containing a potential second mask in healpix form.
        In the case where you want the covariance between observables measured on different areas of the sky.
        PySSC will use healpy to compute the mask cross-spectrum.
        Again, it is faster to directly give clmask if you have it.
        If mask is set and mask2 is None, PySSC assumes that all observables share the same mask.

    var_tol : float, default 0.05
         Float that drives the target precision for the sum over angular multipoles.
         Default is 5%. Lowering it means increasing the number of multipoles thus increasing computational time.
         Only implemented if `sky`  is set to psky.

    verbose : bool, default False
        Verbosity of the routine.
        Defaults to False

    debug : bool, default False
        Debuging options to look for incoherence in the routine.
        Defaults to False.

    Returns
    -------
    array_like
        Sij matrix, shape (nbins,nbins).

    Notes
    -----
    Equation used:

    .. math::
        S_{ij} = \\frac{1}{(4\pi f_{\mathrm{sky}})^2} \sum_\ell (2\ell+1) \ C(\ell,\mathrm{mask}) \ C_S(\ell,i,j) 

    where \
    :math:`C_S(\ell,i,j) = \\frac{2}{\pi} \int k^2 dk \ P(k) \\frac{U(i;k,\ell)}{I_\mathrm{norm}(i)} \\frac{U(j;k,\ell)}{I_\mathrm{norm}(j)}` \n
    with :math:`I_\mathrm{norm}(i) = \int  dX \  W(i,z)^\mathrm{order}`
    and  :math:`U(i;k,\ell) = \int dX \  W(i,z)^\mathrm{order} \ G(z) \ j_\ell(k r)`
    """

    from scipy.special import spherical_jn as jn

    # Find number of redshifts and bins    
    nz    = z_arr.size
    nbins = kernels.shape[0]

    # compute Cl(mask) and fsky computed from user input (mask(s) or clmask)
    ell, cl_mask, fsky = get_mask_quantities(clmask=clmask,mask=mask,mask2=mask2,verbose=verbose)

    # Search of the best lmax for all later sums on ell
    lmax = find_lmax(ell,cl_mask,var_tol,debug=debug)
    if verbose:
        print('lmax = %i' %(lmax))

    # Cut ell and Cl_mask to lmax, for all later computations
    cl_mask = cl_mask[:(lmax+1)]
    nell    = lmax+1
    ell     = np.arange(nell) #0..lmax

    #Get cosmology, comoving distances etc from dedicated auxiliary routine
    cosmo, h, comov_dist, dcomov_dist, growth = get_cosmo(z_arr, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class)

    #Get element of z integration, depending on kernel convention
    dX_dz = get_dX_dz(comov_dist, dcomov_dist, convention=convention)
    
    # Compute normalisations
    Inorm       = np.zeros(nbins)
    for i1 in range(nbins):
        integrand = dX_dz * kernels[i1,:]**order 
        Inorm[i1] = integrate.simps(integrand,z_arr)


    #Full sky computation for debugging
    if debug:
        keq         = 0.02/h                                          #Equality matter radiation in 1/Mpc (more or less)
        klogwidth   = 10                                              #Factor of width of the integration range. 10 seems ok
        kmin        = min(keq,1./comov_dist.max())/klogwidth
        kmax        = max(keq,1./comov_dist.min())*klogwidth
        nk          = 2**precision                                    #10 seems to be enough. Increase to test precision, reduce to speed up.
        #kk          = np.linspace(kmin,kmax,num=nk)                   #linear grid on k
        logkmin     = np.log(kmin) ; logkmax   = np.log(kmax)
        logk        = np.linspace(logkmin,logkmax,num=nk)
        kk          = np.exp(logk)                                     #logarithmic grid on k
        Pk          = np.zeros(nk)
        for ik in range(nk):
            Pk[ik] = cosmo.pk(kk[ik],0.)                              #In Mpc^3
        Uarr        = np.zeros((nbins,nk))
        for ibin in range(nbins):
            for ik in range(nk):
                kr            = kk[ik]*comov_dist
                integrand     = dX_dz * kernels[ibin,:]**order * growth * np.sin(kr)/kr
                Uarr[ibin,ik] = integrate.simps(integrand,z_arr)
        Cl_zero     = np.zeros((nbins,nbins))
        #For i<=j
        for ibin in range(nbins):
            U1 = Uarr[ibin,:]/Inorm[ibin]
            for jbin in range(ibin,nbins):
                U2 = Uarr[jbin,:]/Inorm[jbin]
                integrand = kk**2 * Pk * U1 * U2
                #Cl_zero[ibin,jbin] = 2/pi * integrate.simps(integrand,kk)     #linear integration
                Cl_zero[ibin,jbin] = 2/pi * integrate.simps(integrand*kk,logk) #log integration
        #Fill by symmetry   
        for ibin in range(nbins):
            for jbin in range(nbins):
                Cl_zero[ibin,jbin] = Cl_zero[min(ibin,jbin),max(ibin,jbin)]

    
    # Compute U(i;k,ell) = int dX kernels(i,z)^order growth(z) j_ell(kk*r)
    keq         = 0.02/h                                          #Equality matter radiation in 1/Mpc (more or less)
    klogwidth   = 10                                              #Factor of width of the integration range. 10 seems ok
    kmin        = min(keq,1./comov_dist.max())/klogwidth
    kmax        = max(keq,1./comov_dist.min())*klogwidth
    # print(kmax)
    # kmax = 0.005
    nk          = 2**precision                                    #10 seems to be enough. Increase to test precision, reduce to speed up.
    #kk          = np.linspace(kmin,kmax,num=nk)                   #linear grid on k
    logkmin     = np.log(kmin) ; logkmax   = np.log(kmax)
    logk        = np.linspace(logkmin,logkmax,num=nk)
    kk          = np.exp(logk)                                     #logarithmic grid on k
    Pk          = np.zeros(nk)
    for ik in range(nk):
        Pk[ik] = cosmo.pk(kk[ik],0.)                              #In Mpc^3
    Uarr        = np.zeros((nbins,nk,nell))
    for ik in range(nk):
        kr            = kk[ik]*comov_dist
        for ll in ell:
            bessel_jl = jn(ll,kr)
            for ibin in range(nbins):
                integrand        = dX_dz * kernels[ibin,:]**order * growth * bessel_jl
                Uarr[ibin,ik,ll] = integrate.simps(integrand,z_arr)

    # Compute Cl(X,Y) = 2/pi \int kk^2 dkk P(kk) U(i;kk,ell)/I_\mathrm{norm}(i) U(j;kk,ell)/I_\mathrm{norm}(j)
    Cl_XY      = np.zeros((nbins,nbins,nell))
    for ll in ell:
        #For i<=j
        for ibin in range(nbins):
            U1 = Uarr[ibin,:,ll]/Inorm[ibin]
            for jbin in range(ibin,nbins):
                U2 = Uarr[jbin,:,ll]/Inorm[jbin]
                integrand = kk**2 * Pk * U1 * U2
                #Cl_XY[ibin,jbin,ll] = 2/pi * integrate.simps(integrand,kk)     #linear integration
                Cl_XY[ibin,jbin,ll] = 2/pi * integrate.simps(integrand*kk,logk) #log integration
        #Fill by symmetry   
        for ibin in range(nbins):
            for jbin in range(nbins):
                Cl_XY[ibin,jbin,ll] = Cl_XY[min(ibin,jbin),max(ibin,jbin),ll]

    if debug:
        truc = (Cl_zero-Cl_XY[:,:,0])/Cl_zero
        print('Debug: minmax of relative difference Cl_zero vs Cl_XY(ell=0)',truc.min(),truc.max())

    # Finally sum over ell to get Sij = sum_ell (2ell+1) C(ell,i,j) C(ell,mask) /(4pi*fsky)^2
    Sij     = np.zeros((nbins,nbins))
    #For i<=j
    for ibin in range(nbins):
        for jbin in range(ibin,nbins):
            Sij[ibin,jbin] = np.sum((2*ell+1)*cl_mask*Cl_XY[ibin,jbin,:])/(4*pi*fsky)**2
            if (debug and ibin==0 and jbin==0):
                print('Debug: fsky,ell,cl_mask',fsky,ell,cl_mask)
                print('Debug: Sij computation',Cl_XY[ibin,jbin,0]/(4*pi),np.sum((2*ell+1)*cl_mask*Cl_XY[ibin,jbin,:])/(4*pi*fsky)**2)

    #Fill by symmetry   
    for ibin in range(nbins):
        for jbin in range(nbins):
            Sij[ibin,jbin] = Sij[min(ibin,jbin),max(ibin,jbin)]
    
    return Sij

##### Sij_flatsky #####
def Sij_flatsky(z_arr, kernels, bin_centres, theta, cosmo_params=default_cosmo_params, cosmo_Class=None, verbose=False):
    """Routine to compute Sij according to the flat-sky approximation
    See Eq. 16 of arXiv:1612.05958

    Parameters
    ----------
    z_arr : array_like
        Redshift array of size nz (must be >0)
    kernels : array_like
        2d array for the collection of kernels, shape (nbins,nz)
    bin_centres : array_like
        Central values of redshift bins. Dimensions: (nbins,)
    theta : float
        Radius of the survey mask in deg.

    cosmo_params : dict, default `default_cosmo_params`
       Dictionary of cosmology or cosmological parameters that can be accepted by ``classy``

    cosmo_Class : classy.Class object, default None
       classy.Class object containing precomputed cosmology.
       If you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    verbose : bool, default False
        Verbosity of the routine.

    Returns
    -------
        Array_like
            Sij matrix of shape (nbins, nbins) in flat-sky approximation

    Notes
    -----
    The mask is assumed to be a circle with radius theta
    """

    from scipy.special import spherical_jn as jn
    from scipy.special import jv as Jn

    # Check inputs
    test_zw(z_arr, kernels)
    # Find number of redshifts and bins    
    nz    = np.size(z_arr)
    nbins = kernels.shape[0]

    theta = theta*np.pi/180. #converts in radians

    #Get cosmology, comoving distances etc from dedicated auxiliary routine
    cosmo, h, comov_dist, dcomov_dist, growth = get_cosmo(z_arr, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class)
    
    keq         = 0.02/h                                          #Equality matter radiation in 1/Mpc (more or less)
    klogwidth   = 10                                              #Factor of width of the integration range. 10 seems ok
    kmin        = min(keq,1./comov_dist.max())/klogwidth
    kmax        = max(keq,1./comov_dist.min())*klogwidth

    # kperp array
    kmin_perp = kmin
    kmax_perp = kmax
    nk_perp   = 50
    lnkperp_arr = np.linspace(np.log10(kmin_perp),np.log10(kmax_perp),nk_perp)
    kperp_arr = 10**(lnkperp_arr)
    # kpar array
    kmin_par = kmin
    kmax_par = kmax
    nk_par   = 50
    lnkpar_arr = np.linspace(np.log10(kmin_par),np.log10(kmax_par),nk_par)
    kpar_arr = 10**(lnkpar_arr)
    # k2 = kperp2 + kpar2
    k_arr = np.sqrt(kperp_arr[:,None]**2+kpar_arr[None,:]**2)
    # growth      = np.zeros(nz)                              #Growth factor
    # for iz in range(nbins):
    #     growth[iz] = cosmo.scale_independent_growth_factor(z_arr[iz])

    if verbose: print('Computing flat-sky approximation')
    Sij = np.zeros((nbins,nbins))
    for ibin in range(nbins):
        z1 = bin_centres[ibin]
        r1 = cosmo.z_of_r([z1])[0][0]
        dr1 = comov_dist[kernels[ibin,:]!=0].max()-comov_dist[kernels[ibin,:]!=0].min() #width of function function

        for jbin in range(nbins):
            z2 = bin_centres[jbin]
            r2 = cosmo.z_of_r([z2])[0][0]
            dr2 = comov_dist[kernels[jbin,:]!=0].max()-comov_dist[kernels[jbin,:]!=0].min() #width of kernel

            z12 = np.mean([bin_centres[ibin],bin_centres[jbin]])

            growth = cosmo.scale_independent_growth_factor(z12)
            Pk = np.array([cosmo.pk(k_arr[i,j],0.) for i in range(nk_perp) for j in range(nk_par)])
            Pk = Pk.reshape(k_arr.shape)

            integ_kperp = kperp_arr * 4. * (Jn(1,kperp_arr*theta*r1)/kperp_arr/theta/r1) * ( Jn(1,kperp_arr*theta*r2)/kperp_arr/theta/r2 )
            integ_kpar = jn(0,kpar_arr*dr1/2) * jn(0,kpar_arr*dr2/2)
            dSij = integ_kperp[:,None] * integ_kpar[None,:] * growth * Pk * np.cos(kpar_arr[None,:]*abs(r1-r2)) 

            Sij[ibin,jbin] = integrate.simps( integrate.simps(dSij,kpar_arr), kperp_arr)/ 2. / np.pi**2

    return Sij

    

####################################################################################################
#################################          ANGPOW ROUTINES         #################################
####################################################################################################

##### Sij_AngPow #####
def Sij_AngPow(z_arr, kernels, clmask=None, mask=None, mask2=None, cosmo_params=AngPow_cosmo_params, var_tol=0.05, machinefile=None, Nn=None, Np='default', AngPow_path=None, verbose=False, debug=False):
    """Routine to compute the Sij matrix in partial sky using AngPow.

    Parameters
    ----------
    z_arr : array_like
        Input array of redshifts of size nz.

    kernels : array_like
        2d array for the collection of kernels, shape (nbins, nz).

    clmask : str or numpy.ndarray, default None
        Array or path to fits file containing the angular power spectrum of the mask.

    mask : str or numpy.ndarray, default None
        Array or path to fits file containing the mask in healpix form.
        In that case PySSC will use healpy to compute the mask power spectrum.
        Thus it is faster to directly give clmask if you have it (or if you compute several Sij matrices for some reason).

    mask2 : str or numpy.ndarray, default None
        Array or path to fits file containing a potential second mask in healpix form.
        In the case where you want the covariance between observables measured on different areas of the sky.
        PySSC will use healpy to compute the mask power spectrum.
        Again, it is faster to directly give clmask if you have it.
        Only implemented if `sky` is set to psky.
        If mask is set and mask2 is None, PySSC assumes that all observables share the same mask.

    cosmo_params : dict, default `default_cosmo_params`
        Dictionary of cosmology or cosmological parameters that can be accepted by ``classy``

    cosmo_Class : classy.Class object, default None
        classy.Class object containing precomputed cosmology, if you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    var_tol : float, default 0.05
         Float that drives the target precision for the sum over angular multipoles.
         Default is 5%. Lowering it means increasing the number of multipoles thus increasing computational time.

    machinefile : str, default None
        Path to text file storing the IP addresses of all the nodes in the cluster network, and associated number of threads.
        machinefile is used for parallel computing in mpi.
        Default is None (running in local). If not None, the `Nn` variable must be set by the user.
        Only implemented if `method` is set to AngPow.

    Nn : int, default None
        Number of threads on which the user wants the AngPow routine to be run in mpi.
        This number should not exceed the maximum number of threads provided in machinefile.
        Default is None. If not None, the machinefile variable must be set by the user.
        Only implemented if `method` is set to AngPow.

    Np : str, default ``'default'``
     Equivalent to set the local environment variable OMP_NUM_THREADS to Np.
     It represents the number of processes AngPow is allowed to use on each machine.
     Default is 'default' : AngPow uses the pre-existing `OMP_NUM_THREADS` value.

    AngPow_path : str, default None
        path to the Angpow binary repertory (finishing by '/').
        Default is None and assumes that AngPow is installed at './AngPow/'.

    verbose : bool, default False
        Verbosity of the routine.
        Defaults to False

    debug : bool, default False
        Debuging options to look for incoherence in the routine.
        Defaults to False.

    Returns
    -------

    array_like
        Sij matrix of shape (nbins, nbins).

    """

    import time
    import os
    import shutil
    
    test_zw(z_arr, kernels)
    test_mask(mask, clmask, mask2=mask2)
    test_inputs_angpow(cosmo_params=cosmo_params)
    
    if AngPow_path is None:
        AngPow_path = os.getcwd() + '/AngPow/' #finishing with '/' 
    
    # compute Cl(mask) and fsky computed from user input (mask(s) or clmask)
    ell, cl_mask, fsky = get_mask_quantities(clmask=clmask,mask=mask,mask2=mask2,verbose=verbose)

    # Search of the best lmax for all later sums on ell
    lmax = find_lmax(ell,cl_mask,var_tol,debug=debug)
    if verbose:
        print('lmax = %i' %(lmax))
        
    #create data to pass to MPI AngPow routine
    rdm = np.random.random()
    os.makedirs(AngPow_path + 'temporary_%s'%rdm)
    np.savez(AngPow_path + 'temporary_%s/ini_files'%rdm, z_arr, kernels, lmax, fsky, cl_mask, AngPow_path)
    np.save(AngPow_path + 'temporary_%s/ini_files.npy'%rdm, cosmo_params) 
    
    present_rep = os.getcwd()
    #run MPI AngPow routine
    if Nn is not None:
        os.system('mpiexec -f %s -n %i python %s/AngPow_tools/PySSC_AP_MPI.py %s %s %s'%(machinefile,Nn,present_rep,rdm,AngPow_path,Np))
    else:
        os.system('python %s/AngPow_tools/PySSC_AP_MPI.py %s %s %s'%(present_rep,rdm,AngPow_path,Np))
    time.sleep(10)
    
    #load Sij result
    file = np.load(AngPow_path + 'temporary_%s/Sij.npz'%(rdm))
    Sij = file['arr_0']
    shutil.rmtree(AngPow_path + 'temporary_%s'%rdm, ignore_errors=True)
    
    return Sij

##### Sij_AngPow_fullsky #####
def Sij_AngPow_fullsky(z_arr,kernels,cosmo_params=AngPow_cosmo_params,machinefile=None,Nn=None,Np='default',AngPow_path=None,verbose=False,debug=False):
    """Routine to compute the Sij matrix in full sky using AngPow.

    Parameters
    ----------
    z_arr : array_like
        Input array of redshifts of size nz.

    kernels : array_like
        2d array for the collection of kernels, shape (nbins, nz).

    cosmo_params : dict, default `default_cosmo_params`
        Dictionary of cosmology or cosmological parameters that can be accepted by ``classy``

    cosmo_Class : classy.Class object, default None
        classy.Class object containing precomputed cosmology, if you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    var_tol : float, default 0.05
         Float that drives the target precision for the sum over angular multipoles.
         Default is 5%. Lowering it means increasing the number of multipoles thus increasing computational time.

    machinefile : str, default None
        Path to text file storing the IP addresses of all the nodes in the cluster network, and associated number of threads.
        machinefile is used for parallel computing in mpi.
        Default is None (running in local). If not None, the `Nn` variable must be set by the user.
        Only implemented if `method` is set to AngPow.

    Nn : int, default None
        Number of threads on which the user wants the AngPow routine to be run in mpi.
        This number should not exceed the maximum number of threads provided in machinefile.
        Default is None. If not None, the machinefile variable must be set by the user.
        Only implemented if `method` is set to AngPow.

    Np : str, default 'default'
     Equivalent to set the local environment variable OMP_NUM_THREADS to Np.
     It represents the number of processes AngPow is allowed to use on each machine.
     Default is 'default' : AngPow uses the pre-existing `OMP_NUM_THREADS` value.

    AngPow_path : str, default None
        path to the Angpow binary repertory (finishing by '/').
        Default is None and assumes that AngPow is installed at './AngPow/'.

    verbose : bool, default False
        Verbosity of the routine.
        Defaults to False

    debug : bool, default False
        Debuging options to look for incoherence in the routine.
        Defaults to False.

    Returns
    -------

    array_like
        Sij matrix of shape (nbins, nbins).

    """
    import healpy as hp
    import os
    import shutil
    
    test_zw(z_arr, kernels)

    if AngPow_path is not None:
        assert os.path.exists(AngPow_path + 'bin/angpow') , 'the angpow executable is not in the provided AngPow_path, please update the path or make sure the angpow compilation has been correctly done'
    else :
        assert os.path.exists('./AngPow/bin/angpow') , 'the angpow executable is not in ./AngPow/bin/angpow, please make sure the angpow compilation has been correctly done (cd AngPow ; make) or give another angpow path in the AngPow_path option'
    if AngPow_path is None:
        AngPow_path = os.getcwd() + '/AngPow/' #finishing with '/' 
    
    # Define the angular power spectrum of a mask that is 1 over the full sky
    Cl_fullsky=np.zeros(10) ; Cl_fullsky[0]=4*pi

    # Write the Cl in a temporary folder
    rdm = np.random.random()
    os.makedirs(AngPow_path + 'temporary_%s'%rdm)
    tmp_file = AngPow_path + 'temporary_%s/Cl_M_PySSC.fits'%(rdm)
    hp.fitsfunc.write_cl(tmp_file,Cl_fullsky)

    # Call Sij_AngPow with that Cl file
    Sij = Sij_AngPow(z_arr,kernels,clmask=tmp_file,cosmo_params=cosmo_params,machinefile=machinefile,Nn=Nn,Np='default',AngPow_path=AngPow_path,verbose=verbose,debug=debug)

    # Remove the temporary folder
    shutil.rmtree(AngPow_path + 'temporary_%s'%rdm, ignore_errors=True)

    # Return the result
    return Sij

####################################################################################################
#################################        AUXILIARY ROUTINES        #################################
####################################################################################################

#################### TEST INPUTS ####################

##### test_zw #####
def test_z(z_arr):
    """
    Assert redshift array has the good type, shape and values.
    """
    assert isinstance(z_arr,np.ndarray), 'z_arr must be a numpy array.'
    assert z_arr.ndim==1, 'z_arr must be a 1-dimensional array.'    
    assert z_arr.min()>0, 'z_arr must have values > 0.'

def test_w(kernels, nz):
    """
    Assert kernels array has the good type and shape.
    """
    assert isinstance(kernels,np.ndarray), 'kernels must be a numpy array.'
    assert kernels.ndim==2, 'kernels must be a 2-dimensional array.'    
    assert kernels.shape[1] == nz, 'kernels must have shape (nbins,nz).'

def test_zw(z_arr, kernels):
    """
    Assert redshift and kernels arrays have the good type, shape and values.
    """
    test_z(z_arr)
    nz = len(z_arr)
    test_w(kernels, nz)

##### test_mask #####
def test_mask(mask, clmask, mask2=None):
    """
    Assert that either the mask or its Cl has been provided. If two masks are provided, check that they have the same resolution.
    """
    assert (mask is not None) or (clmask is not None), 'You need to provide either the mask or its angular power spectrum Cl.'
    if mask is not None:
        assert isinstance(mask,str) or isinstance(mask,np.ndarray), 'mask needs to be either a filename or a numpy array'
    if clmask is not None:
        assert isinstance(clmask,str) or isinstance(clmask,np.ndarray), 'Clmask needs to be either a filename or a numpy array'
    if mask2 is not None:
        import healpy
        if isinstance(mask,str):
                map_mask = healpy.read_map(mask, dtype=np.float64)
        elif isinstance(mask,np.ndarray):
                map_mask = mask
        nside     = healpy.pixelfunc.get_nside(map_mask)
        if isinstance(mask2,str):
                map_mask2 = healpy.read_map(mask2, dtype=np.float64)
        elif isinstance(mask2,np.ndarray):
                map_mask2 = mask2
        nside2    = healpy.pixelfunc.get_nside(map_mask2)
        assert nside==nside2, 'The resolutions (nside) of both masks need to be the same.'

##### test_multimask #####
def test_multimask(multimask):
    """
    Assert that multimask has the good structure: a list of dictionnaries, all of the form {'mask':'mask.fits', 'kernels':kernels_array}.
    """
    assert isinstance(multimask,list), 'multimask needs to be a list (of dictionnaries).'
    for dico in multimask:
        assert isinstance(dico,dict), 'The elements of multimask must be dictionnaries.'
        assert 'mask' in dico.keys(), 'The dictionnaries must contain the key "mask".'
        assert 'kernels' in dico.keys(), 'The dictionnaries must contain the key "kernels".'
        assert isinstance(dico.mask,str), 'The key "mask" must contain a string (pointing to a healpix fits file).'
        assert isinstance(dico.kernels,np.ndarray), 'The key "kernels" must contain a numpy array.'

##### test_inputs_angpow #####
def test_inputs_angpow(cosmo_params=AngPow_cosmo_params, cosmo_Class=None, order=2, convention=0, machinefile=None, Nn=None, Np='default', AngPow_path=None):
    """
    Asserts that the various inputs to the AngPow routine are correct
    """
    # Cosmological inputs
    if cosmo_params!=AngPow_cosmo_params:
        assert 'P_k_max_h/Mpc' in cosmo_params, 'If you supply your own cosmological parameters, you must give a high enough P_k_max_h/Mpc.'
    if cosmo_Class is not None:
            raise Exception('Precomputed cosmology cannot (yet) be passed for the AngPow method. Provide cosmological parameters instead to be run by Class.')
    if convention!=0:
        raise Exception('Only the default kernel convention is supported for the AngPow method.')
    if order!=2:
        raise Exception('Only order=2 is supported for the AngPow method.')
    # Parallelisation inputs
    import os
    if machinefile is not None:
        assert Nn is not None, 'a machinefile has been provided, but not the number of node Nn.  Nn should not exceed the number of line in the machinefile'
        assert os.path.exists(machinefile) , 'the proposed machinefile has not been found, please update the path'
    if Nn is not None:
        assert int(Nn) == Nn , 'the number of nodes Nn must be integer'
        assert machinefile is not None, 'a number of nodes Nn has been provided, but not the associated machinefile. Please create an adapted machinefile (see machinefile_example)'
    if AngPow_path is not None:
        assert os.path.exists(AngPow_path + 'bin/angpow') , 'the angpow executable is not in the provided AngPow_path, please update the path or make sure the angpow compilation has been correctly done'
    else:
        assert os.path.exists('./AngPow/bin/angpow') , 'the angpow executable is not in ./AngPow/bin/angpow, please make sure the angpow compilation has been correctly done or give another angpow path in the AngPow_path option'
    if Np != 'default':
        assert int(Np) == Np , 'the number of process per node Np must be integer'

#################### COMPUTE STUFF ####################

##### get_cosmo #####
def get_cosmo(z_arr, cosmo_params=default_cosmo_params, cosmo_Class=None):
    """Auxiliary routine to run CLASS if needed, then compute arrays of comoving distance, volume etc necessary for Sij routines.

    Parameters
    ----------
    z_arr : array_like
        Input array of redshifts of size nz.

    cosmo_params : dict, default `default_cosmo_params`
        Dictionary of cosmology or cosmological parameters that can be accepted by ``classy``

    cosmo_Class : classy.Class object, default None
        classy.Class object containing precomputed cosmology, if you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    Returns
    -------
    tuple
        cosmo, h, comov_dist, dcomov_dist, growth
    """

    nz = z_arr.size

    # If the cosmology is not provided (in the same form as CLASS), run CLASS
    if cosmo_Class is None:
        from classy import Class
        cosmo = Class()
        dico_for_CLASS = cosmo_params
        dico_for_CLASS['output'] = 'mPk'
        cosmo.set(dico_for_CLASS)
        cosmo.compute()
    else:
        cosmo = cosmo_Class
    
    h = cosmo.h() #for  conversions Mpc/h <-> Mpc
    
    # Define arrays of r(z), k, P(k)...
    zofr        = cosmo.z_of_r(z_arr)
    comov_dist  = zofr[0]                                   #Comoving distance r(z) in Mpc
    dcomov_dist = 1/zofr[1]                                 #Derivative dr/dz in Mpc
    growth      = np.zeros(nz)                              #Growth factor
    for iz in range(nz):
        growth[iz] = cosmo.scale_independent_growth_factor(z_arr[iz])

    return cosmo, h, comov_dist, dcomov_dist, growth

##### get_dX_dz #####
def get_dX_dz(comov_dist, dcomov_dist, convention=0):
    """Auxiliary routine to compute the element of integration for z integrals in Sij routines.

    Parameters
    ----------
    comov_dist : numpy.ndarray
        Input 1D array of comoving distance r(z).

    dcomov_dist : numpy.ndarray
        Input 1D array of derivative of comoving distance dr/dz. Same size as comov_dist.

    convention : int, default 0
        Integer to dictate the convention used in the definition of the kernels.
        0 = Lacasa & Grain 2019.
        1 = Cosmosis, Euclid Forecasts (Blanchard et al 2020).

    Returns
    -------
    float
        dX_dz
    """
    if convention==0:    # Default: comoving volume dV = r^2 dr. Convention of Lacasa & Grain 2019
        dX_dz = comov_dist**2 * dcomov_dist
    elif convention==1:  # Convention of Cosmosis, Euclid Forecasts (Blanchard et al 2020)
        dX_dz = dcomov_dist / comov_dist**2
    else:
        raise ValueError('convention must be either 0 or 1')
    
    return dX_dz

def get_mask_quantities(clmask=None,mask=None,mask2=None,verbose=False):
    """Auxiliary routine to compute different mask quantities (ell,Cl,fsky) for partial sky Sij routines.

    Parameters
    ----------
    clmask : str or numpy.ndarray, default None
        Array or path to fits file containing the angular power spectrum of the mask.
        To be used when the observable(s) have a single mask.

    mask : str or numpy.ndarray, default None
        Array or path to fits file containing the mask in healpix form.
        PySSC will use healpy to compute the mask power spectrum.
        It is faster to directly give clmask if you have it (particularly when calling PySSC several times).
        To be used when the observable(s) have a single mask.

    mask2 : str or numpy.ndarray, default None
        Array or path to fits file containing a potential second mask in healpix form.
        In the case where you want the covariance between observables measured on different areas of the sky.
        PySSC will use healpy to compute the mask cross-spectrum.
        Again, it is faster to directly give clmask if you have it.
        If mask is set and mask2 is None, PySSC assumes that all observables share the same mask.

    verbose : bool, default False
        Verbosity of the routine.

    Returns
    -------
    tuple
        ell, cl_mask, fsky
    """
    import healpy
    if mask is None: # User gives Cl(mask)
        if verbose:
            print('Using given Cls')
        if isinstance(clmask,str):
            cl_mask = healpy.fitsfunc.read_cl(str(clmask))
        elif isinstance(clmask,np.ndarray):
            cl_mask = clmask
        ell = np.arange(len(cl_mask))
        lmaxofcl = ell.max()
    else : # User gives mask as a map
        if verbose:
            print('Using given mask map')
        if isinstance(mask,str):
            map_mask = healpy.read_map(mask, dtype=np.float64)
        elif isinstance(mask,np.ndarray):
            map_mask = mask
        nside    = healpy.pixelfunc.get_nside(map_mask)
        lmaxofcl = 2*nside
        if mask2 is None:
            map_mask2 = copy.copy(map_mask)
        else:
            if isinstance(mask2,str):
                map_mask2 = healpy.read_map(mask2, dtype=np.float64)
            elif isinstance(mask2,np.ndarray):
                map_mask2 = mask2
        cl_mask  = healpy.anafast(map_mask, map2=map_mask2, lmax=lmaxofcl)
        ell      = np.arange(lmaxofcl+1)

    # Compute fsky from the mask
    fsky = np.sqrt(cl_mask[0]/(4*np.pi))
    if verbose:
        print('f_sky = %.4f' %(fsky))

    return ell, cl_mask, fsky

##### find_lmax #####
def find_lmax(ell, cl_mask, var_tol, debug=False):
    """Auxiliary routine to search the best lmax for all later sums on multipoles.

    Computes the smallest lmax so that we reach convergence of the variance
    ..math ::
        var = \sum_\ell  \\frac{(2\ell + 1)}{4\pi} C_\ell^{mask}

    Parameters
    ----------
    ell : array_like
        Full vector of multipoles. As large as possible of shape (nell,)
    cl_mask : array_like
        power spectrum of the mask at the supplied multipoles of shape (nell,).

    Returns
    -------
    float
        lmax
    """

    assert ell.ndim==1, 'ell must be a 1-dimensional array'
    assert cl_mask.ndim==1, 'cl_mask must be a 1-dimensional array'
    assert len(ell)==len(cl_mask), 'ell and cl_mask must have the same size'
    lmaxofcl      = ell.max()
    summand       = (2*ell+1)/(4*pi)*cl_mask
    var_target    = np.sum(summand)
    #Initialisation
    lmax          = 0
    var_est       = np.sum(summand[:(lmax+1)])
    while (abs(var_est - var_target)/var_target > var_tol and lmax < lmaxofcl):
        lmax      = lmax +1
        var_est   = np.sum(summand[:(lmax+1)])
        if debug:
            print('In lmax search',lmax,abs(var_est - var_target)/var_target,var_target,var_est)
    lmax = min(lmax,lmaxofcl) #make sure we didnt overshoot at the last iteration
    return lmax

####################################################################################################
#################################           OLD ROUTINES           #################################
####################################################################################################   

##### turboSij #####
def turboSij(zstakes=default_zstakes, cosmo_params=default_cosmo_params, cosmo_Class=None):
    """Routine to compute the Sij matrix with top-hat disjoint redshift kernels.
    example : galaxy clustering with perfect/spectroscopic redshift determinations so that bins are sharp.

    Parameters
    ----------
    zstakes : array_like, default `default_zstakes`
        Stakes of the redshift bins (nz,)
    cosmo_params : dict, default `default_cosmo_params`
    cosmo_Class : classy.Class object , default None

    Returns
    -------
    array_like
        Sij matrix (nz,nz)
    """
    
    from classy import Class

    # If the cosmology is not provided (in the same form as CLASS), run CLASS
    if cosmo_Class is None:
        cosmo = Class()
        dico_for_CLASS = cosmo_params
        dico_for_CLASS['output'] = 'mPk'
        cosmo.set(dico_for_CLASS)
        cosmo.compute()
    else:
        cosmo = cosmo_Class

    h = cosmo.h() #for  conversions Mpc/h <-> Mpc

    # Define arrays of z, r(z), k, P(k)...
    nzbins     = len(zstakes)-1
    nz_perbin  = 10
    z_arr      = np.zeros((nz_perbin,nzbins))
    comov_dist = np.zeros((nz_perbin,nzbins))
    for j in range(nzbins):
        z_arr[:,j] = np.linspace(zstakes[j],zstakes[j+1],nz_perbin)
        comov_dist[:,j] = (cosmo.z_of_r(z_arr[:,j]))[0] #In Mpc
    keq       = 0.02/h                        #Equality matter radiation in 1/Mpc (more or less)
    klogwidth = 10                            #Factor of width of the integration range. 10 seems ok ; going higher needs to increase nk_fft to reach convergence (fine cancellation issue noted in Lacasa & Grain)
    kmin      = min(keq,1./comov_dist.max())/klogwidth
    kmax = max(keq,1./comov_dist.min())*klogwidth
    nk_fft    = 2**11                         #seems to be enough. Increase to test precision, reduce to speed up.
    k_4fft    = np.linspace(kmin,kmax,nk_fft) #linear grid on k, as we need to use an FFT
    Deltak    = kmax - kmin
    Dk        = Deltak/nk_fft
    Pk_4fft   = np.zeros(nk_fft)
    for ik in range(nk_fft):
        Pk_4fft[ik] = cosmo.pk(k_4fft[ik],0.)  #In Mpc^3
    dr_fft    = np.linspace(0,nk_fft//2,nk_fft//2+1)*2*pi/Deltak 

    # Compute necessary FFTs and make interpolation functions
    fft2      = np.fft.rfft(Pk_4fft/k_4fft**2)*Dk
    dct2      = fft2.real ; dst2 = -fft2.imag
    km2Pk_dct = interp1d(dr_fft,dct2,kind='cubic')
    fft3      = np.fft.rfft(Pk_4fft/k_4fft**3)*Dk
    dct3      = fft3.real ; dst3 = -fft3.imag
    km3Pk_dst = interp1d(dr_fft,dst3,kind='cubic')
    fft4      = np.fft.rfft(Pk_4fft/k_4fft**4)*Dk
    dct4      = fft4.real ; dst4 = -fft4.imag
    km4Pk_dct = interp1d(dr_fft,dct4,kind='cubic')

    # Compute Sij finally
    Sij = np.zeros((nzbins,nzbins))
    for j1 in range(nzbins):
        rmin1   = (comov_dist[:,j1]).min()
        rmax1   = (comov_dist[:,j1]).max()
        zmean1  = ((z_arr[:,j1]).min()+(z_arr[:,j1]).max())/2
        growth1 = cosmo.scale_independent_growth_factor(zmean1)
        pref1   = 3. * growth1 / (rmax1**3-rmin1**3)
        for j2 in range(nzbins):
            rmin2      = (comov_dist[:,j2]).min()
            rmax2      = (comov_dist[:,j2]).max()
            zmean2     = ((z_arr[:,j2]).min()+(z_arr[:,j2]).max())/2
            growth2    = cosmo.scale_independent_growth_factor(zmean2)
            pref2      = 3. * growth2 / (rmax2**3-rmin2**3)
            #p1p2: rmax1 & rmax2
            rsum       = rmax1+rmax2
            rdiff      = abs(rmax1-rmax2)
            rprod      = rmax1*rmax2
            Icp2       = km2Pk_dct(rsum) ; Icm2 = km2Pk_dct(rdiff)
            Isp3       = km3Pk_dst(rsum) ; Ism3 = km3Pk_dst(rdiff)
            Icp4       = km4Pk_dct(rsum) ; Icm4 = km4Pk_dct(rdiff)
            Fp1p2      = -Icp4 + Icm4 - rsum*Isp3 + rdiff*Ism3 + rprod*(Icp2+Icm2)
            #p1m2: rmax1 & rmin2
            rsum       = rmax1+rmin2
            rdiff      = abs(rmax1-rmin2)
            rprod      = rmax1*rmin2
            Icp2       = km2Pk_dct(rsum) ; Icm2 = km2Pk_dct(rdiff)
            Isp3       = km3Pk_dst(rsum) ; Ism3 = km3Pk_dst(rdiff)
            Icp4       = km4Pk_dct(rsum) ; Icm4 = km4Pk_dct(rdiff)
            Fp1m2      = -Icp4 + Icm4 - rsum*Isp3 + rdiff*Ism3 + rprod*(Icp2+Icm2)
            #m1p2: rmin1 & rmax2
            rsum       = rmin1+rmax2
            rdiff      = abs(rmin1-rmax2)
            rprod      = rmin1*rmax2
            Icp2       = km2Pk_dct(rsum) ; Icm2 = km2Pk_dct(rdiff)
            Isp3       = km3Pk_dst(rsum) ; Ism3 = km3Pk_dst(rdiff)
            Icp4       = km4Pk_dct(rsum) ; Icm4 = km4Pk_dct(rdiff)
            Fm1p2      = -Icp4 + Icm4 - rsum*Isp3 + rdiff*Ism3 + rprod*(Icp2+Icm2) 
            #m1m2: rmin1 & rmin2
            rsum       = rmin1+rmin2
            rdiff      = abs(rmin1-rmin2)
            rprod      = rmin1*rmin2
            Icp2       = km2Pk_dct(rsum) ; Icm2 = km2Pk_dct(rdiff)
            Isp3       = km3Pk_dst(rsum) ; Ism3 = km3Pk_dst(rdiff)
            Icp4       = km4Pk_dct(rsum) ; Icm4 = km4Pk_dct(rdiff)
            Fm1m2      = -Icp4 + Icm4 - rsum*Isp3 + rdiff*Ism3 + rprod*(Icp2+Icm2)
            #now group everything
            Fsum       = Fp1p2 - Fp1m2 - Fm1p2 + Fm1m2
            Sij[j1,j2] = pref1 * pref2 * Fsum/(4*pi**2)

    return Sij


####################################################################################################
#################################           DEPRECATED ROUTINES           ##########################
####################################################################################################   

##### Sijkl wrapper #####
def Sijkl(z_arr, kernels, sky='full', cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0, precision=10,
          clmask=None, mask=None, mask2=None, var_tol=0.05, tol=1e-3, verbose=False, debug=False):
    """[DEPRECATED] Wrapper routine to compute the Sijkl matrix.
    It calls different routines depending on the inputs : full sky or partial sky methods.

    Parameters
    ----------
    z_arr : array_like
        Input array of redshifts of size nz.

    kernels : array_like
        2d array for the collection of kernels, shape (nbins, nz).

    sky : str, default ``'full'``
        Choice of survey geometry, given as a case-insensitive string.
        Valid choices: \n
        (i) ``'full'``/``'fullsky'``/``'full sky'``/``'full-sky'``,
        (ii) ``'psky'``/``'partial sky'``/``'partial-sky'``/``'partial'``/``'masked'``.

    cosmo_params : dict, default `default_cosmo_params`
        Dictionary of cosmology or cosmological parameters that can be accepted by `classy``

    cosmo_Class : classy.Class object, default None
        classy.Class object containing precomputed cosmology, if you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    convention : int, default 0
        Integer to dictate the convention used in the definition of the kernels.
        0 = Lacasa & Grain 2019.
        1 = Cosmosic , Euclid Forecasts.
        Defaults to 0.

    precision : int, default 10
        Integer which drives the number of Fourier wavenumbers in internal integrals such as : Nk = 2**precision.

    clmask : str or numpy.ndarray, default None
        Array or path to fits file containing the angular power spectrum of the mask.
        Only implemented if `sky` is set to psky.

    mask : str or numpy.ndarray, default None
        Array or path to fits file containing the mask in healpix form.
        In that case PySSC will use healpy to compute the mask power spectrum.
        Thus it is faster to directly give clmask if you have it (or if you compute several Sij matrices for some reason).
        Only implemented if `sky` is set to 'psky'.

    mask2 : str or numpy.ndarray, default None
        Array or path to fits file containing a potential second mask in healpix form.
        In the case where you want the covariance between observables measured on different areas of the sky.
        PySSC will use healpy to compute the mask power spectrum.
        Again, it is faster to directly give clmask if you have it.
        Only implemented if `sky` is set to psky.
        If mask is set and mask2 is None, PySSC assumes that all observables share the same mask.

    var_tol : float, default 0.05
         Float that drives the target precision for the sum over angular multipoles.
         Default is 5%. Lowering it means increasing the number of multipoles thus increasing computational time.
         Only implemented if `sky`  is set to psky.

    tol : float, default 1e-3
        Tolerance value telling PySSC to cut off (i.e. set Sijkl=0) the matrix elements where there is too small
        overlap between the kernels rendering the computation unreliable.

    verbose : bool, default False
        Verbosity of the routine.
        Defaults to False

    debug : bool, default False
        Debuging options to look for incoherence in the routine.
        Defaults to False.

    Returns
    -------

    Array_like
        Sijkl matrix of shape (nbins,nbins,nbins,nbins).

    """

    #Raise deprecation warning
    warnings.warn("The Sijkl functions are now deprecated. Please move to using Sij by feeding the kernel products, see the documentation at https://pyssc.readthedocs.io/en/latest/notebooks/Main-Examples.html#General-case-with-cross-spectra", DeprecationWarning, stacklevel=2)

    test_zw(z_arr,kernels)
    
    if sky.casefold() in ['full','fullsky','full sky','full-sky']:
        Sijkl=Sijkl_fullsky(z_arr, kernels, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, convention=convention, precision=precision, tol=tol)
    elif sky.casefold() in ['psky','partial sky','partial-sky','partial','masked']:
        test_mask(mask, clmask, mask2=mask2)
        Sijkl=Sijkl_psky(z_arr, kernels, clmask=clmask, mask=mask, mask2=mask2, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, convention=convention, precision=precision, tol=tol, var_tol=var_tol, verbose=verbose, debug=debug)
    else:
        raise Exception('Invalid string given for sky geometry parameter. Must be either of : full, fullsky, full sky, full-sky.')
    return Sijkl

##### Sijkl_fullsky #####
def Sijkl_fullsky(z_arr, kernels, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0, precision=10, tol=1e-3):
    """[DEPRECATED] Routine to compute the Sijkl matrix in full sky.

    Parameters
    ----------
    z_arr : array_like
       Input array of redshifts of size nz.

    kernels : array_like
       2d array for the collection of kernels, shape (nbins, nz).

    cosmo_params : dict, default `default_cosmo_params`
       Dictionary of cosmology or cosmological parameters that can be accepted by ``classy``

    cosmo_Class : classy.Class object, default None
       classy.Class object containing precomputed cosmology.
       If you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    convention : int, default 0
       Integer to dictate the convention used in the definition of the kernels.
       0 = Lacasa & Grain 2019.
       1 = Cosmosic , Euclid Forecasts
       Defaults to 0.

    precision : int, default 10
        Integer which drives the number of Fourier wavenumbers in internal integrals.
        Nk = 2**precision.

    tol : float, default 1e-3
        Tolerance value telling PySSC to cut off (i.e. set Sijkl=0) the matrix elements where there is too small
        overlap between the kernels rendering the computation unreliable.

    Returns
    -------
    array_like
        Sijkl matrix, shape (nbins,nbins,nbins,nbins).

    Notes
    -----
    Equation used (using indices :math:`(\\alpha,\\beta,\gamma,\delta)` instead of :math:`(i,j,k,l)` to avoid confusion with the Fourier wavevector and multipole):

    .. math::
        S_{\\alpha \\beta \gamma \delta}=\\frac{1}{2\pi^2} \int k^2 dk \ P(k) \\frac{U(\\alpha,\\beta ; k,\ell=0)}{I_\mathrm{norm}(\\alpha,\\beta)}
                  \\frac{U(\gamma,\delta;k,\ell=0)}{I_\mathrm{norm}(\gamma,\delta)}

    with: \n
    :math:`I_\mathrm{norm}(\\alpha,\\beta) = \int dX \ W(\\alpha,z) \ W(\\beta,z)`  and 
    :math:`U(\\alpha,\\beta;k,\ell) = \int dX \ W(\\alpha,z) \ W(\\beta,z) \ G(z) \ j_\ell(k r)`.
    """

    #Raise deprecation warning
    warnings.warn("The Sijkl functions are now deprecated. Please move to using Sij by feeding the kernel products, see the documentation at https://pyssc.readthedocs.io/en/latest/notebooks/Main-Examples.html#General-case-with-cross-spectra", DeprecationWarning, stacklevel=2)

    # Find number of redshifts and bins    
    nz    = z_arr.size
    nbins = kernels.shape[0]
    
    #Get cosmology, comoving distances etc from dedicated auxiliary routine
    cosmo, h, comov_dist, dcomov_dist, growth = get_cosmo(z_arr, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class)

    #Get element of z integration, depending on kernel convention
    dX_dz = get_dX_dz(comov_dist, dcomov_dist, convention=convention)

    #Index pairs of bins
    npairs      = (nbins*(nbins+1))//2
    pairs       = np.zeros((2,npairs),dtype=int)
    count       = 0
    for ibin in range(nbins):
        for jbin in range(ibin,nbins):
            pairs[0,count] = ibin
            pairs[1,count] = jbin
            count +=1
        
    # Compute normalisations
    Inorm       = np.zeros(npairs)
    Inorm2D     = np.zeros((nbins,nbins))
    for ipair in range(npairs):
        ibin               = pairs[0,ipair]
        jbin               = pairs[1,ipair]
        integrand          = dX_dz * kernels[ibin,:]* kernels[jbin,:]
        integral           = integrate.simps(integrand,z_arr)
        Inorm[ipair]       = integral
        Inorm2D[ibin,jbin] = integral
        Inorm2D[jbin,ibin] = integral
    #Flag pairs with too small overlap as unreliable
    #Note: this will also speed up later computations
    #Default tolerance : tol=1e-3
    flag        = np.zeros(npairs,dtype=int)
    for ipair in range(npairs):
        ibin               = pairs[0,ipair]
        jbin               = pairs[1,ipair]
        ratio              = abs(Inorm2D[ibin,jbin])/np.sqrt(abs(Inorm2D[ibin,ibin]*Inorm2D[jbin,jbin]))
        if ratio<tol:
            flag[ipair]=1
    
    # Compute U(i,j;kk)
    keq         = 0.02/h                                          #Equality matter radiation in 1/Mpc (more or less)
    klogwidth   = 10                                              #Factor of width of the integration range. 10 seems ok
    kmin        = min(keq,1./comov_dist.max())/klogwidth
    kmax        = max(keq,1./comov_dist.min())*klogwidth
    nk          = 2**precision                                    #10 seems to be enough. Increase to test precision, reduce to speed up.
    #kk          = np.linspace(kmin,kmax,num=nk)                   #linear grid on k
    logkmin     = np.log(kmin) ; logkmax   = np.log(kmax)
    logk        = np.linspace(logkmin,logkmax,num=nk)
    kk          = np.exp(logk)                                     #logarithmic grid on k    
    Pk          = np.zeros(nk)
    for ik in range(nk):
        Pk[ik]  = cosmo.pk(kk[ik],0.)                              #In Mpc^3
    Uarr        = np.zeros((npairs,nk))
    for ipair in range(npairs):
        if flag[ipair]==0:
            ibin = pairs[0,ipair]
            jbin = pairs[1,ipair]
            for ik in range(nk):
                kr             = kk[ik]*comov_dist
                integrand      = dX_dz * kernels[ibin,:] * kernels[jbin,:] * growth * np.sin(kr)/kr
                Uarr[ipair,ik] = integrate.simps(integrand,z_arr)
            
    # Compute Sijkl finally
    Cl_zero     = np.zeros((nbins,nbins,nbins,nbins))
    #For ipair<=jpair
    for ipair in range(npairs):
        if flag[ipair]==0:
            U1 = Uarr[ipair,:]/Inorm[ipair]
            ibin = pairs[0,ipair]
            jbin = pairs[1,ipair]
            for jpair in range(ipair,npairs):
                if flag[jpair]==0:
                    U2 = Uarr[jpair,:]/Inorm[jpair]
                    kbin = pairs[0,jpair]
                    lbin = pairs[1,jpair]
                    integrand = kk**2 * Pk * U1 * U2
                    #integral = 2/(i * integrate.simps(integrand,kk)     #linear integration
                    integral = 2/pi * integrate.simps(integrand*kk,logk) #log integration
                    #Run through all valid symmetries to fill the 4D array
                    #Symmetries: i<->j, k<->l, (i,j)<->(k,l)
                    Cl_zero[ibin,jbin,kbin,lbin] = integral
                    Cl_zero[ibin,jbin,lbin,kbin] = integral
                    Cl_zero[jbin,ibin,kbin,lbin] = integral
                    Cl_zero[jbin,ibin,lbin,kbin] = integral
                    Cl_zero[kbin,lbin,ibin,jbin] = integral
                    Cl_zero[kbin,lbin,jbin,ibin] = integral
                    Cl_zero[lbin,kbin,ibin,jbin] = integral
                    Cl_zero[lbin,kbin,jbin,ibin] = integral
    Sijkl = Cl_zero / (4*pi)       
    
    return Sijkl

##### Sijkl_psky #####
def Sijkl_psky(z_arr, kernels, clmask=None, mask=None, mask2=None, cosmo_params=default_cosmo_params, cosmo_Class=None,
               convention=0, precision=10, var_tol=0.05, tol=1e-3, verbose=False, debug=False):
    """[DEPRECATED] Routine to compute the Sijkl matrix in partial sky.

    Parameters
    ----------
    z_arr : array_like
        Input array of redshifts of size nz.

    kernels : array_like
        2d array for the collection of kernels, shape (nbins, nz).

    clmask : str or numpy.ndarray, default None
        Array or path to fits file containing the angular power spectrum of the mask.
        Only implemented if `sky` is set to psky.

    mask : str or numpy.ndarray, default None
        Array or path to fits file containing the mask in healpix form.
        In that case PySSC will use healpy to compute the mask power spectrum.
        Thus it is faster to directly give clmask if you have it (or if you compute several Sij matrices for some reason).
        Only implemented if `sky` is set to 'psky'.
    
    mask2 : str or numpy.ndarray, default None
        Array or path to fits file containing a potential second mask in healpix form.
        In the case where you want the covariance between observables measured on different areas of the sky.
        PySSC will use healpy to compute the mask power spectrum.
        Again, it is faster to directly give clmask if you have it.
        Only implemented if `sky` is set to psky.
        If mask is set and mask2 is None, PySSC assumes that all observables share the same mask.

    cosmo_params : dict, default `default_cosmo_params`
        Dictionary of cosmology or cosmological parameters that can be accepted by `classy``

    cosmo_Class : classy.Class object, default None
        classy.Class object containing precomputed cosmology, \
        if you already have it and do not want PySSC to lose time recomputing cosmology with CLASS.

    convention : int, default 0
        Integer to dictate the convention used in the definition of the kernels.
        0 = Lacasa & Grain 2019.
        1 = Cosmosic , Euclid Forecasts
        Defaults to 0.

    precision : int, default 10
        Integer which drives the number of Fourier wavenumbers in internal integrals such as : Nk = 2**precision.

    var_tol : float, default 0.05
         Float that drives the target precision for the sum over angular multipoles.
         Default is 5%. Lowering it means increasing the number of multipoles thus increasing computational time.
         Only implemented if `sky`  is set to psky.

    tol : float, default 1e-3
        Tolerance value telling PySSC to cut off (i.e. set Sijkl=0) the matrix elements where there is too small
        overlap between the kernels rendering the computation unreliable.

    verbose : bool, default False
        Verbosity of the routine.
        Defaults to False

    debug : bool, default False
        Debuging options to look for incoherence in the routine.
        Defaults to False.

    Returns
    -------

    Array_like
        Sijkl matrix of shape (nbins,nbins,nbins,nbins).

    Notes
    -----

    Equation used (using indices :math:`(\\alpha,\\beta,\gamma,\delta)` instead of :math:`(i,j,k,l)` to avoid confusion with the Fourier wavevector and multipole):

    .. math::
        S_{\\alpha \\beta \gamma \delta} = \\frac{1}{(4\pi f_{\mathrm{sky}})^2} \sum_\ell (2\ell+1) \ C(\ell,\mathrm{mask}) \ C_S(\ell,\\alpha,\\beta,\gamma,\delta) 

    where \
    :math:`C(\ell,\\alpha,\\beta,\gamma,\delta) = \\frac{2}{\pi} \int k^2 dk \ P(k) \\frac{U(\\alpha,\\beta;k,\ell)}{I_\mathrm{norm}(\\alpha,\\beta)} \\frac{U(\gamma,\delta;k,\ell)}{I_\mathrm{norm}(\gamma,\delta)}`
    with :math:`I_\mathrm{norm}(\\alpha,\\beta) = \int dX \ W(\\alpha,z) \ W(\\beta,z)`  and 
    :math:`U(\\alpha,\\beta;k,\ell) = \int dX \ W(\\alpha,z) \ W(\\beta,z) \ G(z) \ j_\ell(k r)`.
    """

    #Raise deprecation warning
    warnings.warn("The Sijkl functions are now deprecated. Please move to using Sij by feeding the kernel products, see the documentation at https://pyssc.readthedocs.io/en/latest/notebooks/Main-Examples.html#General-case-with-cross-spectra", DeprecationWarning, stacklevel=2)

    from scipy.special import spherical_jn as jn

    # Find number of redshifts and bins    
    nz    = z_arr.size
    nbins = kernels.shape[0]

    # compute Cl(mask) and fsky computed from user input (mask(s) or clmask)
    ell, cl_mask, fsky = get_mask_quantities(clmask=clmask,mask=mask,mask2=mask2,verbose=verbose)

    # Search of the best lmax for all later sums on ell
    lmax = find_lmax(ell,cl_mask,var_tol,debug=debug)
    if verbose:
        print('lmax = %i' %(lmax))

    # Cut ell and Cl_mask to lmax, for all later computations
    cl_mask = cl_mask[:(lmax+1)]
    nell    = lmax+1
    ell     = np.arange(nell) #0..lmax

    #Get cosmology, comoving distances etc from dedicated auxiliary routine
    cosmo, h, comov_dist, dcomov_dist, growth = get_cosmo(z_arr, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class)

    #Get element of z integration, depending on kernel convention
    dX_dz = get_dX_dz(comov_dist, dcomov_dist, convention=convention)

    #Index pairs of bins
    npairs      = (nbins*(nbins+1))//2
    pairs       = np.zeros((2,npairs),dtype=int)
    count       = 0
    for ibin in range(nbins):
        for jbin in range(ibin,nbins):
            pairs[0,count] = ibin
            pairs[1,count] = jbin
            count +=1
        
    # Compute normalisations I_\mathrm{norm}(i,j) = int dX kernels(i,z) kernels(j,z)
    Inorm       = np.zeros(npairs)
    Inorm2D     = np.zeros((nbins,nbins))
    for ipair in range(npairs):
        ibin               = pairs[0,ipair]
        jbin               = pairs[1,ipair]
        integrand          = dX_dz * kernels[ibin,:]* kernels[jbin,:]
        integral           = integrate.simps(integrand,z_arr)
        Inorm[ipair]       = integral
        Inorm2D[ibin,jbin] = integral
        Inorm2D[jbin,ibin] = integral
    #Flag pairs with too small overlap as unreliable
    #Note: this will also speed up later computations
    #Default tolerance : tol=1e-3
    flag        = np.zeros(npairs,dtype=int)
    for ipair in range(npairs):
        ibin               = pairs[0,ipair]
        jbin               = pairs[1,ipair]
        ratio              = abs(Inorm2D[ibin,jbin])/np.sqrt(abs(Inorm2D[ibin,ibin]*Inorm2D[jbin,jbin]))
        if ratio<tol:
            flag[ipair]=1
    
    # Compute U(i,j;kk,ell) = int dX kernels(i,z) kernels(j,z) growth(z) j_ell(kk*r)  
    keq         = 0.02/h                                          #Equality matter radiation in 1/Mpc (more or less)
    klogwidth   = 10                                              #Factor of width of the integration range. 10 seems ok
    kmin        = min(keq,1./comov_dist.max())/klogwidth
    kmax        = max(keq,1./comov_dist.min())*klogwidth
    nk          = 2**precision                                    #10 seems to be enough. Increase to test precision, reduce to speed up.
    #kk          = np.linspace(kmin,kmax,num=nk)                   #linear grid on k
    logkmin     = np.log(kmin) ; logkmax   = np.log(kmax)
    logk        = np.linspace(logkmin,logkmax,num=nk)
    kk          = np.exp(logk)                                     #logarithmic grid on k    
    Pk          = np.zeros(nk)
    for ik in range(nk):
        Pk[ik]  = cosmo.pk(kk[ik],0.)                              #In Mpc^3
    Uarr        = np.zeros((npairs,nk,nell))
    for ik in range(nk):
        kr            = kk[ik]*comov_dist
        for ll in ell:
            bessel_jl = jn(ll,kr)
            for ipair in range(npairs):
                if flag[ipair]==0:
                    ibin = pairs[0,ipair]
                    jbin = pairs[1,ipair]
                    integrand        = dX_dz * kernels[ibin,:] * kernels[jbin,:] * growth * bessel_jl
                    Uarr[ipair,ik,ll] = integrate.simps(integrand,z_arr)

    # Compute Cl(X,Y) = 2/pi \int kk^2 dkk P(kk) U(i,j;kk,ell)/I_\mathrm{norm}(i,j) U(k,l;kk,ell)/I_\mathrm{norm}(k,l)
    Cl_XY      = np.zeros((npairs,npairs,nell))
    for ll in ell:
        #For ipair<=jpair
        for ipair in range(npairs):
            if flag[ipair]==0:
                U1 = Uarr[ipair,:,ll]/Inorm[ipair]
                for jpair in range(ipair,npairs):
                    if flag[jpair]==0:
                        U2 = Uarr[jpair,:,ll]/Inorm[jpair]
                        integrand = kk**2 * Pk * U1 * U2
                        #Cl_XY[ipair,jpair,ll] = 2/pi * integrate.simps(integrand,kk)     #linear integration
                        Cl_XY[ipair,jpair,ll] = 2/pi * integrate.simps(integrand*kk,logk) #log integration
        #Fill by symmetry   
        for ipair in range(npairs):
            for jpair in range(npairs):
                Cl_XY[ipair,jpair,ll] = Cl_XY[min(ipair,jpair),max(ipair,jpair),ll]

    # Finally sum over ell to get Sijkl = sum_ell (2ell+1) C(ell;i,j;k;l) C(ell,mask) /(4pi*fsky)^2
    Sijkl   = np.zeros((nbins,nbins,nbins,nbins))
    #For ipair<=jpair
    for ipair in range(npairs):
        if flag[ipair]==0:
            ibin = pairs[0,ipair]
            jbin = pairs[1,ipair]
            for jpair in range(ipair,npairs):
                if flag[jpair]==0:
                    kbin = pairs[0,jpair]
                    lbin = pairs[1,jpair]
                    sum_ell = np.sum((2*ell+1)*cl_mask*Cl_XY[ipair,jpair,:])/(4*pi*fsky)**2
                    #Run through all valid symmetries to fill the 4D array
                    #Symmetries: i<->j, k<->l, (i,j)<->(k,l)
                    Sijkl[ibin,jbin,kbin,lbin] = sum_ell
                    Sijkl[ibin,jbin,lbin,kbin] = sum_ell
                    Sijkl[jbin,ibin,kbin,lbin] = sum_ell
                    Sijkl[jbin,ibin,lbin,kbin] = sum_ell
                    Sijkl[kbin,lbin,ibin,jbin] = sum_ell
                    Sijkl[kbin,lbin,jbin,ibin] = sum_ell
                    Sijkl[lbin,kbin,ibin,jbin] = sum_ell
                    Sijkl[lbin,kbin,jbin,ibin] = sum_ell
    
    return Sijkl

####################################################################################################
#################################                   END                   ##########################
####################################################################################################

if __name__ == "__main__":
    print("test")           #To make the file executable for readthedocs compilation
	
# End of PySSC.py
