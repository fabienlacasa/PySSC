#!/usr/bin/python
# Filename: PySSC.py

####################################################################################################
#################################     IMPORT NECESSARY MODULES    #################################
####################################################################################################

import math ; pi=math.pi
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import os
from classy import Class

##################################################

# Default values for redshift bin, cosmo parameters etc
default_zstakes = [0.9,1]
default_cosmo_params = {'omega_b':0.022,'omega_cdm':0.12,'H0':67.,'n_s':0.96,'sigma8':0.81}
default_cosmo_params = {'z_max_pk': 0,'P_k_max_h/Mpc': 20,'H0':67.,'omega_b':0.022, 'omega_cdm':0.12 ,'n_s': 0.96, 'A_s' : 2.1265e-9,'output' : 'mPk'}

####################################################################################################
#################################          MAIN WRAPPERS           #################################
####################################################################################################

def Sij(z_arr, windows, sky='full', clmask=None, mask=None, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0, precision=10, var_tol=0.05, verbose=False, debug=False):
    
    test_zw(z_arr,windows)
    
    if sky.casefold() in ['full','fullsky','full sky','full-sky']:
        Sij=Sij_fullsky(z_arr, windows, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, convention=convention, precision=precision)
    elif sky.casefold() in ['psky','partial sky','partial-sky','partial','masked']:
        test_mask(mask, clmask)
        Sij=Sij_psky(z_arr, windows, clmask=clmask, mask=mask, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, convention=convention, precision=precision, var_tol=var_tol, verbose=verbose, debug=debug)
    else:
        raise Exception('Invalid string given for sky geometry parameter. Main possibilities : full sky or partial sky (or abbreviations, see code for details).')
    return Sij   

def Sij_alt(z_arr, windows, sky='full', cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0):
    
    test_zw(z_arr,windows)
    
    if sky.casefold() in ['full','fullsky','full sky','full-sky']:
        Sij=Sij_alt_fullsky(z_arr, windows, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, convention=convention)
    elif sky.casefold() in ['psky','partial sky','partial-sky','partial','masked']:
        raise Exception('No implementation of Sij_alt for partial sky. Use Sij instead.')
    else:
        raise Exception('Invalid string given for sky geometry parameter. Must be full sky (or abbreviations, see code for details).')
    return Sij

def Sijkl(z_arr, windows, sky='full', clmask=None, mask=None, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0, precision=10, tol=1e-3, var_tol=0.05, verbose=False, debug=False):
    
    test_zw(z_arr,windows)
    
    if sky.casefold() in ['full','fullsky','full sky','full-sky']:
        Sijkl=Sijkl_fullsky(z_arr, windows, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, convention=convention, precision=precision, tol=tol)
    elif sky.casefold() in ['psky','partial sky','partial-sky','partial','masked']:
        test_mask(mask, clmask)
        Sijkl=Sijkl_psky(z_arr, windows, clmask=clmask, mask=mask, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class, convention=convention, precision=precision, tol=tol, var_tol=var_tol, verbose=verbose, debug=debug)
    else:
        raise Exception('Invalid string given for sky geometry parameter. Must be either of : full, fullsky, full sky, full-sky.')
    return Sijkl

####################################################################################################
#################################        FULL SKY ROUTINES         #################################
####################################################################################################

##### Sij_fullsky #####
def Sij_fullsky(z_arr, windows, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0, precision=10):
    """
    Routine to compute the Sij matrix with general window functions given as tables
    example : weak lensing, or galaxy clustering with redshift errors

    Inputs :
    - redshifts and window functions
      Format : one table of redshifts with size nz, one 2D table for the collection of window functions with shape (nbins,nz)
    - cosmology or cosmological parameters
      Format : dictionnary with format of CLASS's wrapper classy
    Output : Sij matrix (shape: nbins x nbins)

    Options :
    - convention : convention used in the definition of the window functions/kernel.
      0 = Lacasa & Grain 2019. 1 = cosmosis, Euclid Forecasts
      Details below
    - precision : drives the number of Fourier wavenumbers in internal integrals. nk=2^precision

    Equation used :  Sij = 1/(2*pi^2) int k^2 dk P(k) U(i,k)/Inorm(i) U(j,k)/Inorm(j)
    with Inorm(i) = int dX window(i,z)^2 and U(i,k) = int dX window(i,z)^2 growth(z) j_0(kr)
    This can also be seen as an angular power spectrum : Sij = C(ell=0,i,j)^S/4pi
    with C(ell=0,i,j)^S = 2/pi int k^2 dk P(k) U(i,k)/Inorm(i) U(j,k)/Inorm(j)

    dX depends on the convention used to define the window functions : Cl(i,j) = int dX window(i,z) window(j,z) P(k=(ell+1/2)/r,z)
    0 : dX = dV = dV/dz dz = r^2(z) dr/dz dz. Used in Lacasa & Grain 2019.
    1 : dX = dchi/chi^2 = dr/dz/r^2(z) dz. Used in cosmosis.
    The convention of the Euclid Forecasts is nearly the same, up to a factor c^2 (or (c/HO)^2 depending on the probe)
    which is a constant so does not matter in the ratio here.
    """
    
    # Find number of redshifts and bins
    zz  = np.asarray(z_arr)
    win = np.asarray(windows)    
    nz    = len(zz)
    nbins = win.shape[0]
    
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
    
    # Define arrays of r(z), k, P(k)...
    zofr        = cosmo.z_of_r(zz)
    comov_dist  = zofr[0]                                   #Comoving distance r(z) in Mpc
    dcomov_dist = 1/zofr[1]                                 #Derivative dr/dz in Mpc
    dV_dz       = comov_dist**2 * dcomov_dist               #Comoving volume per solid angle dV/dz in Mpc^3/sr
    growth      = np.zeros(nz)                              #Growth factor
    for iz in range(nz):
        growth[iz] = cosmo.scale_independent_growth_factor(zz[iz])
    
    if convention==0:
        dX_dz = dV_dz
    elif convention==1:
        dX_dz = dcomov_dist / comov_dist**2
    else:
        raise ValueError('convention must be either 0 or 1')
    
    # Compute normalisations
    Inorm       = np.zeros(nbins)
    for i1 in range(nbins):
        integrand = dX_dz * windows[i1,:]**2
        Inorm[i1] = integrate.simps(integrand,zz)
    
    # Compute U(i,k), numerator of Sij (integral of Window**2 * matter )
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
            integrand     = dX_dz * windows[ibin,:]**2 * growth * np.sin(kr)/kr
            Uarr[ibin,ik] = integrate.simps(integrand,zz)
    
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
def Sij_alt_fullsky(z_arr, windows, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0):
    """
    Alternative routine to compute the Sij matrix with general window functions given as tables
    Inputs :
    - redshifts and window functions
      Format : one table of redshifts with size nz, one 2D table for the collection of window functions with shape (nbins,nz)
    - cosmology or cosmological parameters
      Format : dictionnary with format of CLASS's wrapper classy
    Output : Sij matrix (shape: nbins x nbins)
    Options :
    - convention : convention used in the definition of the window functions/kernel.
      0 = Lacasa & Grain 2019. 1 = cosmosis, Euclid Forecasts
      Details in Sij above

    Equation used : Sij = int dX1 dX2 window(i,z1)^2/Inorm(i) window(j,z2)^2/Inorm(j) sigma2(z1,z2)
    with Inorm(i) = int dX window(i,z)^2 and sigma2(z1,z2) = 1/(2*pi^2) int k^2 dk P(k|z1,z2) j_0(kr1) j_0(kr2)
    which can be rewritten as sigma2(z1,z2) = 1/(2*pi^2*r1r2) G(z1) G(z2) int dk P(k,z=0) [cos(k(r1-r2))-cos(k(r1+r2))]/2
    which can be computed with an FFT
    """

    # Find number of redshifts and bins
    zz  = np.asarray(z_arr)
    win = np.asarray(windows)    
    nz    = len(zz)
    nbins = win.shape[0]
    
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
    
    # Define arrays of r(z), k, P(k)...
    zofr        = cosmo.z_of_r(zz)
    comov_dist  = zofr[0]                                   #Comoving distance r(z) in Mpc
    dcomov_dist = 1/zofr[1]                                 #Derivative dr/dz in Mpc
    dV_dz       = comov_dist**2 * dcomov_dist               #Comoving volume per solid angle in Mpc^3/sr
    growth      = np.zeros(nz)                              #Growth factor
    for iz in range(nz):
        growth[iz] = cosmo.scale_independent_growth_factor(zz[iz])

    if convention==0:
        dX_dz = dV_dz
    elif convention==1:
        dX_dz = dcomov_dist / comov_dist**2
    else:
        raise ValueError('convention must be either 0 or 1')
    
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
    #First with P(k,z=0) and z1<=z2
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

    # Compute normalisations
    Inorm       = np.zeros(nbins)
    for i1 in range(nbins):
        integrand = dX_dz * windows[i1,:]**2
        Inorm[i1] = integrate.simps(integrand,zz)
    
    
    # Compute Sij finally
    prefactor  = sigma2 * (dX_dz * dX_dz[:,None])
    Sij        = np.zeros((nbins,nbins))
    #For i<=j
    for i1 in range(nbins):
        for i2 in range(i1,nbins):
            integrand  = prefactor * (windows[i1,:]**2 * windows[i2,:,None]**2)
            Sij[i1,i2] = integrate.simps(integrate.simps(integrand,zz),zz)/(Inorm[i1]*Inorm[i2])
    #Fill by symmetry   
    for i1 in range(nbins):
        for i2 in range(nbins):
            Sij[i1,i2] = Sij[min(i1,i2),max(i1,i2)]
    
    return Sij

##### Sijkl_fullsky #####
def Sijkl_fullsky(z_arr, windows, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0, precision=10, tol=1e-3):
    """
    Routine to compute the Sijkl matrix, i.e. the most general case with cross-spectra

    Inputs :
    - redshifts and window functions
      Format : one table of redshifts with size nz, one 2D table for the collection of window functions with shape (nbins,nz)
    - cosmology or cosmological parameters
      Format : dictionnary with format of CLASS's wrapper classy
    Output : Sijkl matrix (shape: nbins x nbins x nbins x nbins)
    Options :
    - convention : convention used in the definition of the window functions/kernel.
      0 = Lacasa & Grain 2019. 1 = cosmosis, Euclid Forecasts
      Details in Sij above
    - precision : drives the number of Fourier wavenumbers in internal integrals. nk=2^precision
    - tol : redshift bin pairs with too small overlap are flagged as unreliable. tol is the threshold of overlap.

    Equation used :  Sij = 1/(2*pi^2) int k^2 dk P(k) U(i,k)/Inorm(i) U(j,k)/Inorm(j)
    with Inorm(i) = int dX window(i,z)^2 and U(i,k) = int dX window(i,z)^2 growth(z) j_0(kr)
    This can also be seen as an angular power spectrum : Sij = C(ell=0,i,j)/4pi
    with C(ell=0,i,j) = 2/pi int k^2 dk P(k) U(i,k)/Inorm(i) U(j,k)/Inorm(j)
    """

    # Find number of redshifts and bins
    zz  = np.asarray(z_arr)
    win = np.asarray(windows)    
    nz    = len(zz)
    nbins = win.shape[0]
    
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
    
    # Define arrays of r(z), k, P(k)...
    zofr        = cosmo.z_of_r(zz)
    comov_dist  = zofr[0]                                   #Comoving distance r(z) in Mpc
    dcomov_dist = 1/zofr[1]                                 #Derivative dr/dz in Mpc
    dV_dz       = comov_dist**2 * dcomov_dist               #Comoving volume per solid angle in Mpc^3/sr
    growth      = np.zeros(nz)                              #Growth factor
    for iz in range(nz):
        growth[iz] = cosmo.scale_independent_growth_factor(zz[iz])

    if convention==0:
        dX_dz = dV_dz
    elif convention==1:
        dX_dz = dcomov_dist / comov_dist**2
    else:
        raise ValueError('convention must be either 0 or 1')

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
        integrand          = dX_dz * windows[ibin,:]* windows[jbin,:]
        integral           = integrate.simps(integrand,zz)
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
                integrand      = dX_dz * windows[ibin,:] * windows[jbin,:] * growth * np.sin(kr)/kr
                Uarr[ipair,ik] = integrate.simps(integrand,zz)
            
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

####################################################################################################
#################################       PARTIAL SKY ROUTINES       #################################
####################################################################################################

##### Sij_psky #####
def Sij_psky(z_arr, windows, clmask=None, mask=None, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0, precision=10, var_tol=0.05, verbose=False, debug=False):
    """
    Sij_psky
    Routine to compute the Sij matrix in partial sky

    Inputs :
    - redshifts and window functions
      Format : one table of redshifts with size nz, one 2D table for the collection of window functions with shape (nbins,nz)
    - cosmology or cosmological parameters
      Format : dictionnary with format of CLASS's wrapper classy
    - mask or its angular power spectrum
      Format : fits file
    Output : Sij matrix (shape: nbins x nbins)
    Options :
    - convention : convention used in the definition of the window functions/kernel.
      0 = Lacasa & Grain 2019. 1 = cosmosis, Euclid Forecasts
      Details in Sij above
    - precision : drives the number of Fourier wavenumbers in internal integrals. nk=2^precision

    Equation used : Sij = sum_ell (2ell+1) C(ell,i,j) C(ell,mask) /(4pi*fsky)^2
    where C(ell,i,j) = 2/pi \int kk^2 dkk P(kk) U(i;kk,ell)/Inorm(i) U(j;kk,ell)/Inorm(j)
    with Inorm(i) = int dX window(i,z)^2
    and U(i;kk,ell) = int dX window(i,z)^2 growth(z) j_ell(kk*r)
    """

    windows[windows<5e-100] = 0.
    import healpy as hp
    from scipy.special import spherical_jn as jn
    from astropy.io import fits

    # Find number of redshifts and bins
    zz  = np.asarray(z_arr)
    win = np.asarray(windows)    
    nz    = len(zz)
    nbins = win.shape[0]

    if mask is None: # User gives Cl(mask)
        if verbose:
            print('Using Cls given as a fits file')
        cl_mask = hp.fitsfunc.read_cl(str(clmask))
        ell = np.arange(len(cl_mask))
        lmaxofcl = ell.max()
    else : # User gives mask as a map
        if verbose:
            print('Using mask map, given as a fits file')
        map_mask = hp.read_map(str(mask), dtype=np.float64, verbose=False)
        nside    = hp.pixelfunc.get_nside(map_mask)
        lmaxofcl = 2*nside
        cl_mask  = hp.anafast(map_mask, lmax=lmaxofcl)
        ell      = np.arange(lmaxofcl+1)
    
    # compute fsky from the mask
    fsky = np.sqrt(cl_mask[0]/(4*pi))
    if verbose:
        print('f_sky = %.4f' %(fsky))

    # Search of the best lmax for all later sums on ell
    lmax = find_lmax(ell,cl_mask,var_tol,debug=debug)
    if verbose:
        print('lmax = %i' %(lmax))

    # Cut ell and Cl_mask to lmax, for all later computations
    cl_mask = cl_mask[:(lmax+1)]
    nell    = lmax+1
    ell     = np.arange(nell) #0..lmax

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
    
    # Define arrays of r(z), k, P(k)...
    zofr        = cosmo.z_of_r(zz)
    comov_dist  = zofr[0]                                   #Comoving distance r(z) in Mpc
    dcomov_dist = 1/zofr[1]                                 #Derivative dr/dz in Mpc
    dV_dz       = comov_dist**2 * dcomov_dist               #Comoving volume per solid angle in Mpc^3/sr
    growth      = np.zeros(nz)                              #Growth factor
    for iz in range(nz):
        growth[iz] = cosmo.scale_independent_growth_factor(zz[iz])

    if convention==0:
        dX_dz = dV_dz
    elif convention==1:
        dX_dz = dcomov_dist / comov_dist**2
    else:
        raise ValueError('convention must be either 0 or 1')
    
    # Compute normalisations
    Inorm       = np.zeros(nbins)
    for i1 in range(nbins):
        integrand = dX_dz * windows[i1,:]**2 
        Inorm[i1] = integrate.simps(integrand,zz)


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
                integrand     = dX_dz * windows[ibin,:]**2 * growth * np.sin(kr)/kr
                Uarr[ibin,ik] = integrate.simps(integrand,zz)
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

    
    # Compute U(i;k,ell) = int dX window(i,z)^2 growth(z) j_ell(kk*r)
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
                integrand        = dX_dz * windows[ibin,:]**2 * growth * bessel_jl
                Uarr[ibin,ik,ll] = integrate.simps(integrand,zz)

    # Compute Cl(X,Y) = 2/pi \int kk^2 dkk P(kk) U(i;kk,ell)/Inorm(i) U(j;kk,ell)/Inorm(j)
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
def Sij_flatsky(z_arr, windows, bin_centres, theta, cosmo_params=default_cosmo_params, cosmo_Class=None, verbose=False):
    """
    Computing Sij according to flat-sky approximation
    See Eq. 16 of arXiv:1612.05958

    Parameters:
       - z_arr : redshift array (must be >0)
       - windows: values of window function on z_arr for each bin. Dimensions (nbins,len(z_arr))
       - bin_centres : central values of redshift bins. Dimensions: (nbins)
       - theta : radius of the survey mask in deg
       Note: the mask is assumed to be a circle with radius theta
       - cosmo_params : Class dictionary of cosmological parameters
    """

    from scipy.special import spherical_jn as jn
    from scipy.special import jv as Jn

    # Find number of redshifts and bins
    zz  = np.asarray(z_arr)
    win = np.asarray(windows)    
    nz    = len(zz)
    nbins = win.shape[0]

    theta = theta*np.pi/180. #converts in radians

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
    
    # Define arrays of r(z), k...
    zofr        = cosmo.z_of_r(zz)
    comov_dist  = zofr[0]                                   #Comoving distance r(z) in Mpc
    
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
    #     growth[iz] = cosmo.scale_independent_growth_factor(zz[iz])

    if verbose: print('Computing flat-sky approximation')
    Sij = np.zeros((nbins,nbins))
    for ibin in range(nbins):
        z1 = zstakes[ibin]
        r1 = cosmo.z_of_r([z1])[0][0]
        dr1 = comov_dist[win[ibin,:]!=0].max()-comov_dist[win[ibin,:]!=0].min() #width of function function

        for jbin in range(nbins):
            z2 = zstakes[jbin]
            r2 = cosmo.z_of_r([z2])[0][0]
            dr2 = comov_dist[win[jbin,:]!=0].max()-comov_dist[win[jbin,:]!=0].min() #width of window function

            z12 = np.mean([zstakes[ibin],zstakes[jbin]])

            growth = cosmo.scale_independent_growth_factor(z12)
            Pk = np.array([cosmo.pk(k_arr[i,j],0.) for i in range(nk_perp) for j in range(nk_par)])
            Pk = Pk.reshape(k_arr.shape)

            integ_kperp = kperp_arr * 4. * (Jn(1,kperp_arr*theta*r1)/kperp_arr/theta/r1) * ( Jn(1,kperp_arr*theta*r2)/kperp_arr/theta/r2 )
            integ_kpar = jn(0,kpar_arr*dr1/2) * jn(0,kpar_arr*dr2/2)
            dSij = integ_kperp[:,None] * integ_kpar[None,:] * growth * Pk * np.cos(kpar_arr[None,:]*abs(r1-r2)) 

            Sij[ibin,jbin] = integrate.simps( integrate.simps(dSij,kpar_arr), kperp_arr)/ 2. / np.pi**2

    return Sij

    
##### Sijkl_psky #####
def Sijkl_psky(z_arr, windows, clmask=None, mask=None, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0, precision=10, var_tol=0.05, tol=1e-3, verbose=False, debug=False):
    """
    Routine to compute the Sijkl matrix, i.e. the most general case with cross-spectra

    Inputs :
    - redshifts and window functions
      Format : one table of redshifts with size nz, one 2D table for the collection of window functions with shape (nbins,nz)
    - cosmology or cosmological parameters
      Format : dictionnary with format of CLASS's wrapper classy
    - mask or its angular power spectrum
      Format : fits file
    Output : Sijkl matrix (shape: nbins x nbins x nbins x nbins)
    Options :
    - convention : convention used in the definition of the window functions/kernel.
      0 = Lacasa & Grain 2019. 1 = cosmosis, Euclid Forecasts
      Details in Sij above
    - precision : drives the number of Fourier wavenumbers in internal integrals. nk=2^precision
    - tol : redshift bin pairs with too small overlap are flagged as unreliable. tol is the threshold of overlap.

     Equation used : Sijkl = sum_ell (2ell+1) C(ell;i,j;k,l) C(ell,mask) /(4pi*fsky)^2
     where C(ell;i,j;k,l) = 2/pi \int kk^2 dkk P(kk) U(i,j;kk,ell)/Inorm(i,j) U(k,l;kk,ell)/Inorm(k,l)
     with Inorm(i,j) = int dX window(i,z) window(j,z)
     and U(i,j;kk,ell) = int dX window(i,z) window(j,z) growth(z) j_ell(kk*r)
    """

    import healpy as hp
    from scipy.special import spherical_jn as jn
    from astropy.io import fits

    # Find number of redshifts and bins
    zz  = np.asarray(z_arr)
    win = np.asarray(windows)    
    nz    = len(zz)
    nbins = win.shape[0]

    if mask is None: # User gives Cl(mask)
        if verbose:
            print('Using Cls given as a fits file')
        cl_mask = hp.fitsfunc.read_cl(str(clmask))
        ell = np.arange(len(cl_mask))
        lmaxofcl = ell.max()
    else : # User gives mask as a map
        if verbose:
            print('Using mask map, given as a fits file')
        map_mask = hp.read_map(str(mask), dtype=np.float64, verbose=False)
        nside    = hp.pixelfunc.get_nside(map_mask)
        lmaxofcl = 2*nside
        cl_mask  = hp.anafast(map_mask, lmax=lmaxofcl)
        ell      = np.arange(lmaxofcl+1)
        
    # compute fsky from the mask
    fsky = np.sqrt(cl_mask[0]/(4*pi))
    if verbose:
        print('f_sky = %.4f' %(fsky))

    # Search of the best lmax for all later sums on ell
    lmax = find_lmax(ell,cl_mask,var_tol,debug=debug)
    if verbose:
        print('lmax = %i' %(lmax))

    # Cut ell and Cl_mask to lmax, for all later computations
    cl_mask = cl_mask[:(lmax+1)]
    nell    = lmax+1
    ell     = np.arange(nell) #0..lmax

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
    
    # Define arrays of r(z), k, P(k)...
    zofr        = cosmo.z_of_r(zz)
    comov_dist  = zofr[0]                                   #Comoving distance r(z) in Mpc
    dcomov_dist = 1/zofr[1]                                 #Derivative dr/dz in Mpc
    dV_dz       = comov_dist**2 * dcomov_dist               #Comoving volume per solid angle in Mpc^3/sr
    growth      = np.zeros(nz)                              #Growth factor
    for iz in range(nz):
        growth[iz] = cosmo.scale_independent_growth_factor(zz[iz])

    if convention==0:
        dX_dz = dV_dz
    elif convention==1:
        dX_dz = dcomov_dist / comov_dist**2
    else:
        raise ValueError('convention must be either 0 or 1')

    #Index pairs of bins
    npairs      = (nbins*(nbins+1))//2
    pairs       = np.zeros((2,npairs),dtype=int)
    count       = 0
    for ibin in range(nbins):
        for jbin in range(ibin,nbins):
            pairs[0,count] = ibin
            pairs[1,count] = jbin
            count +=1
        
    # Compute normalisations Inorm(i,j) = int dX window(i,z) window(j,z)
    Inorm       = np.zeros(npairs)
    Inorm2D     = np.zeros((nbins,nbins))
    for ipair in range(npairs):
        ibin               = pairs[0,ipair]
        jbin               = pairs[1,ipair]
        integrand          = dX_dz * windows[ibin,:]* windows[jbin,:]
        integral           = integrate.simps(integrand,zz)
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
    
    # Compute U(i,j;kk,ell) = int dX window(i,z) window(j,z) growth(z) j_ell(kk*r)  
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
                    integrand        = dX_dz * windows[ibin,:] * windows[jbin,:] * growth * bessel_jl
                    Uarr[ipair,ik,ll] = integrate.simps(integrand,zz)

    # Compute Cl(X,Y) = 2/pi \int kk^2 dkk P(kk) U(i,j;kk,ell)/Inorm(i,j) U(k,l;kk,ell)/Inorm(k,l)
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
#################################          ANGPOW ROUTINES         #################################
####################################################################################################

##### Sij_AngPow #####
def Sij_AngPow(z_arr,windows,clmask=None,mask=None,cosmo_params=default_cosmo_params,var_tol=0.05,machinefile=None,Nn=None,Np='default',AngPow_path=None,verbose=False,debug=False):

    import time
    import os
    import shutil
    import healpy as hp
    
    test_zw(z_arr, windows)
    test_mask(mask, clmask)
    
    zz  = np.asarray(z_arr)
    win = np.asarray(windows) 
    
    #testing machinefile, Nn and AngPow_path
    if machinefile is not None:
        assert Nn is not None, 'a machinefile has been provided, but not the number of node Nn.  Nn should not exceed the number of line in the machinefile'
        assert os.path.exists(machinefile) , 'the proposed machinefile has not been found, please update the path'
    if Nn is not None:
        assert int(Nn) == Nn , 'the number of nodes Nn must be integer'
        assert machinefile is not None, 'a number of nodes Nn has been provided, but not the associated machinefile. Please create an adapted machinefile (see machinefile_example)'
    if AngPow_path is not None:
        assert os.path.exists(AngPow_path + 'bin/angpow') , 'the angpow executable is not in the provided AngPow_path, please update the path or make sure the angpow compilation has been correctly done'
    else :
        assert os.path.exists('./AngPow/AngPow/bin/angpow') , 'the angpow executable is not in ./AngPow/AngPow/bin/angpow, please make sure the angpow compilation has been correctly done or give another angpow path in the AngPow_path option'
    if AngPow_path is None:
        AngPow_path = os.getcwd() + '/AngPow/AngPow/' #finishing with '/' 
    if Np is not 'default':
        assert int(Np) == Np , 'the number of process per node Np must be integer'
    
    #compute the lmax for AngPow    
    if mask is None: # User gives Cl(mask)
        if verbose:
            print('Using Cls given as a fits file')
        cl_mask = hp.fitsfunc.read_cl(str(clmask))
        ell = np.arange(len(cl_mask))
        lmaxofcl = ell.max()
    else : # User gives mask as a map
        if verbose:
            print('Using mask map, given as a fits file')
        map_mask = hp.read_map(str(mask),verbose=False)
        nside    = hp.pixelfunc.get_nside(map_mask)
        lmaxofcl = 2*nside
        cl_mask  = hp.anafast(map_mask, lmax=lmaxofcl)
        ell      = np.arange(lmaxofcl+1)

    # compute fsky from the mask
    fsky = np.sqrt(cl_mask[0]/(4*np.pi))
    if verbose:
        print('f_sky = %.4f' %(fsky))

    # Search of the best lmax for all later sums on ell
    lmax = find_lmax(ell,cl_mask,var_tol,debug=debug)

    if verbose:
        print('lmax = %i' %(lmax))
        
    #create data to pass to MPI AngPow routine
    rdm = np.random.random()
    os.makedirs(AngPow_path + 'temporary_%s'%rdm)
    np.savez(AngPow_path + 'temporary_%s/ini_files'%rdm, zz, win, lmax, fsky, cl_mask, AngPow_path)
    np.save(AngPow_path + 'temporary_%s/ini_files.npy'%rdm, cosmo_params) 
    
    present_rep = os.getcwd()
    #run MPI AngPow routine
    if Nn is not None:
        os.system('mpiexec -f %s -n %i python %s/AngPow/PySSC_AP_MPI.py %s %s %s'%(machinefile,Nn,present_rep,rdm,AngPow_path,Np))
    else:
        os.system('python %s/AngPow/PySSC_AP_MPI.py %s %s %s'%(present_rep,rdm,AngPow_path,Np))
    time.sleep(10)
    
    #load Sij result
    file = np.load(AngPow_path + 'temporary_%s/Sij.npz'%(rdm))
    Sij = file['arr_0']
    shutil.rmtree(AngPow_path + 'temporary_%s'%rdm, ignore_errors=True)
    
    return Sij

##### Sij_AngPow_fullsky #####
def Sij_AngPow_fullsky(z_arr,windows,cosmo_params=default_cosmo_params,var_tol=0.05,machinefile=None,Nn=None,Np='default',AngPow_path=None,verbose=False,debug=False):

    import healpy as hp
    import os
    
    test_zw(z_arr, windows)
    
    # Define the angular power spectrum of a mask that is 1 over the full sky
    Cl_fullsky=np.zeros(10) ; Cl_fullsky[0]=4*pi

    # Write the Cl to a temporary file
    rdm = np.random.random()
    tmp_file = './tmp_Cl_M_PySSC_%s'%rdm+'.fits'
    hp.fitsfunc.write_cl(tmp_file,Cl_fullsky)

    # Call Sij_AngPow with that Cl file
    Sij = Sij_AngPow(z_arr,windows,clmask=tmp_file,cosmo_params=cosmo_params,var_tol=var_tol,machinefile=machinefile,Nn=Nn,Np='default',AngPow_path=AngPow_path,verbose=verbose,debug=debug)

    # Remove the temporary Cl file
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    # Return the result
    return Sij

####################################################################################################
#################################        AUXILIARY ROUTINES        #################################
####################################################################################################

##### test_zw #####
def test_zw(z_arr, windows):
    """
    Assert redshift and windows arrays have the good type and shape
    """
    zz  = np.asarray(z_arr)
    win = np.asarray(windows)
    assert zz.ndim==1, 'z_arr must be a 1-dimensional array'
    assert win.ndim==2, 'windows must be a 2-dimensional array'    
    nz    = len(zz)
    assert win.shape[1]==nz, 'windows must have shape (nbins,nz)'
    assert zz.min()>0, 'z_arr must have values > 0'

##### test_mask #####
def test_mask(mask, clmask):
    assert (mask is not None) or (clmask is not None), 'You need to provide either the mask or its angular power spectrum Cl.'

##### find_lmax #####
def find_lmax(ell, cl_mask, var_tol, debug=False):
    """
    Routine to search the best lmax for all later sums on multipoles
    Inputs :
      - ell : full vectors of multipoles. As large as possible
      - cl_mask : power spectrum of the mask at the supplied multipoles
    Method: smallest lmax so that we have convergence of the variance var = sum_ell (2*ell+1)/4pi * Clmask
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
    """
    Routine to compute the Sij matrix with top-hat disjoint redshift window functions
    example : galaxy clustering with perfect/spectroscopic redshift determinations so that bins are sharp.

    Inputs : stakes of the redshift bins (array), cosmological parameters (dictionnary as in CLASS's wrapper classy)
    Output : Sij matrix (size: nbins x nbins)
    """
    
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

# End of PySSC.py