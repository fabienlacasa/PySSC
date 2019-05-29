#!/usr/bin/python
# Filename: pyssc.py

# Modules necessary for computation
import math ; pi=math.pi
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from classy import Class

##################################################

# Default values for redshift bin, cosmo parameters etc
default_zstakes = [0.9,1]
default_cosmo_params = {'omega_b':0.022,'omega_cdm':0.12,'H0':67.,'n_s':0.96,'sigma8':0.81}

# Routine to compute the Sij matrix with top-hat disjoint redshift window functions
# example : galaxy clustering with perfect/spectroscopic redshift determinations so that bins are sharp.
# Inputs : stakes of the redshift bins (array), cosmological parameters (dictionnary as in CLASS's wrapper classy)
# Output : Sij matrix (size: nbins x nbins)
def turboSij(zstakes=default_zstakes, cosmo_params=default_cosmo_params):

    # Run CLASS
    cosmo = Class()
    dico_for_CLASS = cosmo_params
    dico_for_CLASS['output'] = 'mPk'
    cosmo.set(dico_for_CLASS)
    cosmo.compute() 
    h = cosmo.h() #for  conversions Mpc/h <-> Mpc

    # Define arrays of z, r(z), k, P(k)...
    nzbins     = len(zstakes)-1
    nz_perbin  = 10
    z_arr      = np.zeros((nz_perbin,nzbins))
    comov_dist = np.zeros((nz_perbin,nzbins))
    for j in range(nzbins):
        z_arr[:,j] = np.linspace(zstakes[j],zstakes[j+1],nz_perbin)
        comov_dist[:,j] = (cosmo.z_of_r(z_arr[:,j]))[0] #In Mpc
    #print nzbins, z_arr
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

# Routine to compute the Sij matrix with general window functions given as tables
# example : weak lensing, or galaxy clustering with redshift errors
# Inputs : window functions (format: see below), cosmological parameters (dictionnary as in CLASS's wrapper classy)
# Format for window functions : one table of redshifts with size nz, one 2D table for the first window function with shape (nbin1,nz), one 2D table for the second window function with shape (nbin2,nz)
# Output : Sij matrix (size: nbin1 x nbin2)
def Sij(z_arr, window1, window2, cosmo_params=default_cosmo_params):

    # Assert everything as the good type and shape, and find number of redshifts, bins etc
    zz   = np.asarray(z_arr)
    win1 = np.asarray(window1)
    win2 = np.asarray(window2)
    
    assert zz.ndim==1, 'z_arr must be a 1-dimensional array'
    assert win1.ndim==2, 'window1 must be a 2-dimensional array'
    assert win2.ndim==2, 'window2 must be a 2-dimensional array'
    
    nz    = len(zz)
    nbin1 = win1.shape[0]
    nbin2 = win2.shape[0]
    assert win1.shape[1]==nz, 'window1 must have shape (nbins,nz)'
    assert win2.shape[1]==nz, 'window2 must have shape (nbins,nz)'
    
    assert zz.min()>0, 'z_arr must have values > 0'
    
    # Run CLASS
    cosmo = Class()
    dico_for_CLASS = cosmo_params
    dico_for_CLASS['output'] = 'mPk'
    cosmo.set(dico_for_CLASS)
    cosmo.compute() 
    h = cosmo.h() #for  conversions Mpc/h <-> Mpc
    
    # Define arrays of r(z), k, P(k)...
    comov_dist = cosmo.z_of_r(zz)[0] #In Mpc
    keq        = 0.02/h                        #Equality matter radiation in 1/Mpc (more or less)
    klogwidth  = 10                            #Factor of width of the integration range. 10 seems ok ; going higher needs to increase nk_fft to reach convergence (fine cancellation issue noted in Lacasa & Grain)
    kmin       = min(keq,1./comov_dist.max())/klogwidth
    kmax       = max(keq,1./comov_dist.min())*klogwidth
    nk_fft     = 2**11                         #seems to be enough. Increase to test precision, reduce to speed up.
    k_4fft     = np.linspace(kmin,kmax,nk_fft) #linear grid on k, as we need to use an FFT
    Deltak     = kmax - kmin
    Dk         = Deltak/nk_fft
    Pk_4fft    = np.zeros(nk_fft)
    for ik in range(nk_fft):
        Pk_4fft[ik] = cosmo.pk(k_4fft[ik],0.)  #In Mpc^3
    dr_fft     = np.linspace(0,nk_fft//2,nk_fft//2+1)*2*pi/Deltak
    
    # Compute necessary FFTs and make interpolation functions
    fft0       = np.fft.rfft(Pk_4fft)*Dk
    dct0       = fft0.real ; dst0 = -fft0.imag
    Pk_dct     = interp1d(dr_fft,dct0,kind='cubic')
    
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
            sigma2_nog[iz,jz] = (Icm0-Icp0)/(2*pi**2*r1*r2)
    #Now fill by symmetry and put back growth functions
    sigma2     = np.zeros((nz,nz))
    for iz in range(nz):
        growth1 = cosmo.scale_independent_growth_factor(zz[iz])
        for jz in range(nz):
            growth2       = cosmo.scale_independent_growth_factor(zz[jz])
            sigma2[iz,jz] = sigma2_nog[min(iz,jz),max(iz,jz)]*growth1*growth2            
    
    # Compute Sij finally
    Sij        = np.zeros((nbin1,nbin2))
    for i1 in range(nbin1):
        for i2 in range(nbin2):
            Sij[i1,i2] = 0
    
    return Sij

# End of pyssc.py