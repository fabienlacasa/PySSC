#!/usr/bin/python
# Filename: PySSC.py

# Modules necessary for computation
import math ; pi=math.pi
import numpy as np
import sys
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from classy import Class

##################################################

# Default values for redshift bin, cosmo parameters etc
default_zstakes = [0.9,1]
default_cosmo_params = {'omega_b':0.022,'omega_cdm':0.12,'H0':67.,'n_s':0.96,'sigma8':0.81}

# Routine to compute the Sij matrix with top-hat disjoint redshift window functions
# example : galaxy clustering with perfect/spectroscopic redshift determinations so that bins are sharp.
#
# Inputs : stakes of the redshift bins (array), cosmological parameters (dictionnary as in CLASS's wrapper classy)
# Output : Sij matrix (size: nbins x nbins)
def turboSij(zstakes=default_zstakes, cosmo_params=default_cosmo_params,cosmo_Class=None):

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


# Routine to compute the Sij matrix with general window functions given as tables
# example : weak lensing, or galaxy clustering with redshift errors
#
# Inputs : window functions (format: see below), cosmological parameters (dictionnary as in CLASS's wrapper classy)
# Format for window functions : one table of redshifts with size nz, one 2D table for the collection of window functions with shape (nbins,nz)
# Output : Sij matrix (shape: nbins x nbins)
#
## Equation used :  Sij = 1/(2*pi^2) int k^2 dk P(k) U(i,k)/Inorm(i) U(j,k)/Inorm(j)
## with Inorm(i) = int dV window(i,z)^2 and U(i,k) = int dV window(i,z)^2 growth(z) j_0(kr)
## This can also be seen as an angular power spectrum : Sij = C(ell=0,i,j)/4pi
## with C(ell=0,i,j) = 2/pi int k^2 dk P(k) U(i,k)/Inorm(i) U(j,k)/Inorm(j)
def Sij(z_arr, windows, cosmo_params=default_cosmo_params,precision=10,cosmo_Class=None):

    # Assert everything as the good type and shape, and find number of redshifts, bins etc
    zz  = np.asarray(z_arr)
    win = np.asarray(windows)
    
    assert zz.ndim==1, 'z_arr must be a 1-dimensional array'
    assert win.ndim==2, 'windows must be a 2-dimensional array'
    
    nz    = len(zz)
    nbins = win.shape[0]
    assert win.shape[1]==nz, 'windows must have shape (nbins,nz)'
    
    assert zz.min()>0, 'z_arr must have values > 0'
    
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
    dV          = comov_dist**2 * dcomov_dist               #Comoving volume per solid angle in Mpc^3/sr
    growth      = np.zeros(nz)                              #Growth factor
    for iz in range(nz):
        growth[iz] = cosmo.scale_independent_growth_factor(zz[iz])
    
    # Compute normalisations
    Inorm       = np.zeros(nbins)
    for i1 in range(nbins):
        integrand = dV * windows[i1,:]**2
        Inorm[i1] = integrate.simps(integrand,zz)
    
    # Compute U(i,k), numerator of Sij (integral of Window**2 * matter )
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
            integrand     = dV * windows[ibin,:]**2 * growth * np.sin(kr)/kr
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
            Cl_zero[ibin,jbin] = 2/pi * integrate.simps(integrand*kk,logk) #log integration
    #Fill by symmetry   
    for ibin in range(nbins):
        for jbin in range(nbins):
            Cl_zero[ibin,jbin] = Cl_zero[min(ibin,jbin),max(ibin,jbin)]
    Sij = Cl_zero / (4*pi)
    
    return Sij

# Alternative routine to compute the Sij matrix with general window functions given as tables
#
# Inputs : window functions, cosmological parameters, same format as Sij()
# Output : Sij matrix (shape: nbins x nbins)
#
## Equation used : Sij = int dV1 dV2 window(i,z1)^2/Inorm(i) window(j,z2)^2/Inorm(j) sigma2(z1,z2)
## with Inorm(i) = int dV window(i,z)^2 and sigma2(z1,z2) = 1/(2*pi^2) int k^2 dk P(k|z1,z2) j_0(kr1) j_0(kr2)
## which can be rewritten as sigma2(z1,z2) = 1/(2*pi^2*r1r2) G(z1) G(z2) int dk P(k,z=0) [cos(k(r1-r2))-cos(k(r1+r2))]/2
## which can be computed with an FFT
def Sij_alt(z_arr, windows, cosmo_params=default_cosmo_params,cosmo_Class=None):

    # Assert everything as the good type and shape, and find number of redshifts, bins etc
    zz  = np.asarray(z_arr)
    win = np.asarray(windows)
    
    assert zz.ndim==1, 'z_arr must be a 1-dimensional array'
    assert win.ndim==2, 'windows must be a 2-dimensional array'
    
    nz    = len(zz)
    nbins = win.shape[0]
    assert win.shape[1]==nz, 'windows must have shape (nbins,nz)'
    
    assert zz.min()>0, 'z_arr must have values > 0'
    
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
    dV          = comov_dist**2 * dcomov_dist               #Comoving volume per solid angle in Mpc^3/sr
    growth      = np.zeros(nz)                              #Growth factor
    for iz in range(nz):
        growth[iz] = cosmo.scale_independent_growth_factor(zz[iz])
    
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
        integrand = dV * windows[i1,:]**2
        Inorm[i1] = integrate.simps(integrand,zz)
    
    
    # Compute Sij finally
    prefactor  = sigma2 * (dV * dV[:,None])
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

# Routine to compute the Sijkl matrix, i.e. the most general case with cross-spectra
#
# Inputs : window functions, cosmological parameters, same format as Sij()
# Output : Sijkl matrix (shape: nbins x nbins x nbins x nbins)
#
## Equation used :  Sij = 1/(2*pi^2) int k^2 dk P(k) U(i,k)/Inorm(i) U(j,k)/Inorm(j)
## with Inorm(i) = int dV window(i,z)^2 and U(i,k) = int dV window(i,z)^2 growth(z) j_0(kr)
## This can also be seen as an angular power spectrum : Sij = \sum_\ell \ell (\ell + 1) C(ell,i,j)
## with C(ell=0,i,j) = 2/pi int k^2 dk P(k) U(i,k)/Inorm(i) U(j,k)/Inorm(j)
def Sijkl(z_arr, windows, cosmo_params=default_cosmo_params,precision=10,tol=1e-3,cosmo_Class=None):

    # Assert everything as the good type and shape, and find number of redshifts, bins etc
    zz  = np.asarray(z_arr)
    win = np.asarray(windows)
    
    assert zz.ndim==1, 'z_arr must be a 1-dimensional array'
    assert win.ndim==2, 'windows must be a 2-dimensional array'
    
    nz    = len(zz)
    nbins = win.shape[0]
    assert win.shape[1]==nz, 'windows must have shape (nbins,nz)'
    
    assert zz.min()>0, 'z_arr must have values > 0'
    
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
    dV          = comov_dist**2 * dcomov_dist               #Comoving volume per solid angle in Mpc^3/sr
    growth      = np.zeros(nz)                              #Growth factor
    for iz in range(nz):
        growth[iz] = cosmo.scale_independent_growth_factor(zz[iz])

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
        integrand          = dV * windows[ibin,:]* windows[jbin,:]
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
                integrand      = dV * windows[ibin,:] * windows[jbin,:] * growth * np.sin(kr)/kr
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


# Routine to compute the Sij matrix in partial sky
#
# Inputs : list of redshift bins, window functions, cl of the mask (fits file), path to the mask of the survey (fits file, readable by healpix), cosmological parameters
# Output : Sij matrix (shape: nbins x nbins)
#
## Equation used : Sijkl = sum_ell (2ell+1) C(ell,i,j,k,l) C(ell,mask) /(4pi*fsky)^2
## where C(ell,i,j,k,l) = 2/pi \int kk^2 dkk P(kk) U(i,j;kk,ell)/Inorm(i,j) U(k,l;kk,ell)/Inorm(k,l)
## with Inorm(i,j) = int dV window(i,z) window(j,z)
## and U(i,j;kk,ell) = int dV window(i,z) window(j,z) growth(z) j_ell(kk*r)
## C(ell,i,j,k,l) is computed via AngPow
def Sij_psky(z_arr, windows, clmask=None,mask=None, cosmo_params=default_cosmo_params,precision=10,cosmo_Class=None):

    import healpy as hp
    from scipy.special import spherical_jn as jn
    from astropy.io import fits

    # Assert everything as the good type and shape, and find number of redshifts, bins etc
    zz  = np.asarray(z_arr)
    win = np.asarray(windows)
    
    assert zz.ndim==1, 'z_arr must be a 1-dimensional array'
    assert win.ndim==2, 'windows must be a 2-dimensional array'
    
    nz    = len(zz)
    nbins = win.shape[0]
    assert win.shape[1]==nz, 'windows must have shape (nbins,nz)'
    
    assert zz.min()>0, 'z_arr must have values > 0'

    if (mask is None) and (clmask is None):
        print('Need either mask or cls of mask')
        sys.exit()
    pre_var = 0.01#precision/100
    if mask is None:
        print('Using Cls given as a fits file')
        cl_mask = hp.fitsfunc.read_cl(str(clmask))
        ell = np.arange(len(cl_mask))
        lmax = ell.max()
        var = np.sum(((2*np.arange(lmax+1)+1)/(4*pi)*cl_mask))
        var_est = np.sum(((2*np.arange(lmax/2)+1)/(4*pi)*cl_mask[:int(lmax/2)]))
        u=0
        while (abs(var - var_est)/var < pre_var):
            if (u>100): 
                lmax = int(nside/2)
                break
            lmax = lmax/2
            var = var_est
            var_est = np.sum(((2*np.arange(lmax)+1)/(4*pi)*cl_mask[:int(lmax)]))
            u=u+1
            # print(abs(var - var_est)/var,lmax)
        lmax = int(2*lmax)
        print('lmax = %i' %(lmax))
        cl_mask = cl_mask[:lmax]
        ell = np.arange(lmax)

    else :
        print('Using mask map, given as a fits file')
        map_mask = hp.read_map(str(mask))
        nside = hp.pixelfunc.get_nside(map_mask)  
        lmax = int(nside/10)
        cl2 = hp.anafast(map_mask, lmax=lmax*2)
        var = np.sum(((2*np.arange(lmax+1)+1)/(4*pi)*cl2[:lmax+1]))
        var_est = np.sum(((2*np.arange(2*lmax+1)+1)/(4*pi)*cl2))
        u = 0
        while (abs(var - var_est)/var > pre_var):
            if (u>100): 
                lmax = int(nside/2)
                break
            lmax = lmax*2
            var = var_est
            var_est = np.sum(((2*np.arange(lmax+1)+1)/(4*pi)*hp.anafast(map_mask, lmax=lmax)))
            u=u+1
        if (lmax!=int(nside/10)): lmax=int(lmax/2)
        print('lmax = %i' %(lmax))
        cl_mask = hp.anafast(map_mask, lmax=lmax)
        ell     = np.arange(len(cl_mask))
    
    # compute fsky from the mask
    fsky = np.sqrt(cl_mask[0]/(4*pi))
    print('f_sky = %.2f' %(fsky))

    def ws(x):
        return np.sum(cl_mask*(2*ell+1)*jn(ell,x[:,None]),axis=1)

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
    dV          = comov_dist**2 * dcomov_dist               #Comoving volume per solid angle in Mpc^3/sr
    growth      = np.zeros(nz)                              #Growth factor
    for iz in range(nz):
        growth[iz] = cosmo.scale_independent_growth_factor(zz[iz])
    
    # Compute normalisations
    Inorm       = np.zeros(nbins)
    for i1 in range(nbins):
        integrand = dV * windows[i1,:]**2 
        Inorm[i1] = integrate.simps(integrand,zz)
    
    # Compute U(i,k), numerator of Sij (integral of Window**2 * matter )
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
            integrand     = dV * windows[ibin,:]**2 * growth * ws(kr)
            Uarr[ibin,ik] = integrate.simps(integrand,zz)
    
    # Compute Sij finally
    Sij     = np.zeros((nbins,nbins))
    #For i<=j
    for ibin in range(nbins):
        U1 = Uarr[ibin,:]/Inorm[ibin]
        for jbin in range(ibin,nbins):
            U2 = Uarr[jbin,:]/Inorm[jbin]
            integrand = kk**2 * Pk * U1 * U2
            #Cl_zero[ibin,jbin] = 2/pi * integrate.simps(integrand,kk)     #linear integration
            Sij[ibin,jbin] = 2/pi * integrate.simps(integrand*kk,logk) #log integration
    #Fill by symmetry   
    for ibin in range(nbins):
        for jbin in range(nbins):
            Sij[ibin,jbin] = Sij[min(ibin,jbin),max(ibin,jbin)]
    
    return Sij

    

# End of PySSC.py