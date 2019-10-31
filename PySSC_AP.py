#!/usr/bin/python
# Filename: PySSC_AP.py

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

# Routine to compute the Sij matrix with general window functions given as tables using AngPow for redshift integration
# Note that the user may type a "make" in PySSC/AngPow_files/AngPow
# So far only in the TopHat case
# example : weak lensing, or galaxy clustering with redshift errors
#
# Inputs : window functions (format: see below), cosmological parameters (dictionnary as in CLASS's wrapper classy)
# Format for window functions : one table of redshifts with size nz, one 2D table for the collection of window functions with shape (nbins,nz)
# Output : Sij matrix (shape: nbins x nbins)
## Equation used : Sij = C(ell=0,i,j)/4pi
## with C(ell=0,i,j) computed by AngPow
def Sij_AngPow(z_arr, windows,Lmax, cosmo_params=default_cosmo_params,precision=10,cosmo_Class=None):
    import subprocess as spr
    import os
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
    zofr        = cosmo.z_of_r(zz)
    comov_dist  = zofr[0] 
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
    #how to obtain zmin zmax:
    mange_min = np.zeros(win.shape[0])
    mange_max = np.zeros(win.shape[0])

    for i in range(win.shape[0]):
        z_ald_bin = win[i,:].astype(bool)*zz
        mange_min[i] = np.amin((z_ald_bin)[z_ald_bin!=0])
        mange_max[i] = np.amax((z_ald_bin)[z_ald_bin!=0])
    Sij = np.zeros((nbins,nbins))
    for bins_1 in range(nbins):
        for bins_2 in range(nbins):
            zmin1  = mange_min[bins_1]     ; zmax1 = mange_max[bins_1]
            zmin2  = mange_min[bins_2]     ; zmax2 = mange_max[bins_2]
            mean1  = zmin1/2 + zmax1/2   ; mean2 = zmin2/2 + zmax2/2
            width1 = mean1 - zmin1       ; width2 = mean2 - zmin2
            Omega_b   = default_cosmo_params["omega_b"]  /h**2
            Omega_cdm = default_cosmo_params["omega_cdm"]/h**2
            Omega_m   = Omega_b + Omega_cdm
            path      = './AngPow_files/'
            name      = 'SSC' #the name of the new .ini file
            np.savetxt('%s%s.txt'%(path,name),np.transpose(np.vstack((kk/h,Pk*(h**3)))))
            cl_kmax   = np.amax(h*kk)*h
            spr.call(['cp', '%sangpow_bench_generator.ini'%path, '%sangpow_bench_%s.ini'%(path,name)])
            ini       = open('%sangpow_bench_%s.ini'%(path,name), "a")
            ini.write("\nh = %s\nomega_matter =%s\nomega_baryon = %s\noutput_dir = %s\ncommon_file_tag = angpow_bench_%s_\npower_spectrum_input_dir = %s\npower_spectrum_input_file = %s.txt\nmean = %lf,%lf\nwidth = %lf,%lf\nLmax = %i\ncl_kmax = %lf"%(h,Omega_m,Omega_b,path,name,path,name,mean1,mean2,width1,width2,Lmax,cl_kmax))
            ini.close()
            os.system('%sAngPow/bin/angpow %sangpow_bench_%s.ini'%(path,path,name))
            l_angpow,cl_angpow = np.loadtxt('%sangpow_bench_%s_cl.txt'%(path,name),ndmin=2,unpack=True)
            Sij[bins_1,bins_2] = cl_angpow[0] / (4*pi)
    return Sij