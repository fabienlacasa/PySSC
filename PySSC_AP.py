#!/usr/bin/python
# Filename: PySSC_AP.py

# Modules necessary for computation
import math ; pi=math.pi
import numpy as np
import sys
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from classy import Class
import json
import os
import subprocess as spr

##################################################

# Default values for redshift bin, cosmo parameters etc
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
def Sij_AngPow(z_arr,windows,Lmax,cosmo_params=default_cosmo_params,precision=12,cosmo_Class=None):
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
    
    h           = cosmo.h() #for  conversions Mpc/h <-> Mpc
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
        Pk[ik]  = cosmo.pk(kk[ik],0.)                              #In Mpc^3
    
    
    Omega_b     = default_cosmo_params["omega_b"]  /h**2
    Omega_cdm   = default_cosmo_params["omega_cdm"]/h**2
    Omega_m     = Omega_b + Omega_cdm
    path        = './AngPow_files/' #loc of all AngPow stuffs
    name        = 'SSC' #the name of the new AngPow .ini file
    np.savetxt('%s%s_Pk.txt'%(path,name),np.transpose(np.vstack((kk/h,Pk*(h**3))))) #AngPow needs Pk(z=0)
    cl_kmax     = np.amax(kk)*h #Cl is a mixing of modes : cl_kmax is the maximum k of the integral made by AngPow
    
    # define all AngPow parameters
    ini = {'Lmin' : 0,'Lmax' : Lmax ,'linearStep' : 40, 'logStep' : 1.15,'algo_type' : 1,'limber_lthr1' : -1,'limber_lthr2' : -1,'wtype':'UserFile,UserFile','mean' : '-1.,-1.','width': '-1.,-1.','w_dir' : './AngPow_files' , 'w_files' : '%s_win1.txt , %s_win2.txt' %(name,name),'cross_depth' : -1,'n_sigma_cut' : '-1.','cl_kmax' : cl_kmax,'radial_quad'  : 'trapezes','radial_order' : 50,'chebyshev_order' : 9,'n_bessel_roots_per_interval' : 100,'h': h,'omega_matter': Omega_m ,'omega_baryon': Omega_b,'hasX' : 0,'omega_X' :'','wX':'','waX':'','cosmo_zmin' : 0.,'cosmo_zmax' : 10.,'cosmo_npts' : 1000,'cosmo_precision' : 0.001,'Lmax_for_xmin' : 2000,'jl_xmin_cut'   : 5e-10,'output_dir' : path,'common_file_tag' : 'angpow_bench_%s_'%name,'quadrature_rule_ios_dir' : '%sAngPow/data/'%(path),'power_spectrum_input_dir': path,'power_spectrum_input_file' : '%s_Pk.txt' %name,'pw_kmin' : 1e-5,'pw_kmax' : 100.,}
    
    # write all AngPow parameters in a .ini file
    out = '%sangpow_bench_%s.ini'%(path,name)
    fo  = open(out, "w")
    for k, v in ini.items():
         fo.write(str(k) + '='+ str(v) + '\n')
    fo.close()
    
    threshold = 1e-100 # put by hands --> comming from instability in the AngPow algo
    Sij       = np.zeros((nbins,nbins))
    
    for bins_1 in range(nbins):
        for bins_2 in range(nbins):
            np.savetxt('%s%s_win1.txt'%(path,name),np.transpose(np.vstack((zz[win[bins_1,:]>threshold],(win[bins_1,:])[win[bins_1,:]>threshold]))))
            np.savetxt('%s%s_win2.txt'%(path,name),np.transpose(np.vstack((zz[win[bins_2,:]>threshold],(win[bins_2,:])[win[bins_2,:]>threshold]))))
            # running AngPow
            os.system('%sAngPow/bin/angpow %sangpow_bench_%s.ini'%(path,path,name))
            # Take the [0-1] column of the txt file (X-C_ell's)
            l_angpow,cl_angpow = np.loadtxt('%sangpow_bench_%s_cl.txt'%(path,name),ndmin=2,unpack=True)
            Sij[bins_1,bins_2] = cl_angpow[0] / (4*pi) #So far in full sky only need C_{ell=0}
    
    os.remove('%sangpow_bench_%s_cl.txt'%(path,name))
    os.remove('%sangpow_bench_%s_ctheta.txt'%(path,name))
    os.remove('%sangpow_bench_%s_used-param.txt'%(path,name))
    os.remove('%sangpow_bench_%s.ini'%(path,name))
    os.remove('%s%s_Pk.txt'%(path,name))
    os.remove('%s%s_win1.txt'%(path,name))
    os.remove('%s%s_win2.txt'%(path,name))
    
    return Sij;