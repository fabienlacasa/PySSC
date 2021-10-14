import numpy as np
from classy import Class
import os
from mpi4py import MPI
import sys

def spliting(arr1,arr2,number_at_the_end):
    redivision = np.linspace(0,len(arr1),number_at_the_end+1,dtype=int)
    arr1_ = np.split(arr1,redivision[1:-1])
    arr2_ = np.split(arr2,redivision[1:-1])
    return arr1_,arr2_;

comm = MPI.COMM_WORLD ; size = comm.Get_size() ; rank = comm.Get_rank()

rdm_rep     = float(sys.argv[1])
AngPow_path = str  (sys.argv[2])

file         = np.load(AngPow_path + 'temporary_%s/ini_files.npz'%rdm_rep, allow_pickle=True)
zz           = file['arr_0']
win          = file['arr_1']
lmax         = file['arr_2']
fsky         = file['arr_3']
cl_mask      = file['arr_4']
AngPow_path  = str(file['arr_5'])

cosmo_params = np.load(AngPow_path + 'temporary_%s/ini_files.npy'%rdm_rep,allow_pickle='TRUE').item()


nz    = len(zz)
nbins = win.shape[0]

# If the cosmology is not provided (in the same form as CLASS), run CLASS
cosmo = Class()
dico_for_CLASS = cosmo_params
dico_for_CLASS['output'] = 'mPk'
cosmo.set(dico_for_CLASS)
cosmo.compute()


h           = cosmo.h()                        #for  conversions Mpc/h <-> Mpc
zofr        = cosmo.z_of_r(zz)
comov_dist  = zofr[0]                          #Comoving distance r(z) in Mpc

'''dcomov_dist = 1/zofr[1]                       #Derivative dr/dz in Mpc
factor_conv = dcomov_dist/comov_dist**2 

W_conv      = np.zeros_like(win)
for bins in range(nbins):
     W_conv[bins,:]      = factor_conv * win[bins,:] / np.trapz(factor_conv * win[bins,:],zz)

win = W_conv*1'''


#computing the linear power spectrum at z=0
name = 'SSC' #the name of the new AngPow .ini file
kk = np.geomspace(1e-5,10,100000)

if rank == 0:
    Pk = np.zeros(len(kk))
    for ik in range(len(kk)): Pk[ik]  = cosmo.pk(kk[ik]*h,0.)
    np.savetxt(AngPow_path + 'temporary_%s/%s_Pk.txt'%(rdm_rep,name),np.transpose(np.vstack((kk,Pk*h**3))))

#distribute computation of Sij elements through the nodes  
todistributebin1 = []
todistributebin2 = []

for bins_1 in range(nbins):
    for bins_2 in range(nbins):
        if bins_1 >= bins_2 :
            todistributebin1.append(bins_1)
            todistributebin2.append(bins_2)

todistributebin1,todistributebin2 = spliting(todistributebin1,todistributebin2,size)  
todistributebin1 = todistributebin1[rank]
todistributebin2 = todistributebin2[rank]

comm.Barrier()

Sij  = np.zeros((nbins,nbins))
for bins in range(len(todistributebin1)):
        bins_1 = todistributebin1[bins]
        bins_2 = todistributebin2[bins]
        assert bins_1 >= bins_2 , 'bad cross-bining implementation'
        
        # define all AngPow parameters
        np.random.seed()
        rdm = np.random.random()
        ini = {'Lmin' : 0,'Lmax' : lmax+1 ,'linearStep' : 1, 'logStep' : 1,'algo_type' : 1,'limber_lthr1' : -1,'limber_lthr2' : -1,'wtype':'UserFile,UserFile','mean' : '-1.,-1.','width': '-1.,-1.','w_dir' : AngPow_path + 'temporary_%s'%rdm_rep , 'w_files' : '%s_win1_%s.txt , %s_win2_%s.txt' %(name,rdm,name,rdm),'cross_depth' : -1,'n_sigma_cut' : '-1.','cl_kmax' : np.amax(kk)*h,'radial_quad'  : 'trapezes','radial_order' : '350,350','chebyshev_order' : 10,'n_bessel_roots_per_interval' : 200,'h': h,'omega_matter': cosmo.Omega_m() ,'omega_baryon': cosmo.Omega_b(),'hasX' : 0,'omega_X' :'','wX':'','waX':'','cosmo_zmin' : 0.,'cosmo_zmax' : 3.,'cosmo_npts' : 1000,'cosmo_precision' : 0.001,'Lmax_for_xmin' : 2000,'jl_xmin_cut'   : 5e-10,'output_dir' : AngPow_path + 'temporary_%s/'%rdm_rep,'common_file_tag' : 'angpow_bench_%s_%s_'%(name,rdm),'quadrature_rule_ios_dir' : '%sAngPow/data/'%AngPow_path,'power_spectrum_input_dir': AngPow_path + 'temporary_%s/'%rdm_rep,'power_spectrum_input_file' : '%s_Pk.txt' %name,'pw_kmin' : kk[0],'pw_kmax' : kk[-1]}
        
        # write all AngPow parameters in a .ini file
        out = AngPow_path + 'temporary_%s/angpow_bench_%s_%s.ini'%(rdm_rep,name,rdm)
        fo  = open(out, "w")
        for k, v in ini.items(): fo.write(str(k) + '=' + str(v) + '\n')
        fo.close()
        
        # save window files for angpow
        z_1 = zz[win[bins_1,:]!=0]
        z_2 = zz[win[bins_2,:]!=0]
        w_1 = (win[bins_1,:])[win[bins_1,:]!=0]
        w_2 = (win[bins_2,:])[win[bins_2,:]!=0]
        comov_dist_ = comov_dist[win[bins_1,:]!=0]
        np.savetxt(AngPow_path + 'temporary_%s/%s_win1_%s.txt'%(rdm_rep,name,rdm),np.transpose(np.vstack((z_1,w_1**2))))
        np.savetxt(AngPow_path + 'temporary_%s/%s_win2_%s.txt'%(rdm_rep,name,rdm),np.transpose(np.vstack((z_2,w_2**2))))
        
        # running AngPow
        os.system(AngPow_path + 'bin/angpow ' + out)
        
        # Take the [0-1] column of the txt file (X-C_ell's)
        l_angpow,cl_angpow = np.loadtxt(AngPow_path + 'temporary_%s/angpow_bench_%s_%s'%(rdm_rep,name,rdm) + '_cl.txt',ndmin=2,unpack=True)
        Sij[bins_1,bins_2] = (1/(4*np.pi*fsky))**2 * np.sum( (2*l_angpow+1)*cl_mask[0:lmax+1]*cl_angpow )
        Sij[bins_2,bins_1] = Sij[bins_1,bins_2]

comm.Barrier()
if rank == 0 : 
    totals = np.zeros_like(Sij)
else: totals = None

comm.Barrier()
comm.Reduce( [Sij, MPI.DOUBLE], [totals, MPI.DOUBLE], op = MPI.SUM,root = 0)
if rank == 0 : np.savez(AngPow_path + 'temporary_%s/Sij'%rdm_rep,totals)