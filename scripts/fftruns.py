from mpi4py import MPI
import numpy as np
from pixell import enmap, utils, lensing
from cmbpix.utils import *
from cmbpix.lensing import fftest as fe
from symlens import qe, interp
from orphics.maps import mask_kspace
from orphics.stats import bin2D
from scipy.interpolate import interp1d
import camb
import pickle

import warnings
warnings.filterwarnings(action='ignore')# Change to once for first time

# MPI stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
Nsims = 100

# Test params
lb = 500 # l bin size
lmin = 6000
lmax = 8000
ldT = 2500
w = 1.0 # In uK-arcmin
b = 1.0 # In arcmin

if rank == 0:
    wn = str(w).replace('.', 'p')
    bn = str(b).replace('.', 'p')
    loc = './sims/victor/estout/CLGK'
    fn = '/FFTout_{}ldT_{}lb_{}-{}_{}uKarcmin_{}bin.pkl'.format(ldT, 
                lb, lmin, lmax, wn, bn)

# # Input CMB info
# X = np.loadtxt("sims/cosmo2017_10K_acc3_scalCls.dat")
# Z = np.loadtxt("lensedPS.txt")

# TCMB = 2.726e6
# ell = X[:,0]
# dltt = X[:,1]
# clkk = X[:,4]/(TCMB**2)/4.
# clpp = 4 * clkk / (ell**2 * (ell+1)**2)
# ellfac = ell * (ell+1) / 2 / np.pi
# cltt = dltt / ellfac
# w2 = (w * np.pi / 10800)**2
# beam = 1.0 * np.pi / 10800 / (2 * np.sqrt(2*np.log(2)))
# Wl = np.exp(-ell*(ell+1)*beam**2)
# Nl = w2 / Wl

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(20000, lens_potential_accuracy=2);
results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

ls = np.arange(powers['unlensed_scalar'].shape[0])
## CAMB outputs l(l+1)/(2pi) * C_l by default, need to rescale
ell_fac = ls*(ls+1.)/(2*np.pi)
ctt_unlensed = (powers['unlensed_scalar'][:,0]/ell_fac)
## CAMB outputs l(l+1)/(2pi) * C_l^{dd} by default, need to rescale to C_l^{phiphi}
ctt_lensed = (powers['total'][:,0]/ell_fac)
ell_fac_phi = (ls*(ls+1.)*ls*(ls+1.))/(2*np.pi)
cphiphi = (powers['lens_potential'][:,0]/ell_fac_phi)

# Noise spectrum
# w2 = (w * np.pi / 10800)**2
# beam = 1.0 * np.pi / 10800 / (2 * np.sqrt(2*np.log(2)))
# Wl = np.exp(-ell*(ell+1)*beam**2)
# ntt = w2 / Wl
Delta_T = 1.  # In muK-arcmin
theta_fwhm = 1.   # in arcmin
ntt = (Delta_T*np.pi/180./60.)**2. * np.exp((theta_fwhm*np.pi/180./60. / np.sqrt(8.*np.log(2)))**2.*ls**2.)

ctt_total = ctt_lensed + ntt
# Replace potential nan here
ctt_unlensed[0] = 0
ctt_lensed[0] = 0
cphiphi[0] = 0

# Simulation size/resolution
width = 10
reso = 0.5
box = np.array([[-5,width/2],[5,-width/2]]) * utils.degree
shape, wcs = enmap.geometry(pos=box, res=reso*utils.arcmin, proj='car')
# shape, wcs = maps.rect_geometry(width_deg=10., px_res_arcmin=0.5)
modlmap = enmap.modlmap(shape, wcs)

# Binner for data power spectrum
Dbins = np.arange(20,20000,20)
Dbinner = bin2D(modlmap,Dbins)

# Initialize objects for QE
XY='TT'
UV='TT'
ellmin = 500 ; ellmax = 10000
Lmin = 100 ; Lmax = 10000
xmask = mask_kspace(shape,wcs,lmin=ellmin,lmax=ellmax)
ymask = xmask
kmask = mask_kspace(shape,wcs,lmin=Lmin,lmax=Lmax)
feed_dict = {}
feed_dict['uC_T_T'] = interp(ls, ctt_lensed)(modlmap)
feed_dict['tC_T_T'] = interp(ls, ctt_total)(modlmap)
feed_dict['nC_T_T'] = interp(ls, ctt_total)(modlmap)
bin_edges_phi = np.arange(100,Lmax,100)
binner_phi = bin2D(modlmap,bin_edges_phi)
Nphi = bin_edges_phi.size - 1
cents_phi, pp_theory_bin = binner_phi.bin(interp(ls,cphiphi)(modlmap))

# Output data goes here
outs = {'unlensed': np.empty([Nsims, 10000//lb], dtype=np.float64), 
        'lensed': np.empty([Nsims, 10000//lb], dtype=np.float64), 
        'cents_qe': cents_phi, 
        'reconst': np.empty([Nsims, Nphi], dtype=np.float64), 
        'N0': np.empty([Nsims, Nphi], dtype=np.float64), 
        'RDN0': np.empty([Nsims, Nphi], dtype=np.float64), 
        'phi_theory': pp_theory_bin}

# Sims
for i in range(Nsims):
    if rank == 0 and i % 10 == 0:
        print("{}% complete".format(i), flush=True)
    # Generate maps
    Tmap = enmap.rand_map(shape, wcs, ctt_unlensed)
    phimap = enmap.rand_map(shape, wcs, cphiphi)
    nmap = enmap.rand_map(shape, wcs, ntt)
    lTmap = lensing.lens_map_flat(Tmap, phimap)
    # Compute power spectrum for this realization
    dcents, C_l_tot = Dbinner.bin(np.abs(enmap.fft(Tmap+nmap, 
        normalize='phys'))**2)
    # Set up dictionary for QE
    feed_dict['X'] = enmap.fft(lTmap+nmap, normalize='phys')
    feed_dict['Y'] = feed_dict['X']
    feed_dict['dC_T_T'] = interp1d(dcents, C_l_tot, 
        bounds_error=False, fill_value=0)(modlmap)

    # Run QE on the lensed realization, then run CLGK estimator
    my_reconst = qe.reconstruct(shape, wcs, feed_dict, 'hu_ok', XY, 
        xmask= xmask, ymask = ymask)
    my_A_l = qe.A_l(shape, wcs, feed_dict, 'hu_ok', XY, xmask, ymask)
    N_l_phi = qe.N_l_optimal(shape,wcs,feed_dict,'hu_ok',XY,
        xmask=xmask,ymask=ymask,field_names=None,kmask=kmask)
    rdn0_2d = qe.RDN0_analytic(shape,wcs,feed_dict,'hu_ok',XY,
        'hu_ok',UV,xmask=xmask,ymask=ymask,kmask=kmask)
    cents_phi, reconst_bin = binner_phi.bin(np.abs(my_reconst)**2)
    cents_phi, N0_l_bin = binner_phi.bin(N_l_phi)
    cents_phi, RDN0_l_bin = binner_phi.bin(rdn0_2d)
    kfac = (cents_phi*(cents_phi+1))**2 / (2*np.pi)
    kk_theory_bin = kfac * pp_theory_bin

    # Compute normalization and prediction on first pass only
    if i == 0:
        ucents, up1d, AL, Phi = fe.FFTest(Tmap+nmap, 
            ldT=ldT, lmin=lmin, lmax=lmax, lbins=lb, 
            uCls=ctt_unlensed, lCls=ctt_lensed, Nls=ntt, Clpp=cphiphi, 
            w=w, sg=b, apply_bias=True, plots=False)
        outs['cents'] = ucents
        outs['AL'] = AL
        # outs['BL'] = BL
        outs['Phi'] = Phi
    else:
        ucents, up1d = fe.FFTest(Tmap+nmap, 
            ldT=ldT, lmin=lmin, lmax=lmax, lbins=lb, 
            uCls=ctt_unlensed, lCls=ctt_lensed, Nls=ntt, Clpp=cphiphi, 
            w=w, sg=b, apply_bias=False, plots=False)

    lcents, lp1d = fe.FFTest(lTmap+nmap, 
        ldT=ldT, lmin=lmin, lmax=lmax, lbins=lb, 
        uCls=ctt_unlensed, lCls=ctt_lensed, Nls=ntt, Clpp=cphiphi, 
        w=w, sg=b, apply_bias=False, plots=False)

    outs['unlensed'][i,:] = up1d
    outs['lensed'][i,:] = lp1d
    outs['reconst'] = reconst_bin
    outs['N0'] = N0_l_bin
    outs['RDN0'] = RDN0_l_bin
    if rank == 0 and i == 0:
        print("Finished 1 loop", flush=True)


specsize = outs['unlensed'][0].size
urec = None
lrec = None
qerec = None
N0rec = None
RDN0rec = None
if rank == 0:
    urec = np.empty([size, Nsims, 10000//lb], dtype=np.float64)
    lrec = np.empty([size, Nsims, 10000//lb], dtype=np.float64)
    qerec = np.empty([size, Nsims, Nphi], dtype=np.float64)
    N0rec = np.empty([size, Nsims, Nphi], dtype=np.float64)
    RDN0rec = np.empty([size, Nsims, Nphi], dtype=np.float64)
comm.Gather(outs['unlensed'], urec, root=0)
comm.Gather(outs['lensed'], lrec, root=0)
comm.Gather(outs['reconst'], qerec, root=0)
comm.Gather(outs['N0'], N0rec, root=0)
comm.Gather(outs['RDN0'], RDN0rec, root=0)

if rank == 0:
    outs['unlensed'] = np.concatenate(urec, axis=0)
    outs['lensed'] = np.concatenate(lrec, axis=0)
    outs['reconst'] = np.concatenate(qerec, axis=0)
    outs['N0'] = np.concatenate(N0rec, axis=0)
    outs['RDN0'] = np.concatenate(RDN0rec, axis=0)

    with open(loc+fn, 'wb') as p:
        pickle.dump(outs, p)