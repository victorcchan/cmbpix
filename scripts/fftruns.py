import mpi4py.rc # These first 3 lines allow mpi4py to work on Scinet
mpi4py.rc.threads = False
mpi4py.rc.finalize = True
from mpi4py import MPI
import numpy as np
from pixell import enmap, utils, lensing
from cmbpix.lensing import fftest as fe
from symlens import qe, interp
from orphics.maps import mask_kspace, FourierCalc, binned_power
from orphics.stats import bin2D
from scipy.interpolate import interp1d
import camb
import pickle
import argparse

import warnings
warnings.filterwarnings(action='ignore')# Change to once for first time

# MPI stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Test params
parser = argparse.ArgumentParser(
    description="Simulate CMB realizations and quantify lensing")
parser.add_argument("-lb", 
    required=True, 
    nargs='+',
    help="Size of L binning for CLGK estimator")
parser.add_argument("--lmin", 
    required=False, 
    default=6000, 
    type=int, 
    help="Minimum l for high-pass filter")
parser.add_argument("--lmax", 
    required=False, 
    default=8000, 
    type=int, 
    help="Maximum l for high-pass filter")
parser.add_argument("--ldT", 
    required=False, 
    default=3000, 
    type=int, 
    help="Maximum l for low-pass filter")
parser.add_argument("--w", 
    required=False, 
    default=1.0, 
    type=float, 
    help="White noise level in uK-arcmin")
parser.add_argument("--b", 
    required=False, 
    default=1.0, 
    type=float, 
    help="Beam size in arcmin")
parser.add_argument("--res", 
    required=False, 
    default=0.5, 
    type=float, 
    help="Resolution of simulated maps")
parser.add_argument("--width", 
    required=False, 
    default=10, 
    type=float, 
    help="Map width in degrees")
parser.add_argument("--Nsims", 
    required=False, 
    default=100, 
    type=int, 
    help="Number of realizaitons to simulate per process")
parser.add_argument("--qe", 
    required=False, 
    action="store_true", 
    help="If true, run QE and save with outputs")
parser.add_argument("--lpa", 
    required=False, 
    default=8, 
    type=int, 
    help="Lens Potential Accuracy parameter for CAMB")
parser.add_argument("--phimin", 
    required=False, 
    default=0, 
    help="Minimum for window filter for lens potential")
parser.add_argument("--phimax", 
    required=False, 
    default=-1, 
    help="Maximum for window filter for lens potential")
args = parser.parse_args()
lb = np.array(args.lb, dtype=int) # l bin size
lmin = args.lmin
lmax = args.lmax
ldT = args.ldT
w = args.w # In uK-arcmin
b = args.b # In arcmin
Nsims = args.Nsims
pmin = int(args.phimin)
pmax = int(args.phimax)
lpa=args.lpa
# lb = 500 # l bin size
# lmin = 6000
# lmax = 8000
# ldT = 3000
# w = 1.0 # In uK-arcmin
# b = 1.0 # In arcmin

if rank == 0:
    wn = str(w).replace('.', 'p')
    bn = str(b).replace('.', 'p')
    loc = '/scratch/p/pen/victorc/lensing/CLGK'
    fn = '/FFTout_{}ldT_{}lb_{}-{}_{}uKarcmin_{}beam_lpa{}_{}-{}phi_{}sqdeg.pkl'.format(ldT, 
                                                                   lb, lmin, lmax, wn, bn, lpa, pmin, pmax, args.width*10)

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
pars.set_for_lmax(20000, lens_potential_accuracy=lpa);
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
cphiphi[:pmin] = 0.
cphiphi[pmax:] = 0.

# Noise spectrum
# w2 = (w * np.pi / 10800)**2
# beam = 1.0 * np.pi / 10800 / (2 * np.sqrt(2*np.log(2)))
# Wl = np.exp(-ell*(ell+1)*beam**2)
# ntt = w2 / Wl
Delta_T = args.w  # In muK-arcmin
theta_fwhm = args.b  # in arcmin
ntt = (Delta_T*np.pi/180./60.)**2. * np.exp((theta_fwhm*np.pi/180./60. / np.sqrt(8.*np.log(2)))**2.*ls**2.)

ctt_total = ctt_lensed + ntt
# Replace potential nan here
ctt_unlensed[0] = 0
ctt_lensed[0] = 0
cphiphi[0] = 0

# Simulation size/resolution
width = args.width
reso = args.res
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
xmask = mask_kspace(shape,wcs,lmin=ellmin,lmax=2000)
ymask = mask_kspace(shape,wcs,lmin=ellmin,lmax=ellmax)
kmask = mask_kspace(shape,wcs,lmin=Lmin,lmax=Lmax)
feed_dict = {}
feed_dict['uC_T_T'] = interp(ls, ctt_lensed)(modlmap)
feed_dict['tC_T_T'] = interp(ls, ctt_total)(modlmap)
feed_dict['nC_T_T'] = interp(ls, ctt_total)(modlmap)
bin_edges_phi = np.arange(100,Lmax,100)
binner_phi = bin2D(modlmap,bin_edges_phi)
Nphi = bin_edges_phi.size - 1
cents_phi, pp_theory_bin = binner_phi.bin(interp(ls,cphiphi)(modlmap))
cents_phi, kk_theory_bin = binner_phi.bin(interp(ls, 2*np.pi*powers['lens_potential'][:,0]/4.)(modlmap))

# Output data goes here
outs = {'cents_qe': cents_phi, 
        'reconst': np.empty([Nsims, Nphi], dtype=np.float64), 
        'phi_real': np.empty([Nsims, Nphi], dtype=np.float64), 
        'k_real': np.empty([Nsims, Nphi], dtype=np.float64), 
        'k_cross': np.empty([Nsims, Nphi], dtype=np.float64), 
        'RDN0': np.empty([Nsims, Nphi], dtype=np.float64), 
        'phi_theory': pp_theory_bin, 
        'k_theory': kk_theory_bin}

# If doing multiple binnings, add separately
for lbin in lb:
    outs['unlensed{}'.format(lbin)] = np.empty([Nsims, 10000//lbin], dtype=np.float64)
    outs['lensed{}'.format(lbin)] = np.empty([Nsims, 10000//lbin], dtype=np.float64)
# Sims
for i in range(Nsims):
    if i % 10 == 0:
        print("{}% complete on rank {}".format(i, rank), flush=True)
    # Generate maps
    Tmap = enmap.rand_map((1,) + shape, wcs, 
        interp(ls, ctt_unlensed)(modlmap)[np.newaxis, np.newaxis, :, :] )[0]
    phimap = enmap.rand_map((1,) + shape, wcs, 
        interp(ls, cphiphi)(modlmap)[np.newaxis, np.newaxis, :, :] )[0]
    nmap = enmap.rand_map((1,) + shape, wcs, 
        interp(ls, ntt)(modlmap)[np.newaxis, np.newaxis, :, :] )[0]
    # Tmap = enmap.rand_map(shape, wcs, ctt_unlensed)
    # phimap = enmap.rand_map(shape, wcs, cphiphi)
    # nmap = enmap.rand_map(shape, wcs, ntt)
    lTmap = lensing.lens_map_flat(Tmap, phimap)
    if args.qe:
        # Compute power spectrum for this realization
        dcents, C_l_tot = Dbinner.bin(np.abs(enmap.fft(lTmap+nmap, 
            normalize='phys'))**2)
        # Set up dictionary for QE
        feed_dict['X'] = enmap.fft(lTmap+nmap, normalize='phys')
        feed_dict['Y'] = feed_dict['X']
        feed_dict['dC_T_T'] = interp1d(dcents, C_l_tot, 
            bounds_error=False, fill_value=0)(modlmap)

        # Run QE on the lensed realization, then run CLGK estimator
        my_reconst = qe.reconstruct(shape, wcs, feed_dict, 'hdv', XY, 
            xmask= xmask, ymask = ymask)
        my_A_l = qe.A_l(shape, wcs, feed_dict, 'hdv', XY, xmask, ymask)
        rdn0_2d = qe.RDN0_analytic(shape,wcs,feed_dict,'hdv',XY,
            'hdv',UV,xmask=xmask,ymask=ymask,kmask=kmask)
        cents_phi, reconst_bin = binner_phi.bin(np.abs(my_reconst)**2)
        cents_phi, RDN0_l_bin = binner_phi.bin(rdn0_2d)
        Krecon = enmap.ifft(my_reconst, normalize='phys')
        Kcor = FourierCalc(shape, wcs)
        dy_p, dx_p = enmap.grad(phimap)
        dyy_p, dyx_p = enmap.grad(dy_p)
        dxy_p, dxx_p = enmap.grad(dx_p)
        k_map = -0.5 * (dyy_p + dxx_p)
        cents_phi, kk_real_bin = binned_power(k_map, bin_edges_phi)
        cents_phi, pp_real_bin = binned_power(phimap, bin_edges_phi)
        cross2d = Kcor.power2d(Krecon, k_map)
        cents_phi, cross_bin = binner_phi.bin(cross2d[0])

    # Compute normalization and prediction on first pass only
    for lbin in lb:
        if i == 0:
            ucents, up1d, AL, Phi = fe.FFTest(Tmap+nmap, 
                ldT=ldT, lmin=lmin, lmax=lmax, lbins=lbin, 
                uCls=ctt_unlensed, lCls=ctt_lensed, Nls=ntt, Clpp=cphiphi, 
                w=w, sg=b, apply_bias=True, plots=False)
            if args.qe:
                N_l_phi = qe.N_l_optimal(shape,wcs,feed_dict,'hdv',XY,
                    xmask=xmask,ymask=ymask,field_names=None,kmask=kmask)
                cents_phi, N0_l_bin = binner_phi.bin(N_l_phi)
                outs['N0'] = N0_l_bin
            outs['cents{}'.format(lbin)] = ucents
            outs['AL_unlensed{}'.format(lbin)] = AL
            # outs['BL'] = BL

            lcents, lp1d = fe.FFTest(lTmap+nmap, 
                ldT=ldT, lmin=lmin, lmax=lmax, lbins=lbin, 
                uCls=ctt_unlensed, lCls=ctt_lensed, Nls=ntt, Clpp=cphiphi, 
                w=w, sg=b, apply_bias=False, plots=False)
            outs['AL_lensed{}'.format(lbin)] = AL
            outs['Phi{}'.format(lbin)] = Phi
            print("Rank {} done initial step for Lbin {}".format(rank, lbin), flush=True)
        else:
            ucents, up1d = fe.FFTest(Tmap+nmap, 
                ldT=ldT, lmin=lmin, lmax=lmax, lbins=lbin, 
                uCls=ctt_unlensed, lCls=ctt_lensed, Nls=ntt, Clpp=cphiphi, 
                w=w, sg=b, apply_bias=False, plots=False)

            lcents, lp1d = fe.FFTest(lTmap+nmap, 
                ldT=ldT, lmin=lmin, lmax=lmax, lbins=lbin, 
                uCls=ctt_unlensed, lCls=ctt_lensed, Nls=ntt, Clpp=cphiphi, 
                w=w, sg=b, apply_bias=False, plots=False)

        outs['unlensed{}'.format(lbin)][i,:] = up1d
        outs['lensed{}'.format(lbin)][i,:] = lp1d
    if args.qe:
        outs['reconst'][i,:] = reconst_bin
        outs['phi_real'][i,:] = pp_real_bin
        outs['k_real'][i,:] = kk_real_bin
        outs['k_cross'][i,:] = cross_bin
        outs['RDN0'][i,:] = RDN0_l_bin
    if rank == 0 and i == 0:
        print("Finished 1 loop", flush=True)

if rank == 0:
    print("Rank 0 all done", flush=True)
comm.Barrier()
# specsize = outs['unlensed'][0].size
#print("Saving individual files", flush=True)
#with open(loc+fn, 'wb') as p:
#    pickle.dump(outs, p)
comm.Barrier()
if rank == 0:
    print("Done saving indivudual files, setting up MPI Gather", flush=True)
    if args.qe:
        qerec = np.empty([size, Nsims, Nphi], dtype=np.float64)
        N0rec = np.empty([size, Nsims, Nphi], dtype=np.float64)
        RDN0rec = np.empty([size, Nsims, Nphi], dtype=np.float64)
        prealrec = np.empty([size, Nsims, Nphi], dtype=np.float64)
        krealrec = np.empty([size, Nsims, Nphi], dtype=np.float64)
        kcrossrec = np.empty([size, Nsims, Nphi], dtype=np.float64)
else:
    urec = None
    lrec = None
    qerec = None
    N0rec = None
    RDN0rec = None
    prealrec = None
    krealrec = None
    kcrossrec = None
comm.Barrier()
if args.qe:
    comm.Gather(outs['reconst'], qerec, root=0)
    comm.Gather(outs['RDN0'], RDN0rec, root=0)
    comm.Gather(outs['phi_real'], prealrec, root=0)
    comm.Gather(outs['k_real'], krealrec, root=0)
    comm.Gather(outs['k_cross'], kcrossrec, root=0)
comm.Barrier()
for lbin in lb:
    if rank == 0:
        urec = np.empty([size, Nsims, 10000//lbin], dtype=np.float64)
        lrec = np.empty([size, Nsims, 10000//lbin], dtype=np.float64)
    else:
        urec = None
        lrec = None
    comm.Barrier()
    comm.Gather(outs['unlensed{}'.format(lbin)], urec, root=0)
    comm.Gather(outs['lensed{}'.format(lbin)], lrec, root=0)
    comm.Barrier()
    if rank == 0:
        outs['unlensed{}'.format(lbin)] = np.concatenate(urec, axis=0)
        outs['lensed{}'.format(lbin)] = np.concatenate(lrec, axis=0)


if rank == 0:
    print("Saving outputs", flush=True)
    if args.qe:
        outs['reconst'] = np.concatenate(qerec, axis=0)
        outs['RDN0'] = np.concatenate(RDN0rec, axis=0)
        outs['phi_real'] = np.concatenate(prealrec, axis=0)
        outs['k_real'] = np.concatenate(krealrec, axis=0)
        outs['k_cross'] = np.concatenate(kcrossrec, axis=0)

    with open(loc+fn, 'wb') as p:
        pickle.dump(outs, p)
