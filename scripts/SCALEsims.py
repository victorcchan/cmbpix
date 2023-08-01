"""SCALE simulations

This script simulates CMB lensing maps according to a relatively vanilla 
cosmological model, and recovers the underlying lensing information using 
the SCALE methodology. The SCALE output(s) along with a copy of the 
normalization factors ALv, and expected output Psi_Lv are saved. Many options 
are available to allow for flexibility in the simulations, SCALE configuration, 
and even applying/saving analogous analysis using the Hu, DeDeo and Vale 
quadratic estimator.

The general flow of the script is as follows:
    * Import relevant packages
        * mpi4py is required for running simulations on multiple cores
        * SCALE submodule in cmbpix.lensing contains the relevant functions
        * numpy is used throughout the script
        * camb is used to generate CMB power spectra for simulated maps
        * pixell is used to generate and manipulate flatsky CMB maps
        * symlens is used to apply quadratic estimators if desired
        * orphics is used to manipulate QE outputs from symlens
        * scipy.interpolate's interp1d method is used to prepare
            inputs for symlens
        * pickle is used to save outputs
        * argparse is used to interpret options from the command prompt
        * pymaster is required if using masking options
    * Set up MPI workflow
    * Read in options from argparse
    * Generate CMB power spectra/lensing spectra with CAMB
    * If applicable, set up quadratic estimator
        * Compute realization-independent N_L
    * Loop over simulations/realizations in each process/core
        * Generate unlensed CMB T map, lens potential (phi) map, noise map
        * Apply lensing to T map with phi map
        * On first loop, compute normalization A_Lv, and expected Psi_Lv
        * Run SCALE on both unlensed and lensed realizations, save C_Lv^ls
        * If applicable, run the quadratic estimator on lensed realization
    * Gather SCALE/QE outputs from all processes to 0th process
    * Save outputs
"""

import mpi4py.rc ## These first 3 lines allow mpi4py to work on Scinet
mpi4py.rc.threads = False
mpi4py.rc.finalize = True
from mpi4py import MPI
import numpy as np
from pixell import enmap, utils, lensing
from cmbpix.lensing import SCALE
from symlens import qe, interp
from orphics.maps import mask_kspace, FourierCalc, binned_power
from orphics.stats import bin2D
from scipy.interpolate import interp1d
import camb
import pickle
import argparse

import warnings
warnings.filterwarnings(action='ignore') ## Change to once for first time

## MPI stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## Test params
parser = argparse.ArgumentParser(
    description="Simulate CMB realizations and quantify lensing")
parser.add_argument("--DLv", 
    required=True, 
    nargs='+',
    help="Size of Lv binning for SCALE estimator")
parser.add_argument("--l1min", 
    required=True, 
    type=int, 
    nargs='+',
    help="Lower l1 limit for high-pass filter")
parser.add_argument("--l1max", 
    required=True, 
    type=int, 
    nargs='+',
    help="Upper l1 limit for high-pass filter")
parser.add_argument("--l2min", 
    required=False, 
    default=0, 
    type=int, 
    help="Lower l2 limit for low-pass filter")
parser.add_argument("--l2max", 
    required=False, 
    default=3000, 
    type=int, 
    help="Upper l2 limit for low-pass filter")
parser.add_argument("--w", 
    required=False, 
    default=1.0, 
    type=float, 
    help="White noise level in uK-arcmin")
parser.add_argument("--b", 
    required=False, 
    default=1.0, 
    type=float, 
    help="Beam size (FWHM) in arcmin")
parser.add_argument("--res", 
    required=False, 
    default=0.5, 
    type=float, 
    help="Resolution of simulated maps in arcmin")
parser.add_argument("--width", 
    required=False, 
    default=10, 
    type=float, 
    help="Map width in degrees")
parser.add_argument("--height", 
    required=False, 
    default=10, 
    type=float, 
    help="Map height in degrees")
parser.add_argument("--Nsims", 
    required=False, 
    default=100, 
    type=int, 
    help="Number of realizaitons to simulate per process")
parser.add_argument("--qe", 
    required=False, 
    action="store_true", 
    help="If used, run QE and save with outputs")
parser.add_argument("--lpa", 
    required=False, 
    default=8, 
    type=int, 
    help="Lens Potential Accuracy parameter for CAMB (>=8 recommended)")
parser.add_argument("--phimin", 
    required=False, 
    default=0, 
    help="Minimum for window filter for lens potential")
parser.add_argument("--phimax", 
    required=False, 
    default=-1, 
    help="Maximum for window filter for lens potential")
parser.add_argument("--delens", 
    required=False, 
    action="store_true", 
    help="If used, use unlensed map to generate lambda maps.")
parser.add_argument("--theory", 
    required=False, 
    action="store_true", 
    help="If used, compute theory integrals for AL, PsiL.")
parser.add_argument("--mask", 
    required=False, 
    type=float, 
    default=0, 
    help="Size of mask apodization in deg.")
args = parser.parse_args()
DLv = np.array(args.DLv, dtype=int) ## Lv bin size
l1min = np.array(args.l1min, dtype=int)
l1max = np.array(args.l1max, dtype=int)
l2min = args.l2min
l2max = args.l2max
w = args.w ## In uK-arcmin
b = args.b ## In arcmin
width = args.width
height = args.height
reso = args.res
Nsims = args.Nsims
pmin = int(args.phimin)
pmax = int(args.phimax)
lpa = args.lpa
ints = args.theory


if rank == 0: # Format some strings to create unique filename
    wn = str(w).replace('.', 'p')
    bn = str(b).replace('.', 'p')
    rn = str(reso).replace('.', 'p')
    mn = str(args.mask).replace('.', 'p')
    DLn = str(DLv)[1:-1].replace('  ', '-').replace(' ', '-')
    if DLn[0] == '-':
        DLn = DLn[1:]
    loc = '/scratch/p/pen/victorc/lensing/CLGK/' ## Change this for file location
    ## Construct unique filename based on argparse options
    l1minstr = ','.join(l1min.astype(str))
    l1maxstr = ','.join(l1max.astype(str))
    l1str = '_{}-{}l1'.format(l1minstr, l1maxstr)
    l2str = '_{}-{}l2'.format(l2min, l2max)
    DLvstr = '_{}DLv'.format(DLn)
    nstr = '_{}uKarcmin_{}arcmin'.format(wn, bn)
    lensstr = '_{}lpa_{}-{}phi'.format(lpa, pmin, pmax)
    mapstr  ='_{}sqdeg_{}arcmin_{}degmask_{}Nsims'.format(int(width*height), rn, mn, 
                                                    int(Nsims*size))
    end = '.pkl'
    if args.delens:
        end = '_delensed' + end
    if args.qe:
        end = '_wQE' + end
    fn = 'SCALE'+l1str+l2str+DLvstr+nstr+lensstr+mapstr+end

## Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
## This function sets up CosmoMC-like settings, 
## with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2.1e-9, ns=0.965, r=0)
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

## Noise spectrum
ntt = (w*np.pi/180./60.)**2. * np.exp((b*np.pi/180./60. / np.sqrt(8.*np.log(2)))**2.*ls**2.)

## Replace potential nan here
ctt_unlensed[:2] = 1e-20
ctt_lensed[:2] = 1e-20
cphiphi[:2] = 1e-20
ntt[:2] = 1e-20
if w == 0 and b == 0:
    ntt += 1e-20
ctt_total = ctt_lensed + ntt

## Simulation size/resolution
box = np.array([[-height/2,width/2],[height/2,-width/2]]) * utils.degree
shape, wcs = enmap.geometry(pos=box, res=reso*utils.arcmin, proj='plain')
modlmap = enmap.modlmap(shape, wcs)
# Create mask if necessary
mask = np.ones(shape)
if args.mask > 0:
    import pymaster as nmt
    mask[0,:] = 0. # Trim edges
    mask[-1,:] = 0.
    mask[:,0] = 0.
    mask[:,-1] = 0.
    mask = nmt.mask_apodization_flat(mask, width*np.pi/180., height*np.pi/180., aposize=args.mask, apotype="C1")

## Binner for data power spectrum
Dbins = np.arange(20,20000,20)
Dbinner = bin2D(modlmap,Dbins)
lowl = np.linspace(1.5, 30.5, 30)
midl = np.arange(31, 311, 10, dtype=int)
highl = np.arange(301, 10000, 36, dtype=int)
Dbinlo = bin2D(modlmap,lowl)
Dbinmd = bin2D(modlmap,midl)
Dbinhi = bin2D(modlmap,highl)
# Dbins2 = np.arange(0, 12000, 36, dtype=int)
# Dbinner2 = bin2D(modlmap,Dbins2)

## Initialize objects for QE
XY='TT'
UV='TT'
ellmin = 2 ; ellmax = 10000
Lmin = 100 ; Lmax = 10000
xmask = mask_kspace(shape,wcs,lmin=2,lmax=3000)
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
## For TT cross-correlation/covariance
tt_cent_lo, ctt_bin_lo = Dbinlo.bin(interp(ls, ctt_total)(modlmap))
tt_cent_md, ctt_bin_md = Dbinmd.bin(interp(ls, ctt_total)(modlmap))
tt_cent_hi, ctt_bin_hi = Dbinhi.bin(interp(ls, ctt_total)(modlmap))
tt_cents = np.concatenate((tt_cent_lo, tt_cent_md, tt_cent_hi))
ctt_bins = np.concatenate((ctt_bin_lo, ctt_bin_md, ctt_bin_hi))
# cents_TT, tt_theory_bin = Dbinner2.bin(interp(ls, ctt_total)(modlmap))

## Output data goes here
outs = {'L_qe': cents_phi, 
        'reconst': np.empty([Nsims, Nphi], dtype=np.float64), 
        'phi_real': np.empty([Nsims, Nphi], dtype=np.float64), 
        'k_real': np.empty([Nsims, Nphi], dtype=np.float64), 
        'k_cross': np.empty([Nsims, Nphi], dtype=np.float64), 
        'RDN0': np.empty([Nsims, Nphi], dtype=np.float64), 
        'phi_theory': pp_theory_bin, 
        'k_theory': kk_theory_bin, 
        'l_TT': tt_cents, 
        'TT_theory': ctt_bins, 
        'TT_total': np.empty([Nsims, tt_cents.size], dtype=np.float64)}

## If doing multiple binnings, add separately
for lbin in DLv:
    for l in range(len(l1min)):
        unlenkey = 'CLv_unlensed_DLv{}_{}-{}'.format(lbin,l1min[l],l1max[l])
        lenkey = 'CLv_lensed_DLv{}_{}-{}'.format(lbin,l1min[l],l1max[l])
        outs[unlenkey] = np.empty([Nsims, l1max[l]//lbin], dtype=np.float64)
        outs[lenkey] = np.empty([Nsims, l1max[l]//lbin], dtype=np.float64)
## Loop over sims
for i in range(Nsims):
    # if i % (Nsims//10) == 0:
    #     print("{}% complete on rank {}".format(i, rank), flush=True)
    ## Generate maps
    Tmap = enmap.rand_map((1,) + shape, wcs, 
        interp(ls, ctt_unlensed)(modlmap)[np.newaxis, np.newaxis, :, :] )[0]
    phimap = enmap.rand_map((1,) + shape, wcs, 
        interp(ls, cphiphi)(modlmap)[np.newaxis, np.newaxis, :, :] )[0]
    nmap = enmap.rand_map((1,) + shape, wcs, 
        interp(ls, ntt)(modlmap)[np.newaxis, np.newaxis, :, :] )[0]
    lTmap = lensing.lens_map_flat(Tmap, phimap)
    tt_cent_lo, ctt_data_lo = Dbinlo.bin(np.abs(enmap.fft(lTmap+nmap, 
                normalize='phys'))**2)
    tt_cent_md, ctt_data_md = Dbinmd.bin(np.abs(enmap.fft(lTmap+nmap, 
                normalize='phys'))**2)
    tt_cent_hi, ctt_data_hi = Dbinhi.bin(np.abs(enmap.fft(lTmap+nmap, 
                normalize='phys'))**2)
    # cents_TT, tt_tot = Dbinner2.bin(np.abs(enmap.fft(lTmap+nmap, 
    #         normalize='phys'))**2)
    outs['TT_total'][i,:]  = np.concatenate((ctt_data_lo, ctt_data_md, 
                                                ctt_data_hi))
    if args.qe: ## Apply QE to lensed+noise map if option selected
        ## Compute power spectrum for this realization
        dcents, C_l_tot = Dbinner.bin(np.abs(enmap.fft(lTmap+nmap, 
            normalize='phys'))**2)
        ## Set up dictionary for QE
        feed_dict['X'] = enmap.fft(lTmap+nmap, normalize='phys')
        feed_dict['Y'] = feed_dict['X']
        feed_dict['dC_T_T'] = interp1d(dcents, C_l_tot, 
            bounds_error=False, fill_value=0)(modlmap)

        ## Run QE on the lensed realization, then run CLGK estimator
        my_reconst = qe.reconstruct(shape, wcs, feed_dict, 'hdv', XY, 
            xmask= xmask, ymask = ymask)
        my_A_l = qe.A_l(shape, wcs, feed_dict, 'hdv', XY, xmask, ymask)
        rdn0_2d = qe.RDN0_analytic(shape,wcs,feed_dict,'hdv',XY,
            'hdv',UV,xmask=xmask,ymask=ymask,kmask=kmask)
        cents_phi, reconst_bin = binner_phi.bin(np.abs(my_reconst)**2)
        cents_phi, RDN0_l_bin = binner_phi.bin(rdn0_2d)
        Krecon = enmap.ifft(my_reconst, normalize='phys')
        Kcor = FourierCalc(shape, wcs)
        dy_p, dx_p = enmap.grad(phimap) ## Compute spectra of actual realization
        dyy_p, dyx_p = enmap.grad(dy_p) ## for comparison
        dxy_p, dxx_p = enmap.grad(dx_p)
        k_map = -0.5 * (dyy_p + dxx_p)
        cents_phi, kk_real_bin = binned_power(k_map, bin_edges_phi)
        cents_phi, pp_real_bin = binned_power(phimap, bin_edges_phi)
        cross2d = Kcor.power2d(Krecon, k_map) ## Cross-spectrum for comparing
        cents_phi, cross_bin = binner_phi.bin(cross2d[0])

    ## Compute normalization and prediction on first pass only
    for lbin in DLv:
        for l in range(len(l1min)):
            unlenkey = 'CLv_unlensed_DLv{}_{}-{}'.format(lbin,l1min[l],l1max[l])
            lenkey = 'CLv_lensed_DLv{}_{}-{}'.format(lbin,l1min[l],l1max[l])
            if i == 0: ## No need to delens unlensed maps
                ## Unlensed cphiphi should technically be 0, 
                ## but this doesn't affect anything we keep
                ucents, up1d, AL, Psi = SCALE.SCALE((Tmap+nmap)*mask, map_delens=None, 
                    l1min=l1min[l], l1max=l1max[l], l2min=l2min, l2max=l2max, 
                    DLv=lbin, uCl=ctt_unlensed, lCl=ctt_unlensed, Nl=ntt, 
                    Clpp=cphiphi, w=w, b=b, compute_bias=ints)
                if args.qe: ## Optimal noise is independent of realization; do once
                    N_l_phi = qe.N_l_optimal(shape,wcs,feed_dict,'hdv',XY,
                        xmask=xmask,ymask=ymask,field_names=None,kmask=kmask)
                    cents_phi, N0_l_bin = binner_phi.bin(N_l_phi)
                    outs['N0'] = N0_l_bin
                outs['Lv_DLv{}'.format(lbin)] = ucents
                if ints:
                    outs['ALv_unlensed_DLv{}_{}-{}'.format(lbin,l1min[l],l1max[l])] = AL
                if args.delens: ## Delensing option --> Pass in unlensed map as well
                    lcents, lp1d, AL, Psi = SCALE.SCALE((lTmap+nmap)*mask, (Tmap+nmap)*mask, 
                        l1min=l1min[l], l1max=l1max[l], l2min=l2min, l2max=l2max, 
                        DLv=lbin, uCl=ctt_unlensed, lCl=ctt_lensed, Nl=ntt, 
                        Clpp=cphiphi, w=w, b=b, compute_bias=ints)
                else:
                    lcents, lp1d, AL, Psi = SCALE.SCALE((lTmap+nmap)*mask, map_delens=None, 
                        l1min=l1min[l], l1max=l1max[l], l2min=l2min, l2max=l2max, 
                        DLv=lbin, uCl=ctt_unlensed, lCl=ctt_lensed, Nl=ntt, 
                        Clpp=cphiphi, w=w, b=b, compute_bias=ints)
                if ints:
                    outs['ALv_lensed_DLv{}_{}-{}'.format(lbin,l1min[l],l1max[l])] = AL
                    outs['PsiLv_DLv{}_{}-{}'.format(lbin,l1min[l],l1max[l])] = Psi
                if l == len(l1min)-1:
                    print("Rank {} done initial step for Lbin {}".format(rank, lbin), flush=True)
            else:
                ucents, up1d,_,_ = SCALE.SCALE((Tmap+nmap)*mask, map_delens=None, 
                    l1min=l1min[l], l1max=l1max[l], l2min=l2min, l2max=l2max, 
                    DLv=lbin, uCl=ctt_unlensed, lCl=ctt_unlensed, Nl=ntt, 
                    Clpp=cphiphi, w=w, b=b, compute_bias=False)
                if args.delens:
                    lcents, lp1d,_,_ = SCALE.SCALE((lTmap+nmap)*mask, (Tmap+nmap)*mask, 
                        l1min=l1min[l], l1max=l1max[l], l2min=l2min, l2max=l2max, 
                        DLv=lbin, uCl=ctt_unlensed, lCl=ctt_lensed, Nl=ntt, 
                        Clpp=cphiphi, w=w, b=b, compute_bias=False)
                else:
                    lcents, lp1d,_,_ = SCALE.SCALE((lTmap+nmap)*mask, map_delens=None, 
                        l1min=l1min[l], l1max=l1max[l], l2min=l2min, l2max=l2max, 
                        DLv=lbin, uCl=ctt_unlensed, lCl=ctt_lensed, Nl=ntt, 
                        Clpp=cphiphi, w=w, b=b, compute_bias=False)

            outs[unlenkey][i,:] = up1d
            outs[lenkey][i,:] = lp1d
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
## Gather results to one process, and save
if rank == 0:
    print("Done saving indivudual files, setting up MPI Gather", flush=True)
    if args.qe:
        qerec = np.empty([size, Nsims, Nphi], dtype=np.float64)
        N0rec = np.empty([size, Nsims, Nphi], dtype=np.float64)
        RDN0rec = np.empty([size, Nsims, Nphi], dtype=np.float64)
        prealrec = np.empty([size, Nsims, Nphi], dtype=np.float64)
        krealrec = np.empty([size, Nsims, Nphi], dtype=np.float64)
        kcrossrec = np.empty([size, Nsims, Nphi], dtype=np.float64)
    Cttrec = np.empty([size, Nsims, tt_cents.size], dtype=np.float64)
else:
    urec = None
    lrec = None
    qerec = None
    N0rec = None
    RDN0rec = None
    prealrec = None
    krealrec = None
    kcrossrec = None
    Cttrec = None
comm.Barrier()
comm.Gather(outs['TT_total'], Cttrec, root=0)
if args.qe:
    comm.Gather(outs['reconst'], qerec, root=0)
    comm.Gather(outs['RDN0'], RDN0rec, root=0)
    comm.Gather(outs['phi_real'], prealrec, root=0)
    comm.Gather(outs['k_real'], krealrec, root=0)
    comm.Gather(outs['k_cross'], kcrossrec, root=0)
comm.Barrier()
for lbin in DLv:
    for l in range(len(l1min)):
        unlenkey = 'CLv_unlensed_DLv{}_{}-{}'.format(lbin,l1min[l],l1max[l])
        lenkey = 'CLv_lensed_DLv{}_{}-{}'.format(lbin,l1min[l],l1max[l])
        if rank == 0:
            urec = np.empty([size, Nsims, l1max[l]//lbin], dtype=np.float64)
            lrec = np.empty([size, Nsims, l1max[l]//lbin], dtype=np.float64)
        else:
            urec = None
            lrec = None
        comm.Barrier()
        comm.Gather(outs[unlenkey], urec, root=0)
        comm.Gather(outs[lenkey], lrec, root=0)
        comm.Barrier()
        if rank == 0:
            outs[unlenkey] = np.concatenate(urec, axis=0)
            outs[lenkey] = np.concatenate(lrec, axis=0)


if rank == 0:
    print("Saving outputs", flush=True)
    outs['TT_total'] = np.concatenate(Cttrec, axis=0)
    if args.qe:
        outs['reconst'] = np.concatenate(qerec, axis=0)
        outs['RDN0'] = np.concatenate(RDN0rec, axis=0)
        outs['phi_real'] = np.concatenate(prealrec, axis=0)
        outs['k_real'] = np.concatenate(krealrec, axis=0)
        outs['k_cross'] = np.concatenate(kcrossrec, axis=0)

    with open(loc+fn, 'wb') as p:
        pickle.dump(outs, p)
