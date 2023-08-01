import numpy as np
from cmbpix.lensing import SCALE
import camb
import pickle
import argparse
import time

import warnings
warnings.filterwarnings(action='ignore') ## Change to once for first time

# Integral params
parser = argparse.ArgumentParser(
    description="Compute SCALE normalization and expectation")
parser.add_argument("--DLv", 
    required=False, 
    default=20, 
    type=int, 
    help="Lcheck spacing")
parser.add_argument("--dl1", 
    required=False, 
    default=100, 
    type=int, 
    help="Spacing for l1 integral")
parser.add_argument("--dl2", 
    required=False, 
    default=100, 
    type=int, 
    help="Spacing for l2 integral")
parser.add_argument("--l1min", 
    required=False, 
    default=6000, 
    type=int, 
    help="Lower bound for l1 integral")
parser.add_argument("--l1max", 
    required=False, 
    default=8000, 
    type=int, 
    help="Upper bound for l1 integral")
parser.add_argument("--l2min", 
    required=False, 
    default=0, 
    type=int, 
    help="Lower bound for l2 integral")
parser.add_argument("--l2max", 
    required=False, 
    default=3000, 
    type=int, 
    help="Upper bound for l2 integral")
parser.add_argument("--nonoise", 
    required=False, 
    action="store_true", 
    help="If used, no noise.")
parser.add_argument("--w", 
    required=False, 
    default=1.0, 
    type=float, 
    help="Noise level in uK-arcmin")
parser.add_argument("--b", 
    required=False, 
    default=1.0, 
    type=float, 
    help="Beam FWHM in arcmin")
args = parser.parse_args()

loc = './' ## Change this for file location
fn = 'SCALEintegral_{}DLv_{}dl1{}-{}_{}dl2{}-{}_{}w{}b.pkl'.format(args.DLv, args.dl1,
    args.l1min, args.l1max, args.dl2, args.l2min, args.l2max, args.w, args.b)
if args.nonoise:
    fn = fn[:-4] + '_nonoise.pkl'

## Set up a new set of parameters for CAMB
print("Running CAMB", flush=True)
pars = camb.CAMBparams()
## This function sets up CosmoMC-like settings, 
## with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(20000, lens_potential_accuracy=8);
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

## Noise spectrum
ntt = (args.w*np.pi/180./60.)**2. * np.exp((args.b*np.pi/180./60. / np.sqrt(8.*np.log(2)))**2.*ls**2.)
if args.nonoise:
    ntt = np.zeros(ntt.size) + 1e-20
## Replace potential nan here
ntt[0] = 1e-20
ctt_unlensed[0] = 1e-20
ctt_lensed[0] = 1e-20
cphiphi[0] = 1e-20

ctt_total = ctt_lensed + ntt
print("Spectra computed, computing SCALE theory", flush=True)

nLv = 3000 // args.DLv
Lv = np.arange(args.DLv/2, (nLv+1)*args.DLv, args.DLv, dtype=int)
# Lv = np.arange(0, nLv*args.DLv+1, args.DLv)

tstart = time.time()

ALv, PsiLv = SCALE.CalcBiasExp(ctt_unlensed, ctt_total, cphiphi, 
	l1min=args.l1min, l1max=args.l1max, l2min=args.l2min, l2max=args.l2max, 
    Lv=Lv, dl1=args.dl1, dl2=args.dl2, useC=True)

tend = time.time()

print("Time taken: {}s".format(tend-tstart), flush=True)

out = {"Lv": Lv, 
	   "ALv": ALv, 
	   "PsiLv": PsiLv}

with open(loc+fn, 'wb') as p:
	pickle.dump(out, p)