import numpy as np
from cmbpix.lensing import SCALE
import camb
import pickle
import argparse
from scipy.stats.qmc import LatinHypercube

import warnings
warnings.filterwarnings(action='ignore') ## Change to once for first time

parser = argparse.ArgumentParser(
    description="Compute SCALE normalization and expectation")
parser.add_argument("--N", 
    required=False, 
    default=100, 
    type=int, 
    help="Number of Latin hypercube samples")
args = parser.parse_args()

# Noise spectrum
ls = np.arange(20051)
Delta_T = 1.  # In muK-arcmin
theta_fwhm = 1.   # in arcmin
ntt = (Delta_T*np.pi/180./60.)**2. * np.exp((theta_fwhm*np.pi/180./60. / np.sqrt(8.*np.log(2)))**2.*ls**2.)
# SCALE binning
dLv = 72
nLv = 2000 // dLv # We will consider up to Lv=2000
Lv = np.arange(dLv/2, (nLv+1)*dLv, dLv, dtype=int)

def getPS(H0=67.5, ombh2=0.022, omch2=0.122, 
          tau=0.06, As=2e-9, ns=0.965, mnu=0.06, 
          dl1=2000, l1m=7000):
    pdict1 = dict(H0=H0, ombh2=ombh2, omch2=omch2, 
                  mnu=mnu, omk=0, tau=tau)
    pdict2 = dict(As=As, ns=ns, r=0)
    pars = camb.CAMBparams()
    pars.set_cosmology(**pdict1)
    pars.InitPower.set_params(**pdict2)
    pars.set_for_lmax(20000, lens_potential_accuracy=8.);
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
    return ctt_unlensed, ctt_lensed, cphiphi

def pRange(mu, sigma):
    """
    Return the [mu-3sigma, mu+3sigma] range.
    """
    return np.array([mu-3*sigma,mu+3*sigma])

def sampleParam(pr, x):
    """
    Given x=[0,1], return the uniform sample 
    within pr.
    """
    return pr[0] + x*(pr[1]-pr[0])

def getPsiA(pars:dict=None, xs=None):
    Ps = dict()
    if xs is not None and pars is not None:
        ks = pars.keys()
        for i, k in enumerate(ks):
            Ps[k] = sampleParam(pRange(*pars[k]), xs[i])
    uCl, lCl, Clpp = np.nan_to_num(getPS(**Ps))
    tCl = lCl + ntt
    pfac = (2*np.pi)**4
    l1m, dl1 = int(Ps['l1m']), int(Ps['dl1'])
    l1min, l1max = l1m-dl1//2, l1m+dl1//2
    # Psi = np.zeros(Lv.size)
    # A = np.zeros(Lv.size)
    # for iL, LL in enumerate(Lv):
    #     Psi[iL], A[iL] = Psi_and_A_cy(LL, uCl, tCl, Clpp, l1min, l1max, 
    #                                     0, 3000, 75, 100)
    ALv, PsiLv = SCALE.CalcBiasExp(uCl, tCl, Clpp, 
        l1min=l1min, l1max=l1max, l2min=0, l2max=3000, 
        Lv=Lv, dl1=75, dl2=100, useC=True)
    # return Psi/pfac, A*pfac, Ps
    return PsiLv, ALv, Ps

# Set range of parameters
prs = dict(H0=[67.36, 0.54], 
           ombh2=[0.02237, 0.00015], 
           omch2=[0.1200, 0.0012], 
           tau=[0.0544, 0.0073], 
           As=[2.100e-9, 0.030e-9], 
           ns=[0.9649, 0.0042], 
           mnu=[0.009, 0.0015], 
           dl1=[2000, 500], # 500 to 3500
           l1m=[7000, 1000] # 5000 to 10000
          )

# Sample parameter space
N = args.N
lh = LatinHypercube(len(prs.keys())).random(n=N)
PsiLvs = np.zeros((N,Lv.size))
ALvs = np.zeros((N,Lv.size))
Pars = dict(H0=[], ombh2=[], omch2=[], tau=[], As=[], ns=[], mnu=[])
for i, samps in enumerate(lh):
    P, A, pars = getPsiA(prs, samps)
    PsiLvs[i,:] = P
    ALvs[i,:] = A
    for key, val in Pars.items():
        Pars[key].append(pars[key])

out = dict(PsiLv=PsiLvs, ALv=ALvs, Lv=Lv, Par=Pars)

with open('SCALEtheoryset.pkl', 'wb') as p:
    pickle.dump(out, p)