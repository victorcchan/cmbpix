import os
import numpy as np
import camb
import lenspyx
from lenspyx.utils_hp import synalm, almxfl, alm2cl
import argparse

## Test params
parser = argparse.ArgumentParser(
    description="Simulate CMB realizations and quantify lensing")
parser.add_argument("--Nsim", 
    required=False, 
    default=10, 
    type=int, 
    help="Number of realizaitons to simulate per process")
parser.add_argument("--Njob", 
    required=False, 
    default=0, 
    type=int, 
    help="ID number for the job")
# These are SCALE parameters
parser.add_argument("--l1min", 
    required=True, 
    type=int, 
    nargs='+', 
    help="Lower limit for small-scale filtering")
parser.add_argument("--l1max", 
    required=True, 
    type=int, 
    nargs='+', 
    help="Upper limit for small-scale filtering")
args = parser.parse_args()
Nsim = args.Nsim
Nj = args.Njob
l1min = np.array(args.l1min, dtype=int)
l1max = np.array(args.l1max, dtype=int)
l1minstr = ','.join(l1min.astype(str))
l1maxstr = ','.join(l1max.astype(str))

fn_suff = '_Nsim'+str(Nsim)+'_l1'+l1minstr+'-'+l1maxstr+'_Njob'+str(Nj)+'.npz'

# This generates the power spectra for the lensed and unlensed CMB
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, omk=0, tau=0.06)# , mnu=0.06
pars.InitPower.set_params(As=2.1e-9, ns=0.965, r=0)
pars.set_for_lmax(35000, lens_potential_accuracy=8);
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
w = 1.0
b = 1.0
ntt = (w*np.pi/180./60.)**2. * np.exp((b*np.pi/180./60. / np.sqrt(8.*np.log(2)))**2.*ls**2.)
## Replace potential nan here
ntt[:2] = 1e-20
ctt_unlensed[:2] = 1e-20
ctt_lensed[:2] = 1e-20
cphiphi[:2] = 1e-20
ctt_total = ctt_lensed + ntt

lmax_len = 24000
dlmax = 2048  # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
epsilon = 1e-6 # target accuracy of the output maps (execution time has a fairly weak dependence on this)
lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax

CttU = np.zeros((Nsim,lmax_len+1))
CttL = np.zeros((Nsim,lmax_len+1))
CLGKs = np.zeros((Nsim, l1min.size, lmax_len+1))
el_len = np.arange(lmax_len+1)

# Geometry on which to produce the lensed map
geom_info = ('healpix', {'nside':8192}) # map accuracy is good to ell~2*NSIDE
geom = lenspyx.get_geom(geom_info)

for i in range(Nsim):
    # Generate raw Tmap and phimap alms
    tlm_unl = synalm(ctt_unlensed, lmax=lmax_unl, mmax=mmax_unl)
    CttU[i] = alm2cl(tlm_unl, tlm_unl, lmax_unl, lmax_unl, lmax_unl)[:lmax_len+1]
    plm = synalm(cphiphi, lmax=lmax_unl, mmax=mmax_unl)

    # We then transform the lensing potential into spin-1 deflection field, and deflect the temperature map.
    dlm = almxfl(plm, np.sqrt(np.arange(lmax_unl + 1, dtype=float) * np.arange(1, lmax_unl + 2)), None, False)
    # Free up some memory
    del plm

    # Lensed T map using deflection field:
    Tlen = lenspyx.alm2lenmap(tlm_unl, dlm, geometry=geom_info, verbose=1, epsilon=epsilon)
    # Free up some memory
    del dlm, tlm_unl
    # Add noise
    nlm = synalm(ntt, lmax=lmax_unl, mmax=mmax_unl)
    Tlen += geom.alm2map(nlm, lmax_unl, mmax_unl, nthreads=os.cpu_count())
    del nlm

    # Computed lensed CMB power spectrum
    tlm_len = geom.map2alm(np.copy(Tlen), lmax_len, lmax_len, nthreads=os.cpu_count())
    CttL[i] = alm2cl(tlm_len, tlm_len, lmax_len, lmax_len, lmax_len)

    del Tlen

    ## Beyond here is SCALE
    # Do low pass once; reuse lambda alms
    LP = np.ones(lmax_len+1)*np.sqrt(np.arange(lmax_len + 1, dtype=float) * np.arange(1, lmax_len + 2)) * ctt_unlensed[:lmax_len+1] / ctt_total[:lmax_len+1]
    LP[(el_len > 3000)] = 0.
    tlm_lp = almxfl(tlm_len, LP, None, False)
    lam = geom.alm2map_spin([tlm_lp, tlm_lp*0.0], spin=1, lmax=lmax_len, mmax=lmax_len, nthreads=os.cpu_count())

    del tlm_lp, LP

    lmap = lam[0]**2 + lam[1]**2
    del lam
    llm = geom.map2alm(lmap, lmax_len, lmax_len, nthreads=os.cpu_count())
    del lmap

    for j in range(l1min.size):
        HP = np.ones(lmax_len+1)*np.sqrt(np.arange(lmax_len + 1, dtype=float) * np.arange(1, lmax_len + 2)) / ctt_total[:lmax_len+1]
        HP[(el_len < l1min[j])|(el_len > l1max[j])] = 0.
        tlm_hp = almxfl(tlm_len, HP, None, False)
        sig = geom.alm2map_spin([tlm_hp, tlm_hp*0.0], spin=1, lmax=lmax_len, mmax=lmax_len, nthreads=os.cpu_count())

        del tlm_hp, HP

        smap = sig[0]**2 + sig[1]**2
        del sig
        slm = geom.map2alm(smap, lmax_len, lmax_len, nthreads=os.cpu_count())

        del smap

        CLGKs[i,j] = alm2cl(slm, llm, lmax_len, lmax_len, lmax_len)

        del slm

    del llm, tlm_len


np.savez("lenspyxNSIDE8192_wSCALE"+fn_suff, CttU=CttU, CttL=CttL, CLGKs=CLGKs)

# var = [10000, 11000, 12000, 13000, 14000, 15000] # desired lmax of the lensed field.
# var = [512, 1024, 2048, 4096]
# var = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
# ClTTs = np.zeros((len(var), 10001))
# for i, epsilon in enumerate(var):
#     lmax_len = 12000
#     dlmax = 2048  # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
#     # epsilon = 1e-8 # target accuracy of the output maps (execution time has a fairly weak dependence on this)
#     lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax

#     # Generate raw Tmap and phimap alms
#     tlm_unl = synalm(ctt_unlensed, lmax=lmax_unl, mmax=mmax_unl)
#     plm = synalm(cphiphi, lmax=lmax_unl, mmax=mmax_unl)

#     # We then transform the lensing potential into spin-1 deflection field, and deflect the temperature map.
#     dlm = almxfl(plm, np.sqrt(np.arange(lmax_unl + 1, dtype=float) * np.arange(1, lmax_unl + 2)), None, False)
#     # Free up some memory
#     del plm

#     # Geometry on which to produce the lensed map
#     geom_info = ('healpix', {'nside':4096}) # here we will use an Healpix grid with nside 2048
#     geom = lenspyx.get_geom(geom_info)

#     # Lensed T map using deflection field:
#     Tlen = lenspyx.alm2lenmap(tlm_unl, dlm, geometry=geom_info, verbose=1, epsilon=epsilon)
#     # Free up some memory
#     del dlm, tlm_unl

#     # Computed lensed CMB power spectrum
#     tlm_len = geom.map2alm(np.copy(Tlen), lmax_len, lmax_len, nthreads=os.cpu_count())
#     CttL = alm2cl(tlm_len, tlm_len, lmax_len, lmax_len, lmax_len)

#     del Tlen, tlm_len
#     ClTTs[i] = CttL[:10001]

# np.save('lenspyxCls_epsilon.npy', ClTTs)