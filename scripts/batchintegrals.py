import numpy as np
from cmbpix.utils import *
from cmbpix.lensing.SCALE import CalcBiasExp
from cmbpix.lensing.qe import N1Kesden
from pytempura import get_norms
import argparse

parser = argparse.ArgumentParser(
    description="Compute part of the training set for lensing emulators")
parser.add_argument("--Njob", 
    required=False, 
    default=0, 
    type=int, 
    help="ID number for the job")
parser.add_argument("--qe", 
    required=False, 
    action="store_true", 
    help="If used, run QE and save with outputs")
parser.add_argument("--suppress", 
    required=False, 
    action="store_true", 
    help="If used, add 3 lensing suppression parameters to the model")
args = parser.parse_args()
Nj = args.Njob
doQE = args.qe
doSuppress = args.suppress

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

def build_cl_dicts(ucl, tcl, cphi):
    """Organize input power spectra into dictionaries that go into pytempura
    """
    ucls = {'TT': ucl, 'EE': ucl, 
            'BB': ucl, 'TE': ucl, 'kk': cphi} # ClTgradT for TT
    # total/lensed (observed) Cls, add noise here if desired
    # tcls = {'TT': rawp['total'][:,0], 'EE': rawp['total'][:,1], 
    #         'BB': rawp['total'][:,2], 'TE': rawp['total'][:,3]}
    tcls = {'TT': tcl, 'EE': tcl, 
            'BB': tcl, 'TE': tcl}
    return ucls, tcls

wfac = 1.5 # Factor to widen the 3-sigma range of each parameter by
prs = dict(H0=[67.36, wfac*0.54], 
           ombh2=[0.02237, wfac*0.00015], 
           omch2=[0.1200, wfac*0.0012], 
           tau=[0.0544, wfac*0.0073], 
           As=[2.100e-9, wfac*0.030e-9], 
           ns=[0.9649, wfac*0.0042], 
           mnu=[0.09, wfac*0.02], 
          )
if doSuppress:
    prs.update(dict(L0=[16000, 4000],# L position of Alens suppression ~4k,16k
                    kLens=[-6.1, 1.04], # Steepness of Alens suppression ~log (1e-4,5e-2)
                    Amax=[0., 1./3.], # 0 for no Alens, 1 for full Alens at high L
                    ))
prs.update(dict(dl1=[2000, 400], # [2000, 500] for 500 to 3500
                l1m=[8000, 1200],  # [7000, 1000] for 4000 to 10000
                ))
ks = list(prs.keys())

lhdir = '/home/p/pen/victorc/1501/CLGK/LH8192/' # 
lhs = np.load(lhdir+'LH8192_12p_{}.npy'.format(Nj))
print('Loaded LH8192_12p_{}.npy'.format(Nj), flush=True)

## Need power spectra at fiducial cosmology
ls, ctt_unlensed0, ctt_lensed0, ntt0, cphiphi0 = getPS()
Lv = np.arange(2002)

pars = np.zeros(lhs.shape)
lCls = np.zeros((lhs.shape[0], ls.size))
cphis = np.zeros((lhs.shape[0], ls.size))
CLvs = np.zeros((lhs.shape[0], Lv.size))
ALvs = np.zeros((lhs.shape[0], Lv.size))

# QE normalization at fiducial cosmology
if doQE:
    lmin = 2; lmax = 3000
    mlmax = 4000
    lq, ctt_response, ctt_lens, ntt_lens, cphirec = getPS(lmax=8000, lensresponse=True)
    uclf, tclf = build_cl_dicts(ctt_response, ctt_lens+ntt_lens, cphirec * (lq*(lq+1)/2.)**2)
    ALs_fid = get_norms(['TT'],uclf,uclf,tclf,lmin,lmax,k_ellmax=mlmax)
    Ls = np.arange(mlmax+1)
    Lkfac = (Ls*(Ls+1)/2.)**2
    N0p_fid = ALs_fid['TT'][0]
    N0k_fid = (Ls*(Ls+1)/2.)**2 * N0p_fid
    ALqs = np.zeros((lhs.shape[0], Lv.size))
    N1s = np.zeros((lhs.shape[0], Lv.size))

comp = 0
for i, samps in enumerate(lhs):
    if i / lhs.shape[0] > comp/100:
        print('Completed {}%'.format(comp), flush=True)
        comp += 1
    for j, s in enumerate(samps):
        pr = pRange(*prs[ks[j]])
        pars[i,j] = sampleParam(pr, s)
    l, uCl, lCl, ntt, cphi = getPS(*pars[i,:-2])
    lCls[i,:] = lCl
    cphis[i,:] = cphi
    l1min = int(pars[i,-1] - pars[i,-2]//2)
    l1max = int(pars[i,-1] + pars[i,-2]//2)
    
    ## Assuming filtering is based on each cosmology
    # ALv, CLv = SCALE.CalcBiasExp(uCl, lCl+ntt, cphi, l1min=l1min, l1max=l1max, 
    #                               l2min=0, l2max=3000, Lv=Lv, useMC=True)
    ## Assuming a single filtering based on fiducial cosmology
    ## cphi and uCl need to change with cosmology; latter due to trispectrum changes
    ALv, CLv = CalcBiasExp(ctt_unlensed0, ctt_lensed0+ntt0, cphi, fCl=uCl, l1min=l1min, l1max=l1max, 
                           l2min=0, l2max=3000, Lv=Lv, useMC=True)
    CLvs[i,:] = CLv
    ALvs[i,:] = ALv
    
    ## Compute QE normalization and N1 bias at each cosmology
    if doQE:
        lq, rClq, lClq, nttq, cphiq = getPS(*pars[i,:-2], lmax=8000, lensresponse=True)
        uclp, tclp = build_cl_dicts(rClq, lClq+nttq, cphiq * (lq*(lq+1)/2.)**2)
        ALs_par = get_norms(['TT'],uclp,uclp,tclp,lmin,lmax,k_ellmax=mlmax)
        ALqs[i,:] = ALs_par['TT'][0][:Lv.size]
        # fCl, tCl fixed at fiducial cosmology
        N1s[i,:] = N1Kesden(Lv, uCl=rClq, tCl=ctt_lens+ntt_lens, Clpp=cphiq, fCl=ctt_response, 
                            lmin=lmin, lmax=lmax, dl=1, n_samps=200000, version=1)

np.savez('batchIntegrals_12pars_{}.npz'.format(Nj), pars=pars, lCls=lCls, 
         cphis=cphis, CLvs=CLvs, ALvs=ALvs, ALqs=ALqs, N1s=N1s)