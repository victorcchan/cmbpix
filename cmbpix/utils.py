import numpy as np
import healpy as hp
from pixell import enmap, utils#, curvedsky
import camb

# Useful constants
r2d = 180. / np.pi # radians to degrees
r2am = r2d * 60. # radians to arcmins

def ang2ell(a):
    """Convert the given angular scale(s) in arcmins to the analogous ell(s).

    Return the ell(s) corresponding to the angular scale(s) set by a.

    Parameters
    ----------
    a: float, array of floats
        The angular scale(s) in arcmins.

    Returns
    -------
    ell: float, array of floats
        The ell mode(s) corresponding to a.
    """
    return (2 * 10800. / a) - 1

def ell2ang(ell):
    """Convert the given ell(s) to its analogous angular scale(s) in arcmins.

    Return the angular scale(s) in arcmins corresponding to the Fourier mode 
    ell(s).

    Parameters
    ----------
    ell: value, array of values
        The ell mode(s).

    Returns
    -------
    a: float, array of floats
        The angular scale(s) corresponding to ell.
    """
    return 2 * 10800. / (ell + 1)

def ellfac(ell, phi2k=False):
    """Return the ell factor for the given ell(s).

    Return the ell factor for the given ell(s) to convert C_ell to D_ell.
    If phi2k is True, return the ell factor for converting C^{phiphi} to 
    C^{kk}.

    Parameters
    ----------
    ell: value, array of values
        The ell mode(s).
    phi2k: bool, default=False
        If True, return the ell factor for converting C^{phiphi} to C^{kk}.
        Otherwise, return the ell factor for converting C_ell to D_ell.
    
    Returns
    -------
    ellfac: float, array of floats
        The ell factor for the given ell(s).
    """
    if phi2k:
        return (ell*(ell+1.)*ell*(ell+1.)) / (2*np.pi)
    else:
        return ell*(ell+1.) / (2*np.pi)
    
def dcltt(ell, cltt, fsky=1., dell=1.):
    """Return the minumum error on C_ell for the given ell(s).

    Return the minimum error on C_ell for the given ell(s) for a given
    sky fraction fsky and bin width dell.

    Parameters
    ----------
    ell: value, array of values
        The ell mode(s).
    cltt: value, array of values
        The C_ell value(s).
    fsky: float, default=1.
        The sky fraction.
    dell: float (or array of matching size), default=1.
        The bin width.

    Returns
    -------
    dcltt: float, array of floats
        The minimum error on C_ell for the given ell(s).
    """
    return np.sqrt(2 / ((2*ell+1.)*dell*fsky)) * cltt

def patches(ind, NSIDEin, NSIDEout, nest=False):
    """Daughter pixel indices in a low resolution HEALPix patch.

    Return HEALPix indices for all pixels of a higher resolution map 
    contained inside the pixel(s) of a lower resolution map. Output pixels 
    are always in the RING ordering scheme.

    Parameters
    ----------
    ind: int or array of ints
        Index of the parent HEALPix patch(es).
    NSIDEin: int
        NSIDE resolution of the parent HEALPix patch(es).
    NSIDEout: int
        NSIDE resolution of the daughter HEALPix pixels.
    nest: bool, optional
        If True, assume ``ind`` are given in NESTED pixel ordering. 
        Otherwise, assume RING ordering. Default: False.

    Returns
    -------
    ipix: 1d-array of int
        Indices of all pixels contained with the parent patch(es). Output 
        is always in RING ordering.

    """
    if NSIDEout/2 == NSIDEin: # Base case
        inds = np.array(ind)
        if nest:
            return hp.nest2ring(NSIDEout, np.tile(np.arange(4), inds.size) + \
                4*inds.repeat(4))
        else:
            return hp.nest2ring(NSIDEout, np.tile(np.arange(4), inds.size) + \
                4*hp.ring2nest(NSIDEin, inds).repeat(4))
    else:
        inds = np.array(ind)
        s = inds.size
        if nest:
            ipix = np.tile(np.arange(4), s) + 4*inds.repeat(4)
        else:
            ipix = np.tile(np.arange(4), s) + \
                4*hp.ring2nest(NSIDEin, inds).repeat(4)
    return np.concatenate((patches(ipix[:s], NSIDEin*2, NSIDEout, True), 
                            patches(ipix[s:2*s], NSIDEin*2, \
                                NSIDEout, True), 
                            patches(ipix[2*s:3*s], NSIDEin*2, \
                                NSIDEout, True), 
                            patches(ipix[3*s:], NSIDEin*2, \
                                NSIDEout, True), 
                            ))

def filter_map(map_in, fl):
    """Apply a filter to the given map.

    Return a filtered version of HEALPix map by applying f_ell in 
    alm space.

    Parameters
    ----------
    map_in: 1d-array
        HEALPix map (of the CMB) to be filtered.
    fl: 1d-array
        The filter to be applied. Should have values for 0 < ell < 
        3*NSIDE - 1.

    Returns
    -------
    map_out: 1d-array
        HEALPix map of the original, but filtered with fl.

    """
    NSIDE_in = hp.npix2nside(len(map_in))
    alms = hp.map2alm(map_in)
    alms = hp.almxfl(alms, fl)
    map_out = hp.alm2map(alms, NSIDE_in)
    return map_out

def numgradient_flat(map_in):
    """Return the gradient maps of map_in.
    
    Perform the central difference derivative on map_in. The values at 
    boundaries contain left/right derivatives.
    
    Parameters
    ----------
    map_in: ndmap, ndarray
        The input map.
    
    Returns
    -------
    dy: ndmap, ndarray
        The gradient in the y-direction
    dx: ndmap, ndarray
        The gradient in the x-direction
    """
    yy, xx = map_in.posmap()
    dy = enmap.zeros(map_in.shape, map_in.wcs)
    dx = enmap.zeros(map_in.shape, map_in.wcs)
    dy[1:-1,:] = (map_in[:-2,:] - map_in[2:,:]) / (yy[:-2,:] - yy[2:,:])
    dx[:,1:-1] = (map_in[:,:-2] - map_in[:,2:]) / (xx[:,:-2] - xx[:,2:])
    dy[0,:] = (map_in[1,:] - map_in[0,:]) / (yy[1,:] - yy[0,:])
    dy[-1,:] = (map_in[-1,:] - map_in[-2,:]) / (yy[-1,:] - yy[-2,:])
    dx[:,0] = (map_in[:,1] - map_in[:,0]) / (xx[:,1] - xx[:,0])
    dx[:,-1] = (map_in[:,-1] - map_in[:,-2]) / (xx[:,-1] - xx[:,-2])
    return dy, dx

def gradient_flat(map_in, lmax=2000):
    """Return the gradient maps of map_in.
    
    Compute the gradient of the map_in in Fourier space. Simultaneously 
    low-pass the maps such that they only include modes ell < lmax.
    
    Parameters
    ----------
    map_in: ndmap, ndarray
        The input map.
    lmax: int
        The cutoff ell for low-passing the maps.
    
    Returns
    -------
    dy: ndmap, ndarray
        The gradient in the y-direction
    dx: ndmap, ndarray
        The gradient in the x-direction
    """
    ly, lx = map_in.lmap()
    lmod = map_in.modlmap()
    lp = enmap.zeros(map_in.shape)
    lp[np.where(lmod < lmax)] = 1.
    map_fft = enmap.fft(map_in)
    dx = enmap.ifft(map_fft*lp*lx*1j).real
    dy = enmap.ifft(map_fft*lp*ly*1j).real
    return dy, dx

def shift2d(map_in):
    """Shift the 2D fft map such that the origin is in the center.
    
    Return map_in with its values rolled such that the origin is in its 
    center.
    
    Parameters
    ----------
    map_in: ndmap, ndarray
        The input 2D fft map.
    
    Returns
    -------
    map_out: ndmap, ndarray
        The rolled 2D fft map.
    """
    s = map_in.shape
    map_out = np.roll(map_in, [s[0]//2, s[1]//2], [0,1])[:,::-1]
    return map_out

def add_noise(data, noise, pol=True):
    """Add pixel noise to the data.
    
    Add pixel noise to the data. If data contains multiple maps, the first 
    is assumed to be T, and the rest are assumed to be polarization.
    
    Parameters
    ----------
    data: array
        The map to add noise to.
    noise: value
        The pixel noise levels in units of uK-arcmin.
    pol: bool, default=True
        If True, assume the first map to be T and the rest polarization. 
        If False, assume all maps to be T.
    
    Returns
    -------
    noisymap: array
        The map with noise added.
    """
    sh = data.shape
    try:
        resol = hp.nside2resol(hp.get_nside(data), arcmin=True)
        form = 'HEALPix'
    except(ValueError, TypeError):
        try:
            resol = np.abs(data.wcs.wcs.cdelt[0])*60 # Convert deg->arcmin
            form = 'enmap'
        except(AttributeError):
            raise ValueError("The data isn't in HEALPix or enmap format.")
    nmap = np.random.normal(0, 1, sh) * noise / resol
    # Statistical penalty for polarization noise
    if form == 'HEALPix' and len(sh) > 1 and pol:
        nmap[1:] *= np.sqrt(2)
    elif form == 'enmap' and len(sh) > 2 and sh[0]> 1 and pol:
        nmap[1:] *= np.sqrt(2)
    return nmap + data

def Asuppress(Lmax=30000, L0=10000, kLens=1e-3, Amax=0.):
    """Return a reverse logistic function for suppressing high ell modes.

    Return a reverse logistic function for suppressing high ell modes of a 
    CMB lensing power spectrum. Apply by multiplying cphiphi by the output 
    and/or feeding the output into camb.results.get_partial_lensing_cls().

    Parameters
    ----------
    Lmax: int, default=30000
        Maximum multipole to compute.
    L0: int, default=10000
        Midpoint of the suppression.
    kLens: float, default=1e-3
        Steepness of the suppression. If negative, use exp(kLens).
    Amax: float, default=0.
        Maximum amount of suppression. Give values between 0 and 1.
        I.e., minimum of the function is (1-Amax)

    Returns
    -------
    ALens: array
        The suppression function.
    """
    L = np.arange(Lmax+1)
    ALens = np.ones(Lmax+1, dtype=np.float64)
    if kLens < 0:
        kLens = np.exp(kLens)
    return ALens * Amax / (1 + np.exp(kLens*(L-L0))) + (1-Amax)

def getPS(H0=67.5, ombh2=0.022, omch2=0.122, 
          tau=0.06, As=2.1e-9, ns=0.965, mnu=0.06, 
          L0=None, kLens=None, Amax=0., lmax=20000, 
          w=1.0, b=1.0, lensresponse=False, removeNaN=True):
    """Return CMB power spectra from CAMB for given cosmological parameters.

    Parameters
    ----------
    H0: float, default=67.5
        Hubble constant in km/s/Mpc.
    ombh2: float, default=0.022
        Physical baryon density.
    omch2: float, default=0.122
        Physical cold dark matter density.
    tau: float, default=0.06
        Optical depth to reionization.
    As: float, default=2.1e-9
        Amplitude of the primordial power spectrum.
    ns: float, default=0.965
        Spectral index of the primordial power spectrum.
    mnu: float, default=0.06
        Sum of the neutrino masses in eV.
    L0: int, default=None
        Midpoint of the lensing suppression, if given.
    kLens: float, default=None
        Steepness of the suppression, if given.
    Amax: float, default=0.
        Maximum amount of suppression, if given.
    lmax: int, default=20000
        Maximum multipole to compute.
    w: float, default=1.0
        White noise level in uK-arcmin.
    b: float, default=1.0
        Beam FWHM in arcmin.
    lensresponse: bool, default=False
        If True, return the ClTgradT as ctt_unlensed.
    removeNaN: bool, default=True
        If True, replace NaNs from the power spectra with 1e-20.

    Returns
    -------
    ls: array
        The multipole values.
    ctt_unlensed: array
        The unlensed CMB temperature power spectrum.
        This is ClTgradT if lensresponse is True.
    ctt_lensed: array
        The lensed CMB temperature power spectrum.
    ntt: array
        The noise power spectrum.
    cphiphi: array
        The lensing potential power spectrum.
    """
    pdict1 = dict(H0=H0, ombh2=ombh2, omch2=omch2, 
                  mnu=mnu, omk=0, tau=tau)
    pdict2 = dict(As=As, ns=ns, r=0)
    pars = camb.CAMBparams()
    pars.set_cosmology(**pdict1)
    pars.InitPower.set_params(**pdict2)
    pars.set_for_lmax(lmax, lens_potential_accuracy=8.);
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

    ls = np.arange(lmax+1)
    ## CAMB outputs l(l+1)/(2pi) * C_l by default, need to rescale
    ctt_unlensed = powers['unlensed_scalar'][:lmax+1,0]/ellfac(ls)
    cphiphi = powers['lens_potential'][:lmax+1,0]/ellfac(ls, phi2k=True)
    ## CAMB outputs l(l+1)/(2pi) * C_l^{dd} by default, need to rescale to C_l^{phiphi}
    if L0 is not None and kLens is not None:
        ALens = Asuppress(lmax, L0, kLens, Amax)
        partp = results.get_partially_lensed_cls(Alens=ALens, raw_cl=True, CMB_unit='muK')
        ctt_lensed = partp[:lmax+1,0]
        cphiphi *= ALens
    else:
        ctt_lensed = powers['total'][:lmax+1,0]/ellfac(ls)
    if lensresponse:
        # lensed gradient function expects a longer array than we have
        cphi_pad = np.pad(cphiphi*ellfac(ls, phi2k=True), (0,500), 'constant', constant_values=0)
        gradp = results.get_lensed_gradient_cls(lmax=lmax, raw_cl=True, CMB_unit='muK', 
                                                clpp=cphi_pad)
        ctt_unlensed = gradp[:lmax+1,0]
    # Compute noise spectrum
    ntt = (w/r2am)**2.*np.exp((b/r2am / np.sqrt(8.*np.log(2)))**2.*ls**2.)
    if removeNaN:
        ctt_unlensed[np.isnan(ctt_unlensed)|(ctt_unlensed==0)] = 1.e-20
        ctt_lensed[np.isnan(ctt_lensed)|(ctt_lensed==0)] = 1.e-20
        ntt[np.isnan(ntt)|(ntt==0)] = 1.e-20
        cphiphi[np.isnan(cphiphi)|(cphiphi==0)] = 1.e-20
    return ls, ctt_unlensed, ctt_lensed, ntt, cphiphi

# def alm2stripe(alm, width, resol, proj='car'):
#     """Return the stripe centered at dec=0 of the map corresponding to alm.
    
#     Return a slice of the map corresponding to alm from declination -width/2 
#     to +width/2, and right ascension -180deg to +180deg.
    
#     Parameters
#     ----------
#     alm: array
#         The set of alms to convert to map-space.
#     width: value
#         The width of the map (centered at dec=0) in degrees.
#     resol: value
#         The resolution of the map in arcmin.
#     proj: str, default='car'
#         The projection of the map. Must be compatible with pixell.
    
#     Returns
#     -------
#     stmap: array
#         The output map.
#     """
#     box = np.array([[-width/2,180], [width/2,-180]]) * utils.degree
#     shape, wcs = enmap.geometry(pos=box, res=resol*utils.arcmin, proj=proj)
#     cmap = enmap.zeros(shape, wcs)
#     return curvedsky.alm2map(alm, cmap, method='cyl')