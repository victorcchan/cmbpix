import numpy as np
import healpy as hp

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
    return (2 / a / 10800.) + 1

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

def _lin(x, *p):
    """A generic line function.

    Return y = mx + b using p[0] = b, p[1] = m.

    Parameters
    ----------
    x: real, array of reals
        Point(s) at which to evaluate function.
    p: array of size 2
        The linear coefficients of the function: p = [b, m].

    Returns
    -------
    y: real, array of reals
        The linear function output(s) evaluated at x.
    """
    return p[0] + p[1]*x