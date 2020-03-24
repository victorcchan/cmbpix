import numpy as np
import healpy as hp

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
        if nest:
            return hp.nest2ring(NSIDEout, np.arange(4) + 4*ind)
        else:
            return hp.nest2ring(NSIDEout, np.arange(4) + \
                                4*hp.ring2nest(NSIDEin, ind))
    else:
        if nest:
            ipix = np.arange(4) + 4*ind
        else:
            ipix = np.arange(4) + 4*hp.ring2nest(NSIDEin, ind)
        return np.concatenate((patches(ipix[0], NSIDEin*2, NSIDEout, True), 
                                patches(ipix[1], NSIDEin*2, NSIDEout, True), 
                                patches(ipix[2], NSIDEin*2, NSIDEout, True), 
                                patches(ipix[3], NSIDEin*2, NSIDEout, True), 
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