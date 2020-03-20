import numpy as np
import healpy as hp

def patches(ind, NSIDEin, NSIDEout):
    """Daughter pixel indices in a low resolution HEALPix patch.

    Return HEALPix indices for all pixels of a higher resolution map 
    contained inside the pixel(s) of a lower resolution map. The input
    indices must be in the RING ordering scheme.

    Parameters
    ----------
    ind: int or array of ints
    	Index of the parent HEALPix patch(es).
    NSIDEin: int
    	NSIDE resolution of the parent HEALPix patch(es).
    MSIDEout: int
    	NSIDE resolution of the daughter HEALPix pixels.

    Returns
    -------
	ipix: list of int
		Indices of all pixels contained with the parent patch(es).
	
    """
    if NSIDEout/2 == NSIDEin: # Base case
        return hp.nest2ring(NSIDEout, np.arange(4) + \
                            4*hp.ring2nest(NSIDEin, ind))
    else:
        ipix = hp.nest2ring(NSIDEin*2, np.arange(4) + \
                            4*hp.ring2nest(NSIDEin, ind))
        return np.concatenate((patches(ipix[0], NSIDEin*2, NSIDEout), 
                                patches(ipix[1], NSIDEin*2, NSIDEout), 
                                patches(ipix[2], NSIDEin*2, NSIDEout), 
                                patches(ipix[3], NSIDEin*2, NSIDEout), 
                                ))