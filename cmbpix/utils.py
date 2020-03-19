import numpy as np
import healpy as hp

def patches(ind, nsideIN, nsideOUT):
    """Daughter pixel indices in a low resolution HEALPix patch.

    Return HEALPix indices for all pixels of a higher resolution map 
    contained inside the pixel(s) of a lower resolution map. The input
    indices must be in the RING ordering scheme.

    Parameters
    ----------
    ind: int or array of ints
    	Index of the parent HEALPix patch(es).
    nsideIN: int
    	NSIDE resolution of the parent HEALPix patch(es).
    nsideOUT: int
    	NSIDE resolution of the daughter HEALPix pixels.

    Returns
    -------
	ipix: list of int
		Indices of all pixels contained with the parent patch(es).
	
    """
    if nsideOUT/2 == nsideIN: # Base case
        return hp.nest2ring(nsideOUT, np.arange(4) + \
                            4*hp.ring2nest(nsideIN, ind))
    else:
        ipix = hp.nest2ring(nsideIN*2, np.arange(4) + \
                            4*hp.ring2nest(nsideIN, ind))
        return np.concatenate((patches(ipix[0], nsideIN*2, nsideOUT), 
                                patches(ipix[1], nsideIN*2, nsideOUT), 
                                patches(ipix[2], nsideIN*2, nsideOUT), 
                                patches(ipix[3], nsideIN*2, nsideOUT), 
                                ))