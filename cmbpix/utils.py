import numpy as np
import healpy as hp

def patches(ind, nsideIN, nsideOUT):
    """
    Return the indices of the subpixels of resolution nsideOUT
    of the patch at index ind of resolution nsideIN.
    """
    if nsideOUT//2 == nsideIN: # Base case
        ipix = []
        for i in np.arange(4) + 4*hp.ring2nest(nsideIN, ind):
            ipix.append(hp.nest2ring(nsideOUT, i))
        return ipix
    else:
        ipix = []
        for i in np.arange(4) + 4*hp.ring2nest(nsideIN, ind):
            ipix = ipix + patches(hp.nest2ring(nsideIN*2, i), nsideIN*2, \
                                  nsideOUT)
        return ipix