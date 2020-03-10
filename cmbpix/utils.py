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

	References
	----------
	[1] G{\'o}rski, K. M., et al. 2005.
	"HEALPix: A Framework for High-Resolution Discretization and Fast 
	Analysis of Data Distributed on the Sphere." 
	The Astrophysical Journal 622, 759.

	History
	-------
	2020-03-02: First added to package.
	2020-03-10: Docstring added.
    """
    if nsideOUT/2 == nsideIN: # Base case
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