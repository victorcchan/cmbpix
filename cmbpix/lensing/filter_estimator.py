import numpy as np
import healpy as hp
from cmbpix.utils import *

class FilterEstimator():
	"""Estimator for small scale lensing in a full sky CMB map.

	An object for reducing a HEALPix map of the cosmic microwave 
	background in (mostly) pixel-space to estimate the effects 
	of small angular scale weak lensing due to structure formation.

	Attributes
	----------
	map_in: 1d-array
		The input CMB map in HEALPix format. The resolution of the map 
		must be of NSIDE > 256. Assumed to be in RING ordering.
	cl_fid: 1d-array
		The fiducial power spectrum for the input CMB map.
	h_ells: 2d-array
		The boundaries used for high-passing the input CMB map.
	Nl: 1d-array
		The noise spectrum calcualted from ``w`` and ``beam`` using 
		the Knox formula.
	map_dtheta: 1d-array
		The gradient of the input map with respect to theta in 
		spherical coordinates. Created after executing the method 
		evaluate_gradient, otherwise initialized as None.
	map_dphi: 1d-array
		The gradient of the input map with respect to phi in 
		spherical coordinates. The map is already divided by 
		sin(theta). Created after executing the method 
		evaluate_gradient, otherwise initialized as None.
	map_filtered: 2d-array
		A collection of maps filtered according to the boundaries 
		provided. Created after running the filter_all_maps method.
	map_dtheta_f: 1d-array
		A low-passed version of map_dtheta. Created after running 
		the filter_all_maps method if map_dtheta exists.
	map_dphi_f: 1d-array
		A low-passed version of map_dphi. Created after running 
		the filter_all_maps method if map_dtheta exists.

	"""
	def __init__(self, cmbmap, fiducial_cls=None, lmax=6000, highell=3000, 
					w=0., beam=0.):
		"""Initiate the estimator.

		Parameters
		----------
		cmbmap: 1d-array
			A CMB map in HEALPix format.
		fiducial_cls: 1d-array or None, optional
			A set of fiducial Cls used for constructing Wiener filters. 
			Should have entries for ell up to at least (3*NSIDE)-1. 
			If None, input map Cls are used. Default: None.
		lmax: int, optional
			The maximum ell the estimator will consider. Default: 6000.
		highell: int or array of ints, optional
			Boundaries for high-pass filtering the input map. The 
			generate_filters method uses these to create top hat-like 
			Wiener filters using a sorted version this input as the 
			boundaries. All entries must be smaller than lmax, which 
			is assumed to be the largest boundary. Default: 3000.
		w: float, optional
			The noise level of the input map in uK*arcmin. Default: 0.
		beam: float, optional
			The FHWM size of the map beam in arcmin. Default: 0.

		"""
		self.map_in = cmbmap
		self.cl_fid = fiducial_cls
		self._NSIDE_small = hp.npix2nside(self.map_in.size)
		self._NSIDE_large = 512 # Fixed
		self.ells = np.arange(3*self._NSIDE_small)
		self._lmax = lmax

		# Set up high-pass filter boundaries
		if type(highell) == int:
			self.h_ells = [[highell, self._lmax]]
		else:
			self.h_ells = []
			_ells = np.sort(highell)
			for i in range(len(highell)):
				if i == len(highell)-1:
					self.h_ells.append([_ells[i], self._lmax])
				else:
					self.h_ells.append([_ells[i], _ells[i+1]])
			del _ells

		# Generate a noise spectrum for Wiener filtering purposes
		self._w = w * np.pi / 10800. # Convert to uK*rad
		self._beam = beam * np.pi / 10800. # Convert to rad
		wl = np.exp(-self.ells*(self.ells+1)*self._beam**2)
		self.Nl = self._w**2 / wl # Knox formula
		self.Nl[:2] = 1e-9 # Prevent divide by zero (Values not used anyway)
		del wl

		self.map_dtheta = None
		self.map_dphi = None

	def generate_filters(self):
		"""Create top hat-like Wiener filters.

		Generate top hat-like Wiener filters using the given high-pass 
		boundaries and set of fiducial Cls. If no fiducial Cls are given, 
		the input map Cls are used. A similar filter is also generated for 
		creating the background temperature gradient with 0 < ell < 2000.

		Returns
		-------
		T_filters: 2d-array
			Top hat-like Wiener filters for every ell range given during 
			initialization.
		grad_filter: 1d-array
			Top hat-like Wiener filter for low-passing the map for 
			0 < ell < 2000. Used for creating a background gradient map.

		"""
		if self.cl_fid is None: # Construct Wiener filter
			total_Cl = hp.anafast(self.map_in)
			Wl = (total_Cl - self.Nl) / total_Cl
			del total_Cl
		else:
			fid = self.cl_fid[:3*self._NSIDE_small]
			Wl = fid / (fid + self.Nl)
			del fid
		T_filters = []
		for ls in self.h_ells:
			f = np.copy(Wl)
			f[:ls[0]] = 0.
			f[ls[1]:] = 0.
			T_filters.append(f)
		grad_filter = np.copy(Wl)
		grad_filter[2000:] = 0.
		return T_filters, grad_filter


	def filter_all_maps(self, verbose=False):
		"""Filter the input map, and also determine the background gradient.

		Perform a high-pass Wiener-like filter on the input CMB map. The 
		input map is also low-passed to produce a map of the large scale CMB 
		variations that dictate the background temperature gradient. 
		The high-pass Wiener filters are constructed using the given fiducial 
		Cls. If no fiducial model is given, then the Cls of the input map are 
		used. An additional factor is included in the filter with respect to 
		the alignment between the appropriate small scale map's gradient and 
		the background gradient directions (dot product, no amplitudes).

		"""
		if verbose:
			print("Generating filters", flush=True)
		T_filters, dT_filter = self.generate_filters()
		if verbose:
			print("Converting input map to alms", flush=True)
		map_alm = hp.map2alm(self.map_in) # Don't use filter_map; convert once
		l_alm = hp.almxfl(map_alm, dT_filter)
		if verbose:
			print("Creating large scale gradient maps", flush=True)
		l_map, self.map_dtheta, self.map_dphi = hp.alm2map_der1(l_alm, \
			self._NSIDE_small)
		del l_map, l_alm # Don't need these
		self.map_filtered = []
		for f in range(len(T_filters)):
			if verbose:
				print("Starting filter for {}".format(self.h_ells[f]), \
					flush=True)
			s_alm = hp.almxfl(map_alm, T_filters[f]) # Perform Wiener filter
			s_map, s_dth, s_dph = hp.alm2map_der1(s_alm, self._NSIDE_small)
			self.map_ss = s_map
			del s_map # Don't need this
			dir_map = (self.map_dtheta*s_dth + self.map_dphi*s_dph)**2 / \
				((self.map_dtheta**2 + self.map_dphi**2) * \
					(s_dth**2 + s_dph**2)
					)
			self.d = dir_map
			self.sth = s_dth
			self.sph = s_dph
			del s_dth, s_dph
			if verbose:
				print("Applying direction filter", flush=True)
			dir_alm = hp.map2alm(dir_map)
			new_alm = s_alm * dir_alm # Direction filtered alms
			del s_alm, dir_alm # Don't need these anymore
			self.map_filtered.append(hp.alm2map(new_alm, self._NSIDE_small))
			del new_alm
		del map_alm
		if verbose:
			print("Finished filtering all maps", flush=True)

	def divide_patches(self):
		"""Divide the sky map into patches larger than the pixel size.

		Return the indices of the original resolution for the input map but 
		reordered such that pixels are grouped together into larger patches 
		on the sky --> A lower HEALPix resolution. The new ordering is NOT 
		a standard HEALPix ordering scheme. Also return the corresponding 
		index of the larger patch for each of the grouped.

		Returns
		-------
		patch_order: 2d-array of int
			The new ordering for the original, higher resolution map with 
			pixels grouped together into larger patches. Entries at 
			patch_order[i] contain [old_index, patch_index], where 
			old_index is the original index of the i-th pixel in the RING 
			ordering and patch_index is the index of its corresponding patch 
			in the lower resolution.

		"""
		large_inds = np.arange(hp.nside2npix(self._NSIDE_large), dtype=int)
		small_inds = patches(large_inds, self._NSIDE_large, self._NSIDE_small)
		large_inds = large_inds.repeat(hp.nside2npix(self._NSIDE_small) // \
										hp.nside2npix(self._NSIDE_large))
		self.patch_order = np.stack((small_inds, large_inds), axis=1)

	def calculate_patch_statistics(self, verbose=False):
		"""Determine the statistics for the lensing estimator within patches.

		Compute the required statistics for the probabilistic lensing 
		estimator within large patches of the sky. Specifically, the means of 
		the filtered, input map, and each gradient map are computed within 
		each patch. A copy of the input map is also re-ordered to group its 
		pixels into their corresponding larger patches.

		"""
		n_small = hp.nside2npix(self._NSIDE_small)
		n_large = hp.nside2npix(self._NSIDE_large)
		groups = (n_large, n_small//n_large) # For reshaping into patches

		# Take mean of filtered, input map in each patch
		if verbose:
			print("Determining patch statistics within SS T maps", flush=True)
		self.map_reordered = []
		self.map_mean = []
		self.map_var = []
		for maps in self.map_filtered:
			self.map_reordered.append(maps[self.patch_order[:,0]])
		self.map_mean.append(np.mean(np.reshape(self.map_reordered, \
			groups), axis=1))
		self.map_var.append(np.var(np.reshape(self.map_reordered, \
			groups), axis=1))

		# Take mean of filtered dtheta map in each patch
		if verbose:
			print("Determining patch statistics within LS grad maps", \
				flush=True)
		dtheta_reordered = self.map_dtheta[self.patch_order[:,0]]
		self.dtheta_mean = np.mean(np.reshape(dtheta_reordered, groups),axis=1)
		del dtheta_reordered

		# Take mean of filtered dphi map in each patch
		dphi_reordered = self.map_dphi[self.patch_order[:,0]]
		self.dphi_mean = np.mean(np.reshape(dphi_reordered, groups),axis=1)
		del dphi_reordered