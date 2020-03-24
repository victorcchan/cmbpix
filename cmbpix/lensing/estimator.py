import numpy as np
import healpy as hp
from cmbpix.utils import *

class LensingEstimator():
	"""Estimator for small scale lensing in a full sky CMB map.

	An object for reducing a HEALPix map of the cosmic microwave 
	background in (mostly) pixel-space to estimate the effects 
	of small angular scale weak lensing due to structure formation.

	"""
	def __init__(self, cmbmap, fiducial_cls=None, lmax=6700, highell=3000, 
					w=0., beam=0.):
		"""Initiate the estimator.

		Parameters
		----------
		cmbmap: 1d-array
			A CMB map in HEALPix format.
		fiducial_cls: 1d-array or None, optional
			A set of fiducial Cls used for constructing Wiener filters. 
			Should have entries for ell up to at least (3*NSIDE)-1.
		lmax: int, optional
			The maximum ell the estimator will consider.
		highell: int or array of ints, optional
			Boundaries for high-pass filtering the input map. The 
			generate_filters method uses these to create top hat-like 
			Wiener filters using a sorted version this input as the 
			boundaries. All entries must be smaller than lmax, which 
			is assumed to be the largest boundary.
		w: float, optional
			The noise level of the input map in uK*arcmin.
		beam: float, optional
			The FHWM size of the map beam in arcmin.

		Attributes
		----------
		map_in: 1d-array
			The input CMB map in HEALPix format.
		cl_fid: 1d-array
			The fiducial power spectrum for the input CMB map.
		h_ells: 2d-array
			The boundaries used for high-passing the input CMB map.
		w: float
			The noise level of the input map in uK*rad.
		beam: float
			The FHWM of the map beam in rad.
		map_dtheta: 1d-array
			The gradient of the input map with respect to theta in 
			spherical coordinates. Created after executing the method 
			evaluate_gradient, otherwise initialized as None.
		map_dphi: 1d-array
			The gradient of the input map with respect to phi in 
			spherical coordinates. The map is already divided by 
			sin(theta). Created after executing the method 
			evaluate_gradient, otherwise initialized as None.

		"""
		self.map_in = cmbmap
		self.cl_fid = fiducial_cls
		self._NSIDE_small = hp.npix2nside(self.map_in.size)
		self._NSIDE_large = 256 # Fixed
		self.ells = np.arange(3*self._NSIDE_small-1)

		# Set up high-pass filter boundaries
		if type(highell) == int:
			self.h_ells = [[highell, 3*self._NSIDE_small-1]]
		else:
			self.h_ells = []
			_ells = np.sort(highell)
			for i in range(len(highell)):
				if i == len(highell)-1:
					self.h_ells.append([_ells[i], 3*self._NSIDE_small-1])
				else:
					self.h_ells.append([_ells[i], _ells[i+1]])
			del _ells

		# Generate a noise spectrum for Wiener filtering purposes
		self.w = w * np.pi / 10800. # Convert to uK*rad
		self.beam = beam * np.pi / 10800. # Convert to rad
		wl = np.exp(-self.ells*(self.ells+1)*self.beam**2)
		self.Nl = self.w**2 / wl # Knox formula
		self.Nl[:2] = 1e-9 # Prevent divide by zero (Values not used anyway)
		del wl

		self.map_dtheta = None
		self.map_dphi = None


	def evaluate_gradient(self):
		"""Evaluate the 2D gradient of the CMB map.
		
		Evaluate the 2D gradient of the original input map across 
		the sky. This method uses the ``healpy`` function that returns 
		d/dtheta and d/dphi/sin(theta).

		"""
		alm = hp.map2alm(self.map_in)
		m, self.map_dtheta, self.map_dphi = hp.alm2map_der1(alm, 
															self._NSIDE_small)
		del m, alm

	def generate_filters(self):
		"""Create top hat-like Wiener filters.

		Generate top hat-like Wiener filters using the given high-pass 
		boundaries and set of fiducial Cls. If no fiducial Cls are given, 
		the input map Cls are used. A top hat filter is also generated for 
		the background temperature gradient for 0 < ell < 2000.

		Returns
		-------
		T_filters: 2d-array
			Top hat-like Wiener filters for every ell range given during 
			initialization.

		"""
		if self.cl_fid is None: # Construct Wiener filter
			total_Cl = hp.anafast(self.map_in)
			Wl = (total_Cl - self.Nl) / total_Cl
			del total_Cl
		else:
			fid = self.cl_fid[:3*self._NSIDE_small-1]
			Wl = fid / (fid + self.Nl)
			del fid
		T_filters = []
		for ls in self.h_ells:
			f = np.copy(Wl)
			f[:ls[0]] = 0.
			f[ls[1]:] = 0.
			T_filters.append(f)
		grad_filter = np.zeros(3*self._NSIDE_small-1)
		grad_filter[:2000] = 1.
		return T_filters, grad_filter


	def filter_all_maps(self, filter_grad=True):
		"""Filter the maps associated with this object.

		Perform a high-pass Wiener filter on the input CMB map. The Wiener 
		filter is constructed using the given fiducial Cls. If no fiducial 
		model is given, then the Cls of the input map are used. If the 
		gradient of the CMB map has already been taken, then this method 
		also low-passes those maps for ell < 2000 unless filter_grad=False.

		"""
		T_filters, dT_filter = self.generate_filters()
		map_alm = hp.map2alm(self.map_in) # Don't use filter_map; convert once
		self.map_filtered = []
		for f in T_filters:
			new_alm = hp.almxfl(map_alm, f)
			self.map_filtered.append(hp.alm2map(new_alm, self._NSIDE_small))
			del new_alm
		del map_alm

		if filter_grad:
			try:
				self.map_dtheta_f = filter_map(self.map_dtheta, dT_filter)
				self.map_dtphi_f = filter_map(self.map_dphi, dT_filter)
			except TypeError:
				print("Gradient not yet evaluated! Skipping filtering step.")
				pass
