import numpy as np
import healpy as hp
#import stan as stan
from cmbpix.utils import *
#from cmbpix.lensing import prob_model as pm

class LensingEstimator():
	"""Estimator for small scale lensing in a CMB map.

	An object for reducing a CMB map of the cosmic microwave 
	background in (mostly) pixel-space to estimate the effects 
	of small angular scale weak lensing due to structure formation.

	Attributes
	----------
	map_in: 1d-array
		The input CMB map.

	"""
	
	def __init__(self, cmbmap):
		"""Initiate the estimator.

		Parameters
		----------
		cmbmap: nd-array
			A CMB map.
		"""
		self.map_in = cmbmap