# Tests for the estimator class in cmbpix

def test_estimator_init():
	# Test that the LensingEstimator class initiates as expected
	from cmbpix.lensing import estimator as le
	import numpy as np
	import healpy as hp

	# Create a "CMB map"
	m = np.zeros(hp.nside2npix(4))
	est = le.LensingEstimator(m)
	assert np.all(est.map_in == m), \
		"LensingEstimator does not keep input map"
	assert est._NSIDE_small == 4, \
		"LensingEstimator does not infer map NSIDE corectly"

def test_estimator_gradient():
	# Test that LensingEstimator takes/saves the correct derivatives
	from cmbpix.lensing import estimator as le
	import numpy as np
	import healpy as hp

	# Create a zero map ==> zero derivatives
	m = np.zeros(hp.nside2npix(4))
	est = le.LensingEstimator(m)
	est.evaluate_gradient()
	assert np.all(est.map_dtheta == m), \
		"LensingEstimator does not take a zero map's dtheta correctly"
	assert np.all(est.map_dphi == m), \
		"LensingEstimator does not take a zero map's dphi correctly"