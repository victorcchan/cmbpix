# Tests for the estimator class in cmbpix

def test_estimator_init():
	# Test that the LensingEstimator class initiates as expected
	from cmbpix.lensing import estimator as le
	import numpy as np
	import healpy as hp

	# Create a "CMB map" at NSIDE=2048
	m = np.zeros(hp.nside2npix(2048))
	est = le.LensingEstimator(m)
	assert np.all(est.map_in == m), \
		"LensingEstimator does not keep input map"
	assert est._NSIDE_small == 2048, \
		"LensingEstimator does not infer map NSIDE corectly"
	assert len(est.ells) == 3*2048, \
		"LensingEstimator does not generate an appropriate range of ells"
	assert np.all(np.abs(est.Nl[2:]) < 1e-14), \
		"LensingEstimator does not give zero noise when expected"

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

def test_estimator_filters():
	# Test that LensingEstimator filters maps properly
	from cmbpix.lensing import estimator as le
	import numpy as np
	import healpy as hp

	# Create a "CMB map" at NSIDE=2048
	m = np.ones(hp.nside2npix(2048))

	# Test filters for default ell range: [3000, 6000]
	est = le.LensingEstimator(m)
	est.evaluate_gradient()
	T_f, dT_f = est.generate_filters()
	assert len(T_f) == 1, \
		"LensingEstimator does not generate 1 filter when given 1 highell"
	assert len(T_f[0] == 3*2048), \
		"LensingEstimator does not generate T filter values for correct ells"
	hpass_th_zero = np.all(T_f[0][:3000] == 0) & np.all(T_f[0][6000:] == 0)
	hpass_th_one = np.all(T_f[0][3000:6000] == 1)
	assert hpass_th_zero & hpass_th_one, \
		"LensingEstimator does not produce tophat filters for 0 instr. noise"
	assert len(dT_f == 3*2048), \
		"LensingEstimator does not generate dT filter values for correct ells"
	lpass_th_zero = np.all(dT_f[2000:] == 0)
	lpass_th_one = np.all(dT_f[:2000] == 1)
	assert lpass_th_zero & lpass_th_one, \
		"LensingEstimator does not produce the correct low pass top hat filter"