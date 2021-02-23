# Tests for utils.py in the cmbpix package

def test_patches_works():
	# Test that patches works for known cases, as well as for self-consistency
	from cmbpix import patches
	import numpy as np

	# Index 0 @ NSIDE=1 contains only these 4 indices @ NSIDE=2
	patch0nside1to2 = np.array([0,4,5,13])
	patch0nside1to4 = np.array([])
	patches_says2 = patches(0, 1, 2)
	patches_says4 = patches(0, 1, 4)
	for i in patch0nside1to2:
		assert i in patches_says2, \
			"cmbpix.patches does not include {} in 0 from NSIDE 1->2".format(i)
		# It also contains these 4 indices @ NSIDE=4 (and more)
		assert i in patches_says4, \
			"cmbpix.patches does not include {} in 0 from NSIDE 1->4".format(i)
		# Use these subindices to build the next level of subindices @ NSIDE=4
		patch0nside1to4 = np.concatenate((patch0nside1to4, patches(i, 2, 4)))
	for i in patch0nside1to4:
		# Check that the multi-level search works
		assert i in patches_says4, \
			"cmbpix.patches does not include {} in 0 from NSIDE 1->4".format(i)

def test_patches_properties():
	# Test that patches returns the expected number of appropriate subindices
	from cmbpix import patches
	import numpy as np
	import healpy as hp

	# Check that patches returns 4 subindices for each NSIDE factor of 2
	assert len(patches(0, 1, 2)) == 4, \
		"cmbpix.patches does not return 4 subindices for NSIDE 1->2"
	assert len(patches(0, 2, 4)) == 4, \
		"cmbpix.patches does not return 4 subindices for NSIDE 2->4"
	patches0nside1to4 = patches(0, 1, 4)
	assert len(patches0nside1to4) == 4**2, \
		"cmbpix.patches does not return 16 subindices for NSIDE 1->4"
	patches0nside1to512 = patches(0, 1, 512)
	assert len(patches0nside1to512) == 4**9,  \
		"cmbpix.patches does not return 262144 subindices for NSIDE 1->512"

	# Check that patches returns valid subindices
	assert np.all(patches0nside1to4 < hp.nside2npix(4)), \
		"cmbpix.patches does not return valid subindices for NSIDE 1->4"
	assert np.all(patches0nside1to512 < hp.nside2npix(512)), \
		"cmbpix.patches does not return valid subindices for NSIDE 1->512"