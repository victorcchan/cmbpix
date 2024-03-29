import numpy as np
from stan import StanModel

lens_code = """
data {
	int<lower=0> N; // Number of large patches
	int<lower=0> M; // Number of small pixels
	vector[M] T; // T map with pixels grouped into patches
	vector[N] T_mean; // Mean T in each patch
	vector[N] dTx; // x-mean T gradient in each patch
	vector[N] dTy; // y-mean T gradient in each patch
	int ii[M]; // Indices to broadcast N-sized vectors to M
}

parameters {
	vector[M] dPsix; // x-lensing deflection in each patch
	vector[M] dPsiy; // y-lensing deflection in each patch
	real dPsi_total; // Log total lensing variance
}

transformed parameters {
	vector<lower=0>[M] T2; // Estimated T variance from lensing/patch
	vector<lower=0>[M] dPsi2; // 
	T2 = (dPsix .*dTx[ii] + dPsiy .*dTy[ii]) .* \
		(dPsix .*dTx[ii] + dPsiy .*dTy[ii]);
}

model{
	T ~ normal(T_mean[ii], T2[ii]);
	dPsix ~ normal(0, exp(dPsix_total));
	dPsiy ~ normal(0, exp(dPsiy_total));
}
"""

def build_lens_model():
	"""Return a StanModel object for the pixel-space CMB lensing estimator.

	Return a StanModel object for the pixel-space CMB lensing estimator. 
	Requires a working version of PyStan to be installed. 
	The object requires the following in its input data: 
	``N`` - (int) The number of large patches, 
	``M`` - (int) The total number of pixels in the map, 
	``T`` - (1d-array) The filtered T map with indices grouped into patches, 
	``T_mean`` - (1d-array) The mean temperature within each patch, 
	``dTx`` - (1d-array) The mean x-background temperature gradient, 
	``dTy`` - (1d-array) The mean y-background temperature gradient, and
	``ii`` - (1d-array) The corresponding patch index for each pixel.
	The model contains the following parameters:
	``dPsix`` - (1d-array) The estimated x-deflection in each patch, 
	``dPsiy`` - (1d-array) The estimated y-deflection in each patch, 
	``dPsix_total`` - (float) The log total variance in x-deflection, and 
	``dPsiy_total`` - (float) The log total variance in y-deflection.

	Returns
	-------
	lens_model: StanModel object
		A StanModel object for the pixel-space CMB lensing estimator.
		
	"""
	lens_model = StanModel(model_code=lens_code)
	return lens_model