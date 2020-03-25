import numpy as np
import pystan as stan

lens_model = """
data {
	int<lower=0> N; // Number of large patches
	int<lower=0> M; // Number of small pixels
	vector[M] T; // T map with pixels grouped into patches
	vector[N] T_mean; // Mean T in each patch
	vector[N] dTx; // x-component of the mean T gradient in each patch
	vector[N] dTy; // y-component of the mean T gradient in each patch
}

parameters {
	vector[N] dPsix; // x-component of the lensing deflection in each patch
	vector[N] dPsiy; // y-component of the lensing deflection in each patch
	real<lower=0> dPsix_total2; // Total variance of lensing deflection in x
	real<lower=0> dPsiy_total2; // Total variance of lensing deflection in y
}

model{
	vector<lower=0>[N] T2; // Estimated T variance per patch due to lensing
	T2 = (dPsix .*dTx + dPsiy .*dTy).*(dPsix .*dTx + dPsiy .*dTy);
	T ~ normal(T_mean, T2)
	dPsix ~ normal(0, dPsix_total2)
	dPsiy ~ normal(0, dPsiy_total2)
}
"""