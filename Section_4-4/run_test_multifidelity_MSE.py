#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from help_functions_multifidelity import compute_reduction, compute_MC, compute_MFMC, compute_sMC_uniform, compute_sMC_uniform_stdprop, compute_sMFMC_uniform, compute_sMFMC_uniform_stdprop

#%% Parameters

np.random.seed(2025)
torch.manual_seed(2025)
torch.set_default_tensor_type(torch.DoubleTensor)

#%% Model and data

d = 2
f_HF = lambda x: torch.exp(0.7*x[0] + 0.3*x[1]) + 0.15*torch.sin(2*np.pi*x[0]) if x.ndim==1 else \
                 torch.exp(0.7*x[:,0] + 0.3*x[:,1]) + 0.15*torch.sin(2*np.pi*x[:,0])
                 
g_LF = lambda x: torch.exp(0.01*x[0] + 0.99*x[1]) + 0.15*torch.sin(3*np.pi*x[1]) if x.ndim==1 else \
                 torch.exp(0.01*x[:,0] + 0.99*x[:,1]) + 0.15*torch.sin(3*np.pi*x[:,1])
              
w = 0.01
iterations = 1000
M = 100
  
data = 2*torch.rand((M, d)) - 1
f_HF_data = f_HF(data)
g_LF_data = g_LF(data)

#%% Dimensionality reduction

r = 1
activation = torch.nn.Tanh()
epochs = 10000
    
layers_AE = 2
neurons_AE = 8
layers_surrogate = 2
neurons_surrogate = 8

autoencoder_HF, surrogate_HF, _ = compute_reduction(d, data, f_HF_data, r, activation, layers_AE, neurons_AE, layers_surrogate, neurons_surrogate, epochs)
autoencoder_LF, surrogate_LF, _ = compute_reduction(d, data, g_LF_data, r, activation, layers_AE, neurons_AE, layers_surrogate, neurons_surrogate, epochs)
    
K = round(1e6)
data_CDF = 2*torch.rand((K, d)) - 1

F_HF = ECDF(torch.squeeze(autoencoder_HF.encoder(data_CDF)).detach())
F_LF = ECDF(torch.squeeze(autoencoder_LF.encoder(data_CDF)).detach())

latent_data_HF = torch.squeeze(autoencoder_HF.encoder(data_CDF)).detach()
sorted_latent_data_HF = torch.sort(latent_data_HF)[0]
Finv_HF = lambda x: sorted_latent_data_HF[torch.floor(K*x).int()*(x != 1) + (K-1)*(x == 1)]  if torch.is_tensor(x) else sorted_latent_data_HF[np.floor(K*x).astype(int)*(x != 1) + (K-1)*(x == 1)]

latent_data_LF = torch.squeeze(autoencoder_LF.encoder(data_CDF)).detach()
sorted_latent_data_LF = torch.sort(latent_data_LF)[0]
Finv_LF = lambda x: sorted_latent_data_LF[torch.floor(K*x).int()*(x != 1) + (K-1)*(x == 1)]  if torch.is_tensor(x) else sorted_latent_data_LF[np.floor(K*x).astype(int)*(x != 1) + (K-1)*(x == 1)]

f_LF = lambda x: g_LF(autoencoder_LF.decoder(torch.unsqueeze(Finv_LF(F_HF(torch.squeeze(autoencoder_HF.encoder(x)).detach())), 1))).detach()

#%% Monte Carlo

N_vec = [125, 250, 500, 1000, 2000]

S = 5

MC = np.empty((len(N_vec), iterations))
sMC_uniform = np.empty((len(N_vec), iterations))
sMC_uniform_stdprop = np.empty((len(N_vec), iterations))

MFMC = np.empty((len(N_vec), iterations))
sMFMC_uniform = np.empty((len(N_vec), iterations))
sMFMC_uniform_stdprop = np.empty((len(N_vec), iterations))

for n, N in enumerate(N_vec):

    for it in tqdm(range(iterations)):
        
        MC[n,it]                    = compute_MC(f_HF, d, N)
        sMC_uniform[n,it]           = compute_sMC_uniform(f_HF, autoencoder_HF, F_HF, d, N, S)
        sMC_uniform_stdprop[n,it]   = compute_sMC_uniform_stdprop(f_HF, autoencoder_HF, surrogate_HF, F_HF, Finv_HF, d, N, S)
    
        MFMC[n,it]                  = compute_MFMC(f_HF, f_LF, surrogate_HF, surrogate_LF, Finv_HF, Finv_LF, d, N, w)
        sMFMC_uniform[n,it]         = compute_sMFMC_uniform(f_HF, f_LF, autoencoder_HF, surrogate_HF, surrogate_LF, F_HF, Finv_HF, Finv_LF, d, N, S, w)
        sMFMC_uniform_stdprop[n,it] = compute_sMFMC_uniform_stdprop(f_HF, f_LF, autoencoder_HF, surrogate_HF, surrogate_LF, F_HF, Finv_HF, Finv_LF, d, N, S, w)

#%% Save

for n, N in enumerate(N_vec):
    results = np.stack((MC[n,:], sMC_uniform[n,:], sMC_uniform_stdprop[n,:], MFMC[n,:], sMFMC_uniform[n,:], sMFMC_uniform_stdprop[n,:]))
    np.savetxt("results/multifidelity_MSE_" + str(N) + ".txt", results) 
    
#%% MSE

Exact = (25/21)*(np.exp(7/10) - np.exp(-7/10))*(np.exp(3/10) - np.exp(-3/10))

MSE_MC = np.empty((len(N_vec),))
MSE_sMC_uniform = np.empty((len(N_vec),))
MSE_sMC_uniform_stdprop = np.empty((len(N_vec),))

MSE_MFMC = np.empty((len(N_vec),))
MSE_sMFMC_uniform = np.empty((len(N_vec),))
MSE_sMFMC_uniform_stdprop = np.empty((len(N_vec),))

for n in range(len(N_vec)):
    
    MSE_MC[n] = np.mean((MC[n,:] - Exact)**2)
    MSE_sMC_uniform[n] = np.mean((sMC_uniform[n,:] - Exact)**2)
    MSE_sMC_uniform_stdprop[n] = np.mean((sMC_uniform_stdprop[n,:] - Exact)**2)
    
    MSE_MFMC[n] = np.mean((MFMC[n,:] - Exact)**2)
    MSE_sMFMC_uniform[n] = np.mean((sMFMC_uniform[n,:] - Exact)**2)
    MSE_sMFMC_uniform_stdprop[n] = np.mean((sMFMC_uniform_stdprop[n,:] - Exact)**2)
    
#%% Plot

plt.figure()

plt.loglog(np.array(N_vec), MSE_MC, label="MC")
plt.loglog(np.array(N_vec), MSE_MFMC, label="MFMC")
plt.loglog(np.array(N_vec), MSE_sMC_uniform, label="sMC")
plt.loglog(np.array(N_vec), MSE_sMFMC_uniform, label="sMFMC")
plt.loglog(np.array(N_vec), 1/np.array(N_vec), label="1/N", linestyle='--')

plt.legend()
plt.show()

plt.figure()

plt.loglog(np.array(N_vec), MSE_MC, label="MC")
plt.loglog(np.array(N_vec), MSE_MFMC, label="MFMC")
plt.loglog(np.array(N_vec), MSE_sMC_uniform_stdprop, label="sMC")
plt.loglog(np.array(N_vec), MSE_sMFMC_uniform_stdprop, label="sMFMC")
plt.loglog(np.array(N_vec), 1/np.array(N_vec), label="1/N", linestyle='--')

plt.legend()
plt.show()