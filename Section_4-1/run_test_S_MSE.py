#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from help_functions_S import compute_reduction, compute_MC, compute_sMC_old, compute_sMC_new

#%% Parameters

np.random.seed(2025)
torch.manual_seed(2025)
torch.set_default_tensor_type(torch.DoubleTensor)

#%% Model and data

d = 2
f = lambda x: torch.exp(0.7*x[0] + 0.3*x[1]) + 0.15*torch.sin(2*np.pi*x[0]) if x.ndim==1 else \
              torch.exp(0.7*x[:,0] + 0.3*x[:,1]) + 0.15*torch.sin(2*np.pi*x[:,0])
              
iterations = 1000
M = 100
  
data = 2*torch.rand((M, d)) - 1
f_data = f(data)

#%% Dimensionality reduction

r = 1
activation = torch.nn.Tanh()
epochs = 10000
    
layers_AE = 2
neurons_AE = 8
layers_surrogate = 2
neurons_surrogate = 8

autoencoder, surrogate, _ = compute_reduction(d, data, f_data, r, activation, layers_AE, neurons_AE, layers_surrogate, neurons_surrogate, epochs)
    
K = round(1e6)
data_CDF = 2*torch.rand((K, d)) - 1

F = ECDF(torch.squeeze(autoencoder.encoder(data_CDF)).detach())

latent_data = torch.squeeze(autoencoder.encoder(data_CDF)).detach()
sorted_latent_data = torch.sort(latent_data)[0]
Finv = lambda x: sorted_latent_data[torch.floor(K*x).int()*(x != 1) + (K-1)*(x == 1)]  if torch.is_tensor(x) else sorted_latent_data[np.floor(K*x).astype(int)*(x != 1) + (K-1)*(x == 1)]

#%% Monte Carlo
    
S_strata = [4, 9, 16, 25]
N_vec = [225, 450, 900, 1800, 3600]

MC = np.empty((1, len(N_vec), iterations))
sMC_old = np.empty((len(S_strata), len(N_vec), iterations))
sMC_new = np.empty((len(S_strata), len(N_vec), iterations))

for n, N in enumerate(N_vec):

    for it in tqdm(range(iterations)):
        MC[0, n, it]  = compute_MC(f, d, N)
    
    for s, S in enumerate(S_strata):
        
        for it in tqdm(range(iterations)):
            sMC_old[s, n, it] = compute_sMC_old(f, d, N, S)
            sMC_new[s, n, it] = compute_sMC_new(f, autoencoder, F, d, N, S)
        
#%% Save

for n, N in enumerate(N_vec):
    results = np.concatenate((MC[:,n,:], sMC_old[:,n,:], sMC_new[:,n,:]))
    np.savetxt("results/S_strata_MSE_" + str(N) + ".txt", results)
    
#%% MSE

Exact = (25/21)*(np.exp(7/10) - np.exp(-7/10))*(np.exp(3/10) - np.exp(-3/10))

MSE_MC = np.empty((len(N_vec),))
MSE_sMC_old = np.empty((len(S_strata), len(N_vec)))
MSE_sMC_new = np.empty((len(S_strata), len(N_vec)))

for n in range(len(N_vec)):
    MSE_MC[n] = np.mean((MC[0,n,:] - Exact)**2)
    for s in range(len(S_strata)):
        MSE_sMC_old[s,n] = np.mean((sMC_old[s,n,:] - Exact)**2)
        MSE_sMC_new[s,n] = np.mean((sMC_new[s,n,:] - Exact)**2)
            
#%% Plot

plt.figure()

plt.loglog(np.array(N_vec), MSE_MC, label="MC")
for s, S in enumerate(S_strata):
    plt.loglog(np.array(N_vec), MSE_sMC_old[s,:], label="sMC old, S = "+str(S))
plt.loglog(np.array(N_vec), 1/np.array(N_vec), label="1/N", linestyle='--')

plt.legend()
plt.show()

plt.figure()

plt.loglog(np.array(N_vec), MSE_MC, label="MC")
for s, S in enumerate(S_strata):
    plt.loglog(np.array(N_vec), MSE_sMC_new[s,:], label="sMC new, S = "+str(S))
plt.loglog(np.array(N_vec), 1/np.array(N_vec), label="1/N", linestyle='--')

plt.legend()
plt.show()