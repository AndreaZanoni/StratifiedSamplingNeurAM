#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from help_functions_comparison import compute_reduction, compute_MC, compute_sMC, compute_LHS, compute_qMC

#%% Parameters

np.random.seed(2025)
torch.manual_seed(2025)
torch.set_default_tensor_type(torch.DoubleTensor)

#%% Model and data

d = 5
f = lambda x: torch.sin(torch.sum(x)) if x.ndim==1 else \
              torch.sin(torch.sum(x, dim=1))
              
iterations = 1000
M = 1000
  
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

S = 16
N_vec = [64, 128, 256, 512, 1024]

MC = np.empty((len(N_vec), iterations))
sMC = np.empty((len(N_vec), iterations))
LHS = np.empty((len(N_vec), iterations))
qMC = np.empty((len(N_vec), iterations))
    
for n, N in enumerate(N_vec):
    for it in tqdm(range(iterations)):
        MC[n,it] = compute_MC(f, d, N)
        sMC[n,it] = compute_sMC(f, autoencoder, F, d, N, S)
        LHS[n,it] = compute_LHS(f, d, N)
        qMC[n,it] = compute_qMC(f, d, N)
        
#%% Save

for n, N in enumerate(N_vec):
    results = np.stack((MC[n,:], sMC[n,:], LHS[n,:], qMC[n,:]))
    np.savetxt("results/comparison_MSE_" + str(N) + "_d" + str(d) + ".txt", results)
    
#%% MSE

Exact = 0

MSE_MC = np.empty((len(N_vec),))
MSE_sMC = np.empty((len(N_vec),))
MSE_LHS = np.empty((len(N_vec),))
MSE_qMC = np.empty((len(N_vec),))

for n in range(len(N_vec)):
    MSE_MC[n] = np.nanmean((MC[n,:] - Exact)**2)
    MSE_sMC[n] = np.nanmean((sMC[n,:] - Exact)**2)
    MSE_LHS[n] = np.nanmean((LHS[n,:] - Exact)**2)
    MSE_qMC[n] = np.nanmean((qMC[n,:] - Exact)**2)
            
#%% Plot

plt.figure()
plt.loglog(np.array(N_vec), MSE_MC, label="MC")
plt.loglog(np.array(N_vec), MSE_sMC, label="sMC")
plt.loglog(np.array(N_vec), MSE_LHS, label="LHS")
plt.loglog(np.array(N_vec), MSE_qMC, label="qMC")
plt.loglog(np.array(N_vec), 1/np.array(N_vec), label="1/N", linestyle='--')
plt.legend()
plt.show()