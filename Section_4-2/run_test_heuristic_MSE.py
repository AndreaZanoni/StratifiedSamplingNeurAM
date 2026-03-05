#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from help_functions_heuristic import compute_reduction, compute_MC, compute_sMC_uniform, compute_sMC_bisection, compute_sMC_optimal, compute_sMC_uniform_stdprop, compute_sMC_bisection_stdprop, compute_sMC_optimal_stdprop

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

N_vec = [250, 500, 1000]

S = 10

MC = np.empty((len(N_vec), iterations))

sMC_uniform = np.empty((len(N_vec), iterations))
sMC_bisection = np.empty((len(N_vec), iterations))
sMC_optimal = np.empty((len(N_vec), iterations))

sMC_uniform_stdprop = np.empty((len(N_vec), iterations))
sMC_bisection_stdprop = np.empty((len(N_vec), iterations))
sMC_optimal_stdprop = np.empty((len(N_vec), iterations))

for n, N in enumerate(N_vec):

    for it in tqdm(range(iterations)):
        
        MC[n,it]                    = compute_MC(f, d, N)
        
        sMC_uniform[n,it]           = compute_sMC_uniform(f, autoencoder, F, d, N, S)
        sMC_bisection[n,it]         = compute_sMC_bisection(f, autoencoder, surrogate, F, Finv, d, N, S)
        sMC_optimal[n,it]           = compute_sMC_optimal(f, autoencoder, surrogate, F, Finv, d, N, S)
        
        sMC_uniform_stdprop[n,it]   = compute_sMC_uniform_stdprop(f, autoencoder, surrogate, F, Finv, d, N, S)
        sMC_bisection_stdprop[n,it] = compute_sMC_bisection_stdprop(f, autoencoder, surrogate, F, Finv, d, N, S)
        sMC_optimal_stdprop[n,it]   = compute_sMC_optimal_stdprop(f, autoencoder, surrogate, F, Finv, d, N, S)

#%% Save

for n, N in enumerate(N_vec):
    results = np.stack((MC[n,:], sMC_uniform[n,:], sMC_bisection[n,:], sMC_optimal[n,:], sMC_uniform_stdprop[n,:], sMC_bisection_stdprop[n,:], sMC_optimal_stdprop[n,:]))
    np.savetxt("results/heuristic_MSE_" + str(N) + ".txt", results)    
    
#%% MSE

Exact = (25/21)*(np.exp(7/10) - np.exp(-7/10))*(np.exp(3/10) - np.exp(-3/10))

MSE_MC = np.empty((len(N_vec),))

MSE_sMC_uniform = np.empty((len(N_vec),))
MSE_sMC_bisection = np.empty((len(N_vec),))
MSE_sMC_optimal = np.empty((len(N_vec),))

MSE_sMC_uniform_stdprop = np.empty((len(N_vec),))
MSE_sMC_bisection_stdprop = np.empty((len(N_vec),))
MSE_sMC_optimal_stdprop = np.empty((len(N_vec),))

for n in range(len(N_vec)):
    
    MSE_MC[n] = np.mean((MC[n,:] - Exact)**2)
    
    MSE_sMC_uniform[n] = np.mean((sMC_uniform[n,:] - Exact)**2)
    MSE_sMC_bisection[n] = np.mean((sMC_bisection[n,:] - Exact)**2)
    MSE_sMC_optimal[n] = np.mean((sMC_optimal[n,:] - Exact)**2)
    
    MSE_sMC_uniform_stdprop[n] = np.mean((sMC_uniform_stdprop[n,:] - Exact)**2)
    MSE_sMC_bisection_stdprop[n] = np.mean((sMC_bisection_stdprop[n,:] - Exact)**2)
    MSE_sMC_optimal_stdprop[n] = np.mean((sMC_optimal_stdprop[n,:] - Exact)**2)

#%% Plot

plt.figure()

plt.loglog(np.array(N_vec), MSE_MC, label="MC")
plt.loglog(np.array(N_vec), MSE_sMC_uniform, label="sMC (u)")
plt.loglog(np.array(N_vec), MSE_sMC_bisection, label="sMC (b)")
plt.loglog(np.array(N_vec), MSE_sMC_optimal, label="sMC (o)")
plt.loglog(np.array(N_vec), 1/np.array(N_vec), label="1/N", linestyle='--')

plt.legend()
plt.show()

plt.figure()

plt.loglog(np.array(N_vec), MSE_MC, label="MC")
plt.loglog(np.array(N_vec), MSE_sMC_uniform_stdprop, label="sMC (u)")
plt.loglog(np.array(N_vec), MSE_sMC_bisection_stdprop, label="sMC (b)")
plt.loglog(np.array(N_vec), MSE_sMC_optimal_stdprop, label="sMC (o)")
plt.loglog(np.array(N_vec), 1/np.array(N_vec), label="1/N", linestyle='--')

plt.legend()
plt.show()