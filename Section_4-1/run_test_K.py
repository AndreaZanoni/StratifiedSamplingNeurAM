#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from help_functions_MK import compute_reduction, compute_MC, compute_sMC

#%% Parameters

np.random.seed(2025)
torch.manual_seed(2025)
torch.set_default_tensor_type(torch.DoubleTensor)

#%% Model and data

d = 2
f = lambda x: torch.exp(0.7*x[0] + 0.3*x[1]) + 0.15*torch.sin(2*np.pi*x[0]) if x.ndim==1 else \
              torch.exp(0.7*x[:,0] + 0.3*x[:,1]) + 0.15*torch.sin(2*np.pi*x[:,0])
              
iterations = 1000
N = 1024
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
    
K_CDF = [round(1e3), round(1e4), round(1e5), round(1e6)]

MC = np.empty((1, iterations))
sMC = np.empty((len(K_CDF), iterations))

for it in tqdm(range(iterations)):
    MC[0, it]  = compute_MC(f, d, N)

for k, K in enumerate(K_CDF):
    
    data_CDF = 2*torch.rand((K, d)) - 1
    
    F = ECDF(torch.squeeze(autoencoder.encoder(data_CDF)).detach())
    
    latent_data = torch.squeeze(autoencoder.encoder(data_CDF)).detach()
    sorted_latent_data = torch.sort(latent_data)[0]
    Finv = lambda x: sorted_latent_data[torch.floor(K*x).int()*(x != 1) + (K-1)*(x == 1)]  if torch.is_tensor(x) else sorted_latent_data[np.floor(K*x).astype(int)*(x != 1) + (K-1)*(x == 1)]
    
    #%% Monte Carlo
    
    S = 2
    
    for it in tqdm(range(iterations)):
        sMC[k, it] = compute_sMC(f, autoencoder, F, d, N, S)
            
#%% Plot

Exact = (25/21)*(np.exp(7/10) - np.exp(-7/10))*(np.exp(3/10) - np.exp(-3/10))

x_min = Exact - 0.03
x_max = Exact + 0.03
x_axis = np.linspace(x_min, x_max, 1000)

plt.figure()

plt.plot(x_axis, norm.pdf(x_axis, np.mean(MC), np.std(MC)), label="MC")
for k, K in enumerate(K_CDF):
    plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC[k,:]), np.std(sMC[k,:])), label="sMC, K = "+str(K))

plt.axvline(x = Exact, color='r', linestyle='--')
plt.legend()
plt.ylim(0, 60)
plt.title('M = ' + str(M))

plt.show()

#%% Save

results = np.concatenate((MC, sMC))
np.savetxt("results/K_CDF.txt", results)
