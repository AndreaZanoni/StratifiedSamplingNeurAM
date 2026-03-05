#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
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
N = 1024
  
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

MC = np.empty((iterations))
sMC = np.empty((iterations))
LHS = np.empty((iterations))
qMC = np.empty((iterations))
    
for it in tqdm(range(iterations)):
    MC[it] = compute_MC(f, d, N)
    sMC[it] = compute_sMC(f, autoencoder, F, d, N, S)
    LHS[it] = compute_LHS(f, d, N)
    qMC[it] = compute_qMC(f, d, N)
        
#%% Save

results = np.stack((MC, sMC, LHS, qMC))
#np.savetxt("results/comparison_d" + str(d) + ".txt", results)
            
#%% Plot

Exact = 0

x_min = Exact - 0.05
x_max = Exact + 0.05
x_axis = np.linspace(x_min, x_max, 1000)

plt.figure()
plt.plot(x_axis, norm.pdf(x_axis, np.mean(MC), np.std(MC)), label="MC")
plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC), np.std(sMC)), label="sMC")
plt.plot(x_axis, norm.pdf(x_axis, np.mean(LHS), np.std(LHS)), label="LHS")
plt.plot(x_axis, norm.pdf(x_axis, np.mean(qMC), np.std(qMC)), label="qMC")
plt.axvline(x = Exact, linestyle='--')
plt.legend()
plt.show()