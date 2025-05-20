#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
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
N = 1000
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

S = 10

MC = np.empty((iterations,))


sMC_uniform = np.empty((iterations,))
sMC_bisection = np.empty((iterations,))
sMC_optimal = np.empty((iterations,))

sMC_uniform_stdprop = np.empty((iterations,))
sMC_bisection_stdprop = np.empty((iterations,))
sMC_optimal_stdprop = np.empty((iterations,))

for it in tqdm(range(iterations)):
    
    MC[it]                    = compute_MC(f, d, N)
    
    sMC_uniform[it]           = compute_sMC_uniform(f, autoencoder, F, d, N, S)
    sMC_bisection[it]         = compute_sMC_bisection(f, autoencoder, surrogate, F, Finv, d, N, S)
    sMC_optimal[it]           = compute_sMC_optimal(f, autoencoder, surrogate, F, Finv, d, N, S)
    
    sMC_uniform_stdprop[it]   = compute_sMC_uniform_stdprop(f, autoencoder, surrogate, F, Finv, d, N, S)
    sMC_bisection_stdprop[it] = compute_sMC_bisection_stdprop(f, autoencoder, surrogate, F, Finv, d, N, S)
    sMC_optimal_stdprop[it]   = compute_sMC_optimal_stdprop(f, autoencoder, surrogate, F, Finv, d, N, S)
            
#%% Plot

Exact = (25/21)*(np.exp(7/10) - np.exp(-7/10))*(np.exp(3/10) - np.exp(-3/10))

x_min = Exact - 0.005
x_max = Exact + 0.005
x_axis = np.linspace(x_min, x_max, 1000)

plt.figure()

plt.plot(x_axis, norm.pdf(x_axis, np.mean(MC), np.std(MC)), label="MC")

plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC_uniform), np.std(sMC_uniform)), label="sMC_uniform")
plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC_bisection), np.std(sMC_bisection)), label="sMC_bisection")
plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC_optimal), np.std(sMC_optimal)), label="sMC_optimal")

plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC_uniform_stdprop), np.std(sMC_uniform_stdprop)), label="sMC_uniform_stdprop")
plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC_bisection_stdprop), np.std(sMC_bisection_stdprop)), label="sMC_bisection_stdprop")
plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC_optimal_stdprop), np.std(sMC_optimal_stdprop)), label="sMC_optimal_stdprop")

plt.axvline(x = Exact, color='r', linestyle='--')
plt.legend()
plt.ylim(0, 350)

plt.show()

#%% Save

results = np.stack((MC, sMC_uniform, sMC_bisection, sMC_optimal, sMC_uniform_stdprop, sMC_bisection_stdprop, sMC_optimal_stdprop))
np.savetxt("results/heuristic.txt", results)