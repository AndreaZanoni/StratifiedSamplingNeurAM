#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from help_functions_AS import compute_reduction, compute_MC, compute_sMC, compute_sMC_AS
from scipy.special import erf, erfinv

#%% Parameters

np.random.seed(2025)
torch.manual_seed(2025)
torch.set_default_tensor_type(torch.DoubleTensor)

#%% Model and data

d = 2
f = lambda x: torch.exp(0.7*x[0] + 0.3*x[1]) + 0.15*torch.sin(2*np.pi*x[0]) if x.ndim==1 else \
              torch.exp(0.7*x[:,0] + 0.3*x[:,1]) + 0.15*torch.sin(2*np.pi*x[:,0])
              
dfdx0 = lambda x: 0.7*torch.exp(0.7*x[0] + 0.3*x[1]) + 0.3*np.pi*torch.cos(2*np.pi*x[0])
dfdx1 = lambda x: 0.3*torch.exp(0.7*x[0] + 0.3*x[1])
              
iterations = 1000
M = 100
  
data = 2*torch.rand((M, d)) - 1
f_data = f(data)

omega = lambda csi: erf(csi/np.sqrt(2))
inv_omega = lambda csi: np.sqrt(2)*erfinv(csi)
domega = lambda csi: np.sqrt(2/np.pi)*torch.exp(-csi**2/2)

grad_f = lambda csi: torch.tensor([dfdx0(torch.squeeze(omega(csi)))*domega(csi)[0], dfdx1(torch.squeeze(omega(csi)))*domega(csi)[1]])

#%% Active subspace

data_AS = inv_omega(data)

C = np.zeros((d, d))
for x in tqdm(data_AS):
    g = grad_f(x).numpy()
    C += np.outer(g, g)
C /= M
Lambda, W = np.linalg.eig(C)
idx = Lambda.argsort()[::-1]
Lambda = Lambda[idx]
W = W[:,idx]
AS = torch.from_numpy(W[:,0])

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

N_vec = [64, 128, 256, 512, 1024]

S = 8

MC = np.empty((len(N_vec), iterations))
sMC = np.empty((len(N_vec), iterations))
sMC_AS = np.empty((len(N_vec), iterations))

for n, N in enumerate(N_vec):

    for it in tqdm(range(iterations)):
        MC[n,it]  = compute_MC(f, d, N)
        sMC[n,it] = compute_sMC(f, autoencoder, F, d, N, S)
        sMC_AS[n,it] = compute_sMC_AS(f, AS, d, N, S)

#%% Save

for n, N in enumerate(N_vec):
    results = np.stack((MC[n,:], sMC[n,:], sMC_AS[n,:]))
    np.savetxt("results/AS_" + str(S) + "_MSE_" + str(N) + ".txt", results)

#%% MSE

Exact = (25/21)*(np.exp(7/10) - np.exp(-7/10))*(np.exp(3/10) - np.exp(-3/10))

MSE_MC = np.empty((len(N_vec),))
MSE_sMC = np.empty((len(N_vec),))
MSE_sMC_AS = np.empty((len(N_vec),))

for n in range(len(N_vec)):
    MSE_MC[n] = np.mean((MC[n,:] - Exact)**2)
    MSE_sMC[n] = np.mean((sMC[n,:] - Exact)**2)
    MSE_sMC_AS[n] = np.mean((sMC_AS[n,:] - Exact)**2)
    
#%% Plot

plt.figure()

plt.loglog(np.array(N_vec), MSE_MC, label="MC")
plt.loglog(np.array(N_vec), MSE_sMC, label="sMC")
plt.loglog(np.array(N_vec), MSE_sMC_AS, label="sMC AS")
plt.loglog(np.array(N_vec), 1/np.array(N_vec), label="1/N", linestyle='--')

plt.legend()
plt.show()