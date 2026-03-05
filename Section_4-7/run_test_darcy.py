#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from help_functions_darcy import compute_reduction, compute_MC, compute_sMC

#%% Parameters

np.random.seed(2025)
torch.manual_seed(2025)
torch.set_default_tensor_type(torch.DoubleTensor)

#%% Model and data

d = 100
L = 5
mu = lambda n: torch.randn(n, d)
    
data = np.load("solutions/dim" + str(d) + "/inputs_dim" + str(d) + "_concatenated.npy")

x_grid = np.load("solutions/dim" + str(d) + "/coords_x_dim" + str(d) + "_concatenated.npy")[0,:]
y_grid = np.load("solutions/dim" + str(d) + "/coords_y_dim" + str(d) + "_concatenated.npy")[0,:]
k = np.load("solutions/dim" + str(d) + "/permeability_dim" + str(d) + "_concatenated.npy")
p = np.load("solutions/dim" + str(d) + "/pressure_dim" + str(d) + "_concatenated.npy")
v_x = np.load("solutions/dim" + str(d) + "/velocity_x_dim" + str(d) + "_concatenated.npy")
v_y = np.load("solutions/dim" + str(d) + "/velocity_y_dim" + str(d) + "_concatenated.npy")

Q = np.mean(v_x**2 + v_y**2, axis=1)

M = 2000
K = round(1e6)
iterations = 20
N = 512

#%% Dimensionality reduction

data_reduction = torch.from_numpy(data[-M:,:])
Q_reduction = torch.from_numpy(Q[-M:])

r = 1
activation = torch.nn.Tanh()
epochs = 10000
    
layers_AE = 2
neurons_AE = 8
layers_surrogate = 2
neurons_surrogate = 8

autoencoder, surrogate, _ = compute_reduction(d, L, data_reduction, Q_reduction, r, activation, layers_AE, neurons_AE, layers_surrogate, neurons_surrogate, epochs)

data_CDF = mu(K)
F = ECDF(torch.squeeze(autoencoder.encoder(data_CDF)).detach())

latent_data = torch.squeeze(autoencoder.encoder(data_CDF)).detach()
sorted_latent_data = torch.sort(latent_data)[0]
Finv = lambda x: sorted_latent_data[torch.floor(K*x).int()*(x != 1) + (K-1)*(x == 1)]  if torch.is_tensor(x) else sorted_latent_data[np.floor(K*x).astype(int)*(x != 1) + (K-1)*(x == 1)]

#%% Monte Carlo

S = 16
N_it = round(N*7/4)

MC = np.empty((iterations,))
sMC = np.empty((iterations,))

for it in tqdm(range(iterations)):
    
    data_it = torch.from_numpy(data[it*N_it:(it+1)*N_it,:])
    Q_it = torch.from_numpy(Q[it*N_it:(it+1)*N_it])
    
    MC[it]  = compute_MC(Q_it, N)
    sMC[it] = compute_sMC(data_it, Q_it, N, autoencoder, F, S)
            
#%% Plot

x_min = np.mean(MC) - 3*np.std(MC)
x_max = np.mean(MC) + 3*np.std(MC)
x_axis = np.linspace(x_min, x_max, 1000)

plt.figure()
plt.plot(x_axis, norm.pdf(x_axis, np.mean(MC), np.std(MC)), label="MC")
plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC), np.std(sMC)), label="sMC")
plt.legend()
plt.ylim(0, 1.1*norm.pdf(x_axis, np.mean(sMC), np.std(sMC)).max())

#%% Save

results = np.stack((MC, sMC))
np.savetxt("results/darcy_dim" + str(d) + "_S" + str(S) + "_M" + str(M) + "_N" + str(N) + "_iterations" + str(iterations) + ".txt", results)

np.savetxt("results/darcy_dim" + str(d) + "_x.txt", x_grid)
np.savetxt("results/darcy_dim" + str(d) + "_y.txt", y_grid)
np.savetxt("results/darcy_dim" + str(d) + "_vx.txt", v_x[0,:])
np.savetxt("results/darcy_dim" + str(d) + "_vy.txt", v_y[0,:])
np.savetxt("results/darcy_dim" + str(d) + "_k.txt", k[0,:])
np.savetxt("results/darcy_dim" + str(d) + "_p.txt", p[0,:])