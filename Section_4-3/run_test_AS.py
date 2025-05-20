#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from help_functions_AS import compute_reduction, compute_MC, compute_sMC, compute_sMC_AS
from scipy.special import erf, erfinv
from colorsys import hls_to_rgb

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
N = 1024
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
C /= N
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

S = 8

MC = np.empty(iterations)
sMC = np.empty(iterations)
sMC_AS = np.empty(iterations)

for it in tqdm(range(iterations)):
    MC[it]  = compute_MC(f, d, N)
    sMC[it] = compute_sMC(f, autoencoder, F, d, N, S)
    sMC_AS[it] = compute_sMC_AS(f, AS, d, N, S)
            
#%% Plot

Exact = (25/21)*(np.exp(7/10) - np.exp(-7/10))*(np.exp(3/10) - np.exp(-3/10))

x_min = Exact - 0.03
x_max = Exact + 0.03
x_axis = np.linspace(x_min, x_max, 1000)

plt.figure()

plt.plot(x_axis, norm.pdf(x_axis, np.mean(MC), np.std(MC)), label="MC")
plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC), np.std(sMC)), label="sMC (NeurAM)")
plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC_AS), np.std(sMC_AS)), label="sMC (AS)")

plt.axvline(x = Exact, color='r', linestyle='--')
plt.legend()
plt.ylim(0, 150)
plt.title('M = ' + str(M))

plt.show()

#%% Save

results = np.stack((MC, sMC, sMC_AS))
np.savetxt("results/AS_" + str(S) + ".txt", results)

#%% Stratification

def get_distinct_colors(n):
    colors = []
    for i in np.arange(0., 360., 360. / n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))
    return colors

points_stratification = 100000
    
cmap = get_distinct_colors(S)
np.savetxt("results/stratification/cmap_S" + str(S) + ".txt", np.array(cmap))
size = 1/S
data_stratification = 2*torch.rand((points_stratification, d)) - 1

plt.figure()
latent_data_stratification = F(torch.squeeze(autoencoder.encoder(data_stratification)).detach())
for i in range(S):
    samples = data_stratification[(latent_data_stratification >= i*size)*(latent_data_stratification < (i+1)*size)]
    np.savetxt("results/stratification/samples_S" + str(S) + str(i) + ".txt", samples)
    plt.scatter(samples[:,0], samples[:,1], color=cmap[i])
plt.title('S = ' + str(S))
plt.show()

plt.figure()
latent_data_stratification = (omega(torch.matmul(inv_omega(data_stratification), AS)).detach() + 1)/2
for i in range(S):
    samples = data_stratification[(latent_data_stratification >= i*size)*(latent_data_stratification < (i+1)*size)]
    np.savetxt("results/stratification/samples_S" + str(S) + str(i) + "_AS.txt", samples)
    plt.scatter(samples[:,0], samples[:,1], color=cmap[S-1-i])
plt.title('S = ' + str(S))
plt.show()