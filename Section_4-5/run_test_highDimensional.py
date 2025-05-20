#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Modules

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from help_functions_highDimensional import compute_reduction, compute_MC, compute_sMC_old, compute_sMC_new

#%% Parameters

np.random.seed(2025)
torch.manual_seed(2025)
torch.set_default_tensor_type(torch.DoubleTensor)

#%% Info

M = 100
iterations = 1000
N = 1024

#%% Model and data

name = "Borehole"

if name == "Ishigami":
    d = 3
    f = lambda x: torch.sin(np.pi*x[0]) + 7*torch.sin(np.pi*x[1])**2 + 0.1*(np.pi*x[2]**4)*torch.sin(np.pi*x[0]) if x.ndim==1 else \
                  torch.sin(np.pi*x[:,0]) + 7*torch.sin(np.pi*x[:,1])**2 + 0.1*(np.pi*x[:,2]**4)*torch.sin(np.pi*x[:,0])
    mu = lambda n: torch.from_numpy(np.random.uniform(-1, 1, (n, d)))

elif name == "HighDimensional":
    d = 10
    f = lambda x: torch.exp(x[0] + 0.05*x[1]) \
                + torch.exp(0.8*x[2]) \
                + torch.exp(0.8*x[3] + 0.05*x[4] + 0.05*x[5]) \
                + torch.exp(0.8*x[6] + 0.05*x[7]) \
                + torch.exp(0.08*x[8] + 0.05*x[9]) if x.ndim==1 else \
                  torch.exp(x[:,0] + 0.05*x[:,1]) \
                + torch.exp(0.8*x[:,2]) \
                + torch.exp(0.8*x[:,3] + 0.05*x[:,4] + 0.05*x[:,5]) \
                + torch.exp(0.8*x[:,6] + 0.05*x[:,7]) \
                + torch.exp(0.08*x[:,8] + 0.05*x[:,9])
    mu = lambda n: torch.from_numpy(np.random.uniform(-1, 1, (n, d)))
                    
elif name == "Borehole":
    d = 8 
    f = lambda x: (2*np.pi*x[2]*(x[3] - x[5]))/(torch.log(x[1]/x[0])*(1 + x[2]/x[4] + (2*x[6]*x[2])/(torch.log(x[1]/x[0])*x[0]**2*x[7]))) if x.ndim==1 else \
                  (2*np.pi*x[:,2]*(x[:,3] - x[:,5]))/(torch.log(x[:,1]/x[:,0])*(1 + x[:,2]/x[:,4] + (2*x[:,6]*x[:,2])/(torch.log(x[:,1]/x[:,0])*x[:,0]**2*x[:,7])))
    left = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
    right = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]
    def mu(n):
        data = np.empty((n, 8))
        data[:,0] = np.random.normal(0.1, 0.0161812, n)
        data[:,1] = np.random.lognormal(7.71, 1.0056, n)
        for i in range(2,8):
            data[:,i] = np.random.uniform(left[i], right[i], n)
        return torch.from_numpy(data)
    
elif name == "Hartmann":
    d = 4
    left = [0.05, 0.5, 0.5, 0.1]
    right = [0.2, 3, 3, 1]
    def omega(x):
        y = torch.empty(x.shape)
        if y.ndim == 1:
            for i in range(d):
                y[i] = torch.exp(np.log(left[i]) + ((np.log(right[i]) - np.log(left[i]))/2)*(x[i] + 1))
        else:
            for i in range(d):
                y[:,i] = torch.exp(np.log(left[i]) + ((np.log(right[i]) - np.log(left[i]))/2)*(x[:,i] + 1))
        return y
    g = lambda y: - y[1]*(y[2]/(y[3]**2))*(1 - (y[3]/torch.sqrt(y[2]*y[0]))*(torch.cosh(y[3]/torch.sqrt(y[2]*y[0]))/torch.sinh(y[3]/torch.sqrt(y[2]*y[0])))) if y.ndim==1 else \
                  - y[:,1]*(y[:,2]/(y[:,3]**2))*(1 - (y[:,3]/torch.sqrt(y[:,2]*y[:,0]))*(torch.cosh(y[:,3]/torch.sqrt(y[:,2]*y[:,0]))/torch.sinh(y[:,3]/torch.sqrt(y[:,2]*y[:,0]))))
    f = lambda x: g(omega(x))
    mu = lambda n: torch.from_numpy(np.random.uniform(-1, 1, (n, d)))

    
data = mu(N)
f_data = f(data)

if name == "Borehole":
    data_reduced = torch.empty(data.shape)
    for j in range(data.shape[1]):
        data_reduced[:,j] = (2*data[:,j] - left[j] - right[j])/(right[j] - left[j])
        
    minimum = f_data.min()
    maximum = f_data.max()
    f_data_reduced = (2*f_data - minimum - maximum)/(maximum - minimum)
else:
    data_reduced = data   
    f_data_reduced = f_data     

#%% Dimensionality reduction

r = 1
activation = torch.nn.Tanh()
epochs = 10000
    
layers_AE = 2
neurons_AE = 8
layers_surrogate = 2
neurons_surrogate = 8

autoencoder, surrogate, _ = compute_reduction(d, data_reduced, f_data_reduced, r, activation, layers_AE, neurons_AE, layers_surrogate, neurons_surrogate, epochs)
f_S = lambda x: torch.squeeze(surrogate(autoencoder.encoder(x))).detach()

K = round(1e6)
data_CDF = mu(K)
if name == "Borehole":
    data_CDF_reduced = torch.empty(data_CDF.shape)
    for j in range(data_CDF.shape[1]):
        data_CDF_reduced[:,j] = (2*data_CDF[:,j] - left[j] - right[j])/(right[j] - left[j])
else:
    data_CDF_reduced = data_CDF

F = ECDF(torch.squeeze(autoencoder.encoder(data_CDF_reduced)).detach())

latent_data = torch.squeeze(autoencoder.encoder(data_CDF_reduced)).detach()
sorted_latent_data = torch.sort(latent_data)[0]
Finv = lambda x: sorted_latent_data[torch.floor(K*x).int()*(x != 1) + (K-1)*(x == 1)]  if torch.is_tensor(x) else sorted_latent_data[np.floor(K*x).astype(int)*(x != 1) + (K-1)*(x == 1)]

#%% Monte Carlo

if name == "Ishigami":
    S = 8
else:
    S = 16

MC = np.empty((iterations,))
sMC_new = np.empty((iterations,))
MC_S = np.empty((iterations,))

if d < 6:
    sMC_old = np.empty((iterations,))

for it in tqdm(range(iterations)):
    
    MC[it]           = compute_MC(f, d, N, mu)
    if name == "Borehole":
        sMC_new[it] = compute_sMC_new(f, autoencoder, F, d, N, S, mu, name, left, right)
    else:
        sMC_new[it] = compute_sMC_new(f, autoencoder, F, d, N, S, mu, name)
    if d < 6:
        sMC_old[it] = compute_sMC_old(f, d, N, S, mu)
    MC_S[it]  = compute_MC(f_S, d, N*10, mu)
            
#%% Plot

x_min = np.mean(MC) - 3*np.std(MC)
x_max = np.mean(MC) + 3*np.std(MC)
x_axis = np.linspace(x_min, x_max, 1000)

plt.figure()
plt.plot(x_axis, norm.pdf(x_axis, np.mean(MC), np.std(MC)), label="MC")
plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC_new), np.std(sMC_new)), label="sMC_new")
if d < 6:
    plt.plot(x_axis, norm.pdf(x_axis, np.mean(sMC_old), np.std(sMC_old)), label="sMC_old")
plt.axvline(x=np.mean(MC_S), label="MC_S")
plt.legend()
plt.ylim(0, 1.1*norm.pdf(x_axis, np.mean(sMC_new), np.std(sMC_new)).max())

#%% Save

if d < 6:
    results = np.stack((MC, sMC_old, sMC_new))
else:
    results = np.stack((MC, sMC_new))
np.savetxt("results/highDimensional/" + name + ".txt", results)

