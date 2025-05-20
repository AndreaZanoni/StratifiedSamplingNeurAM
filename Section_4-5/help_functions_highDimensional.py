# -*- coding: utf-8 -*-

#%% Modules 

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Autoencoder and surrogate model

def compute_reduction(d, data, f_data, r, activation, layers_AE, neurons_AE, layers_surrogate, neurons_surrogate, epochs, show=True, X_test=None, Y_test=None):

    class Autoencoder(torch.nn.Module):
        
        def __init__(self):
            super().__init__()
            
            model_structure = []
            for i in range(layers_AE-1):
                model_structure += [torch.nn.Linear(neurons_AE, neurons_AE), activation]
    
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(d, neurons_AE),
                activation,
                *model_structure,
                torch.nn.Linear(neurons_AE, r),
            )
    
            self.decoder_ = torch.nn.Sequential(
                torch.nn.Linear(r, neurons_AE),
                activation,
                *model_structure,
                torch.nn.Linear(neurons_AE, d),
                torch.nn.Sigmoid()
            )
          
        def decoder(self, t):
            return 2*self.decoder_(t) - 1
          
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        
    class Surrogate(torch.nn.Module):
        
        def __init__(self):
            super().__init__()
            
            model_structure = [torch.nn.Linear(1, neurons_surrogate), activation]
            for i in range(layers_surrogate-1):
                model_structure += [torch.nn.Linear(neurons_surrogate, neurons_surrogate), activation]
            model_structure += [torch.nn.Linear(neurons_surrogate, 1)]
            self.net = torch.nn.Sequential(*model_structure)
          
        def forward(self, x):
            output = self.net(x)
            return output
    
    autoencoder = Autoencoder()
    surrogate = Surrogate()
    
    optimizer = torch.optim.Adam(list(autoencoder.parameters()) + list(surrogate.parameters()))
    loss_function = torch.nn.MSELoss()
    losses = [[], []]
    
    for epoch in tqdm(range(epochs), disable = not show):  
        reconstructed = autoencoder(data)   
        loss =   loss_function(f_data, torch.squeeze(surrogate(autoencoder.encoder(reconstructed)))) \
               + loss_function(f_data, torch.squeeze(surrogate(autoencoder.encoder(data)))) \
               + loss_function(reconstructed, autoencoder(reconstructed))
        losses[0].append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if X_test is not None:
            reconstructed_test = autoencoder(X_test)
            loss_output =  (loss_function(Y_test, torch.squeeze(surrogate(autoencoder.encoder(reconstructed_test)))) \
                          + loss_function(Y_test, torch.squeeze(surrogate(autoencoder.encoder(X_test)))) \
                          + loss_function(reconstructed_test, autoencoder(reconstructed_test))).item()
        else:
            loss_output = losses[0][-1]
        losses[1].append(loss_output)
        
    if show:
        plt.figure()
        plt.semilogy(losses[0], label='Train')
        if X_test is not None:
            plt.semilogy(losses[1], label='Test')
        plt.legend()
        plt.show(block=False)
        plt.close()
    
    return autoencoder, surrogate, losses[1][-1]

#%% Monte Carlo

def compute_MC(f, d, N, mu):
    
    samples = mu(N)
    MC = torch.mean(f(samples)).item()

    return MC
    
def compute_sMC_old(f, d, N, S, mu):
    
    S_dim = round(np.power(S, 1./d))
    N_element = round(N/S)
    size = 2/S_dim
    sMC_old = 0
    
    for i in range(S):
        multi_index = np.unravel_index(i, [S_dim] * d)
        bounds = [(-1 + idx*size, -1 + (idx + 1)*size) for idx in multi_index]
        samples = torch.rand(N_element, d)
        for j in range(d):
            samples[:,j] = samples[:,j]*(bounds[j][1] - bounds[j][0]) + bounds[j][0]
        f_samples = f(samples)
        sMC_old += torch.mean(f_samples)
    
    return sMC_old/S

def compute_sMC_new(f, autoencoder, F, d, N, S, mu, name, left=0, right=0):
    
    size = 1./S
    B = round(N/S)
    
    samples = mu(2*N)
    if name == "Borehole":
        samples_reduced = torch.empty(samples.shape)
        for j in range(samples.shape[1]):
            samples_reduced[:,j] = (2*samples[:,j] - left[j] - right[j])/(right[j] - left[j])
    else:
        samples_reduced = samples
    latent_samples = F(torch.squeeze(autoencoder.encoder(samples_reduced)).detach())
    
    sMC_new = 0
    
    for i in range(S):
        stratum_samples = samples[(latent_samples >= i*size)*(latent_samples < (i+1)*size)][:B]
        f_stratum_samples = f(stratum_samples)
        sMC_new += size*torch.mean(f_stratum_samples)
    
    return sMC_new