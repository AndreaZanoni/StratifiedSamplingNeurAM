# -*- coding: utf-8 -*-

#%% Modules 

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Autoencoder and surrogate model

def compute_reduction(d, L, data, f_data, r, activation, layers_AE, neurons_AE, layers_surrogate, neurons_surrogate, epochs, show=True, X_test=None, Y_test=None):

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
            return 2*L*self.decoder_(t) - L
          
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

def compute_MC(Q, N):
    
    MC = torch.mean(Q[:N]).item()

    return MC

def compute_sMC(data, Q, N, autoencoder, F, S):
    
    size = 1./S
    B = round(N/S)
    
    latent_data = F(torch.squeeze(autoencoder.encoder(data)).detach())
    sMC_new = 0
    
    for i in range(S):
        Q_stratum = Q[(latent_data >= i*size)*(latent_data < (i+1)*size)][:B]
        sMC_new += size*torch.mean(Q_stratum)
    
    return sMC_new