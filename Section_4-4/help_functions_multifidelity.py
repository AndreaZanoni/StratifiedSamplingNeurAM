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

def compute_MC(f, d, N):
    
    samples = torch.from_numpy(np.random.uniform(-1, 1, (N, d)))
    MC = torch.mean(f(samples)).item()

    return MC

def compute_MFMC(f_HF, f_LF, surrogate_HF, surrogate_LF, Finv_HF, Finv_LF, d, N, w):
    
    pilot_samples = torch.from_numpy(np.random.uniform(0, 1, (N,)))
    surrogate_HF_pilot_samples = torch.squeeze(surrogate_HF(torch.unsqueeze(Finv_HF(pilot_samples), 1))).detach()
    surrogate_LF_pilot_samples = torch.squeeze(surrogate_LF(torch.unsqueeze(Finv_LF(pilot_samples), 1))).detach()
    
    cov_matrix = np.cov(surrogate_HF_pilot_samples, surrogate_LF_pilot_samples)
    rho = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])
    beta = cov_matrix[0,1]/cov_matrix[1,1]
    
    gamma = np.sqrt((rho**2)/(w*(1 - rho**2)))
    N_HF = int(round(N/(1 + w*gamma)))
    N_LF = int(round(gamma*N_HF))
    
    samples = torch.from_numpy(np.random.uniform(-1, 1, (N_LF, d)))
    f_HF_samples = f_HF(samples[:N_HF])
    f_LF_samples = f_LF(samples)
    
    MFMC = (torch.mean(f_HF_samples) - beta*(torch.mean(f_LF_samples[:N_HF]) - torch.mean(f_LF_samples))).item()
    
    return MFMC

def compute_sMC_uniform(f, autoencoder, F, d, N, S):
    
    size = 1./S
    B = round(N/S)
    
    samples = torch.from_numpy(np.random.uniform(-1, 1, (2*N, d)))
    latent_samples = F(torch.squeeze(autoencoder.encoder(samples)).detach())
    
    sMC_uniform = 0
    
    for i in range(S):
        stratum_samples = samples[(latent_samples >= i*size)*(latent_samples < (i+1)*size)][:B]
        f_stratum_samples = f(stratum_samples)
        sMC_uniform += size*torch.mean(f_stratum_samples)
    
    return sMC_uniform

def compute_sMC_uniform_stdprop(f, autoencoder, surrogate, F, Finv, d, N, S):
    
    pilot_samples = torch.from_numpy(np.random.uniform(0, 1, (N,)))
    surrogate_pilot_samples = torch.squeeze(surrogate(torch.unsqueeze(Finv(pilot_samples), 1))).detach()
    
    size = 1./S
    strata = np.linspace(0., 1., S+1)
    stds = np.empty(S)
    for i in range(S):
        surrogate_pilot_samples_stratum = surrogate_pilot_samples[(pilot_samples >= i*size)*(pilot_samples < (i+1)*size)]
        stds[i] = torch.std(surrogate_pilot_samples_stratum).item()
    
    samples = torch.from_numpy(np.random.uniform(-1, 1, (2*N, d)))
    latent_samples = F(torch.squeeze(autoencoder.encoder(samples)).detach())
    
    sMC_uniform_stdprop = 0
    
    for i in range(S):
        B = round(N*stds[i]/np.sum(stds))
        stratum_samples = samples[(latent_samples >= strata[i])*(latent_samples < strata[i+1])][:B]
        f_stratum_samples = f(stratum_samples)
        sMC_uniform_stdprop += size*torch.mean(f_stratum_samples)
    
    return sMC_uniform_stdprop

def compute_sMFMC_uniform(f_HF, f_LF, autoencoder_HF, surrogate_HF, surrogate_LF, F_HF, Finv_HF, Finv_LF, d, N, S, w):
    
    size = 1./S
    B = round(N/S)
    
    pilot_samples = torch.from_numpy(np.random.uniform(0, 1, (N,)))
    surrogate_HF_pilot_samples = torch.squeeze(surrogate_HF(torch.unsqueeze(Finv_HF(pilot_samples), 1))).detach()
    surrogate_LF_pilot_samples = torch.squeeze(surrogate_LF(torch.unsqueeze(Finv_LF(pilot_samples), 1))).detach()
    
    samples = torch.from_numpy(np.random.uniform(-1, 1, (round(2*N/w), d)))
    latent_samples = F_HF(torch.squeeze(autoencoder_HF.encoder(samples)).detach())
    
    sMFMC_uniform = 0
    
    for i in range(S):
        
        surrogate_HF_pilot_samples_stratum = surrogate_HF_pilot_samples[(pilot_samples >= i*size)*(pilot_samples < (i+1)*size)]
        surrogate_LF_pilot_samples_stratum = surrogate_LF_pilot_samples[(pilot_samples >= i*size)*(pilot_samples < (i+1)*size)]
        
        cov_matrix = np.cov(surrogate_HF_pilot_samples_stratum, surrogate_LF_pilot_samples_stratum)
        rho = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])
        beta = cov_matrix[0,1]/cov_matrix[1,1]
        
        gamma = np.sqrt((rho**2)/(w*(1 - rho**2)))
        N_HF = int(round(B/(1 + w*gamma)))
        N_LF = int(round(gamma*N_HF))
        
        stratum_samples = samples[(latent_samples >= i*size)*(latent_samples < (i+1)*size)][:N_LF]
        f_HF_stratum_samples = f_HF(stratum_samples[:N_HF])
        f_LF_stratum_samples = f_LF(stratum_samples)
        
        sMFMC_uniform += size*(torch.mean(f_HF_stratum_samples) - beta*(torch.mean(f_LF_stratum_samples[:N_HF]) - torch.mean(f_LF_stratum_samples))).item()
    
    return sMFMC_uniform

def compute_sMFMC_uniform_stdprop(f_HF, f_LF, autoencoder_HF, surrogate_HF, surrogate_LF, F_HF, Finv_HF, Finv_LF, d, N, S, w):
    
    pilot_samples = torch.from_numpy(np.random.uniform(0, 1, (N,)))
    surrogate_HF_pilot_samples = torch.squeeze(surrogate_HF(torch.unsqueeze(Finv_HF(pilot_samples), 1))).detach()
    surrogate_LF_pilot_samples = torch.squeeze(surrogate_LF(torch.unsqueeze(Finv_LF(pilot_samples), 1))).detach()
    
    size = 1./S
    stds = np.empty(S)
    rhos = np.empty(S)
    coeff_rhos = np.empty(S)
    betas = np.empty(S)
    for i in range(S):
        surrogate_HF_pilot_samples_stratum = surrogate_HF_pilot_samples[(pilot_samples >= i*size)*(pilot_samples < (i+1)*size)]
        surrogate_LF_pilot_samples_stratum = surrogate_LF_pilot_samples[(pilot_samples >= i*size)*(pilot_samples < (i+1)*size)]
        stds[i] = torch.std(surrogate_HF_pilot_samples_stratum).item()
        
        cov_matrix = np.cov(surrogate_HF_pilot_samples_stratum, surrogate_LF_pilot_samples_stratum)
        rho = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0]*cov_matrix[1,1])
        beta = cov_matrix[0,1]/cov_matrix[1,1]
        
        rhos[i] = rho
        coeff_rhos[i] = np.sqrt(1 - rho**2) + np.sqrt(w*rho**2)
        betas[i] = beta
        
    samples = torch.from_numpy(np.random.uniform(-1, 1, (round(2*N/w), d)))
    latent_samples = F_HF(torch.squeeze(autoencoder_HF.encoder(samples)).detach())
    
    sMFMC_uniform_stdprop = 0
    
    for i in range(S):
        
        B = round(N*stds[i]*coeff_rhos[i]/np.sum(stds*coeff_rhos))
        
        rho = rhos[i]
        beta = betas[i]
        
        gamma = np.sqrt((rho**2)/(w*(1 - rho**2)))
        N_HF = int(round(B/(1 + w*gamma)))
        N_LF = int(round(gamma*N_HF))
        
        stratum_samples = samples[(latent_samples >= i*size)*(latent_samples < (i+1)*size)][:N_LF]
        f_HF_stratum_samples = f_HF(stratum_samples[:N_HF])
        f_LF_stratum_samples = f_LF(stratum_samples)
        
        sMFMC_uniform_stdprop += size*(torch.mean(f_HF_stratum_samples) - beta*(torch.mean(f_LF_stratum_samples[:N_HF]) - torch.mean(f_LF_stratum_samples))).item()
    
    return sMFMC_uniform_stdprop