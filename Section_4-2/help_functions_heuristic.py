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

#%% Optimal stratification
    
def func_var_tilde(surrogate, Finv, a, u, b):
    B = 1000
    samples1 = torch.from_numpy(np.random.uniform(a, u, (B,)))
    samples2 = torch.from_numpy(np.random.uniform(u, b, (B,)))
    var1 = torch.var(surrogate(torch.unsqueeze(Finv(samples1), 1))).item()
    var2 = torch.var(surrogate(torch.unsqueeze(Finv(samples2), 1))).item()
    return (u - a)*var1 + (b - u)*var2

def func_std_tilde(surrogate, Finv, a, u, b):
    B = 1000
    samples1 = torch.from_numpy(np.random.uniform(a, u, (B,)))
    samples2 = torch.from_numpy(np.random.uniform(u, b, (B,)))
    std1 = torch.std(surrogate(torch.unsqueeze(Finv(samples1), 1))).item()
    std2 = torch.std(surrogate(torch.unsqueeze(Finv(samples2), 1))).item()
    return (u - a)*std1 + (b - u)*std2

#%% Monte Carlo

def compute_MC(f, d, N):
    
    samples = torch.from_numpy(np.random.uniform(-1, 1, (N, d)))
    MC = torch.mean(f(samples)).item()

    return MC

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

def compute_sMC_bisection(f, autoencoder, surrogate, F, Finv, d, N, S):
    
    pilot_samples = torch.from_numpy(np.random.uniform(0, 1, (2*N,)))
    surrogate_pilot_samples = torch.squeeze(surrogate(torch.unsqueeze(Finv(pilot_samples), 1))).detach()
    
    strata = np.array([0., 1.])
    variances = np.array([torch.var(surrogate_pilot_samples).item()])
    lengths = np.array([1.])
    maximum = np.argmax(lengths*variances)
    
    for i in range(S-1):
        strata = np.insert(strata, maximum+1, (strata[maximum] + strata[maximum+1])/2)
        surrogate_pilot_samples_1 = surrogate_pilot_samples[(pilot_samples >= strata[maximum])*(pilot_samples < strata[maximum+1])]
        surrogate_pilot_samples_2 = surrogate_pilot_samples[(pilot_samples >= strata[maximum+1])*(pilot_samples < strata[maximum+2])]
        var_1 = torch.var(surrogate_pilot_samples_1).item()
        var_2 = torch.var(surrogate_pilot_samples_2).item()
        variances[maximum] = var_1
        variances = np.insert(variances, maximum+1, var_2)
        lengths[maximum] = strata[maximum+1] - strata[maximum]
        lengths = np.insert(lengths, maximum+1, strata[maximum+2] - strata[maximum+1])
        maximum = np.argmax(lengths*variances)
        
    samples = torch.from_numpy(np.random.uniform(-1, 1, (2*N, d)))
    latent_samples = F(torch.squeeze(autoencoder.encoder(samples)).detach())
    
    sMC_bisection = 0
    
    for i in range(S):
        B = round(lengths[i]*N)
        stratum_samples = samples[(latent_samples >= strata[i])*(latent_samples < strata[i+1])][:B]
        f_stratum_samples = f(stratum_samples)
        sMC_bisection += lengths[i]*torch.mean(f_stratum_samples)
    
    return sMC_bisection

def compute_sMC_optimal(f, autoencoder, surrogate, F, Finv, d, N, S):
    
    pilot_samples = torch.from_numpy(np.random.uniform(0, 1, (2*N,)))
    surrogate_pilot_samples = torch.squeeze(surrogate(torch.unsqueeze(Finv(pilot_samples), 1))).detach()
    
    strata = np.array([0., 1.])
    variances = np.array([torch.var(surrogate_pilot_samples).item()])
    lengths = np.array([1.])
    maximum = np.argmax(lengths*variances)
    
    for i in range(S-1):
        func_var = lambda u: func_var_tilde(surrogate, Finv, strata[maximum], u, strata[maximum+1])
        uu = np.linspace(strata[maximum], strata[maximum+1], 102)
        uu = uu[1:-1]
        func_var_uu = np.empty(uu.shape)
        for i, u in enumerate(uu):
            func_var_uu[i] = func_var(u)
        optimal_division = uu[np.argmin(func_var_uu)]
        strata = np.insert(strata, maximum+1, optimal_division)
        surrogate_pilot_samples_1 = surrogate_pilot_samples[(pilot_samples >= strata[maximum])*(pilot_samples < strata[maximum+1])]
        surrogate_pilot_samples_2 = surrogate_pilot_samples[(pilot_samples >= strata[maximum+1])*(pilot_samples < strata[maximum+2])]
        var_1 = torch.var(surrogate_pilot_samples_1).item()
        var_2 = torch.var(surrogate_pilot_samples_2).item()
        variances[maximum] = var_1
        variances = np.insert(variances, maximum+1, var_2)
        lengths[maximum] = strata[maximum+1] - strata[maximum]
        lengths = np.insert(lengths, maximum+1, strata[maximum+2] - strata[maximum+1])
        maximum = np.argmax(lengths*variances)
        
    samples = torch.from_numpy(np.random.uniform(-1, 1, (2*N, d)))
    latent_samples = F(torch.squeeze(autoencoder.encoder(samples)).detach())
    
    sMC_optimal = 0
    
    for i in range(S):
        B = round(lengths[i]*N)
        stratum_samples = samples[(latent_samples >= strata[i])*(latent_samples < strata[i+1])][:B]
        f_stratum_samples = f(stratum_samples)
        sMC_optimal += lengths[i]*torch.mean(f_stratum_samples)
    
    return sMC_optimal

def compute_sMC_uniform_stdprop(f, autoencoder, surrogate, F, Finv, d, N, S):
    
    pilot_samples = torch.from_numpy(np.random.uniform(0, 1, (2*N,)))
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

def compute_sMC_bisection_stdprop(f, autoencoder, surrogate, F, Finv, d, N, S):
    
    pilot_samples = torch.from_numpy(np.random.uniform(0, 1, (2*N,)))
    surrogate_pilot_samples = torch.squeeze(surrogate(torch.unsqueeze(Finv(pilot_samples), 1))).detach()
    
    strata = np.array([0., 1.])
    stds = np.array([torch.std(surrogate_pilot_samples).item()])
    lengths = np.array([1.])
    maximum = np.argmax(lengths*stds)
    
    for i in range(S-1):
        strata = np.insert(strata, maximum+1, (strata[maximum] + strata[maximum+1])/2)
        surrogate_pilot_samples_1 = surrogate_pilot_samples[(pilot_samples >= strata[maximum])*(pilot_samples < strata[maximum+1])]
        surrogate_pilot_samples_2 = surrogate_pilot_samples[(pilot_samples >= strata[maximum+1])*(pilot_samples < strata[maximum+2])]
        std_1 = torch.std(surrogate_pilot_samples_1).item()
        std_2 = torch.std(surrogate_pilot_samples_2).item()
        stds[maximum] = std_1
        stds = np.insert(stds, maximum+1, std_2)
        lengths[maximum] = strata[maximum+1] - strata[maximum]
        lengths = np.insert(lengths, maximum+1, strata[maximum+2] - strata[maximum+1])
        maximum = np.argmax(lengths*stds)
        
    samples = torch.from_numpy(np.random.uniform(-1, 1, (2*N, d)))
    latent_samples = F(torch.squeeze(autoencoder.encoder(samples)).detach())
    
    sMC_bisection_stdprop = 0
    
    for i in range(S):
        B = round(N*lengths[i]*stds[i]/np.sum(lengths*stds))
        stratum_samples = samples[(latent_samples >= strata[i])*(latent_samples < strata[i+1])][:B]
        f_stratum_samples = f(stratum_samples)
        sMC_bisection_stdprop += lengths[i]*torch.mean(f_stratum_samples)
    
    return sMC_bisection_stdprop

def compute_sMC_optimal_stdprop(f, autoencoder, surrogate, F, Finv, d, N, S):
    
    pilot_samples = torch.from_numpy(np.random.uniform(0, 1, (2*N,)))
    surrogate_pilot_samples = torch.squeeze(surrogate(torch.unsqueeze(Finv(pilot_samples), 1))).detach()
    
    strata = np.array([0., 1.])
    stds = np.array([torch.std(surrogate_pilot_samples).item()])
    lengths = np.array([1.])
    maximum = np.argmax(lengths*stds)
    
    for i in range(S-1):
        func_std = lambda u: func_var_tilde(surrogate, Finv, strata[maximum], u, strata[maximum+1])
        uu = np.linspace(strata[maximum], strata[maximum+1], 102)
        uu = uu[1:-1]
        func_std_uu = np.empty(uu.shape)
        for i, u in enumerate(uu):
            func_std_uu[i] = func_std(u)
        optimal_division = uu[np.argmin(func_std_uu)]
        strata = np.insert(strata, maximum+1, optimal_division)
        surrogate_pilot_samples_1 = surrogate_pilot_samples[(pilot_samples >= strata[maximum])*(pilot_samples < strata[maximum+1])]
        surrogate_pilot_samples_2 = surrogate_pilot_samples[(pilot_samples >= strata[maximum+1])*(pilot_samples < strata[maximum+2])]
        std_1 = torch.std(surrogate_pilot_samples_1).item()
        std_2 = torch.std(surrogate_pilot_samples_2).item()
        stds[maximum] = std_1
        stds = np.insert(stds, maximum+1, std_2)
        lengths[maximum] = strata[maximum+1] - strata[maximum]
        lengths = np.insert(lengths, maximum+1, strata[maximum+2] - strata[maximum+1])
        maximum = np.argmax(lengths*stds)
        
    samples = torch.from_numpy(np.random.uniform(-1, 1, (2*N, d)))
    latent_samples = F(torch.squeeze(autoencoder.encoder(samples)).detach())
    
    sMC_optimal_stdprop = 0
    
    for i in range(S):
        B = round(N*lengths[i]*stds[i]/np.sum(lengths*stds))
        stratum_samples = samples[(latent_samples >= strata[i])*(latent_samples < strata[i+1])][:B]
        f_stratum_samples = f(stratum_samples)
        sMC_optimal_stdprop += lengths[i]*torch.mean(f_stratum_samples)
    
    return sMC_optimal_stdprop