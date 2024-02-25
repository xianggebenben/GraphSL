
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import ipdb
import scipy.sparse as sp



class InverseModel(nn.Module):
    def __init__(self, vae_model: nn.Module, gnn_model: nn.Module, propagate: nn.Module):
        super(InverseModel, self).__init__()
        
        self.vae_model = vae_model
        self.gnn_model = gnn_model 
        self.propagate = propagate

        self.reg_params = list(filter(lambda x: x.requires_grad, self.gnn_model.parameters()))
    
    def forward(self, input_pair, seed_vec, adj):
        device = next(self.gnn_model.parameters()).device
        seed_idx = torch.LongTensor(np.argwhere(seed_vec.cpu().detach().numpy() == 1)).to(device)
        
        seed_hat, mean, log_var = self.vae_model(input_pair, adj)
        predictions = self.gnn_model(seed_hat)
        predictions = self.propagate(predictions, seed_idx)
        
        return seed_hat, mean, log_var, predictions
    
    def loss(self, x, x_hat, mean, log_var, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')
        #reproduction_loss = F.mse_loss(x_hat, x)
        KLD = -0.5*torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        total_loss = forward_loss+reproduction_loss+KLD
        return KLD, reproduction_loss, forward_loss, total_loss


class ForwardModel(nn.Module):
    def __init__(self, gnn_model: nn.Module, propagate: nn.Module):
        super(ForwardModel, self).__init__()
        self.gnn_model = gnn_model 
        self.propagate = propagate
    
        self.reg_params = list(filter(lambda x: x.requires_grad, self.gnn_model.parameters()))
    
    def forward(self, seed_vec):
        device = next(self.gnn_model.parameters()).device
        #seed_idx = torch.LongTensor(np.argwhere(seed_vec.cpu().detach().numpy() == 1)).to(device)
        seed_idx = (seed_vec==1).nonzero(as_tuple=False)
        
        #seed_hat, mean, log_var = self.vae_model(input_pair)
        predictions = self.gnn_model(seed_vec)
        predictions = self.propagate(predictions, seed_idx)
        
        return predictions
    
    def loss(self, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        return forward_loss