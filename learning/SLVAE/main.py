import torch.nn as nn
import torch
import torch.nn.functional as F
class slvae(nn.Module):
    def __init__(self, vae: nn.Module, gnn: nn.Module, propagate: nn.Module):
        super(slvae, self).__init__()

        self.vae = vae
        self.gnn = gnn
        self.propagate = propagate

        self.reg_params = list(filter(lambda x: x.requires_grad, self.gnn.parameters()))

    def forward(self, seed_vec):
        seed_idx = torch.LongTensor(torch.argwhere(seed_vec == 1)).to(seed_vec.device)
        seed_hat, mean, log_var = self.vae(seed_vec)
        predictions = self.gnn(seed_hat)
        predictions = self.propagate(predictions, seed_idx)

        return seed_hat, mean, log_var,predictions

    def loss(self, x, x_hat, mean, log_var, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')
        # reproduction_loss = F.mse_loss(x_hat, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        total_loss = forward_loss + reproduction_loss + KLD
        return total_loss