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

    def forward(self, seed_vec,train_mode):
        seed_idx = torch.LongTensor(torch.argwhere(seed_vec == 1)).to(seed_vec.device)
        seed_hat, mean, log_var = self.vae(seed_vec)
        if train_mode:
            seed_hat.clamp(0,1)
            predictions = self.gnn(seed_hat)
            predictions = self.propagate(predictions, seed_idx)
        else:
            seed_vec.clamp(0,1)
            predictions = self.gnn(seed_vec)
            predictions = self.propagate(predictions, seed_idx)
        return seed_hat, mean, log_var, predictions

    def train_loss(self, x, x_hat, mean, log_var, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        total_loss = forward_loss + reproduction_loss + KLD
        return total_loss

    def infer_loss(self, y_true, y_hat, x_hat, train_pred):
        device = y_true.device
        BN = nn.BatchNorm1d(1, affine=False).to(device)
        forward_loss = F.mse_loss(y_hat, y_true)
        log_pmf = []
        for pred in train_pred:
            log_lh = torch.zeros(1).to(device)
            for i, x_i in enumerate(x_hat[0]):
                temp = x_i*torch.log(pred[i])+(1-x_i)*torch.log(1-pred[i]).to(torch.double)
                log_lh += temp
            log_pmf.append(log_lh)

        log_pmf = torch.stack(log_pmf)
        log_pmf = BN(log_pmf.float())

        pmf_max = torch.max(log_pmf)

        pdf_sum = pmf_max + torch.logsumexp(log_pmf-pmf_max, dim=0)

        total_loss = forward_loss - pdf_sum

        return total_loss