import torch.nn as nn
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from typing import List
class Encoder(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=256):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True

    def forward(self, x):
        x = self.LeakyReLU(self.linear1(x))
        x = self.LeakyReLU(self.linear2(x))

        mean = self.mean(x)
        log_var = self.var(x)
        return mean, log_var


class Decoder(nn.Module):

    def __init__(self, output_dim=784, hidden_dim=512, latent_dim=256):
        super(Decoder, self).__init__()

        self.linear2 = nn.Linear(latent_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.LeakyReLU(self.linear2(x))
        x = self.LeakyReLU(self.linear1(x))

        x_hat = torch.sigmoid(self.output(x))
        return x_hat


class VAE(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=512, latent_dim=256):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var,device):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def forward(self, x):
        device = x.device
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var),device)
        x_hat = self.decode(z)
        return x_hat, mean, log_var


class GNN(nn.Module):
    def __init__(self, prob_matrix,input_dim=5, hiddenunits: List[int]=[64,64], num_classes=1, bias=True, drop_prob=0.5):
        super(GNN, self).__init__()

        self.input_dim = input_dim

        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()

        self.prob_matrix = nn.Parameter((torch.FloatTensor(prob_matrix)), requires_grad=False)

        fcs = [nn.Linear(input_dim, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i]))
        fcs.append(nn.Linear(hiddenunits[-1], num_classes))

        self.fcs = nn.ModuleList(fcs)

        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(drop_prob)

        self.act_fn = nn.ReLU()

    def forward(self, seed_vec):

        for i in range(self.input_dim - 1):
            if i == 0:
                mat = self.prob_matrix.T @ seed_vec
                attr_mat = torch.cat((seed_vec.unsqueeze(0), mat.unsqueeze(0)), 0)
            else:
                mat = self.prob_matrix.T @ attr_mat[-1]
                attr_mat = torch.cat((attr_mat, mat.unsqueeze(0)), 0)

        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_mat.T)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = torch.sigmoid(self.fcs[-1](self.dropout(layer_inner)))
        return res

    def loss(self, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        return forward_loss


class DiffusionPropagate(nn.Module):
    def __init__(self, prob_matrix, niter):
        super(DiffusionPropagate, self).__init__()

        self.niter = niter

        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()

        self.register_buffer('prob_matrix', torch.FloatTensor(prob_matrix))

    def forward(self, preds, seed_idx):
        # import ipdb; ipdb.set_trace()
        # prop_preds = torch.ones((preds.shape[0], preds.shape[1])).to(device)
        device = preds.device

        for i in range(preds.shape[0]):
            prop_pred = preds[i]
            for j in range(self.niter):
                P2 = self.prob_matrix.T * prop_pred.view((1, -1)).expand(self.prob_matrix.shape)
                P3 = torch.ones(self.prob_matrix.shape).to(device) - P2
                prop_pred = torch.ones((self.prob_matrix.shape[0],)).to(device) - torch.prod(P3, dim=1)
                # prop_pred[seed_idx[seed_idx[:,0] == i][:, 1]] = 1
                prop_pred = prop_pred.unsqueeze(0)
            if i == 0:
                prop_preds = prop_pred
            else:
                prop_preds = torch.cat((prop_preds, prop_pred), 0)

        return prop_preds



class ForwardModel(nn.Module):
    def __init__(self, gnn_model: nn.Module, propagate: nn.Module):
        super(ForwardModel, self).__init__()
        self.gnn_model = gnn_model
        self.propagate = propagate
        self.relu = nn.ReLU(inplace=True)

        self.reg_params = list(filter(lambda x: x.requires_grad, self.gnn_model.parameters()))

    def forward(self, seed_vec):
        seed_idx = (seed_vec == 1).nonzero(as_tuple=False)

        predictions = self.gnn_model(seed_vec)
        predictions = self.propagate(predictions, seed_idx)

        predictions = self.relu(predictions)

        return predictions

    def loss(self, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        return forward_loss