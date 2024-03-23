import torch.nn as nn
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from typing import List
class Encoder(nn.Module):
    """
    Encoder module for a variational autoencoder (VAE).

    Attributes:
    - input_dim (int): Dimension of the input.
    - hidden_dim (int): Dimension of the hidden layer.
    - latent_dim (int): Dimension of the latent space.
    """

    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=256):
        """
        Initialize the Encoder.

        Args:
        - input_dim (int): Dimension of the input.
        - hidden_dim (int): Dimension of the hidden layer.
        - latent_dim (int): Dimension of the latent space.
        """
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass of the Encoder.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - Tuple: Tuple containing the mean and log variance.
        """
        x = self.LeakyReLU(self.linear1(x))
        x = self.LeakyReLU(self.linear2(x))

        mean = self.mean(x)
        log_var = self.var(x)
        return mean, log_var


class Decoder(nn.Module):
    """
    Decoder module for a variational autoencoder (VAE).

    Attributes:
    - output_dim (int): Dimension of the output.
    - hidden_dim (int): Dimension of the hidden layer.
    - latent_dim (int): Dimension of the latent space.
    """

    def __init__(self, output_dim=784, hidden_dim=512, latent_dim=256):
        """
        Initialize the Decoder.

        Args:
        - output_dim (int): Dimension of the output.
        - hidden_dim (int): Dimension of the hidden layer.
        - latent_dim (int): Dimension of the latent space.
        """
        super(Decoder, self).__init__()

        self.linear2 = nn.Linear(latent_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass of the Decoder.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Decoded output tensor.
        """
        x = self.LeakyReLU(self.linear2(x))
        x = self.LeakyReLU(self.linear1(x))

        x_hat = torch.sigmoid(self.output(x))
        return x_hat



class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.

    Attributes:
    - input_dim (int): Dimension of the input.
    - hidden_dim (int): Dimension of the hidden layer.
    - latent_dim (int): Dimension of the latent space.
    """

    def __init__(self, input_dim=1, hidden_dim=512, latent_dim=256):
        """
        Initialize the VAE model.

        Args:
        - input_dim (int): Dimension of the input.
        - hidden_dim (int): Dimension of the hidden layer.
        - latent_dim (int): Dimension of the latent space.
        """
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        # Latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encode input data into latent space.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - Tuple: Tuple containing the mean and log variance.
        """
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var, device):
        """
        Reparameterization trick to sample from the latent space.

        Args:
        - mean (torch.Tensor): Mean of the latent space.
        - var (torch.Tensor): Variance of the latent space.
        - device (torch.device): Device to be used for computation.

        Returns:
        - torch.Tensor: Sampled latent vector.
        """
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        """
        Decode latent vector into output space.

        Args:
        - x (torch.Tensor): Latent vector.

        Returns:
        - torch.Tensor: Decoded output tensor.
        """
        return self.decoder(x)

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - Tuple: Tuple containing the reconstructed input, mean, and log variance.
        """
        device = x.device
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var, device)
        x_hat = self.decode(z)
        return x_hat, mean, log_var


class GNN(nn.Module):
    """
    Graph Neural Network (GNN) model.

    Attributes:
    - input_dim (int): Dimension of the input.
    - prob_matrix (torch.Tensor): Probability matrix representing graph connectivity.
    - hiddenunits (List[int]): List of hidden units for each layer.
    - num_classes (int): Number of output classes.
    - bias (bool): Whether to include bias in linear layers.
    - drop_prob (float): Dropout probability.
    """

    def __init__(self, prob_matrix, input_dim=5, hiddenunits=[64, 64], num_classes=1, bias=True, drop_prob=0.5):
        """
        Initialize the GNN model.

        Args:
        - prob_matrix (torch.Tensor): Probability matrix representing graph connectivity.
        - input_dim (int): Dimension of the input.
        - hiddenunits (List[int]): List of hidden units for each layer.
        - num_classes (int): Number of output classes.
        - bias (bool): Whether to include bias in linear layers.
        - drop_prob (float): Dropout probability.
        """
        super(GNN, self).__init__()

        self.input_dim = input_dim

        # Convert sparse matrix to dense if needed
        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()

        self.prob_matrix = nn.Parameter(torch.FloatTensor(prob_matrix), requires_grad=False)

        # Define fully connected layers
        fcs = [nn.Linear(input_dim, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i]))
        fcs.append(nn.Linear(hiddenunits[-1], num_classes))

        self.fcs = nn.ModuleList(fcs)

        # Define dropout layer
        if drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(drop_prob)

        # Activation function
        self.act_fn = nn.ReLU()

    def forward(self, seed_vec):
        """
        Forward pass of the GNN.

        Args:
        - seed_vec (torch.Tensor): Input seed vector.

        Returns:
        - torch.Tensor: Predicted output.
        """
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
        """
        Calculate loss.

        Args:
        - y (torch.Tensor): Ground truth.
        - y_hat (torch.Tensor): Predicted output.

        Returns:
        - torch.Tensor: Forward loss.
        """
        forward_loss = F.mse_loss(y_hat, y)
        return forward_loss


class DiffusionPropagate(nn.Module):
    """
    Diffusion Propagation module for graph data.

    Attributes:
    - prob_matrix (torch.Tensor): Probability matrix representing graph connectivity.
    - niter (int): Number of diffusion iterations.
    """

    def __init__(self, prob_matrix, niter):
        """
        Initialize the DiffusionPropagate module.

        Args:
        - prob_matrix (torch.Tensor): Probability matrix representing graph connectivity.
        - niter (int): Number of diffusion iterations.
        """
        super(DiffusionPropagate, self).__init__()

        self.niter = niter

        # Convert sparse matrix to dense if needed
        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()

        self.register_buffer('prob_matrix', torch.FloatTensor(prob_matrix))

    def forward(self, preds):
        """
        Forward pass of the DiffusionPropagate module.

        Args:
        - preds (torch.Tensor): Predictions.

        Returns:
        - torch.Tensor: Propagated predictions.
        """
        device = preds.device

        for i in range(preds.shape[0]):
            prop_pred = preds[i]
            for j in range(self.niter):
                P2 = self.prob_matrix.T * prop_pred.view((1, -1)).expand(self.prob_matrix.shape)
                P3 = torch.ones(self.prob_matrix.shape).to(device) - P2
                prop_pred = torch.ones((self.prob_matrix.shape[0],)).to(device) - torch.prod(P3, dim=1)
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