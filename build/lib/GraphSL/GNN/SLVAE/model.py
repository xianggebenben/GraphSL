import torch.nn as nn
import torch
import torch.nn.functional as F
import scipy.sparse as sp


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

            - mean (torch.Tensor): The mean of the latent space.

            - log_var (torch.Tensor): The log variance of the latent space.

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

            - x_hat (torch.Tensor): Decoded output tensor.
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

            - mean (torch.Tensor):  The mean of latent space

            - logvar (torch.Tensor):   Log variance of latent space.
        """
        self.encoder.to(x.device)
        self.mean_layer.to(x.device)
        self.logvar_layer.to(x.device)

        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var, device):
        """
        Reparameterization trick to sample from the latent space.

        Args:

            - mean (torch.Tensor): Mean of the latent space.

            - var (torch.Tensor): Variance of the latent space.

            - device (torch.device): Device to be used for computation, cpu or cuda.

        Returns:

            - z (torch.Tensor): Sampled latent vector.
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

            - x (torch.Tensor): Decoded output tensor.
        """
        self.decoder.to(x.device)
        return self.decoder(x)

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:

            - x (torch.Tensor): Input tensor.

        Returns:

            - x_hat (tensor.Tensor): The reconstructed input.

            - mean (tensor.Tensor): The mean of the latent space.

            - log_var (tensor.Tensor): The log variance of the latent space.

        """
        device = x.device
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var, device)
        x_hat = self.decode(z)
        return x_hat, mean, log_var


class GCNLayer(nn.Module):
    """
    A single layer of a Graph Convolutional Network (GCN).

    Attributes:
        - in_features (int): Number of input features for each node.

        - out_features (int): Number of output features for each node.

        - bias (bool): Whether to include a bias term in the layer.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize a GCN layer.

        Args:
            - in_features (int): Number of input features for each node.

            - out_features (int): Number of output features for each node.

            - bias (bool): Whether to include a bias term in the layer.
        """
        super(GCNLayer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define a linear transformation for the layer
        self.linear = nn.Linear(in_features, out_features, bias=bias).to(self.device)

    def forward(self, x, adj):
        """
        Forward pass of the GCN layer.

        Args:
            - x (torch.Tensor): Input feature matrix of shape (num_nodes, in_features).

            - adj (torch.Tensor): Adjacency matrix of shape (num_nodes, num_nodes).

        Returns:
            - x(torch.Tensor): Output feature matrix of shape (num_nodes, out_features).
        """
        # Perform the graph convolution operation
        x = torch.matmul(adj, x)
        x = self.linear(x)
        return x


class GNN(nn.Module):
    """
    Graph Neural Network (GNN) model using GCN layers.

    Attributes:
        - adj_matrix (torch.Tensor): Adjacency matrix representing graph connectivity.

        - input_dim (int): Dimension of the input.

        - hiddenunits (List[int]): List of hidden units for each layer.

        - num_classes (int): Number of output classes.

        - bias (bool): Whether to include bias in linear layers.

        - drop_prob (float): Dropout probability.
    """

    def __init__(
            self,
            adj_matrix,
            input_dim=1,
            hiddenunits=[
                64,
                64],
            out_dim=1,
            bias=True,
            drop_prob=0.5):
        """
        Initialize the GNN model.

        Args:
            - adj_matrix (torch.Tensor): Adjacency matrix representing graph connectivity.

            - input_dim (int): Dimension of the input.

            - hiddenunits (List[int]): List of hidden units for each layer.

            - out_dim (int): Dimension of the output.

            - bias (bool): Whether to include bias in linear layers.

            - drop_prob (float): Dropout probability.
        """
        super(GNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim

        # Convert sparse matrix to dense if needed
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.toarray()

        self.adj_matrix = nn.Parameter(
            torch.Tensor(adj_matrix).to(self.device),
            requires_grad=False)

        # Define GCN layers
        gcn_layers = [GCNLayer(input_dim, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            gcn_layers.append(GCNLayer(hiddenunits[i - 1], hiddenunits[i], bias=bias))
        gcn_layers.append(GCNLayer(hiddenunits[-1], out_dim, bias=bias))

        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Define dropout layer
        self.dropout = nn.Dropout(drop_prob).to(self.device) if drop_prob > 0 else lambda x: x

        # Activation function
        self.act_fn = nn.ReLU().to(self.device)

    def forward(self, seed_vec):
        """
        Forward pass of the GNN.

        Args:
            - seed_vec (torch.Tensor): Input seed vector.

        Returns:
            - x (torch.Tensor): Predicted output.
        """
        # Initial feature matrix is the seed vector
        x = seed_vec

        # Apply each GCN layer
        for layer in self.gcn_layers[:-1]:
            x = self.act_fn(layer(x, self.adj_matrix))
            x = self.dropout(x)
        
        # Final layer with no activation function
        res = torch.sigmoid(self.gcn_layers[-1](x, self.adj_matrix))
        return res

    def loss(self, y, y_hat):
        """
        Calculate loss.

        Args:
            - y (torch.Tensor): Ground truth.
            - y_hat (torch.Tensor): Predicted output.

        Returns:
            - forward_loss (torch.Tensor): Forward loss.
        """
        forward_loss = F.mse_loss(y_hat, y)
        return forward_loss