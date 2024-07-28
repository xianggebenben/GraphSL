import torch
import torch.nn as nn
import torch.nn.functional as F

class I_GCNLayer(nn.Module):
    """
    Invertible Graph Convolutional Network Layer
    """
    def __init__(self, in_features, out_features):
        """
        Initialize an I_GCNLayer.

        Arguments:

        - in_features (int): Number of input features for each node.

        - out_features (int): Number of output features for each node.
        """
        super(I_GCNLayer, self).__init__()
        # Determine whether to use GPU or CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Define a linear transformation layer
        self.linear = nn.Linear(in_features, out_features).to(self.device)

    def forward(self, x, adj):
        """
        Perform a forward pass through the graph convolution layer.

        Arguments:

        - x (torch.Tensor): Input feature matrix of shape (num_nodes, in_features).

        - adj (torch.Tensor): Adjacency matrix of the graph of shape (num_nodes, num_nodes).

        Returns:

        - x (torch.Tensor): Output feature matrix of shape (num_nodes, out_features).
        """
        # Perform graph convolution: multiply adjacency matrix with input features
        x = torch.matmul(adj, x)
        # Apply linear transformation
        x = self.linear(x)
        return x

class I_GCN(nn.Module):
    """
        Invertible Graph Convolutional Network
    """
    def __init__(self, hidden_dim=32, num_layers=3, d=2):
        """
        Initialize an I_GCN model.

        Arguments:

        - hidden_dim (int): Number of hidden units in each layer.

        - num_layers (int): Total number of layers in the network.

        - d (int): Number of diffusion steps for feature construction.
        """
        super(I_GCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.iter_num = 5  # Number of iterations for fixed-point iteration in backward pass
        self.d = d  # Number of diffusion steps to consider in feature construction

        # Define the input layer with specified dimensions
        self.input_layer = I_GCNLayer(d + 1, hidden_dim).to(self.device)

        # Define the hidden layers for the GCN
        self.hidden_layers = nn.ModuleList([
            I_GCNLayer(hidden_dim, hidden_dim).to(self.device) for _ in range(num_layers - 2)
        ])

        # Define the output layer
        self.output_layer = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, adj, seed_vector):
        """
        Perform a forward pass through the I_GCN model.

        Arguments:

        - adj (scipy.sparse matrix): Adjacency matrix of the graph.

        - seed_vector (torch.Tensor): Seed vector representing initial node activations.

        Returns:

        - x (torch.Tensor): Output vector after passing through the network.
        """
        # Convert adjacency matrix to tensor and move to device
        adj = torch.Tensor(adj.toarray()).to(self.device)

        # Move seed vector to device
        seed_vector = seed_vector.to(self.device)

        # Feature construction: concatenate seed vector with its diffusion features
        features = torch.cat([seed_vector] + [torch.matmul(adj.T, seed_vector) for _ in range(self.d)], dim=1)

        # Forward pass through the I_GCN
        x = F.relu(self.input_layer(features, adj))

        for layer in self.hidden_layers:
            x = F.relu(layer(x, adj))

        # Apply sigmoid activation to the output layer
        x = torch.sigmoid(self.output_layer(x))
        return x
    
    def backward(self, adj, influ_vector):
        """
        Perform a backward pass (inverse) using fixed-point iteration.

        Arguments:

        - adj (scipy.sparse matrix): Adjacency matrix of the graph.

        - influ_vector (torch.Tensor): Influence vector obtained from forward pass.

        Returns:

        - seed_vector (torch.Tensor): Seed vector approximated using fixed-point iteration.
        """
        # Move influence vector to device
        influ_vector = influ_vector.to(self.device)

        # Initialize seed vector with zeros
        seed_vector = torch.zeros(influ_vector.shape).to(self.device)

        # Perform fixed-point iteration
        for _ in range(self.iter_num):
            seed_vector = influ_vector - self(adj, seed_vector)
        
        return seed_vector
