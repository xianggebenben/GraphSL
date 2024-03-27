import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
import copy
import numpy as np
from Prescribed import LPSI
class GCNConv(MessagePassing):
    """
    Defines a Graph Convolutional Network (GCN) layer.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes the GCNConv layer with input and output channel dimensions.

        Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        """
        super(GCNConv, self).__init__(aggr='add')  # Setting the aggregation method for message passing
        self.lin = torch.nn.Linear(in_channels, out_channels)  # Initializing a linear transformation

    def forward(self, x, edge_index):
        """
        Perform the forward pass of the GCNConv layer.

        Args:
        - x (torch.Tensor): Input node features.
        - edge_index (torch.Tensor): Edge indices representing connectivity.

        Returns:
        - Output tensor after the GCN layer computation.
        """
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

class GCNSI_model(torch.nn.Module):
    """
    Defines the model of Graph Convolutional Networks based Source Identification (GCNSI).
    """

    def __init__(self):
        super(GCNSI_model, self).__init__()
        self.conv1 = GCNConv(4, 128)  # Initializing the first GCN layer
        self.conv2 = GCNConv(128, 128)  # Initializing the second GCN layer
        self.fc = torch.nn.Linear(128, 2)  # Initializing a linear transformation layer

    def forward(self, alpha, laplacian, num_node, diff_vec, edge_index):
        """
        Performs the forward pass of the GCNSI model.

        Args:
        - alpha (float): The fraction of label information that node gets from its neighbors..
        - laplacian (numpy.ndarray): The Laplacian matrix of the graph.
        - num_node (int): Number of nodes in the graph.
        - diff_vec (torch.Tensor): The difference vector.
        - edge_index (torch.Tensor): Edge indices representing connectivity.

        Returns:
        - A tensor representing identified source nodes.
        """
        lpsi = LPSI()  # Initializing LPSI module
        V3 = copy.deepcopy(diff_vec)
        V4 = copy.deepcopy(diff_vec)
        V3[diff_vec < 0.5] = 0.5
        V4[diff_vec >= 0.5] = 0.5
        d1 = copy.deepcopy(diff_vec)
        d1 = d1[:, np.newaxis]
        d2 = lpsi.predict(laplacian, num_node,alpha, diff_vec)
        d2 = d2[:, np.newaxis]
        d3 = lpsi.predict(laplacian, num_node,alpha, V3)
        d3 = d3[:, np.newaxis]
        d4 = lpsi.predict(laplacian, num_node,alpha, V4)
        d4 = d4[:, np.newaxis]
        x = np.concatenate((d1, d2, d3, d4), axis=1)
        x = torch.tensor(x, dtype=torch.float)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x
