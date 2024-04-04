import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
class GCNConv(MessagePassing):
    """
    Define a Graph Convolutional Network (GCN) layer.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the GCNConv layer with input and output channel dimensions.

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

        - torch.Tensor: Tensor after the GCN layer computation.
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
    Define the model of Graph Convolutional Networks based Source Identification (GCNSI).
    """

    def __init__(self):
        super(GCNSI_model, self).__init__()
        self.conv1 = GCNConv(4, 32)  # Initializing the first GCN layer
        self.conv2 = GCNConv(32, 32)  # Initializing the second GCN layer
        self.fc = torch.nn.Linear(32, 2)  # Initializing a linear transformation layer
        #self.softmax=torch.nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        """
        Performs the forward pass of the GCNSI model.

        Args:

        - x (numpy.ndarray): The input features augmented by LPSI.

        - edge_index (torch.Tensor): Edge indices representing connectivity.

        Returns:

        - x (torch.Tensor): A tensor representing identified source nodes.
        """
        
        x = torch.tensor(x, dtype=torch.float)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        #x = self.softmax(x)
        return x
