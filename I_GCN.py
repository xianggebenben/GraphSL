import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from GraphSL.utils import diffusion_generation, download_dataset, load_dataset, split_dataset
import os

class I_GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(I_GCNLayer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear = nn.Linear(in_features, out_features).to(self.device)

    def forward(self, x, adj):
        # Perform graph convolution
        x = torch.matmul(adj, x)
        x = self.linear(x)
        return x

class I_GCN(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=3, d=2):
        super(I_GCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.iter_num = 100
        self.d = d        
        # I_GCN layers
        self.input_layer = I_GCNLayer(d + 1, hidden_dim).to(self.device)

        self.hidden_layers = nn.ModuleList([
            I_GCNLayer(hidden_dim, hidden_dim).to(self.device) for _ in range(num_layers - 2)
        ])

        self.output_layer = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, adj,seed_vector):
        # Feature construction: seed_vector concatenated with its diffusion features
        adj = adj.to(self.device)

        seed_vector = seed_vector.to(self.device)

        features = torch.cat([seed_vector] + [torch.matmul(adj.T, seed_vector) for _ in range(self.d)], dim=1)

        # Forward pass through the I_GCN
        x = F.relu(self.input_layer(features, adj))

        for layer in self.hidden_layers:
            x = F.relu(layer(x, adj))

        x = torch.sigmoid(self.output_layer(x))
        return x
    
    def backward(self, adj, influ_vector):
        adj = adj.to(self.device)
        influ_vector = influ_vector.to(self.device)
        seed_vector = torch.zeros(influ_vector.shape).to(self.device)

        for i in range(self.iter_num):
                seed_vector = influ_vector-self(adj,seed_vector)
        
        return seed_vector

def train_diffusion(adj, model, dataset, lr=1e-4, num_epoch=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    model.train()

    for epoch in range(num_epoch):
        train_loss = 0

        for influ_mat in dataset:
            seed_vector = influ_mat[:, 0].unsqueeze(-1).to(device)
            diff_vector = influ_mat[:, 1].unsqueeze(-1).to(device)

            optimizer.zero_grad()

            output = model(adj,seed_vector)
            loss = loss_function(output, diff_vector)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(dataset)

        print(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {train_loss:.4f}')

    print("Training complete!")

def test(adj, model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    loss_function = nn.MSELoss()
    test_loss = 0

    with torch.no_grad():
        for influ_mat in dataset:
            seed_vector = influ_mat[:, 0].unsqueeze(-1).to(device)
            diff_vector = influ_mat[:, 1].unsqueeze(-1).to(device)
            output = model(adj,seed_vector)
            loss = loss_function(output, diff_vector)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(dataset)
    print(f'Test Loss: {avg_test_loss:.4f}')

def test_seed(adj, model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    loss_function = nn.MSELoss()
    test_loss = 0

    with torch.no_grad():
        for influ_mat in dataset:
            seed_vector = influ_mat[:, 0].unsqueeze(-1).to(device)
            diff_vector = influ_mat[:, 1].unsqueeze(-1).to(device)
            output = model.backward(adj,diff_vector)
            loss = loss_function(output, seed_vector)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(dataset)
    print(f'Test Seed Loss: {avg_test_loss:.4f}')

# Example usage

# Example graph with random adjacency matrix
curr_dir = os.getcwd()
download_dataset(curr_dir)
# Load datasets ('karate', 'dolphins', 'jazz', 'netscience', 'cora_ml', 'power_grid')
data_name = 'power_grid'
graph = load_dataset(data_name, data_dir=curr_dir)

num_nodes = graph['adj_mat'].shape[0]  # Set num_nodes based on the dataset

# Generate diffusion data
dataset = diffusion_generation(graph=graph, sim_num=1000, diff_type='IC', seed_ratio=0.1, infect_prob=0.3)

adj, train_dataset, test_dataset = split_dataset(dataset)

adj = torch.Tensor(adj.toarray())

# Initialize and train I_GCN model
model = I_GCN()
train_diffusion(adj, model,train_dataset)

# Evaluate model on the test dataset
test(adj, model, test_dataset)

test_seed(adj, model, train_dataset)

test_seed(adj, model, test_dataset)

