import torch.nn as nn
import numpy as np
import networkx as nx
import copy
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_score,recall_score
from scipy.sparse import csgraph
from scipy.sparse import coo_matrix
from evaluation import Metric

class LPSI(nn.Module):
    """
    Defines a module for Label Propagation based Source Identification (LPSI).
    """

    def __init__(self):
        """
          Initializes the LPSI module.
          """
        super().__init__()

    def forward(self, laplacian, num_node,alpha, diff_vec):

        x = (1 - alpha) * np.matmul(np.linalg.inv(np.eye(N=num_node) - alpha * laplacian), diff_vec)
        return x

    def train(self,adj,train_dataset,alpha_list=[0.01, 0.1, 1],thres_list=[0.1,0.3,0.5,0.7,0.9]):
        laplacian = csgraph.laplacian(adj, normed = False)
        laplacian = np.array(coo_matrix.todense(laplacian))
        num_node = adj.shape[0]
        train_num = len(train_dataset)
        opt_auc = 0
        opt_alpha = 0
        for alpha in alpha_list:
            train_auc = 0
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                influ_vec = influ_mat[:, -1]
                x = self.forward(laplacian,num_node,alpha, influ_vec)
                train_auc += roc_auc_score(seed_vec, x)
            train_auc = train_auc / train_num
            if train_auc > opt_auc:
                opt_auc = train_auc
                opt_alpha = alpha

        opt_f1 = 0
        opt_thres = 0
        for thres in thres_list:
            train_f1 = 0
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                influ_vec = influ_mat[:, -1]
                x = self.forward(laplacian,num_node,opt_alpha, influ_vec)
                train_f1 += f1_score(seed_vec, x>=thres)
            train_f1 = train_f1 / train_num
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres
        return opt_alpha, opt_thres

    def test(self, adj, test_dataset, alpha, thres):
        laplacian = csgraph.laplacian(adj, normed = False)
        laplacian = np.array(coo_matrix.todense(laplacian))
        num_node = adj.shape[0]
        test_num = len(test_dataset)
        test_acc = 0
        test_pr = 0
        test_re = 0
        test_f1 = 0
        test_auc = 0
        for influ_mat in test_dataset:
            seed_vec = influ_mat[:, 0]
            influ_vec = influ_mat[:, -1]
            x = self.forward(laplacian,num_node,alpha, influ_vec)
            test_acc += accuracy_score(seed_vec, x>=thres)
            test_pr += precision_score(seed_vec, x>=thres)
            test_re += recall_score(seed_vec, x>=thres)
            test_f1 += f1_score(seed_vec, x >= thres)
            test_auc += roc_auc_score(seed_vec, x)

        test_acc = test_acc / test_num
        test_pr = test_pr / test_num
        test_re = test_re / test_num
        test_f1 = test_f1 / test_num
        test_auc = test_auc / test_num
        metric = Metric(test_acc,test_pr,test_re,test_f1,test_auc)
        return metric

class NetSleuth(nn.Module):
    """
    Defines a module for NetSleuth.
    """

    def __init__(self):
        """
        Initializes the NetSleuth module with a given graph.

        Args:
        - G (networkx.Graph): The input graph.
        """
        super().__init__()  # Call the constructor of the superclass

    def forward(self, G, k, diff_vec):
        """
        Performs the forward pass of the NetSleuth module.

        Args:
        - k (int): Number of source nodes to identify.
        - diff_vec (numpy.ndarray): The diffusion vector.

        Returns:
        - seed_vec (torch.Tensor): A tensor representing identified source nodes.
        """
        g = copy.deepcopy(G)  # Creating a deep copy of the input graph
        g.remove_nodes_from([n for n in g if n not in np.where(diff_vec == 1)[0]])  # Removing non-relevant nodes
        lap = nx.laplacian_matrix(g).toarray()  # Computing the Laplacian matrix of the modified graph

        seed = []
        while len(seed) < k:
            value, vector = np.linalg.eig(lap)
            index = np.argmax(vector[np.argmin(value)])
            seed_index = list(g.nodes)[index]
            seed.append(seed_index)
            g.remove_node(seed_index)
            if len(g.nodes) == 0:
                break
            lap = nx.laplacian_matrix(g).toarray()

        seed_vec = torch.zeros(diff_vec.shape)
        seed_vec[seed] = 1
        return seed_vec

    def train(self,adj,train_dataset,k_list=[5, 10, 50, 100],thres_list=[0.1,0.3,0.5,0.7,0.9]):
        G = nx.from_numpy_array(adj)
        opt_auc = 0
        opt_k = 0
        train_num=len(train_dataset)
        for k in k_list:
            train_auc = 0
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                influ_vec = influ_mat[:, -1]
                x = self.forward(G,k, influ_vec)
                train_auc += roc_auc_score(seed_vec, x)
            train_auc = train_auc / train_num
            if train_auc > opt_auc:
                opt_auc = train_auc
                opt_k = k
        opt_f1 = 0
        opt_thres = 0
        for thres in thres_list:
            train_f1 = 0
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                influ_vec = influ_mat[:, -1]
                x = self.forward(G,opt_k, influ_vec)
                train_f1 += f1_score(seed_vec, x>=thres)
            train_f1 = train_f1 / train_num
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres
        return opt_k,opt_thres

    def test(self, adj, test_dataset, k, thres):
        G = nx.from_numpy_array(adj)
        test_num=len(test_dataset)
        test_acc = 0
        test_pr = 0
        test_re = 0
        test_f1 = 0
        test_auc = 0
        for influ_mat in test_dataset:
            seed_vec = influ_mat[:, 0]
            influ_vec = influ_mat[:, -1]
            x = self.forward(G, k, influ_vec)
            test_acc += accuracy_score(seed_vec, x >= thres)
            test_pr += precision_score(seed_vec, x >= thres)
            test_re += recall_score(seed_vec, x >= thres)
            test_f1 += f1_score(seed_vec, x >= thres)
            test_auc += roc_auc_score(seed_vec, x)

        test_acc = test_acc / test_num
        test_pr = test_pr / test_num
        test_re = test_re / test_num
        test_f1 = test_f1 / test_num
        test_auc = test_auc / test_num
        metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
        return metric

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
        Performs the forward pass of the GCNConv layer.

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

    def message(self, x_j, norm):
        """
        Defines the message passing operation.

        Args:
        - x_j (torch.Tensor): Node features of neighboring nodes.
        - norm (torch.Tensor): Normalization factor.

        Returns:
        - Normalized node features.
        """
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class GCNSI(torch.nn.Module):
    """
    Defines a Graph Convolutional Networks based Source Identification (GCNSI).
    """

    def __init__(self):
        super(GCNSI, self).__init__()
        self.conv1 = GCNConv(4, 128)  # Initializing the first GCN layer
        self.conv2 = GCNConv(128, 128)  # Initializing the second GCN layer
        self.fc = torch.nn.Linear(128, 2)  # Initializing a linear transformation layer

    def forward(self, alpha, laplacian, num_node, threshold, diff_vec, edge_index):
        """
        Performs the forward pass of the GCNSI model.

        Args:
        - alpha (float): The fraction of label information that node gets from its neighbors..
        - laplacian (numpy.ndarray): The Laplacian matrix of the graph.
        - num_node (int): Number of nodes in the graph.
        - threshold (float): Threshold value.
        - diff_vec (numpy.ndarray): The difference vector.
        - edge_index (torch.Tensor): Edge indices representing connectivity.

        Returns:
        - A tensor representing identified source nodes.
        """
        lpsi = LPSI(laplacian, num_node)  # Initializing LPSI module
        V3 = copy.deepcopy(diff_vec)
        V4 = copy.deepcopy(diff_vec)
        V3[diff_vec < threshold] = threshold
        V4[diff_vec >= threshold] = threshold
        d1 = copy.deepcopy(diff_vec)
        d1 = d1[:, np.newaxis]
        d2 = lpsi(alpha, diff_vec)
        d2 = d2[:, np.newaxis]
        d3 = lpsi(alpha, V3)
        d3 = d3[:, np.newaxis]
        d4 = lpsi(alpha, V4)
        d4 = d4[:, np.newaxis]
        x = np.concatenate((d1, d2, d3, d4), axis=1)
        x = torch.tensor(x, dtype=torch.float)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x


class OJC(nn.Module):
    """
    Defines a module for identifying potential source nodes using Optimal-Jordan-Cover (OJC) algorithm.
    """

    def __init__(self):
        """
        Initializes the OJC module with a given graph.

        Args:
        - G (networkx.Graph): The input graph.
        """
        super().__init__()  # Call the constructor of the superclass

    def get_K_list(self, G, Y, I, target):
        """
        Helper function to get the list of potential source nodes.

        Args:
        - Y (int): Number of desired source nodes.
        - I (list): diffused nodes.
        - target (numpy.ndarray): Target vector.

        Returns:
        - K (list): List of potential source nodes.
        """
        K = list()
        for n1 in G.nodes():
            count = 0
            for n2 in G[n1]:
                if target[n2] == 1:
                    count += 1
                if count == Y:
                    K.append(n1)
                    break
        K = list(set(K + I))
        return K

    def Candidate(self, G, Y, I, target):
        """
        Identifies potential source nodes based on the given criteria.

        Args:
        - Y (int): Number of desired source nodes.
        - I (list): List of diffused nodes.
        - target (numpy.ndarray): Target vector.

        Returns:
        - K (list): List of potential source nodes.
        - G_bar (networkx.Graph): Subgraph containing potential source nodes.
        """
        K = self.get_K_list(G, Y, I, target)
        G_prime = G.subgraph(K)
        # unforzen
        G_prime = nx.Graph(G_prime)
        if nx.is_connected(G_prime):
            G_bar = G_prime
        else:
            component = nx.connected_components(G_prime)
            R = [list(c)[0] for c in component]
            for n in R[1:]:
                if nx.has_path(G, n, R[0]):
                    path = nx.shortest_path(G, R[0], n)
                    nx.add_path(G_prime, path)
            G_bar = G_prime
        nx.is_connected(G_bar)
        return K, G_bar

    def forward(self, G, Y, I, target, num_source):
        """
        Performs the forward pass of the OJC module.

        Args:
        - Y (int): Number of desired source nodes.
        - I (list): List of diffused nodes.
        - target (numpy.ndarray): Target vector.
        - num_source (int): Number of potential source nodes to return.

        Returns:
        - x (numpy.ndarray): Binary vector representing identified potential source nodes.
        """
        K, G_bar = self.Candidate(G, Y, I, target)
        ecc = list()
        for n in K:
            long_path = 0
            for i in I:
                if nx.has_path(G_bar, n, i):
                    path = nx.shortest_path_length(G_bar, n, i)
                else:
                    path = 0
                if path > long_path:
                    long_path = path
            ecc.append(long_path)
        ecc_arr = np.array(ecc)
        index = np.argsort(ecc_arr).tolist()
        index = index[:num_source]
        x = np.zeros(target.shape)
        x[index] = 1
        return x

    def train(self,adj,train_dataset,Y_list=[1, 2, 3, 4, 5, 10, 20, 50],thres_list=[0.1,0.3,0.5,0.7,0.9],seed=0):

        G = nx.from_scipy_sparse_array(adj)
        train_num = len(train_dataset)
        opt_auc = 0
        opt_Y = 0
        for Y in Y_list:
            train_auc = 0
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                influ_vec = influ_mat[:, -1]
                num_source = len(influ_vec[influ_vec == 1])
                I = (influ_vec == 1).nonzero()[0].tolist()
                x = self.forward(G,Y, I, influ_vec, num_source)
                train_auc += roc_auc_score(seed_vec, x)
            train_auc = train_auc / train_num
            if train_auc > opt_auc:
                opt_auc = train_auc
                opt_Y = Y
        opt_f1 = 0
        opt_thres = 0
        for thres in thres_list:
            train_f1 = 0
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                influ_vec = influ_mat[:, -1]
                num_source = len(influ_vec[influ_vec == 1])
                I = (influ_vec == 1).nonzero()[0].tolist()
                x = self.forward(G,opt_Y, I, influ_vec, num_source)
                train_f1 += f1_score(seed_vec, x >= thres)
            train_f1 = train_f1 / train_num
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres
        return opt_Y,opt_thres

    def test(self, adj, test_dataset, Y, thres):
        G = nx.from_scipy_sparse_array(adj)
        test_num = len(test_dataset)
        test_acc = 0
        test_pr = 0
        test_re = 0
        test_f1 = 0
        test_auc = 0
        for influ_mat in test_dataset:
            seed_vec = influ_mat[:, 0]
            influ_vec = influ_mat[:, -1]
            num_source = len(influ_vec[influ_vec == 1])
            I = (influ_vec == 1).nonzero()[0].tolist()
            x = self.forward(G, Y, I, influ_vec, num_source)
            test_acc += accuracy_score(seed_vec, x >= thres)
            test_pr += precision_score(seed_vec, x >= thres)
            test_re += recall_score(seed_vec, x >= thres)
            test_f1 += f1_score(seed_vec, x >= thres)
            test_auc += roc_auc_score(seed_vec, x)

        test_acc = test_acc / test_num
        test_pr = test_pr / test_num
        test_re = test_re / test_num
        test_f1 = test_f1 / test_num
        test_auc = test_auc / test_num
        metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
        return metric
