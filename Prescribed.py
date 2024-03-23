import torch.nn as nn
import numpy as np
import networkx as nx
import copy
import torch
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_score,recall_score
from scipy.sparse import csgraph,coo_matrix
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

    def forward(self, laplacian, num_node, alpha, diff_vec):
        """
        Forward pass of the LPSI module.

        Args:
        - laplacian (numpy.ndarray): The Laplacian matrix of the graph.
        - num_node (int): Number of nodes in the graph.
        - alpha (float): Label propagation parameter.
        - diff_vec (numpy.ndarray): The difference vector.

        Returns:
        - x (numpy.ndarray): The output of the label propagation.
        """
        x = (1 - alpha) * np.matmul(np.linalg.inv(np.eye(N=num_node) - alpha * laplacian), diff_vec)
        return x

    def train(self, adj, train_dataset, alpha_list=[0.01, 0.1, 1], thres_list=[0.1, 0.3, 0.5, 0.7, 0.9]):
        """
        Trains the LPSI module.

        Args:
        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.
        - train_dataset (list): List of training datasets.
        - alpha_list (list): List of alpha values to try.
        - thres_list (list): List of threshold values to try.

        Returns:
        - opt_alpha (float): Optimal alpha value.
        - opt_thres (float): Optimal threshold value.
        - opt_auc (float): Optimal Area Under the Curve (AUC) value.
        - opt_f1 (float): Optimal F1 score value.
        """
        laplacian = csgraph.laplacian(adj, normed=False)
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
                x = self.forward(laplacian, num_node, alpha, influ_vec)
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
                x = self.forward(laplacian, num_node, opt_alpha, influ_vec)
                train_f1 += f1_score(seed_vec, x >= thres)
            train_f1 = train_f1 / train_num
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres
        return opt_alpha, opt_thres, opt_auc, opt_f1

    def test(self, adj, test_dataset, alpha, thres):
        """
        Tests the LPSI module.

        Args:
        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.
        - test_dataset (list): List of testing datasets.
        - alpha (float): Alpha value.
        - thres (float): Threshold value.

        Returns:
        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC.
        """
        laplacian = csgraph.laplacian(adj, normed=False)
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
            x = self.forward(laplacian, num_node, alpha, influ_vec)
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

class NetSleuth(nn.Module):
    """
    Defines a module for NetSleuth.
    """

    def __init__(self):
        """
        Initializes the NetSleuth module with a given graph.
        """
        super().__init__()  # Call the constructor of the superclass

    def forward(self, G, k, diff_vec):
        """
        Performs the forward pass of the NetSleuth module.

        Args:
        - G (networkx.Graph): The input graph.
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

    def train(self, adj, train_dataset, k_list=[5, 10, 50, 100], thres_list=[0.1, 0.3, 0.5, 0.7, 0.9]):
        """
        Trains the NetSleuth module.

        Args:
        - adj (numpy.ndarray): The adjacency matrix of the graph.
        - train_dataset (list): List of training datasets.
        - k_list (list): List of k values to try.
        - thres_list (list): List of threshold values to try.

        Returns:
        - opt_k (int): Optimal k value.
        - opt_thres (float): Optimal threshold value.
        - opt_auc (float): Optimal Area Under the Curve (AUC) value.
        - opt_f1 (float): Optimal F1 score value.
        """
        G = nx.from_numpy_array(adj)
        opt_auc = 0
        opt_k = 0
        train_num = len(train_dataset)
        for k in k_list:
            train_auc = 0
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                influ_vec = influ_mat[:, -1]
                x = self.forward(G, k, influ_vec)
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
                x = self.forward(G, opt_k, influ_vec)
                train_f1 += f1_score(seed_vec, x >= thres)
            train_f1 = train_f1 / train_num
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres
        return opt_k, opt_thres, opt_auc, opt_f1

    def test(self, adj, test_dataset, k, thres):
        """
        Tests the NetSleuth module.

        Args:
        - adj (numpy.ndarray): The adjacency matrix of the graph.
        - test_dataset (list): List of testing datasets.
        - k (int): Number of source nodes.
        - thres (float): Threshold value.

        Returns:
        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC.
        """
        G = nx.from_numpy_array(adj)
        test_num = len(test_dataset)
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



class OJC(nn.Module):
    """
    Defines a module for identifying potential source nodes using Optimal-Jordan-Cover (OJC) algorithm.
    """

    def __init__(self):
        """
        Initializes the OJC module.
        """
        super().__init__()  # Call the constructor of the superclass

    def get_K_list(self, G, Y, I, target):
        """
        Helper function to get the list of potential source nodes.

        Args:
        - G (networkx.Graph): The input graph.
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
        - G (networkx.Graph): The input graph.
        - Y (int): Number of desired source nodes.
        - I (list): List of diffused nodes.
        - target (numpy.ndarray): Target vector.

        Returns:
        - K (list): List of potential source nodes.
        - G_bar (networkx.Graph): Subgraph containing potential source nodes.
        """
        K = self.get_K_list(G, Y, I, target)
        G_prime = G.subgraph(K)
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
        return K, G_bar

    def forward(self, G, Y, I, target, num_source):
        """
        Performs the forward pass of the OJC module.

        Args:
        - G (networkx.Graph): The input graph.
        - Y (int): Number of desired source nodes.
        - I (list): List of diffused nodes.
        - target (numpy.ndarray): Target vector.
        - num_source (int): Number of potential source nodes to return.

        Returns:
        - x (numpy.ndarray): Binary vector representing identified potential source nodes.
        """
        K, G_bar = self.Candidate(G, Y, I, target)
        ecc = []
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

    def train(self, adj, train_dataset, Y_list=[1, 2, 3, 4, 5, 10, 20, 50], thres_list=[0.1, 0.3, 0.5, 0.7, 0.9]):
        """
        Trains the OJC module.

        Args:
        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.
        - train_dataset (list): List of training datasets.
        - Y_list (list): List of Y values to try.
        - thres_list (list): List of threshold values to try.

        Returns:
        - opt_Y (int): Optimal Y value.
        - opt_thres (float): Optimal threshold value.
        - opt_auc (float): Optimal Area Under the Curve (AUC) value.
        - opt_f1 (float): Optimal F1 score value.
        """
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
                x = self.forward(G, Y, I, influ_vec, num_source)
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
                x = self.forward(G, opt_Y, I, influ_vec, num_source)
                train_f1 += f1_score(seed_vec, x >= thres)
            train_f1 = train_f1 / train_num
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres
        return opt_Y, opt_thres, opt_auc, opt_f1

    def test(self, adj, test_dataset, Y, thres):
        """
        Tests the OJC module.

        Args:
        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.
        - test_dataset (list): List of testing datasets.
        - Y (int): Number of desired source nodes.
        - thres (float): Threshold value.

        Returns:
        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC.
        """
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
