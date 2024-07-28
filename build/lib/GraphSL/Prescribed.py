import numpy as np
import networkx as nx
import copy
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from GraphSL.utils import Metric
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

class LPSI:
    """
    Implement the Label Propagation based Source Identification (LPSI) algorithm.

    Wang, Zheng, et al. "Multiple source detection without knowing the underlying propagation model." 
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 31. No. 1. 2017.
    """

    def __init__(self):
        """
        Initialize the LPSI module.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, laplacian, num_node, alpha, diff_vec):
        """
        Prediction of the LPSI algorithm.

        Args:

            - laplacian (torch.Tensor): The Laplacian matrix of the graph.

            - num_node (int): Number of nodes in the graph.

            - alpha (float): The fraction of label information that a node gets from its neighbors (between 0 and 1).

            - diff_vec (torch.Tensor): The diffusion vector.

        Returns:

            - x (torch.Tensor): Prediction of source nodes.
        """
        I = torch.eye(num_node, device=self.device)
        inv_matrix = torch.linalg.inv(I - alpha * laplacian)
        x = (1 - alpha) * inv_matrix @ diff_vec
        return x

    def train(self, adj, train_dataset, alpha_list=[0.001, 0.01, 0.1], num_thres=10):
        """
        Train the LPSI algorithm.

        Args:

            - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

            - train_dataset (torch.utils.data.dataset.Subset): The training dataset.

            - alpha_list (list): List of alpha values to try.

            - num_thres (int): Number of threshold values to try.

        Returns:

            - opt_alpha (float): Optimal fraction of label information that a node gets from its neighbors.

            - opt_thres (float): Optimal threshold value.

            - opt_auc (float): Optimal Area Under the Curve (AUC) value.

            - opt_f1 (float): Optimal F1 score value.

            - opt_pred (torch.Tensor): Prediction of training seed vector given opt_alpha.
        """
        laplacian = csgraph_laplacian(adj, normed=True).toarray()
        laplacian = torch.tensor(laplacian, dtype=torch.float32, device=self.device)
        num_node = adj.shape[0]
        train_num = len(train_dataset)

        # Initialize optimal values
        opt_auc = 0
        opt_alpha = 0
        opt_pred = torch.zeros((num_node, train_num), device=self.device)

        # Evaluate performance for different alphas
        for alpha in alpha_list:
            auc_scores = []
            for i, influ_mat in enumerate(train_dataset):
                seed_vec = torch.tensor(influ_mat[:, 0], dtype=torch.float32, device=self.device)
                influ_vec = torch.tensor(influ_mat[:, -1], dtype=torch.float32, device=self.device)
                x = self.predict(laplacian, num_node, alpha, influ_vec)
                auc_scores.append(roc_auc_score(seed_vec.cpu().numpy(), x.cpu().detach().numpy()))
            avg_auc = np.mean(auc_scores)
            print(f"alpha = {alpha}, train_auc = {avg_auc:.3f}")

            if avg_auc > opt_auc:
                opt_auc = avg_auc
                opt_alpha = alpha

        # Compute predictions for optimal alpha
        for i, influ_mat in enumerate(train_dataset):
            seed_all = torch.tensor(influ_mat[:, 0], dtype=torch.float32, device=self.device)
            influ_vec = torch.tensor(influ_mat[:, -1], dtype=torch.float32, device=self.device)
            opt_pred[:, i] = self.predict(laplacian, num_node, opt_alpha, influ_vec)

        # Determine the optimal threshold
        pred_min, pred_max = opt_pred.min(), opt_pred.max()
        thresholds = np.linspace(pred_min.item(), pred_max.item(), num=num_thres+2)[1:-1]
        f1_scores = []

        for thres in thresholds:
            predictions = (opt_pred >= thres).cpu().numpy()
            f1_scores.append(np.mean([f1_score(seed_all.cpu().numpy(), predictions[:, i], zero_division=1) for i, influ_mat in enumerate(train_dataset)]))
            print(f"thres = {thres:.3f}, train_f1 = {f1_scores[-1]:.3f}")

        opt_f1 = max(f1_scores)
        opt_thres = thresholds[f1_scores.index(opt_f1)]

        return opt_alpha, opt_thres, opt_auc, opt_f1, opt_pred

    def test(self, adj, test_dataset, alpha, thres):
        """
        Test the LPSI algorithm.

        Args:

            - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

            - test_dataset (torch.utils.data.dataset.Subset): The test dataset.

            - alpha (float): The fraction of label information that a node gets from its neighbors.

            - thres (float): Threshold value.

        Returns:

            - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC.
        """
        laplacian = csgraph_laplacian(adj, normed=True).toarray()
        laplacian = torch.tensor(laplacian, dtype=torch.float32, device=self.device)
        num_node = adj.shape[0]
        test_num = len(test_dataset)

        metrics = np.zeros(5)  # [accuracy, precision, recall, f1, auc]

        # Compute metrics
        for influ_mat in test_dataset:
            seed_vec = torch.tensor(influ_mat[:, 0], dtype=torch.float32, device=self.device)
            influ_vec = torch.tensor(influ_mat[:, -1], dtype=torch.float32, device=self.device)
            x = self.predict(laplacian, num_node, alpha, influ_vec)
            predictions = (x >= thres).cpu().numpy()
            metrics += np.array([
                accuracy_score(seed_vec.cpu().numpy(), predictions),
                precision_score(seed_vec.cpu().numpy(), predictions, zero_division=1),
                recall_score(seed_vec.cpu().numpy(), predictions, zero_division=1),
                f1_score(seed_vec.cpu().numpy(), predictions, zero_division=1),
                roc_auc_score(seed_vec.cpu().numpy(), x.cpu().detach().numpy())
            ])

        metrics /= test_num
        return Metric(*metrics)


class NetSleuth:
    """
    Implement the NetSleuth algorithm.

    Prakash, B. Aditya, Jilles Vreeken, and Christos Faloutsos. "Spotting culprits in epidemics: How many and which ones?." 2012 IEEE 12th international conference on data mining. IEEE, 2012.
    """

    def __init__(self):
        """
        Initialize the NetSleuth.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def predict(self, G, k, diff_vec):
        """
        Prediction of the NetSleuth algorithm.

        Args:

        - G (networkx.Graph): The input graph.

        - k (int): Number of source nodes to identify.

        - diff_vec (torch.Tensor): The diffusion vector.

        Returns:

        - seed_vec (torch.Tensor): A binary tensor representing identified source nodes.
        """
        # Create a copy of the graph and filter nodes
        g = G.subgraph([n for n in G.nodes if diff_vec[n] == 1]).copy()
        if len(g.nodes) == 0:
            return torch.zeros(len(diff_vec), dtype=torch.float32)

        # Compute the Laplacian matrix
        lap = nx.laplacian_matrix(g).toarray()
        lap = torch.tensor(lap, dtype=torch.float32).to(self.device)

        # Initialize the seed list
        seed = []
        while len(seed) < k and len(g.nodes) > 0:
            # Compute eigenvalues and eigenvectors
            values, vectors = torch.linalg.eig(lap)
            values = values.real  # Use only real part of eigenvalues
            vectors = vectors.real  # Use only real part of eigenvectors

            # Find the index of the maximum eigenvalue
            index = torch.argmax(values)
            seed_index = list(g.nodes)[index]
            seed.append(seed_index)

            # Remove the selected node and update the Laplacian matrix
            g.remove_node(seed_index)
            if len(g.nodes) > 0:
                lap = nx.laplacian_matrix(g).toarray()
                lap = torch.tensor(lap, dtype=torch.float32).to(self.device)

        # Create the seed vector
        seed_vec = torch.zeros(len(diff_vec), dtype=torch.float32).to(self.device)
        seed_vec[seed] = 1
        return seed_vec

    def train(self, adj, train_dataset, k_list=[2, 5, 10]):
        """
        Train the NetSleuth algorithm.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - train_dataset (torch.utils.data.dataset.Subset): The training dataset.

        - k_list (list): List of the numbers of source nodes to try.

        Returns:

        - opt_k (int): Optimal number of source nodes.

        - opt_auc (float): Optimal Area Under the Curve (AUC) value.

        - train_f1 (float): Training F1 score value.
        """
        num_node = adj.shape[0]
        k_list = [k for k in k_list if k <= num_node]

        G = nx.from_numpy_array(adj.toarray())
        opt_auc = 0
        opt_k = 0
        train_num = len(train_dataset)

        for k in k_list:
            auc_scores = []
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                influ_vec = torch.tensor(influ_mat[:, -1], dtype=torch.float32)
                x = self.predict(G, k, influ_vec)
                auc_scores.append(roc_auc_score(seed_vec, x.cpu().numpy()))
            avg_auc = np.mean(auc_scores)
            print(f"k = {k}, train_auc = {avg_auc:.3f}")

            if avg_auc > opt_auc:
                opt_auc = avg_auc
                opt_k = k

        f1_scores = []
        for influ_mat in train_dataset:
            seed_vec = influ_mat[:, 0]
            influ_vec = torch.tensor(influ_mat[:, -1], dtype=torch.float32)
            x = self.predict(G, opt_k, influ_vec)
            f1_scores.append(f1_score(seed_vec, x.cpu().numpy(), zero_division=1))
        train_f1 = np.mean(f1_scores)

        return opt_k, opt_auc, train_f1

    def test(self, adj, test_dataset, k):
        """
        Test the NetSleuth algorithm.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - test_dataset (torch.utils.data.dataset.Subset): The test dataset.

        - k (int): Number of source nodes.

        Returns:

        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC.
        """
        G = nx.from_numpy_array(adj.toarray())
        test_num = len(test_dataset)
        metrics = {'acc': 0, 'pr': 0, 're': 0, 'f1': 0, 'auc': 0}

        for influ_mat in test_dataset:
            seed_vec = influ_mat[:, 0]
            influ_vec = torch.tensor(influ_mat[:, -1], dtype=torch.float32)
            x = self.predict(G, k, influ_vec)
            metrics['acc'] += accuracy_score(seed_vec, x.cpu().numpy())
            metrics['pr'] += precision_score(seed_vec, x.cpu().numpy(), zero_division=1)
            metrics['re'] += recall_score(seed_vec, x.cpu().numpy(), zero_division=1)
            metrics['f1'] += f1_score(seed_vec, x.cpu().numpy(), zero_division=1)
            metrics['auc'] += roc_auc_score(seed_vec, x.cpu().numpy())

        for key in metrics:
            metrics[key] /= test_num

        return Metric(metrics['acc'], metrics['pr'], metrics['re'], metrics['f1'], metrics['auc'])

class OJC:
    """
    Implement the Optimal-Jordan-Cover (OJC) algorithm.

    Zhu, Kai, Zhen Chen, and Lei Ying. "Catch’em all: Locating multiple diffusion sources in networks with partial observations." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 31. No. 1. 2017.
    """

    def __init__(self):
        """
        Initialize the OJC module.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_K_list(self, G, Y, I, target):
        """
        Get the list of potential source nodes.

        Args:

        - G (networkx.Graph): The input graph.

        - Y (int): Number of desired source nodes.

        - I (list): List of diffused nodes.

        - target (torch.Tensor): Target vector.

        Returns:

        - K (list): List of potential source nodes.
        """


        K = []
        target_set = set(torch.where(target == 1)[0])  # Create a set for quick lookup

        for n1 in G.nodes():
            count = sum(1 for n2 in G.neighbors(n1) if n2 in target_set)
            if count == Y:
                K.append(n1)

        K = list(set(K + I))
        return K

    def Candidate(self, G, Y, I, target):
        """
        Identify potential source nodes based on the given criteria.

        Args:

        - G (networkx.Graph): The input graph.

        - Y (int): Number of desired source nodes.

        - I (list): List of diffused nodes.

        - target (torch.Tensor): Target vector.

        Returns:

        - K (list): List of potential source nodes.
        
        - G_bar (networkx.Graph): Subgraph containing potential source nodes.
        """
        K = self.get_K_list(G, Y, I, target)
        G_prime = G.subgraph(K).copy()  # Make sure to use a copy

        if not nx.is_connected(G_prime):
            # Connect components with shortest paths
            components = list(nx.connected_components(G_prime))
            for component in components[1:]:
                source = next(iter(components[0]))  # Pick a node from the first component
                for node in component:
                    if nx.has_path(G, source, node):
                        path = nx.shortest_path(G, source, node)
                        G_prime.add_edges_from(zip(path[:-1], path[1:]))  # Add path edges

        return K, G_prime

    def predict(self, G, Y, I, target, num_source):
        """
        Prediction of the OJC algorithm.

        Args:

        - G (networkx.Graph): The input graph.

        - Y (int): Number of source nodes.

        - I (list): List of diffused nodes.

        - target (torch.Tensor): Target vector.

        - num_source (int): Maximal number of source nodes.

        Returns:

        - x (torch.Tensor): A binary vector representing identified potential source nodes.
        """
        K, G_bar = self.Candidate(G, Y, I, target)
        ecc = []

        for n in K:
            # Compute eccentricity in one pass
            eccentricity = max(
                (nx.shortest_path_length(G_bar, n, i) if nx.has_path(G_bar, n, i) else 0)
                for i in I
            )
            ecc.append(eccentricity)

        ecc_arr = torch.Tensor(ecc).to(self.device)
        indices = torch.argsort(ecc_arr)[-num_source:]  # Get the top num_source indices
        x = torch.zeros_like(target, dtype=torch.float32).to(self.device)
        x[indices] = 1
        return x

    def train(self, adj, train_dataset, Y_list=[2, 5, 10]):
        """
        Train the OJC algorithm.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - train_dataset (torch.utils.data.dataset.Subset): The train dataset.

        - Y_list (list): List of numbers of source nodes to try.

        Returns:

        - opt_Y (int): Optimal number of source nodes.

        - opt_auc (float): Optimal Area Under the Curve (AUC) value.

        - train_f1 (float): Training F1 score value.
        """
        num_node = adj.shape[0]
        Y_list = [Y for Y in Y_list if Y <= num_node]
        G = nx.from_scipy_sparse_array(adj)  # Correct method for conversion

        opt_auc = 0
        opt_Y = 0

        for Y in Y_list:
            auc_scores = []

            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0].cpu().numpy()
                influ_vec = influ_mat[:, -1].to(self.device)


                num_source = torch.sum(influ_vec == 1)
                I = torch.where(influ_vec == 1)[0].tolist()
                x = self.predict(G, Y, I, influ_vec, num_source).cpu().numpy()
                auc_scores.append(roc_auc_score(seed_vec, x))

            avg_auc = np.mean(auc_scores)
            print(f"Y = {Y}, train_auc = {avg_auc:.3f}")

            if avg_auc > opt_auc:
                opt_auc = avg_auc
                opt_Y = Y

        f1_scores = []
        for influ_mat in train_dataset:
            seed_vec = influ_mat[:, 0].cpu().numpy()
            influ_vec = influ_mat[:, -1].to(self.device)

            num_source = torch.sum(influ_vec == 1)
            I = torch.where(influ_vec == 1)[0].tolist()
            x = self.predict(G, opt_Y, I, influ_vec, num_source).cpu().numpy()
            f1_scores.append(f1_score(seed_vec, x, zero_division=1))

        train_f1 = np.mean(f1_scores)
        return opt_Y, opt_auc, train_f1

    def test(self, adj, test_dataset, Y):
        """
        Test the OJC algorithm.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - test_dataset (torch.utils.data.dataset.Subset): The test dataset.

        - Y (int): Number of source nodes.

        Returns:

        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC.
        """
        G = nx.from_scipy_sparse_array(adj)  # Correct method for conversion
        test_num = len(test_dataset)
        metrics = {'acc': 0, 'pr': 0, 're': 0, 'f1': 0, 'auc': 0}

        for influ_mat in test_dataset:
            seed_vec = influ_mat[:, 0].cpu().numpy()
            influ_vec = influ_mat[:, -1].to(self.device)

            num_source = torch.sum(influ_vec == 1)
            I = torch.where(influ_vec == 1)[0].tolist()
            x = self.predict(G, Y, I, influ_vec, num_source).cpu().numpy()
            metrics['acc'] += accuracy_score(seed_vec, x)
            metrics['pr'] += precision_score(seed_vec, x, zero_division=1)
            metrics['re'] += recall_score(seed_vec, x, zero_division=1)
            metrics['f1'] += f1_score(seed_vec, x, zero_division=1)
            metrics['auc'] += roc_auc_score(seed_vec, x)

        for key in metrics:
            metrics[key] /= test_num

        return Metric(metrics['acc'], metrics['pr'], metrics['re'], metrics['f1'], metrics['auc'])
