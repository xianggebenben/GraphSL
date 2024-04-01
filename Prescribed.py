import numpy as np
import networkx as nx
import copy
import torch
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_score,recall_score
from scipy.sparse import csgraph,coo_matrix
from Evaluation import Metric

class LPSI:
    """
    Implement the Label Propagation based Source Identification (LPSI) algorithm.

    Wang, Zheng, et al. "Multiple source detection without knowing the underlying propagation model." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 31. No. 1. 2017.
    """

    def __init__(self):
        """
        Initialize the LPSI module.
        """

    def predict(self, laplacian, num_node, alpha, diff_vec):
        """
        Prediction of the LPSI algorithm.

        Args:

        - laplacian (numpy.ndarray): The Laplacian matrix of the graph.

        - num_node (int): Number of nodes in the graph.

        - alpha (float): the fraction of label information that a node gets from its neighbors (between 0 and 1).

        - diff_vec (numpy.ndarray): The diffusion vector.

        Returns:

        - x (numpy.ndarray): Prediction of source nodes.
        """
        x = (1 - alpha) * np.matmul(np.linalg.pinv(np.eye(N=num_node) - alpha * laplacian,hermitian=True), diff_vec)
        return x

    def train(self, adj, train_dataset, alpha_list=[0.001, 0.01, 0.1], thres_list=[0.1, 0.3, 0.5, 0.7, 0.9]):
        """
         Train the LPSI algorithm.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - train_dataset (torch.utils.data.dataset.Subset): the training dataset (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).

        - alpha_list (list): List of the fraction of label information that a node gets from its neighbors (between 0 and 1) to try.

        - thres_list (list): List of threshold values to try.

        Returns:

        - opt_alpha (float): Optimal fraction of label information that a node gets from its neighbors, between 0 and 1.

        - opt_thres (float): Optimal threshold value.

        - opt_auc (float): Optimal Area Under the Curve (AUC) value.

        - opt_f1 (float): Optimal F1 score value.

        - opt_pred (numpy.ndarray): Prediction of training seed vector given opt_alpha, every column is the prediction of every simulation. It is used to adjust thres_list.

        Example:

        from GraphSL.data.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.Prescribed import LPSI

        data_name = 'karate'

        graph = load_dataset(data_name)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        lpsi = LPSI()

        alpha, thres, auc, f1, pred =lpsi.train(adj, train_dataset)

        print("LPSI:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")
        """
        laplacian = csgraph.laplacian(adj, normed=True)
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
                x = self.predict(laplacian, num_node, alpha, influ_vec)
                train_auc += roc_auc_score(seed_vec, x)
            train_auc = train_auc / train_num
            print(f"alpha = {alpha}, train_auc = {train_auc:.3f}")
            if train_auc > opt_auc:
                opt_auc = train_auc
                opt_alpha = alpha
        
        opt_pred =np.zeros((num_node,train_num))
        seed_all = np.zeros((num_node,train_num))
        for i,influ_mat in enumerate(train_dataset):
                seed_all[:,i] = influ_mat[:, 0]
                influ_vec = influ_mat[:, -1]
                opt_pred[:,i] = self.predict(laplacian, num_node, opt_alpha, influ_vec)

        opt_f1 = 0
        opt_thres = 0
        for thres in thres_list:
            print(f"thres = {thres:.3f}")
            train_f1 = 0
            for i in range(train_num):
                train_f1 += f1_score(seed_all[:,i], opt_pred[:,i] >= thres, zero_division = 1)
            train_f1 = train_f1 / train_num
            print(f"thres = {thres:.3f}, train_f1 = {train_f1:.3f}")
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres
        return opt_alpha, opt_thres, opt_auc, opt_f1, opt_pred

    def test(self, adj, test_dataset, alpha, thres):
        """
        Test the LPSI algorithm.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - test_dataset (torch.utils.data.dataset.Subset): The test dataset (number of simulations * number of graph nodes * 2(the first column is seed vector and the second column is diffusion vector)).

        - alpha (float): The fraction of label information that a node gets from its neighbors (between 0 and 1).

        - thres (float): Threshold value.

        Returns:

        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC.

        Example:

        from GraphSL.data.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.Prescribed import LPSI

        data_name = 'karate'

        graph = load_dataset(data_name)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset = split_dataset(dataset)

        lpsi = LPSI()

        alpha, thres, auc, f1, pred = lpsi.train(adj, train_dataset)

        print("LPSI:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

        metric=lpsi.test(adj, test_dataset, alpha, thres)

        print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")
        """
        laplacian = csgraph.laplacian(adj, normed=True)
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
            x = self.predict(laplacian, num_node, alpha, influ_vec)
            test_acc += accuracy_score(seed_vec, x >= thres)
            test_pr += precision_score(seed_vec, x >= thres, zero_division = 1)
            test_re += recall_score(seed_vec, x >= thres, zero_division = 1)
            test_f1 += f1_score(seed_vec, x >= thres, zero_division = 1)
            test_auc += roc_auc_score(seed_vec, x)

        test_acc = test_acc / test_num
        test_pr = test_pr / test_num
        test_re = test_re / test_num
        test_f1 = test_f1 / test_num
        test_auc = test_auc / test_num
        metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
        return metric

class NetSleuth:
    """
    Implement the NetSleuth algorithm.

    Prakash, B. Aditya, Jilles Vreeken, and Christos Faloutsos. "Spotting culprits in epidemics: How many and which ones?." 2012 IEEE 12th international conference on data mining. IEEE, 2012.
    """

    def __init__(self):

        """
        Initialize the NetSleuth.
        """

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

    def train(self, adj, train_dataset, k_list=[5, 10, 50, 100]):
        """
        Train the NetSleuth algorithm.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - train_dataset (torch.utils.data.dataset.Subset): The training dataset (number of simulations * number of graph nodes * 2(the first column is seed vector and the second column is diffusion vector)).

        - k_list (list): List of the numbers of source nodes to try.

        Returns:

        - opt_k (int): Optimal number of source nodes.

        - opt_auc (float): Optimal Area Under the Curve (AUC) value.

        - train_f1 (float): Training F1 score value.

        Example:

        from GraphSL.data.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.Prescribed import NetSleuth

        data_name = 'karate'

        graph = load_dataset(data_name)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        netSleuth = NetSleuth()

        k, auc, f1=netSleuth.train(adj, train_dataset)

        print("NetSleuth:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")
        """
        # Y should be no more than number of nodes
        num_node = adj.shape[0]
        condition = lambda k: k <= num_node
        k_list = list(filter(condition, k_list))

        G = nx.from_numpy_array(adj)
        opt_auc = 0
        opt_k = 0
        train_num = len(train_dataset)
        for k in k_list:
            train_auc = 0
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                influ_vec = influ_mat[:, -1]
                x = self.predict(G, k, influ_vec)
                train_auc += roc_auc_score(seed_vec, x)
            train_auc = train_auc / train_num
            print(f"k = {k}, train_auc = {train_auc:.3f}")

            if train_auc > opt_auc:
                opt_auc = train_auc
                opt_k = k

        train_f1 = 0
        for influ_mat in train_dataset:
            seed_vec = influ_mat[:, 0]
            influ_vec = influ_mat[:, -1]
            x = self.predict(G, opt_k, influ_vec)
            train_f1 += f1_score(seed_vec, x, zero_division = 1)
        train_f1 = train_f1 / train_num

        return opt_k, opt_auc, train_f1

    def test(self, adj, test_dataset, k):
        """
        Test the NetSleuth algorithm.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - test_dataset (torch.utils.data.dataset.Subset): The test dataset (number of simulations * number of graph nodes * 2(the first column is seed vector and the second column is diffusion vector)).

        - k (int): Number of source nodes.


        Returns:

        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC.

        Example:

        from GraphSL.data.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.Prescribed import NetSleuth

        data_name = 'karate'

        graph = load_dataset(data_name)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        netSleuth = NetSleuth()

        k, auc, f1=netSleuth.train(adj, train_dataset)

        print("NetSleuth:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

        metric = netSleuth.test(adj, test_dataset, k)

        print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")
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
            x = self.predict(G, k, influ_vec)
            test_acc += accuracy_score(seed_vec, x)
            test_pr += precision_score(seed_vec, x, zero_division = 1)
            test_re += recall_score(seed_vec, x, zero_division = 1)
            test_f1 += f1_score(seed_vec, x, zero_division = 1)
            test_auc += roc_auc_score(seed_vec, x)

        test_acc = test_acc / test_num
        test_pr = test_pr / test_num
        test_re = test_re / test_num
        test_f1 = test_f1 / test_num
        test_auc = test_auc / test_num
        metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
        return metric



class OJC:
    """
    Implement the Optimal-Jordan-Cover (OJC) algorithm.

    Zhu, Kai, Zhen Chen, and Lei Ying. "Catch’em all: Locating multiple diffusion sources in networks with partial observations." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 31. No. 1. 2017.
    """

    def __init__(self):
        """
        Initialize the OJC module.
        """

    def get_K_list(self, G, Y, I, target):
        """
         Get the list of potential source nodes.

        Args:

        - G (networkx.Graph): The input graph.

        - Y (int): Number of desired source nodes.

        - I (list): list of diffused nodes.

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

    def predict(self, G, Y, I, target, num_source):
        """
        Prediction of the OJC algorithm.

        Args:

        - G (networkx.Graph): The input graph.

        - Y (int): Number of source nodes.

        - I (list): List of diffused nodes.

        - target (numpy.ndarray): Target vector.

        - num_source (int): Maximal number of source nodes.

        Returns:

        - x (numpy.ndarray): A binary vector representing identified potential source nodes.
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

    def train(self, adj, train_dataset, Y_list=[5, 10, 20, 50]):
        """
        Train the OJC algorithm.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - train_dataset (torch.utils.data.dataset.Subset): The train dataset (number of simulations * number of graph nodes * 2(the first column is seed vector and the second column is diffusion vector)).

        - Y_list (list): List of numbers of source nodes to try.


        Returns:

        - opt_Y (int): Optimal number of source nodes.

        - opt_auc (float): Optimal Area Under the Curve (AUC) value.

        - train_f1 (float): Training F1 score value.

        Example:

        from GraphSL.data.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.Prescribed import OJC

        data_name = 'karate'

        graph = load_dataset(data_name)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        ojc = OJC()

        Y, auc, f1 =ojc.train(adj, train_dataset)

        print("OJC:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")
        """
        # Y should be no more than number of nodes
        num_node = adj.shape[0]
        condition = lambda k: k <= num_node
        Y_list = list(filter(condition, Y_list))
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
                I = (influ_vec == 1).nonzero().squeeze(-1).tolist()
                x = self.predict(G, Y, I, influ_vec, num_source)
                train_auc += roc_auc_score(seed_vec, x)
            train_auc = train_auc / train_num
            print(f"Y = {Y}, train_auc = {train_auc:.3f}")
            if train_auc > opt_auc:
                opt_auc = train_auc
                opt_Y = Y

        train_f1 = 0
        for influ_mat in train_dataset:
            seed_vec = influ_mat[:, 0]
            influ_vec = influ_mat[:, -1]
            num_source = len(influ_vec[influ_vec == 1])
            I = (influ_vec == 1).nonzero()[0].tolist()
            x = self.predict(G, opt_Y, I, influ_vec, num_source)
            train_f1 += f1_score(seed_vec, x, zero_division = 1)
        train_f1 = train_f1 / train_num

        return opt_Y, opt_auc, train_f1

    def test(self, adj, test_dataset, Y):
        """
        Test the OJC algorithm.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - test_dataset (torch.utils.data.dataset.Subset): The test dataset (number of simulations * number of graph nodes * 2(the first column is seed vector and the second column is diffusion vector)).

        - Y (int): Number of source nodes.

        Returns:

        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC.

        Example:

        from GraphSL.data.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.Prescribed import OJC

        data_name = 'karate'

        graph = load_dataset(data_name)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        ojc = OJC()

        Y, auc, f1 =ojc.train(adj, train_dataset)

        print("OJC:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

        metric=ojc.test(adj, test_dataset, Y)

        print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")
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
            x = self.predict(G, Y, I, influ_vec, num_source)
            test_acc += accuracy_score(seed_vec, x)
            test_pr += precision_score(seed_vec, x, zero_division = 1)
            test_re += recall_score(seed_vec, x, zero_division = 1)
            test_f1 += f1_score(seed_vec, x, zero_division = 1)
            test_auc += roc_auc_score(seed_vec, x)

        test_acc = test_acc / test_num
        test_pr = test_pr / test_num
        test_re = test_re / test_num
        test_f1 = test_f1 / test_num
        test_auc = test_auc / test_num
        metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
        return metric
