import torch
from scipy.sparse import csgraph, coo_matrix
import copy
import numpy as np
from GraphSL.GNN.GCNSI.model import GCNSI_model
from GraphSL.Prescribed import LPSI
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from GraphSL.Evaluation import Metric


class GCNSI:
    """
    Implement the Graph Convolutional Networks based Source Identification (GCNSI).

    Dong, Ming, et al. "Multiple rumor source detection with graph convolutional networks." Proceedings of the 28th ACM international conference on information and knowledge management. 2019.
    """

    def __init__(self):
        """
        Initializes the GCNSI module.
        """

    def train(self,
              adj,
              train_dataset,
              alpha=0.01,
              thres_list=[0.1,
                          0.3,
                          0.5,
                          0.7,
                          0.9],
              lr=1e-3,
              num_epoch=100,
              print_epoch=10,
              weight=torch.tensor([1.0,
                                   3.0]),
              random_seed=0):
        """
        Train the GCNSI model.

        Args:

        - adj (scipy.sparse.csr_matrix): Adjacency matrix of the graph.

        - train_dataset (torch.utils.data.dataset.Subset): the training dataset (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).

        - alpha (float): The fraction of label information that a node gets from its neighbors (between 0 and 1) to try.

        - thres_list (list): List of threshold values to try.

        - lr (float): Learning rate.

        - num_epoch (int): Number of training epochs.

        - print_epoch (int): Number of epochs every time to print loss.

        - weight (torch.Tensor): Weight tensor for loss computation, the first and second values are loss weights for non-seed and seed, respectively.

        - random_seed (int): Random seed.
        
        Returns:

        - gcnsi_model (GCNSI_model): GCNSI model.

        - opt_thres (float): Optimal threshold value.

        - train_auc (float): Training AUC score.

        - opt_f1 (float): Optimal F1 score.

        - opt_pred (numpy.ndarray): Predicted seed vector of the training set, every column is the prediction of every simulation. It is used to adjust thres_list.

        Example:

        import os

        curr_dir = os.getcwd()

        from GraphSL.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.GNN.GCNSI.main import GCNSI

        data_name = 'karate'

        graph = load_dataset(data_name, data_dir=curr_dir)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        gcnsi = GCNSI()

        gcnsi_model, thres, auc, f1, pred =gcnsi.train(adj, train_dataset)

        print("GCNSI:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")
        """
        # Compute Laplacian matrix
        S = csgraph.laplacian(adj, normed=True)
        S = np.array(coo_matrix.todense(S))
        num_node = adj.shape[0]
        train_num = len(train_dataset)
        # Convert adjacency matrix to edge index tensor
        coo = adj.tocoo()
        row = torch.from_numpy(coo.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(coo.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        # Iterate over alpha values
        torch.manual_seed(random_seed)
        gcnsi_model = GCNSI_model()
        optimizer = torch.optim.Adam(gcnsi_model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        # Augmented features by LPSI

        lpsi = LPSI()  # Initializing LPSI module
        lpsi_input = np.zeros((train_num, num_node, 4))

        for i, influ_mat in enumerate(train_dataset):
            diff_vec = influ_mat[:, -1]
            V3 = copy.deepcopy(diff_vec)
            V4 = copy.deepcopy(diff_vec)
            V3[diff_vec < 0.5] = 0.5
            V4[diff_vec >= 0.5] = 0.5
            d1 = copy.deepcopy(diff_vec)
            d1 = d1[:, np.newaxis]
            d2 = lpsi.predict(S, num_node, alpha, diff_vec)
            d2 = d2[:, np.newaxis]
            d3 = lpsi.predict(S, num_node, alpha, V3)
            d3 = d3[:, np.newaxis]
            d4 = lpsi.predict(S, num_node, alpha, V4)
            d4 = d4[:, np.newaxis]
            lpsi_input[i, :, :] = np.concatenate((d1, d2, d3, d4), axis=1)
        # Training loop
        print("train GCNSI:")
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            total_loss = 0
            for i, influ_mat in enumerate(train_dataset):
                seed_vec = influ_mat[:, 0].unsqueeze(-1)
                seed_vec_onehot = torch.concat((1 - seed_vec, seed_vec), dim=1)
                diff_vec = influ_mat[:, -1]
                pred = gcnsi_model(lpsi_input[i, :, :], edge_index)
                loss = criterion(pred, seed_vec_onehot)
                total_loss += loss
                loss.backward()
                optimizer.step()
            average_loss = total_loss / train_num
            if epoch % print_epoch == 0:
                print(f"Epoch [{epoch}/{num_epoch}], loss = {average_loss:.3f}")
        train_auc = 0
        # Compute AUC score on training data
        for i, influ_mat in enumerate(train_dataset):
            seed_vec = influ_mat[:, 0]
            seed_vec = seed_vec.squeeze(-1).long()
            pred = gcnsi_model(lpsi_input[i, :, :], edge_index)
            pred = torch.softmax(pred, dim=1)
            pred = pred[:, 1].squeeze(-1).detach().numpy()
            train_auc += roc_auc_score(seed_vec, pred)
        train_auc = train_auc / train_num
        print(f"train_auc = {train_auc:.3f}")

        opt_pred = np.zeros((num_node, train_num))
        seed_all = np.zeros((num_node, train_num))
        for i, influ_mat in enumerate(train_dataset):
            seed_all[:, i] = influ_mat[:, 0]
            diff_vec = influ_mat[:, -1]
            pred = gcnsi_model(lpsi_input[i, :, :], edge_index)
            pred = torch.softmax(pred, dim=1)
            pred = pred[:, 1].squeeze(-1).detach().numpy()
            opt_pred[:, i] = pred

        opt_f1 = 0
        # Find optimal threshold and F1 score
        for thres in thres_list:
            train_f1 = 0
            for i in range(train_num):
                train_f1 += f1_score(seed_all[:, i], opt_pred[:, i] >= thres)
            train_f1 = train_f1 / train_num
            print(f"thres = {thres:.3f}, train_f1 = {train_f1:.3f}")
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres

        return gcnsi_model, opt_thres, train_auc, opt_f1, opt_pred

    def test(self, adj, test_dataset, gcnsi_model, thres, alpha=0.01):
        """
        Test the GCNSI model.

        Args:

        - adj (scipy.sparse.csr_matrix): Adjacency matrix of the graph.

        - test_dataset (torch.utils.data.dataset.Subset): the test dataset (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).

        - gcnsi_model (GCNSI_model): Trained GCNSI model.

        - thres (float): Threshold value.

        - alpha (float): The fraction of label information that a node gets from its neighbors (between 0 and 1).


        Returns:

        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC score.

        Example:

        import os

        curr_dir = os.getcwd()

        from data.utils import load_dataset, diffusion_generation, split_dataset

        from GNN.GCNSI.main import GCNSI

        data_name = 'karate'

        graph = load_dataset(data_name, data_dir=curr_dir)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        gcnsi = GCNSI()

        gcnsi_model, thres, auc, f1, pred =gcnsi.train(adj, train_dataset)

        print("GCNSI:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

        metric = gcnsi.test(adj, test_dataset, gcnsi_model, thres)

        print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")
        """
        # Compute Laplacian matrix
        S = csgraph.laplacian(adj, normed=True)
        S = np.array(coo_matrix.todense(S))
        num_node = adj.shape[0]
        test_num = len(test_dataset)
        # Convert adjacency matrix to edge index tensor
        coo = adj.tocoo()
        row = torch.from_numpy(coo.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(coo.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        lpsi = LPSI()  # Initializing LPSI module
        lpsi_input = np.zeros((test_num, num_node, 4))

        for i, influ_mat in enumerate(test_dataset):
            diff_vec = influ_mat[:, -1]
            V3 = copy.deepcopy(diff_vec)
            V4 = copy.deepcopy(diff_vec)
            V3[diff_vec < 0.5] = 0.5
            V4[diff_vec >= 0.5] = 0.5
            d1 = copy.deepcopy(diff_vec)
            d1 = d1[:, np.newaxis]
            d2 = lpsi.predict(S, num_node, alpha, diff_vec)
            d2 = d2[:, np.newaxis]
            d3 = lpsi.predict(S, num_node, alpha, V3)
            d3 = d3[:, np.newaxis]
            d4 = lpsi.predict(S, num_node, alpha, V4)
            d4 = d4[:, np.newaxis]
            lpsi_input[i, :, :] = np.concatenate((d1, d2, d3, d4), axis=1)
        test_acc = 0
        test_pr = 0
        test_re = 0
        test_f1 = 0
        test_auc = 0
        # Evaluate on test data
        for influ_mat in test_dataset:
            seed_vec = influ_mat[:, 0]
            diff_vec = influ_mat[:, -1]
            pred = gcnsi_model(lpsi_input[i, :, :], edge_index)
            pred = torch.softmax(pred, dim=1)
            pred = pred[:, 1].squeeze(-1).detach().numpy()
            test_acc += accuracy_score(seed_vec, pred >= thres)
            test_pr += precision_score(seed_vec,
                                       pred >= thres, zero_division=1)
            test_re += recall_score(seed_vec, pred >= thres, zero_division=1)
            test_f1 += f1_score(seed_vec, pred >= thres, zero_division=1)
            test_auc += roc_auc_score(seed_vec, pred)

        test_acc = test_acc / test_num
        test_pr = test_pr / test_num
        test_re = test_re / test_num
        test_f1 = test_f1 / test_num
        test_auc = test_auc / test_num
        metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
        return metric
