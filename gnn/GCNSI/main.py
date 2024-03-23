import torch
from scipy.sparse import csgraph,coo_matrix
import numpy as np
from gnn.GCNSI.model import GCNSI_model
import copy
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_score,recall_score
from evaluation import Metric
class GCNSI(torch.nn.Module):
    """
    Defines a module for Graph Convolutional Networks based Source Identification (GCNSI).
    """

    def __init__(self):
        """
        Initializes the GCNSI module.
        """
        super(GCNSI, self).__init__()

    def train(self, adj, train_dataset, alpha_list=[0.01, 0.1, 1], thres_list=[0.1,0.3,0.5,0.7,0.9], num_epoch=500, weight=torch.tensor([1.0,3.0])):
        """
        Trains the GCNSI model.

        Args:
        - adj (scipy.sparse.csr_matrix): Adjacency matrix of the graph.
        - train_dataset (list): List of training data matrices.
        - alpha_list (list): List of alpha values for training.
        - thres_list (list): List of threshold values for training.
        - num_epoch (int): Number of training epochs.
        - weight (torch.Tensor): Weight tensor for loss computation.

        Returns:
        - opt_gcnsi_model (GCNSI_model): Optimized GCNSI model.
        - opt_alpha (float): Optimal alpha value.
        - opt_thres (float): Optimal threshold value.
        - opt_auc (float): Optimal AUC score.
        - opt_f1 (float): Optimal F1 score.
        """
        # Compute Laplacian matrix
        S = csgraph.laplacian(adj, normed=False)
        S = np.array(coo_matrix.todense(S))
        num_node = adj.shape[0]
        train_num = len(train_dataset)
        # Convert adjacency matrix to edge index tensor
        coo = adj.tocoo()
        row = torch.from_numpy(coo.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(coo.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        opt_auc = 0
        # Iterate over alpha values
        for alpha in alpha_list:
            gcnsi_model = GCNSI_model()
            optimizer = torch.optim.SGD(gcnsi_model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = torch.nn.CrossEntropyLoss(weight=weight)
            # Training loop
            for epoch in range(num_epoch):
                optimizer.zero_grad()
                total_loss = 0
                for influ_mat in train_dataset:
                    seed_vec = influ_mat[:, 0]
                    diff_vec = influ_mat[:, -1]
                    seed_vec = seed_vec.squeeze(-1).long()
                    pred = gcnsi_model(alpha, S, num_node, diff_vec, edge_index)
                    loss = criterion(pred, seed_vec)
                    total_loss += loss
                    loss.backward()
                    optimizer.step()
            train_auc = 0
            # Compute AUC score on training data
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                diff_vec = influ_mat[:, -1]
                seed_vec = seed_vec.squeeze(-1).long()
                pred = gcnsi_model(alpha, S, num_node, diff_vec, edge_index)
                pred = torch.softmax(pred, dim=1)
                pred = pred[:, 1].squeeze(-1).detach().numpy()
                train_auc += roc_auc_score(seed_vec, pred)
            train_auc = train_auc / train_num
            if train_auc > opt_auc:
                opt_auc = train_auc
                opt_alpha = alpha
                opt_gcnsi_model = copy.deepcopy(gcnsi_model)

        opt_f1 = 0
        # Find optimal threshold and F1 score
        for thres in thres_list:
            train_f1 = 0
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                diff_vec = influ_mat[:, -1]
                seed_vec = seed_vec.squeeze(-1).long()
                pred = opt_gcnsi_model(opt_alpha, S, num_node, diff_vec, edge_index)
                pred = torch.softmax(pred, dim=1)
                pred = pred[:, 1].squeeze(-1).detach().numpy()
                train_f1 += f1_score(seed_vec, pred >= thres)
            train_f1 = train_f1 / train_num
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres

        return opt_gcnsi_model, opt_alpha, opt_thres, opt_auc, opt_f1

    def test(self, adj, test_dataset, gcnsi_model, alpha, thres):
        """
        Tests the GCNSI model.

        Args:
        - adj (scipy.sparse.csr_matrix): Adjacency matrix of the graph.
        - test_dataset (list): List of testing data matrices.
        - gcnsi_model (GCNSI_model): Trained GCNSI model.
        - alpha (float): Alpha value.
        - thres (float): Threshold value.

        Returns:
        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC score.
        """
        # Compute Laplacian matrix
        S = csgraph.laplacian(adj, normed=False)
        S = np.array(coo_matrix.todense(S))
        num_node = adj.shape[0]
        test_num = len(test_dataset)
        # Convert adjacency matrix to edge index tensor
        coo = adj.tocoo()
        row = torch.from_numpy(coo.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(coo.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        test_acc = 0
        test_pr = 0
        test_re = 0
        test_f1 = 0
        test_auc = 0
        # Evaluate on test data
        for influ_mat in test_dataset:
            seed_vec = influ_mat[:, 0]
            diff_vec = influ_mat[:, -1]
            pred = gcnsi_model(alpha, S, num_node, diff_vec, edge_index)
            pred = torch.softmax(pred, dim=1)
            pred = pred[:, 1].squeeze(-1).detach().numpy()
            test_acc += accuracy_score(seed_vec, pred >= thres)
            test_pr += precision_score(seed_vec, pred >= thres)
            test_re += recall_score(seed_vec, pred >= thres)
            test_f1 += f1_score(seed_vec, pred >= thres)
            test_auc += roc_auc_score(seed_vec, pred)

        test_acc = test_acc / test_num
        test_pr = test_pr / test_num
        test_re = test_re / test_num
        test_f1 = test_f1 / test_num
        test_auc = test_auc / test_num
        metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
        return metric
