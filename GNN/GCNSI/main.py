import torch
from scipy.sparse import csgraph,coo_matrix
import numpy as np
from GNN.GCNSI.model import GCNSI_model
import copy
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_score,recall_score
from Evaluation import Metric
class GCNSI:
    """
    Implement the Graph Convolutional Networks based Source Identification (GCNSI).

    Dong, Ming, et al. "Multiple rumor source detection with graph convolutional networks." Proceedings of the 28th ACM international conference on information and knowledge management. 2019.
    """

    def __init__(self):
        """
        Initializes the GCNSI module.
        """

    def train(self, adj, train_dataset, alpha_list=[0.001,0.01, 0.1], thres_list=[0.1,0.3,0.5,0.7,0.9], num_epoch=50,print_epoch=10, weight=torch.tensor([1.0,3.0])):
        """
        Train the GCNSI model.

        Args:
        - adj (scipy.sparse.csr_matrix): Adjacency matrix of the graph.

        - train_dataset (torch.utils.data.dataset.Subset): the training dataset (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).
        
        - alpha_list (list): List of the fraction of label information that a node gets from its neighbors (between 0 and 1) to try.
        
        - thres_list (list): List of threshold values to try.
        
        - num_epoch (int): Number of training epochs.
        
        - print_epoch (int): Number of epochs every time to print loss.

        - weight (torch.Tensor): Weight tensor for loss computation, the first and second values are loss weights for non-seed and seed, respectively.

        Returns:
        - opt_gcnsi_model (GCNSI_model): Optimized GCNSI model.

        - opt_alpha (float): Optimal alpha value.

        - opt_thres (float): Optimal threshold value.

        - opt_auc (float): Optimal AUC score.

        - opt_f1 (float): Optimal F1 score.

        - opt_pred (numpy.ndarray): Predicted seed vector of the training set given opt_alpha, every column is the prediction of every simulation. It is used to adjust thres_list.

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
        opt_auc = 0
        # Iterate over alpha values
        for alpha in alpha_list:
            print(f"alpha = {alpha:.3f}")
            gcnsi_model = GCNSI_model()
            optimizer = torch.optim.SGD(gcnsi_model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = torch.nn.CrossEntropyLoss(weight=weight)
            # Training loop
            for epoch in range(num_epoch):
                optimizer.zero_grad()
                total_loss = 0
                for influ_mat in train_dataset:
                    seed_vec = influ_mat[:, 0].unsqueeze(-1)
                    seed_vec_onehot =torch.concat((seed_vec,1-seed_vec),dim=1)
                    diff_vec = influ_mat[:, -1]
                    pred = gcnsi_model(alpha, S, num_node, diff_vec, edge_index)
                    loss = criterion(pred, seed_vec_onehot)
                    total_loss += loss
                    loss.backward()
                    optimizer.step()
                average_loss =total_loss/train_num
                if epoch%print_epoch==0:
                    print(f"epoch = {epoch}")
                    print(f"loss = {average_loss:.3f}")
            train_auc = 0
            # Compute AUC score on training data
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                diff_vec = influ_mat[:, -1]
                seed_vec = seed_vec.squeeze(-1).long()
                pred = gcnsi_model(alpha, S, num_node, diff_vec, edge_index)
                pred = pred[:, 1].squeeze(-1).detach().numpy()
                train_auc += roc_auc_score(seed_vec, pred)
            train_auc = train_auc / train_num
            if train_auc > opt_auc:
                opt_auc = train_auc
                opt_alpha = alpha
                opt_gcnsi_model = copy.deepcopy(gcnsi_model)
        
        opt_pred = np.zeros((num_node,train_num))
        seed_all = np.zeros((num_node,train_num))
        for i,influ_mat in enumerate(train_dataset):
                seed_all[:,i] = influ_mat[:, 0]
                diff_vec = influ_mat[:, -1]
                pred = opt_gcnsi_model(opt_alpha, S, num_node, diff_vec, edge_index)
                pred = torch.softmax(pred, dim=1)
                pred = pred[:, 1].squeeze(-1).detach().numpy()
                opt_pred[:,i] = pred

        opt_f1 = 0
        # Find optimal threshold and F1 score
        for thres in thres_list:
            print(f"thres = {thres:.3f}")
            train_f1 = 0
            for i in range(train_num):
                train_f1 += f1_score(seed_all[:,i], opt_pred[:,i] >= thres)
            train_f1 = train_f1 / train_num
            print(f" train_f1 = {train_f1:.3f}")
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres

        return opt_gcnsi_model, opt_alpha, opt_thres, opt_auc, opt_f1, opt_pred

    def test(self, adj, test_dataset, gcnsi_model, alpha, thres):
        """
        Test the GCNSI model.

        Args:
        - adj (scipy.sparse.csr_matrix): Adjacency matrix of the graph.
        - test_dataset (torch.utils.data.dataset.Subset): the test dataset (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).
        - gcnsi_model (GCNSI_model): Trained GCNSI model.
        - alpha (float): The fraction of label information that a node gets from its neighbors (between 0 and 1) to try..
        - thres (float): Threshold value.

        Returns:
        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC score.
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
