import torch
from scipy.sparse import csgraph, coo_matrix
import numpy as np
from GraphSL.GNN.GCNSI.model import GCNSI_model
from GraphSL.Prescribed import LPSI
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from GraphSL.utils import Metric


class GCNSI:
    """
    Implement the Graph Convolutional Networks based Source Identification (GCNSI).

    Dong, Ming, et al. "Multiple rumor source detection with graph convolutional networks." Proceedings of the 28th ACM international conference on information and knowledge management. 2019.
    """

    def __init__(self):
        """
        Initializes the GCNSI module.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self,
              adj,
              train_dataset,
              alpha=0.01,
              num_thres=10,
              lr=1e-3,
              num_epoch=100,
              print_epoch=10,
              random_seed=0):
        """
        Train the GCNSI model.

        Args:

        - adj (scipy.sparse.csr_matrix): Adjacency matrix of the graph.

        - train_dataset (torch.utils.data.dataset.Subset): The training dataset (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).

        - alpha (float): The fraction of label information that a node gets from its neighbors (between 0 and 1) to try.

        - num_thres (int): Number of threshold values to try.

        - lr (float): Learning rate.

        - num_epoch (int): Number of training epochs.

        - print_epoch (int): Number of epochs every time to print loss.

        - random_seed (int): Random seed.

        Returns:

        - gcnsi_model (GCNSI_model): GCNSI model.

        - opt_thres (float): Optimal threshold value.

        - train_auc (float): Training AUC score.

        - opt_f1 (float): Optimal F1 score.

        - opt_pred (numpy.ndarray): Predicted seed vector of the training set, every column is the prediction of every simulation. It is used to adjust thres_list.
        """
        # Compute Laplacian matrix
        S = csgraph.laplacian(adj, normed=True)
        S = torch.tensor(coo_matrix.todense(S), dtype=torch.float).to(self.device)
        num_node = adj.shape[0]
        train_num = len(train_dataset)

        # Convert adjacency matrix to edge index tensor
        coo = adj.tocoo()
        row = torch.from_numpy(coo.row.astype(np.int64)).to(self.device)
        col = torch.from_numpy(coo.col.astype(np.int64)).to(self.device)
        edge_index = torch.stack([row, col], dim=0)

        # Initialize model and optimizer
        torch.manual_seed(random_seed)
        gcnsi_model = GCNSI_model().to(self.device)
        optimizer = torch.optim.Adam(gcnsi_model.parameters(), lr=lr)
        num_seed = torch.sum(train_dataset[0][:, 0]).item()
        weight = torch.tensor([1, (num_node - num_seed) / num_seed], dtype=torch.float).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        # Augmented features by LPSI
        lpsi = LPSI()  # Initializing LPSI module
        lpsi_input = torch.zeros((train_num, num_node, 4), dtype=torch.float).to(self.device)

        for i, influ_mat in enumerate(train_dataset):
            diff_vec = influ_mat[:, -1].to(self.device)
            V3 = diff_vec.clone()
            V4 = diff_vec.clone()
            V3[diff_vec < 0.5] = 0.5
            V4[diff_vec >= 0.5] = 0.5

            d1 = diff_vec.unsqueeze(1)
            d2 = torch.tensor(lpsi.predict(S, num_node, alpha, diff_vec), dtype=torch.float, device=self.device).unsqueeze(1)
            d3 = torch.tensor(lpsi.predict(S, num_node, alpha, V3), dtype=torch.float, device=self.device).unsqueeze(1)
            d4 = torch.tensor(lpsi.predict(S, num_node, alpha, V4), dtype=torch.float, device=self.device).unsqueeze(1)

            lpsi_input[i] = torch.cat((d1, d2, d3, d4), dim=1)

        # Training loop
        print("train GCNSI:")
        for epoch in range(num_epoch):
            gcnsi_model.train()
            optimizer.zero_grad()
            total_loss = 0
            for i, influ_mat in enumerate(train_dataset):
                seed_vec = influ_mat[:, 0].unsqueeze(-1).to(self.device)
                seed_vec_onehot = torch.cat((1 - seed_vec, seed_vec), dim=1)
                pred = gcnsi_model(lpsi_input[i], edge_index)
                loss = criterion(pred, seed_vec_onehot)
                total_loss += loss
                loss.backward()
                optimizer.step()
            average_loss = total_loss / train_num
            if epoch % print_epoch == 0:
                print(f"Epoch [{epoch}/{num_epoch}], loss = {average_loss:.3f}")

        # Compute AUC score on training data
        train_auc = 0
        gcnsi_model.eval()
        with torch.no_grad():
            for i, influ_mat in enumerate(train_dataset):
                seed_vec = influ_mat[:, 0].long()
                pred = gcnsi_model(lpsi_input[i], edge_index)
                pred = torch.softmax(pred, dim=1)[:, 1].cpu().numpy()
                train_auc += roc_auc_score(seed_vec.numpy(), pred)
        train_auc /= train_num
        print(f"train_auc = {train_auc:.3f}")

        # Compute optimal F1 score and threshold
        opt_pred = np.zeros((num_node, train_num))
        seed_all = np.zeros((num_node, train_num))
        for i, influ_mat in enumerate(train_dataset):
            seed_all[:, i] = influ_mat[:, 0].numpy()
            pred = gcnsi_model(lpsi_input[i], edge_index)
            pred = torch.softmax(pred, dim=1)[:, 1].detach().cpu().numpy()
            opt_pred[:, i] = pred

        opt_f1 = -1
        pred_min = opt_pred.min()
        pred_max = opt_pred.max()
        thres_list = np.linspace(pred_min, pred_max, num=num_thres + 2)[1:-1]
        for thres in thres_list:
            train_f1 = 0
            for i in range(train_num):
                train_f1 += f1_score(seed_all[:, i], opt_pred[:, i] >= thres)
            train_f1 /= train_num
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

        - test_dataset (torch.utils.data.dataset.Subset): Test dataset containing simulations and graph nodes.

        - gcnsi_model (GCNSI_model): Trained GCNSI model.

        - thres (float): Threshold value.

        - alpha (float): Fraction of label information from neighbors.

        Returns:

        - metric (Metric): Evaluation metric containing accuracy, precision, recall, F1 score, and AUC score.
        """
        gcnsi_model = gcnsi_model.to(self.device)

        # Compute Laplacian matrix
        S = csgraph.laplacian(adj, normed=True)
        S = torch.tensor(coo_matrix.todense(S), dtype=torch.float).to(self.device)
        num_node = adj.shape[0]
        test_num = len(test_dataset)

        # Convert adjacency matrix to edge index tensor
        coo = adj.tocoo()
        row = torch.from_numpy(coo.row.astype(np.int64)).to(self.device)
        col = torch.from_numpy(coo.col.astype(np.int64)).to(self.device)
        edge_index = torch.stack([row, col], dim=0)

        # Augment features by LPSI
        lpsi = LPSI()
        lpsi_input = torch.zeros((test_num, num_node, 4), dtype=torch.float).to(self.device)

        for i, influ_mat in enumerate(test_dataset):
            diff_vec = influ_mat[:, -1].to(self.device)
            V3 = diff_vec.clone()
            V4 = diff_vec.clone()
            V3[diff_vec < 0.5] = 0.5
            V4[diff_vec >= 0.5] = 0.5

            d1 = diff_vec.unsqueeze(1)
            d2 = torch.tensor(lpsi.predict(S, num_node, alpha, diff_vec), dtype=torch.float, device=self.device).unsqueeze(1)
            d3 = torch.tensor(lpsi.predict(S, num_node, alpha, V3), dtype=torch.float, device=self.device).unsqueeze(1)
            d4 = torch.tensor(lpsi.predict(S, num_node, alpha, V4), dtype=torch.float, device=self.device).unsqueeze(1)

            lpsi_input[i] = torch.cat((d1, d2, d3, d4), dim=1)

        gcnsi_model.eval()
        lpsi_input = lpsi_input.to(self.device)

        # Evaluate on test data
        test_acc = 0
        test_pr = 0
        test_re = 0
        test_f1 = 0
        test_auc = 0

        with torch.no_grad():
            for influ_mat in test_dataset:
                seed_vec = influ_mat[:, 0].numpy()
                diff_vec = influ_mat[:, -1].to(self.device)
                pred = gcnsi_model(lpsi_input[i], edge_index)
                pred = torch.softmax(pred, dim=1)[:, 1].cpu().numpy()

                test_acc += accuracy_score(seed_vec, pred >= thres)
                test_pr += precision_score(seed_vec, pred >= thres, zero_division=1)
                test_re += recall_score(seed_vec, pred >= thres, zero_division=1)
                test_f1 += f1_score(seed_vec, pred >= thres, zero_division=1)
                test_auc += roc_auc_score(seed_vec, pred)

        test_acc /= test_num
        test_pr /= test_num
        test_re /= test_num
        test_f1 /= test_num
        test_auc /= test_num

        metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
        return metric
