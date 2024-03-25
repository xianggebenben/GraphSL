import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from GNN.SLVAE.model import VAE,GNN,DiffusionPropagate
from torch.optim import Adam
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_score,recall_score
from Evaluation import Metric
class SLVAE_model(nn.Module):
    """
        Source Localization Variational Autoencoder (SLVAE) model combining VAE, GNN, and propagation modules.

    Attributes:
    - vae (nn.Module): Variational Autoencoder module.
    - GNN (nn.Module): Graph Neural Network module.
    - propagate (nn.Module): Propagation module.
    - reg_params (list): List of parameters requiring gradients.
    """

    def __init__(self, vae: nn.Module, gnn: nn.Module, propagate: nn.Module):
        """
        Initialize the SLVAE_model.

        Args:
        - vae (nn.Module): Variational Autoencoder module.
        - GNN (nn.Module): Graph Neural Network module.
        - propagate (nn.Module): Propagation module.
        """
        super(SLVAE_model, self).__init__()
        self.vae = vae
        self.gnn = gnn
        self.propagate = propagate
        self.reg_params = list(filter(lambda x: x.requires_grad, self.gnn.parameters()))

    def forward(self, seed_vec, train_mode):
        """
        Forward pass method of the model.

        Args:
        - seed_vec (Tensor): Seed vector tensor.
        - train_mode (bool): Flag indicating whether in training mode.

        Returns:
        - Tuple[Tensor, Tensor, Tensor, Tensor]: Tuple containing seed_hat, mean, log_var, and predictions tensors.
        """
        seed_hat, mean, log_var = self.vae(seed_vec)
        if train_mode:
            seed_hat.clamp(0, 1)
            predictions = self.gnn(seed_hat)
            predictions = self.propagate(predictions)
        else:
            seed_vec.clamp(0, 1)
            predictions = self.gnn(seed_vec)
            predictions = self.propagate(predictions)
        return seed_hat, mean, log_var, predictions

    def train_loss(self, x, x_hat, mean, log_var, y, y_hat):
        """
        Compute training loss.

        Args:
        - x (Tensor): Input tensor.
        - x_hat (Tensor): Reconstructed input tensor.
        - mean (Tensor): Mean tensor.
        - log_var (Tensor): Log variance tensor.
        - y (Tensor): Target tensor.
        - y_hat (Tensor): Predicted tensor.

        Returns:
        - Tensor: Total loss tensor.
        """
        forward_loss = F.mse_loss(y_hat, y)
        reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        total_loss = forward_loss + reproduction_loss + KLD
        return total_loss

    def infer_loss(self, y_true, y_hat, x_hat, train_pred):
        """
        Compute inference loss.

        Args:
        - y_true (Tensor): True label tensor.
        - y_hat (Tensor): Predicted label tensor.
        - x_hat (Tensor): Reconstructed input tensor.
        - train_pred (Tensor): Predicted tensor during training.

        Returns:
        - Tensor: Total loss tensor.
        """
        device = y_true.device
        BN = nn.BatchNorm1d(1, affine=False).to(device)
        forward_loss = F.mse_loss(y_hat, y_true)
        log_pmf = []
        for pred in train_pred:
            log_lh = torch.zeros(1).to(device)
            for i, x_i in enumerate(x_hat[0]):
                temp = x_i * torch.log(pred[i]) + (1 - x_i) * torch.log(1 - pred[i]).to(torch.double)
                log_lh += temp
            log_pmf.append(log_lh)

        log_pmf = torch.stack(log_pmf)
        log_pmf = BN(log_pmf.float())

        pmf_max = torch.max(log_pmf)

        pdf_sum = pmf_max + torch.logsumexp(log_pmf - pmf_max, dim=0)

        total_loss = forward_loss - pdf_sum

        return total_loss

class SLVAE(nn.Module):
    """
    Source Localization Variational Autoencoder (SLVAE) model.

    Attributes:
    - None
    """

    def __init__(self):
        """
        Initialize the SLVAE model.
        """
        super(SLVAE, self).__init__()

    def train(self, adj, train_dataset, thres_list=[0.1, 0.3, 0.5, 0.7, 0.9], num_epoch=50):
        """
        Train the SLVAE model.

        Args:
        - adj (torch.Tensor): Adjacency matrix tensor.
        - train_dataset (List[torch.Tensor]): List of tensors containing training data.
        - infect_prob (float): Infection probability.
        - thres_list (List[float]): List of threshold values.
        - num_epoch (int): Number of training epochs.

        Returns:
        - Tuple: Tuple containing SLVAE model, seed , optimal threshold, train AUC, and optimal F1 score.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_node = adj.shape[0]
        adj_coo = adj.tocoo()
        values = adj_coo.data
        indices = np.vstack((adj_coo.row, adj_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj.shape

        adj_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
        train_num = len(train_dataset)
        vae = VAE().to(device)
        gnn = GNN(adj_matrix=adj_matrix)
        propagate = DiffusionPropagate(adj_matrix, niter=2)

        slvae_model = SLVAE_model(vae, gnn, propagate).to(device)

        optimizer = Adam(slvae_model.parameters(), lr=1e-3)

        # Train SLVAE
        slvae_model.train()
        for epoch in range(num_epoch):
            overall_loss = 0
            for influ_mat in train_dataset:
                seed_vec = influ_mat[:, 0]
                influ_vec = influ_mat[:, -1]
                influ_vec = influ_vec.unsqueeze(-1).float()
                seed_vec = seed_vec.unsqueeze(-1).float()
                optimizer.zero_grad()
                seed_vec_hat, mean, log_var, influ_vec_hat = slvae_model(seed_vec, True)
                loss = slvae_model.train_loss(seed_vec, seed_vec_hat, mean, log_var, influ_vec, influ_vec_hat)

                overall_loss += loss.item()

                loss.backward()
                optimizer.step()

        # Evaluation
        slvae_model.eval()
        for param in slvae_model.parameters():
            param.requires_grad = False

        seed_vae_train = torch.zeros(size=(train_num, num_node))
        for i, influ_mat in enumerate(train_dataset):
            seed_vec = influ_mat[:, 0].unsqueeze(-1).float()
            seed_vae_train[i, :] = slvae_model.vae(seed_vec)[0].squeeze(-1)
        seed_infer = []
        seed_vae_mean = torch.mean(seed_vae_train, 0).unsqueeze(-1).to(device)
        for i in range(train_num):
            seed_vec_hat, _, _, influ_vec_hat = slvae_model(seed_vae_mean, False)
            seed_infer.append(seed_vec_hat)

        for seed in seed_infer:
            seed.requires_grad = True

        optimizer = Adam(seed_infer, lr=1e-3)

        for epoch in range(num_epoch):
            overall_loss = 0
            for i, influ_mat in enumerate(train_dataset):
                influ_vec = influ_mat[:, -1]
                influ_vec = influ_vec.unsqueeze(-1).float()
                optimizer.zero_grad()
                seed_vec_hat, _, _, influ_vec_hat = slvae_model(seed_infer[i], False)
                loss = slvae_model.infer_loss(influ_vec, influ_vec_hat, seed_vec_hat, seed_vae_train)

                overall_loss += loss.item()

                loss.backward()
                optimizer.step()

        train_auc = 0
        for i, influ_mat in enumerate(train_dataset):
            seed_vec = influ_mat[:, 0]
            seed_vec = seed_vec.squeeze(-1).detach().numpy()
            seed_pred = seed_infer[i].detach().numpy()
            train_auc += roc_auc_score(seed_vec, seed_pred)
        train_auc = train_auc / train_num

        opt_f1 = 0
        opt_thres = 0
        for thres in thres_list:
            train_f1 = 0
            for i, influ_mat in enumerate(train_dataset):
                seed_vec = influ_mat[:, 0]
                seed_vec = seed_vec.squeeze(-1).detach().numpy()
                seed_pred = seed_infer[i].detach().numpy()
                train_f1 += f1_score(seed_vec, seed_pred >= thres)
            train_f1 = train_f1 / train_num
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres

        return slvae_model, seed_vae_train, opt_thres, train_auc, opt_f1

    def infer(self, test_dataset, slvae_model, seed_vae_train, thres, num_epoch=50):
        """
        Infer using the SLVAE model.

        Args:
        - test_dataset (List[torch.Tensor]): List of tensors containing test data.
        - slvae_model (SLVAE_model): Trained SLVAE model.
        - seed_vae_train (torch.Tensor): Seed VAE training data.
        - thres (float): Threshold value.
        - num_epoch (int): Number of epochs.

        Returns:
        - Metric: Evaluation metric containing accuracy, precision, recall, F1 score, and AUC.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_num = len(test_dataset)
        slvae_model.eval()
        for param in slvae_model.parameters():
            param.requires_grad = False

        seed_infer = []
        seed_mean = torch.mean(seed_vae_train, 0).unsqueeze(-1).to(device)
        for i in range(test_num):
            seed_vec_hat, _, _, influ_vec_hat = slvae_model(seed_mean, False)
            seed_infer.append(seed_vec_hat)

        for seed in seed_infer:
            seed.requires_grad = True

        optimizer = Adam(seed_infer, lr=1e-3)

        for epoch in range(num_epoch):
            overall_loss = 0
            for i, influ_mat in enumerate(test_dataset):
                influ_vec = influ_mat[:, -1]
                influ_vec = influ_vec.unsqueeze(-1).float()
                optimizer.zero_grad()
                seed_vec_hat, _, _, influ_vec_hat = slvae_model(seed_infer[i], False)
                loss = slvae_model.infer_loss(influ_vec, influ_vec_hat, seed_vec_hat, seed_vae_train)

                overall_loss += loss.item()

                loss.backward()
                optimizer.step()

        test_acc = 0
        test_pr = 0
        test_re = 0
        test_f1 = 0
        test_auc = 0

        for i, influ_mat in enumerate(test_dataset):
            seed_vec = influ_mat[:, 0]
            seed_vec = seed_vec.squeeze(-1).detach().numpy()
            seed_pred = seed_infer[i].detach().numpy()
            test_acc += accuracy_score(seed_vec, seed_pred >= thres)
            test_pr += precision_score(seed_vec, seed_pred >= thres)
            test_re += recall_score(seed_vec, seed_pred >= thres)
            test_f1 += f1_score(seed_vec, seed_pred >= thres)
            test_auc += roc_auc_score(seed_vec, seed_pred)

        test_acc = test_acc / test_num
        test_pr = test_pr / test_num
        test_re = test_re / test_num
        test_f1 = test_f1 / test_num
        test_auc = test_auc / test_num

        metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
        return metric

