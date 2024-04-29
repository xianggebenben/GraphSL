import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from GraphSL.GNN.SLVAE.model import VAE,GNN,DiffusionPropagate
from torch.optim import Adam
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_score,recall_score
from GraphSL.Evaluation import Metric
class SLVAE_model(nn.Module):
    """
    Source Localization Variational Autoencoder (SLVAE) model combining VAE, GNN, and propagation modules.

    Attributes:
    - vae (nn.Module): Variational Autoencoder module.

    - gnn (nn.Module): Graph Neural Network module.

    - propagate (nn.Module): Propagation module.

    - reg_params (list): List of parameters requiring gradients.
    """

    def __init__(self, vae: nn.Module, gnn: nn.Module, propagate: nn.Module):
        """
        Initialize the SLVAE_model.

        Args:
        - vae (nn.Module): Variational Autoencoder module.

        - gnn (nn.Module): Graph Neural Network module.

        - propagate (nn.Module): Propagation module.
        """
        super(SLVAE_model, self).__init__()
        self.vae = vae
        self.gnn = gnn
        self.propagate = propagate
        self.reg_params = list(filter(lambda x: x.requires_grad, self.gnn.parameters()))

    def forward(self, seed_vec, train_mode):
        """
        Forward pass method of the SLVAE model.

        Args:

        - seed_vec (torch.Tensor): Seed vector.

        - train_mode (bool): Flag indicating whether in training mode.

        Returns:

        - seed_hat (torch.Tensor): reconstructed seed vector.

        - mean (torch.Tensor): Mean of the VAE.

        - log_var (torch.Tensor): Log variance of the VAE.

        - predictions (torch.Tensor): Predictions made by the SLVAE model.
        """
        # Pass seed_vec through VAE to obtain reconstructed seed vector, mean, and log variance
        seed_hat, mean, log_var = self.vae(seed_vec)

        if train_mode:
            # Ensure values of seed_hat are within range [0, 1]
            seed_hat.clamp(0, 1)
            # Pass seed_hat through GNN and perform propagation
            predictions = self.gnn(seed_hat)
            predictions = self.propagate(predictions)
        else:
            # Ensure values of seed_vec are within range [0, 1]
            seed_vec.clamp(0, 1)
            # Pass seed_vec through GNN and perform propagation
            predictions = self.gnn(seed_vec)
            predictions = self.propagate(predictions)

        predictions = torch.transpose(predictions, 0, 1)

        # Return reconstructed seed vector, mean, log variance, and predictions
        return seed_hat, mean, log_var, predictions

    def train_loss(self, x, x_hat, mean, log_var, y, y_hat):
        """
        Compute training loss.

        Args:

        - x (torch.Tensor): Seed vector.

        - x_hat (torch.Tensor): Reconstructed seed tensor.

        - mean (torch.Tensor): Mean of the VAE.

        - log_var (torch.Tensor): Log variance of the VAE.

        - y (torch.Tensor): Diffusion vector.

        - y_hat (torch.Tensor): Predicted Diffusion vector.

        Returns:

        - total_loss (torch.Tensor): Total loss is the sum of prediction loss, reconstruction loss and KL divergence.
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

        - y_true (torch.Tensor): True label tensor.

        - y_hat (torch.Tensor): Predicted label tensor.

        - x_hat (torch.Tensor): Reconstructed input tensor.

        - train_pred (torch.Tensor): Predicted tensor during training.

        Returns:

        - total_loss (torch.Tensor): Total loss tensor.
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

class SLVAE:
    """
    Implement the Source Localization Variational Autoencoder (SLVAE) model.

    Ling C, Jiang J, Wang J, et al. Source localization of graph diffusion via variational autoencoders for graph inverse problems[C]//Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 2022: 1010-1020.
    """

    def __init__(self):
        """
        Initialize the SLVAE model.
        """

    def train(self, adj, train_dataset, thres_list=[0.1, 0.3, 0.5, 0.7, 0.9], lr = 1e-3, weight_decay = 1e-4, num_epoch = 50,print_epoch =10):
        """
        Train the SLVAE model.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - train_dataset (torch.utils.data.dataset.Subset): the training dataset (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).

        - thres_list (list): List of threshold values to try.

        - lr (float): Learning rate.

        - weight_decay (float): Weight decay.

        - num_epoch (int): Number of training epochs.

        - print_epoch (int): Number of epochs every time to print loss.

        Returns:

        - slvae_model (SLVAE_model): Trained SLVAE model.

        - seed_vae_train (torch.Tensor): The latent representations of training seed vector from VAE, which is used to initialize seed vector in the test set.

        - opt_thres (float): Optimal threshold.

        - train_auc (float): Train AUC.

        - opt_f1 (float): Optimal F1 score.

        - pred (numpy.ndarray): Predicted seed vector of the training set, every column is the prediction of every simulation. It is used to adjust thres_list.

        Example:

        from GraphSL.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.GNN.SLVAE.main import SLVAE

        data_name = 'karate'

        graph = load_dataset(data_name)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        slave = SLVAE()

        slvae_model, seed_vae_train, thres, auc, f1, pred = slave.train(adj, train_dataset)

        print("SLVAE:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_node = adj.shape[0]
        adj_coo = adj.tocoo()
        values = adj_coo.data
        indices = np.vstack((adj_coo.row, adj_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj.shape

        adj_matrix = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to_dense()
        train_num = len(train_dataset)
        vae = VAE().to(device)
        gnn = GNN(adj_matrix=adj_matrix)
        propagate = DiffusionPropagate(adj_matrix, niter=2)

        slvae_model = SLVAE_model(vae, gnn, propagate).to(device)

        optimizer = Adam(slvae_model.parameters(), lr=lr)

        # Train SLVAE
        print("train SLVAE:")
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
            average_loss = overall_loss/train_num
            if epoch % print_epoch == 0:
                print(f"epoch = {epoch}, loss = {average_loss:.3f}")

        # Evaluation
        print("infer seed from training set:")

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

        optimizer = Adam(seed_infer, lr = lr, weight_decay = weight_decay)

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
            
            average_loss = overall_loss/train_num


            if epoch % print_epoch ==0:
                print(f"epoch = {epoch}, obj = {average_loss:.4f}")

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
                train_f1 += f1_score(seed_vec, seed_pred >= thres, zero_division = 1)
            train_f1 = train_f1 / train_num
            print(f"thres = {thres:.3f}, train_f1 = {train_f1:.3f}")
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres
        
        pred = np.zeros((num_node,train_num))

        for i in range(train_num):
            pred[:,i] = seed_infer[i].squeeze(-1).detach().numpy()


        return slvae_model, seed_vae_train, opt_thres, train_auc, opt_f1, pred

    def infer(self, test_dataset, slvae_model, seed_vae_train, thres, lr=0.001,num_epoch = 50, print_epoch = 10):
        """
        Infer using the SLVAE model.

        Args:

        - test_dataset (torch.utils.data.dataset.Subset): the test dataset (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).

        - slvae_model (SLVAE_model): Trained SLVAE model.

        - seed_vae_train (torch.Tensor): The latent representations of training seed vector from VAE, which is used to initialize
          seed vector in the test set.

        - thres (float): Threshold value.

        - lr (float): Learning rate.

        - num_epoch (int): Number of epochs.

        - print_epoch (int): Number of epochs every time to print loss.


        Returns:

        - Metric: Evaluation metric containing accuracy, precision, recall, F1 score, and AUC.

        Example:

        from GraphSL.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.GNN.SLVAE.main import SLVAE

        data_name = 'karate'

        graph = load_dataset(data_name)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        slave = SLVAE()

        slvae_model, seed_vae_train, thres, auc, f1, pred = slave.train(adj, train_dataset)

        print("SLVAE:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

        metric = slave.infer(test_dataset, slvae_model, seed_vae_train, thres)

        print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")
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

        optimizer = Adam(seed_infer, lr=lr)

        print("infer seed from test set:")
        for epoch in range(num_epoch):
            overall_loss = 0
            for i, influ_mat in enumerate(test_dataset):
                influ_vec = influ_mat[:, -1]
                influ_vec = influ_vec.unsqueeze(-1).float()
                optimizer.zero_grad()
                seed_vec_hat, _, _, influ_vec_hat = slvae_model(seed_infer[i], False)
                loss = slvae_model.infer_loss(influ_vec, influ_vec_hat, seed_vec_hat, seed_vae_train)

                overall_loss += loss.item()

                average_loss = overall_loss / test_num

                loss.backward()
                optimizer.step()

            
            if epoch % print_epoch == 0:
                print(f"epoch = {epoch}, obj = {average_loss:.4f}")

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
            test_pr += precision_score(seed_vec, seed_pred >= thres, zero_division = 1)
            test_re += recall_score(seed_vec, seed_pred >= thres, zero_division = 1)
            test_f1 += f1_score(seed_vec, seed_pred >= thres, zero_division = 1)
            test_auc += roc_auc_score(seed_vec, seed_pred)

        test_acc = test_acc / test_num
        test_pr = test_pr / test_num
        test_re = test_re / test_num
        test_f1 = test_f1 / test_num
        test_auc = test_auc / test_num

        metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
        return metric

