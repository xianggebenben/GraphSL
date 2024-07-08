import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from GraphSL.GNN.IVGD.correction import correction
from GraphSL.GNN.IVGD.i_deepis import i_DeepIS, DiffusionPropagate
from GraphSL.GNN.IVGD.training import FeatureCons, get_idx_new_seeds
from GraphSL.GNN.IVGD.model.MLP import MLPTransform
from GraphSL.GNN.IVGD.training import train_model
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from GraphSL.Evaluation import Metric
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")


class IVGD_model(torch.nn.Module):
    """
     Invertible Validity-aware Graph Diffusion (IVGD) Model.
    """

    def __init__(self, alpha, tau, rho):
        """
        Initializes the IVGD_model.

        Args:

        - alpha (float): Value of alpha parameter.

        - tau (float): Value of tau parameter.

        - rho (float): Value of rho parameter.
        """
        super(IVGD_model, self).__init__()
        self.number_layer = 5
        self.alpha1 = alpha
        self.alpha2 = alpha
        self.alpha3 = alpha
        self.alpha4 = alpha
        self.alpha5 = alpha

        self.tau1 = tau
        self.tau2 = tau
        self.tau3 = tau
        self.tau4 = tau
        self.tau5 = tau

        self.net1 = correction()
        self.net2 = correction()
        self.net3 = correction()
        self.net4 = correction()
        self.net5 = correction()

        self.rho1 = rho
        self.rho2 = rho
        self.rho3 = rho
        self.rho4 = rho
        self.rho5 = rho

    def forward(self, x, label, lamda):
        """
        Perform the forward pass of IVGD_model.

        Args:

        - x (torch.Tensor): Input tensor.

        - label (torch.Tensor): Label tensor.

        - lamda (float): Value of lambda parameter.

        Returns:

        - x (torch.Tensor): Output tensor after forward pass.
        """
        self.net1.to(x.device)
        self.net2.to(x.device)
        self.net3.to(x.device)
        self.net4.to(x.device)
        self.net5.to(x.device)
        sum = torch.sum(label)
        label = torch.cat((1 - label, label), dim=1)
        x = torch.cat((1 - x, x), dim=1)
        prob = x[:, 1].unsqueeze(-1)
        x = (self.tau1 * self.net1(prob) - label * torch.softmax(x, dim=1) / label.shape[0] - lamda
             - self.rho1 * (torch.sum(x) - sum) + self.alpha1 * x) / (
            self.tau1 + self.alpha1)
        prob = x[:, 1].unsqueeze(-1)
        lamda = lamda + self.rho1 * (torch.sum(prob) - sum)
        x = (self.tau2 * self.net2(prob) - label * torch.softmax(x, dim=1) / label.shape[0] - lamda
             - self.rho2 * (torch.sum(x) - sum) + self.alpha2 * x) / (
            self.tau2 + self.alpha2)
        prob = x[:, 1].unsqueeze(-1)
        lamda = lamda + self.rho2 * (torch.sum(prob) - sum)
        x = (self.tau3 * self.net3(prob) - label * torch.softmax(x, dim=1) / label.shape[0] - lamda
             - self.rho3 * (torch.sum(x) - sum) + self.alpha3 * x) / (
            self.tau3 + self.alpha3)
        prob = x[:, 1].unsqueeze(-1)
        lamda = lamda + self.rho3 * (torch.sum(prob) - sum)
        x = (self.tau4 * self.net4(prob) - label * torch.softmax(x, dim=1) / label.shape[0] - lamda
             - self.rho4 * (torch.sum(x) - sum) + self.alpha4 * x) / (
            self.tau4 + self.alpha4)
        prob = x[:, 1].unsqueeze(-1)
        lamda = lamda + self.rho4 * (torch.sum(prob) - sum)
        x = (self.tau5 * self.net5(prob) - label * torch.softmax(x, dim=1) / label.shape[0] - lamda
             - self.rho5 * (torch.sum(x) - sum) + self.alpha5 * x) / (
            self.tau5 + self.alpha5)
        return x

    def correction(self, pred):
        """
        Corrects predictions based on input.

        Args:

        - pred (torch.Tensor): Input tensor of predictions.

        Returns:

        - torch.Tensor: Corrected predictions.
        """
        temp = pred[:, 0].unsqueeze(-1)
        return (self.net1(temp) + self.net2(temp) + self.net3(temp) +
                self.net4(temp) + self.net5(temp)) / self.number_layer


class IVGD:
    """
    Implement the Invertible Validity-aware Graph Diffusion (IVGD) model.

    Wang, Junxiang, Junji Jiang, and Liang Zhao. "An invertible graph diffusion neural network for source localization." Proceedings of the ACM Web Conference 2022. 2022.
    """

    def __init__(self):
        """
        Initializes the IVGD model.
        """

    def train_diffusion(self, adj, train_dataset):
        """
        Train the diffusion model.

        Args:

        - adj (scipy.sparse.csr_matrix): Adjacency matrix of the graph.

        - train_dataset (torch.utils.data.dataset.Subset): the training dataset (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).

        Returns:

        - diffusion_model (torch.nn.Module): Trained diffusion model.

        Example:

        import os

        curr_dir = os.getcwd()

        from GraphSL.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.GNN.IVGD.main import IVGD

        data_name = 'karate'

        graph = load_dataset(data_name, data_dir=curr_dir)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        ivgd = IVGD()

        diffusion_model = ivgd.train_diffusion(adj, train_dataset)

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_node = adj.shape[0]
        adj_coo = adj.tocoo()
        values = adj_coo.data
        indices = np.vstack((adj_coo.row, adj_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj.shape

        prob_matrix = torch.sparse.FloatTensor(
            i, v, torch.Size(shape)).to_dense()
        prob_matrix = prob_matrix + torch.eye(n=prob_matrix.shape[0])
        prob_matrix = prob_matrix / prob_matrix.sum(dim=1, keepdims=True)
        ndim = 5
        niter = 2
        propagate_model = DiffusionPropagate(prob_matrix, niter=niter)
        fea_constructor = FeatureCons(ndim=ndim)
        fea_constructor.prob_matrix = prob_matrix
        args_dict = {
            'learning_rate': 1e-8,
            'λ': 0,
            'γ': 0,
            'idx_split_args': {
                'ntraining': int(
                    num_node / 2),
                'nstopping': int(
                    num_node / 6),
                'nval': int(
                    num_node / 3)},
            'test': False,
            'device': device,
            'print_interval': 10}
        gnn_model = MLPTransform(
            input_dim=ndim, hiddenunits=[
                ndim, ndim], num_classes=1, device=device)
        diffusion_model = i_DeepIS(
            gnn_model=gnn_model,
            propagate=propagate_model)
        diffusion_model, result = train_model(
            diffusion_model, fea_constructor, prob_matrix, train_dataset, **args_dict)
        print(f"train mean error:{result['train']['mean error']:.3f}")
        print(
            f"early_stopping mean error:{result['early_stopping']['mean error']:.3f}")
        print(f"validation mean error:{result['valtest']['mean error']:.3f}")
        print(f"run time:{result['runtime']:.3f} seconds")
        print(f"run time per epoch:{result['runtime_perepoch']:.3f} seconds")
        return diffusion_model

    def train(
            self,
            adj,
            train_dataset,
            diffusion_model,
            thres_list=[
                0.1,
                0.3,
                0.5,
                0.7,
                0.9],
            lr=1e-4,
            weight_decay=1e-4,
            num_epoch=10):
        """
        Train the IVGD model.

        Args:

        - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

        - train_dataset (torch.utils.data.dataset.Subset): the training dataset (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).

        - diffusion_model (torch.nn.Module): Trained diffusion model.

        - thres_list (list): List of threshold values.

        - lr (float): Learning rate.

        - weight_decay (float): Weight decay.

        - num_epoch (int): Number of epochs for training.

        Returns:

        - ivgd (torch.nn.Module): Trained IVGD model.

        - opt_thres (float): Optimal threshold value.

        - train_auc (float): Training AUC.

        - opt_f1 (float): Optimal F1 score.

        - pred (numpy.ndarray): Predicted seed vector of the training set, every column is the prediction of every simulation. It is used to adjust thres_list.

        Example:

        import os

        curr_dir = os.getcwd()

        from GraphSL.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.GNN.IVGD.main import IVGD

        data_name = 'karate'

        graph = load_dataset(data_name, data_dir=curr_dir)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        ivgd = IVGD()

        diffusion_model = ivgd.train_diffusion(adj, train_dataset)

        ivgd_model, thres, auc, f1, pred =ivgd.train(adj, train_dataset, diffusion_model)

        print("IVGD:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        # train_num = len(train_dataset)
        num_node = adj.shape[0]
        alpha = 1
        tau = 1
        rho = 1e-3
        lamda = 0
        ivgd = IVGD_model(alpha=alpha, tau=tau, rho=rho)
        optimizer = optim.Adam(
            ivgd.parameters(),
            lr=lr,
            weight_decay=weight_decay)
        ivgd.train()
        train_num = len(train_dataset)
        for i, influ_mat in enumerate(train_dataset):
            seed_vec = influ_mat[:, 0].to(device)
            influ_vec = influ_mat[:, -1].to(device)
            seed_preds = get_idx_new_seeds(diffusion_model, influ_vec)
            seed_preds = seed_preds.unsqueeze(-1).float()
            seed_vec = seed_vec.unsqueeze(-1).float()
            for epoch in range(num_epoch):
                optimizer.zero_grad()
                seed_correction = ivgd(seed_preds, seed_vec, lamda)
                loss = criterion(seed_correction, seed_vec.squeeze(-1).long())
                loss.backward(retain_graph=True)
                optimizer.step()
            print(f"optimize simulation {i+1}: loss = {loss:.3f}")

        ivgd.eval()
        train_auc = 0
        for influ_mat in train_dataset:
            seed_vec = influ_mat[:, 0]
            influ_vec = influ_mat[:, -1]
            seed_preds = get_idx_new_seeds(diffusion_model, influ_vec)
            seed_preds = torch.tensor(seed_preds).unsqueeze(-1).float()
            seed_vec = seed_vec.unsqueeze(-1).float()
            seed_correction = ivgd(seed_preds, seed_preds, lamda)
            seed_correction = F.softmax(seed_correction, dim=1)
            seed_correction = seed_correction[:, 1].unsqueeze(-1)
            seed_correction = self.normalize(seed_correction)
            seed_correction = seed_correction.cpu().squeeze(-1).detach().numpy()
            seed_vec = seed_vec.detach().numpy()
            train_auc += roc_auc_score(seed_vec, seed_correction)
        train_auc = train_auc / train_num

        pred = np.zeros((num_node, train_num))
        seed_all = np.zeros((num_node, train_num))
        for i, influ_mat in enumerate(train_dataset):
            seed_all[:, i] = influ_mat[:, 0]
            influ_vec = influ_mat[:, -1]
            seed_preds = get_idx_new_seeds(diffusion_model, influ_vec)
            seed_preds = torch.tensor(seed_preds).unsqueeze(-1).float()
            seed_correction = ivgd(seed_preds, seed_preds, lamda)
            seed_correction = F.softmax(seed_correction, dim=1)
            seed_correction = seed_correction[:, 1].unsqueeze(-1)
            seed_correction = self.normalize(seed_correction)
            seed_correction = seed_correction.squeeze(
                -1).cpu().detach().numpy()
            pred[:, i] = seed_correction

        opt_f1 = 0
        opt_thres = 0
        # Find optimal threshold and F1 score
        for thres in thres_list:
            train_f1 = 0
            for i in range(train_num):
                train_f1 += f1_score(seed_all[:, i], pred[:, i] >= thres)
            train_f1 = train_f1 / train_num
            print(f"thres = {thres:.3f}, train_f1 = {train_f1:.3f}")
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres

        return ivgd, opt_thres, train_auc, opt_f1, pred

    def test(self, test_dataset, diffusion_model, IVGD_model, thres):
        """
        Test the IVGD model on the given test dataset.

        Args:

        - test_dataset (torch.utils.data.dataset.Subset): the test dataset (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).

        - diffusion_model (torch.nn.Module): Trained diffusion model.

        - IVGD_model (torch.nn.Module): Trained IVGD model.

        - thres (float): Threshold value.

        Returns:

        - metric (Metric): Object containing test metrics.

        Example:

        import os

        curr_dir = os.getcwd()

        from GraphSL.utils import load_dataset, diffusion_generation, split_dataset

        from GraphSL.GNN.IVGD.main import IVGD

        data_name = 'karate'

        graph = load_dataset(data_name, data_dir=curr_dir)

        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

        adj, train_dataset, test_dataset =split_dataset(dataset)

        ivgd = IVGD()

        diffusion_model = ivgd.train_diffusion(adj, train_dataset)

        ivgd_model, thres, auc, f1, pred =ivgd.train(adj, train_dataset, diffusion_model)

        print("IVGD:")

        print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

        metric = ivgd.test(test_dataset, diffusion_model, ivgd_model, thres)

        print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")
        """
        test_num = len(test_dataset)
        test_acc = 0
        test_pr = 0
        test_re = 0
        test_f1 = 0
        test_auc = 0

        lamda = 0
        # Loop through each test dataset
        for influ_mat in test_dataset:
            seed_vec = influ_mat[:, 0]
            influ_vec = influ_mat[:, -1]

            # Get seed predictions from the diffusion model
            seed_preds = get_idx_new_seeds(diffusion_model, influ_vec)
            seed_preds = torch.tensor(seed_preds).unsqueeze(-1).float()
            seed_vec = seed_vec.unsqueeze(-1).float()

            # Obtain seed correction predictions from the IVGD model
            seed_correction = IVGD_model(seed_preds, seed_preds, lamda)
            seed_correction = F.softmax(seed_correction, dim=1)
            seed_correction = seed_correction[:, 1].unsqueeze(-1)
            seed_correction = self.normalize(seed_correction)
            seed_correction = seed_correction.squeeze(
                -1).cpu().detach().numpy()
            seed_vec = seed_vec.squeeze(-1).detach().cpu().numpy()

            # Compute metrics
            test_acc += accuracy_score(seed_vec, seed_correction >= thres)
            test_pr += precision_score(seed_vec,
                                       seed_correction >= thres,
                                       zero_division=1)
            test_re += recall_score(seed_vec,
                                    seed_correction >= thres,
                                    zero_division=1)
            test_f1 += f1_score(seed_vec, seed_correction >=
                                thres, zero_division=1)
            test_auc += roc_auc_score(seed_vec, seed_correction)

        # Compute average metrics
        test_acc = test_acc / test_num
        test_pr = test_pr / test_num
        test_re = test_re / test_num
        test_f1 = test_f1 / test_num
        test_auc = test_auc / test_num

        # Create Metric object containing test metrics
        metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
        return metric

    def normalize(self, x):
        """
        The input tensor x is normalized to between 0 and 1.

        Args:

        - x (torch.Tensor): Input tensor to be normalized.

        Returns:

        - x (torch.Tensor): Normalized tensor.
        """
        # Compute minimum and maximum values along each dimension
        min_vals, _ = torch.min(x, dim=0)
        max_vals, _ = torch.max(x, dim=0)

        # Compute the range and handle cases where the range is too small
        rang = max_vals - min_vals
        rang[torch.lt(rang, 1e-6)] = 1e-6

        # Perform normalization
        x = (x - min_vals) / rang
        return x
