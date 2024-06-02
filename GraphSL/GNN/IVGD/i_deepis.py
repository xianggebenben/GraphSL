import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp


class Identity(nn.Module):
    """
    Identity module to select specific elements from input tensor.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, preds, idx):
        """
        Forward pass to select specific elements from input tensor.

        Args:

        - preds (torch.Tensor): Input tensor.

        - idx (torch.Tensor): Indices of elements to select.

        Returns:

        - torch.Tensor: Selected elements from input tensor.
        """
        return preds[idx]


class DiffusionPropagate(nn.Module):
    """
    Module for diffusion propagation.
    """

    def __init__(self, prob_matrix, niter):
        """
        Initializes the DiffusionPropagate module.

        Args:

        - prob_matrix (torch.Tensor): Probability matrix for diffusion.

        - niter (int): Number of iterations for diffusion propagation.
        """
        super(DiffusionPropagate, self).__init__()
        self.niter = niter
        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()
        self.register_buffer('prob_matrix', torch.FloatTensor(prob_matrix))

    def forward(self, preds, seed_idx, idx):
        """
        Performs forward pass for diffusion propagation.

        Args:

        - preds (torch.Tensor): Input tensor of predictions.

        - seed_idx (torch.Tensor): Indices of seed nodes.

        - idx (torch.Tensor): Indices of nodes to propagate to.

        Returns:

        - torch.Tensor: Resulting propagated predictions.
        """
        temp = preds
        temp = temp.flatten()
        device = preds.device
        for i in range(self.niter):
            P2 = self.prob_matrix.T * \
                preds.view((1, -1)).expand(self.prob_matrix.shape)
            P3 = torch.ones(self.prob_matrix.shape).to(device) - P2
            preds = torch.ones((self.prob_matrix.shape[0],)).to(
                device) - torch.prod(P3, dim=1)
            # preds[seed_idx] = 1
        preds = (preds + temp) / 2
        return preds[idx]

    def backward(self, preds):
        """
        Perform backward pass for diffusion propagation.

        Args:

        - preds (torch.Tensor):  Prediction of diffusion.

        Returns:

        - res (torch.Tensor): Prediction of seeds.
        """
        device = preds.device
        res = preds
        temp = preds
        for j in range(10):
            for i in range(self.niter):
                P2 = self.prob_matrix.T * \
                    res.view((1, -1)).expand(self.prob_matrix.shape)
                P3 = torch.ones(self.prob_matrix.shape).to(device) - P2
                temp = torch.ones((self.prob_matrix.shape[0],)).to(
                    device) - torch.prod(P3, dim=1)
                # temp[preds == 1] = 1
            res = 2 * preds - temp
            res = torch.maximum(
                torch.minimum(
                    res,
                    torch.tensor(1)),
                torch.tensor(0))
        return res


class i_DeepIS(nn.Module):
    """
    Invertible Deep Influence Spread (i_DeepIS) module for graph source localization.
    """

    def __init__(self, gnn_model: nn.Module, propagate: nn.Module):
        """
        Initializes the i_DeepIS module.

        Args:

        - gnn_model (nn.Module): Graph neural network model.

        - propagate (nn.Module): Propagation module for diffusion.
        """
        super(i_DeepIS, self).__init__()
        self.gnn_model = gnn_model
        self.propagate = propagate

        self.reg_params = list(
            filter(
                lambda x: x.requires_grad,
                self.gnn_model.parameters()))

    def forward(self, idx: torch.LongTensor):
        """
        Forward pass for i_DeepIS module.

        Args:

        - idx (torch.LongTensor): Indices of nodes to fetch predictions for.

        Returns:

        - torch.Tensor: Predictions for selected nodes.
        """
        device = next(self.gnn_model.parameters()).device
        total_node_nums = self.gnn_model.features.weight.shape[0]
        total_nodes = torch.LongTensor(np.arange(total_node_nums)).to(device)
        seed = self.gnn_model.features.weight[:, 0]
        seed_idx = torch.LongTensor(np.argwhere(
            seed.detach().cpu().numpy() == 1)).to(device)
        seed = torch.unsqueeze(seed, 1)
        # predict all, for prediction propagation
        predictions = self.gnn_model(total_nodes)
        predictions = (predictions + seed) / 2

        predictions = self.propagate(predictions, seed_idx, idx)  # then select

        return predictions.flatten()

    def backward(self, prediction: torch.LongTensor):
        """
        Backward pass for i_DeepIS module.

        Args:

        - prediction (torch.LongTensor): Predictions.

        Returns:

        - torch.Tensor: Resulting propagated predictions after backward pass.
        """
        device = next(self.gnn_model.parameters()).device
        total_node_nums = self.gnn_model.features.weight.shape[0]
        total_nodes = torch.LongTensor(np.arange(total_node_nums)).to(device)
        res = self.propagate.backward(prediction)
        self.gnn_model.features.weight[:, 0] = res
        for i in range(10):
            temp = self.gnn_model(total_nodes).squeeze()
            res = 2 * prediction - temp
            self.gnn_model.features.weight[:, 0] = res.float()
        return res

    def loss(self, predictions, labels, λ, γ):
        """
        Computes the loss function for i_DeepIS module.

        Args:

        - predictions (torch.Tensor): Predicted values.

        - labels (torch.Tensor): Ground truth labels.

        - λ (float): Influence spread coefficient.

        - γ (float): Regularization coefficient.

        Returns:

        - Loss (torch.Tensor): Computed loss value.
        """
        L1 = torch.sum(torch.abs(predictions - labels)) / \
            len(labels)  # node-level error
        L2 = torch.abs(torch.sum(predictions) - torch.sum(labels)) / (
            torch.sum(labels) + 1e-5)  # influence spread error
        Reg = sum(torch.sum(param ** 2) for param in self.reg_params)
        Loss = L1 + λ * L2 + γ * Reg
        return Loss
