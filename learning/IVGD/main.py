import torch
import torch.nn.functional as F
from learning.IVGD.correction import correction


class ivgd(torch.nn.Module):
    def __init__(self, alpha, tau, rho):
        super(ivgd, self).__init__()
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
        temp = pred[:, 0].unsqueeze(-1)
        return (self.net1(temp) + self.net2(temp) + self.net3(temp) + self.net4(temp) + self.net5(
            temp)) / self.number_layer
