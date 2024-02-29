
import torch.nn as nn
import numpy as np
class LPSI(nn.Module):
    def __init__(self, alpha,laplacian,num_node):
        super().__init__()
        self.alpha = alpha
        self.laplacian=laplacian
        self.num_node=num_node

    def forward(self, diff_vec):
        x = (1 - self.alpha) * np.matmul(np.linalg.inv(np.eye(N=self.num_node) - self.alpha * self.laplacian), diff_vec)
        return x