
import torch.nn as nn
import numpy as np
import networkx as nx
import copy
import torch
class LPSI(nn.Module):
    def __init__(self, alpha, laplacian, num_node):
        super().__init__()
        self.alpha = alpha
        self.laplacian = laplacian
        self.num_node = num_node

    def forward(self, diff_vec):
        x = (1 - self.alpha) * np.matmul(np.linalg.inv(np.eye(N=self.num_node) - self.alpha * self.laplacian), diff_vec)
        return x

class NetSleuth(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G
    def forward(self, k, diff_vec):
        G = copy.deepcopy(self.G)
        G.remove_nodes_from([n for n in G if n not in np.where(diff_vec == 1)[0]])
        lap = nx.laplacian_matrix(G).toarray()

        seed = []
        while len(seed) < k:
            value, vector = np.linalg.eig(lap)
            index = np.argmax(vector[np.argmin(value)])
            seed_index = list(G.nodes)[index]
            seed.append(seed_index)
            G.remove_node(seed_index)
            if len(G.nodes) == 0:
                break
            lap = nx.laplacian_matrix(G).toarray()
        seed_vec = torch.zeros(diff_vec.shape)
        seed_vec[seed] = 1
        return seed_vec