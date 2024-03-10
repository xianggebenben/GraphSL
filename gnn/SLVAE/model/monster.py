import torch
from dgl import function as fn
from torch import nn
from torch.nn import functional as F

class Conv(nn.Module):
    r"""We modified existing implementation of GraphSAGE from DGL
    (https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/sageconv.py)
    """
    def __init__(self, in_feats, out_feats, norm=None, activation=None):
        super(Conv, self).__init__()
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.norm = norm
        self.activation = activation
        
        self.fc_pool = nn.Linear(in_feats, in_feats, bias=True)
        self.fc_neigh = nn.Linear(in_feats + in_feats, out_feats, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat):
        graph = graph.local_var()
        graph.ndata['h'] = self.fc_pool(feat)
        graph.edata['_w'] = graph.edata['weight'].unsqueeze(1)
        graph.update_all(message_func = fn.u_mul_e('h', '_w', 'm'),
                         reduce_func = fn.max('m', 'neigh'))
        
        h_neigh = graph.ndata['neigh']
        degs = graph.in_degrees()
        h_neigh[degs == 0, :] = 0
            
        rst = self.fc_neigh(torch.cat((feat, h_neigh), dim=1))
        if self.activation is not None:
            rst = self.activation(rst)
        if self.norm is not None:
            rst = self.norm(rst)
        return rst

class MONSTOR(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers):
        super(MONSTOR, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        dims = [in_feats, *[n_hidden for _ in range(n_layers - 1)], 1]
        
        for i in range(n_layers):
            self.layers.append(Conv(dims[i], dims[i+1]))
            self.acts.append(nn.ReLU())
        
    def forward(self, g, features):
        graph = g.local_var()
        h = features.clone()
        for act, layer in zip(self.acts, self.layers):
            h = act(layer(graph, h))
        
        # compute upper bound of influence
        prv_diff, now = features[:, -2], features[:, -1]
        graph.ndata['prv'] = prv_diff
        graph.update_all(fn.u_mul_e('prv', 'weight', 'm'), fn.sum('m', 'delta_ub'))
        lb = now
        ub = torch.clamp((lb + graph.ndata['delta_ub'].squeeze()), min=0, max=1)
        
        return torch.min(lb + h.squeeze(), ub)