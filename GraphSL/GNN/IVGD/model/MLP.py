import torch
import torch.nn as nn
from typing import List
from torch.nn.functional import normalize


class MLPTransform(nn.Module):
    def __init__(
            self,
            input_dim,
            hiddenunits: List[int],
            num_classes,
            bias=True,
            drop_prob=0.5,
            n_power_iterations=10,
            eps=1e-12,
            coeff=0.9,
            device='cuda'):
        super(MLPTransform, self).__init__()
        # Here features is just a placeholder, each time before forward, we will substutute the embedding layer with desired node feature matrix
        # and when saving model params, we will first pop self.features.weight
        self.features = None

        fcs = [nn.Linear(input_dim, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i]))
        fcs.append(nn.Linear(hiddenunits[-1], num_classes))

        self.fcs = nn.ModuleList(fcs)

        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(drop_prob)
        self.act_fn = nn.ReLU()
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.coeff = coeff
        self.device = device

    def forward(self, nodes: torch.LongTensor):
        # ipdb.set_trace()
        layer_inner = self.act_fn(
            self.fcs[0](
                self.dropout(
                    self.features(nodes))))
        for fc in self.fcs[1:-1]:
            weight = self.compute_weight(fc)
            fc.weight.data.copy_(weight)
            layer_inner = self.act_fn(fc(layer_inner))

        res = torch.sigmoid(self.fcs[-1](self.dropout(layer_inner)))
        return res

    def compute_weight(self, module):
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important bahaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is alreay on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = module.weight.clone()
        u = torch.rand(weight.shape[0], device=self.device)
        v = torch.rand(weight.shape[0], device=self.device)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                v = normalize(
                    torch.mv(
                        weight.t(),
                        u),
                    dim=0,
                    eps=self.eps,
                    out=v)
                u = normalize(torch.mv(weight, v), dim=0, eps=self.eps, out=u)
            if self.n_power_iterations > 0:
                # See above on why we need to clone
                u = u.clone()
                v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor

        return weight
