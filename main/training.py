
from typing import Type, Tuple
import time, numpy as np, ipdb, logging, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import scipy.sparse as sp
import copy, sys
sys.path.append('../../')
from .preprocessing import gen_seeds, normalize_attributes, gen_splits_
from .earlystopping import EarlyStopping, stopping_args

class FeatureCons:
    """Initial feature constructor for different models"""
    __module__ = __name__
    __qualname__ = 'FeatureCons'

    def __init__(self, name, ndim=None):
        if not name in ('deepis', 'gcn', 'graphsage', 'gat', 'sgc', 'monstor'):
            raise AssertionError
        else:
            if name == 'deepis':
                assert isinstance(ndim, int) and ndim > 0, AssertionError('Assign an initial feature iteration number for DeepIS')
            self.prob_matrix = None
            self.name = name
            self.ndim = ndim
            if self.name == 'deepis':
                self.ndim = ndim
            else:
                self.ndim = 1

    def __deepis_fea(self, seed_vec):
        seed_vec = seed_vec.reshape((-1, 1))
        import scipy.sparse as sp
        if sp.isspmatrix(self.prob_matrix):
            self.prob_matrix = self.prob_matrix.toarray()
        assert seed_vec.shape[0] == self.prob_matrix.shape[0], 'Seed vector is illegal'
        attr_mat = [seed_vec]
        for i in range(self.ndim-1):
            attr_mat.append(self.prob_matrix.T @ attr_mat[(-1)])

        attr_mat = np.concatenate(attr_mat, axis=(-1))
        return attr_mat

    @staticmethod
    def show_names():
        string = ' '.join(['deepis', 'gcn', 'graphsage', 'gat', 'sgc', 'monstor'])
        print(string)

    def __call__(self, seed_vec):
        if self.name == 'deepis':
            return self._FeatureCons__deepis_fea(seed_vec)
        if self.name == 'gcn':
            return self._FeatureCons__deepis_fea(seed_vec)
        if self.name == 'graphsage':
            return self._FeatureCons__deepis_fea(seed_vec)
        if self.name == 'gat':
            return self._FeatureCons__deepis_fea(seed_vec)
        if self.name == 'sgc':
            return self._FeatureCons__deepis_fea(seed_vec)
        return seed_vec.reshape((-1, 1))


def get_dataloaders(idx, labels_np, batch_size=None):
    labels = torch.FloatTensor(labels_np)
    if batch_size is None:
        batch_size = max((val.numel() for val in idx.values()))
    datasets = {phase:TensorDataset(torch.LongTensor(ind), labels[ind]) for phase, ind in idx.items()}
    dataloaders = {phase:DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) for phase, dataset in datasets.items()}
    return dataloaders


def construct_attr_mat(prob_matrix, seed_vec, order=5):
    lanczos_flag = False
    if lanczos_flag:
        return lanczos_algo(prob_matrix, seed_vec, order)
    seed_vec = seed_vec.reshape((-1, 1))
    assert seed_vec.shape[0] == prob_matrix.shape[0]
    attr_mat = [
     seed_vec]
    if sp.isspmatrix(prob_matrix):
        prob_matrix = prob_matrix.toarray()
    for i in range(1, order + 1):
        attr_mat.append(prob_matrix.T @ attr_mat[(-1)])

    attr_mat = np.concatenate(attr_mat, axis=(-1))
    return attr_mat


def lanczos_algo(prob_matrix, seed_vec, order=5, epsilon=0.001):
    S = prob_matrix.T
    seed_vec = seed_vec.flatten()
    beta = np.zeros((order + 1,))
    gamma = np.zeros((order + 1,))
    q = [np.zeros((len(seed_vec),)), seed_vec / np.linalg.norm(seed_vec, ord=2)]
    for j in range(1, order + 1):
        z = S @ q[j]
        gamma[j] = q[j].reshape((1, -1)) @ z
        z = z - gamma[j] * q[j] - beta[(j - 1)] * q[(j - 1)]
        beta[j] = np.linalg.norm(z, ord=2)
        if beta[j] < epsilon:
            break
        q.append(z / beta[j])

    q = q[1:]
    q = np.array(q).T
    return q


def update_embedding(model, feature_mat):
    """Use new feature matrix to update embedding layer.
    """
    assert getattr(model, 'gnn_model', None) is not None, 'Object model should have a submodule `gnn_model` '
    device = next(model.parameters()).device
    if model.gnn_model.features is None:
        new_embedding_layer = nn.Embedding(feature_mat.shape[0], feature_mat.shape[1])
        new_embedding_layer.weight = nn.Parameter((torch.FloatTensor(feature_mat)), requires_grad=False)
        model.gnn_model.features = new_embedding_layer
    else:
        assert feature_mat.shape[1] == model.gnn_model.features.weight.shape[1], 'New dim of new embedding weights is not consistent with the old dim'
        model.gnn_model.features.weight = nn.Parameter((torch.FloatTensor(feature_mat)), requires_grad=False)
        model.gnn_model.features.num_embeddings = feature_mat.shape[0]
        model.gnn_model.features.embedding_dim = feature_mat.shape[1]
    model = model.to(device)
    return model


def PIteration(prob_matrix, predictions, seed_idx, substitute=True, piter=10):
    """Final prediction iteration to fit the ideal equation system.
    """

    def one_iter(prob_matrix, predictions):
        P2 = np.multiply(prob_matrix.T, np.broadcast_to(predictions.reshape((1, -1)), prob_matrix.shape))
        P3 = np.ones(prob_matrix.shape) - P2
        one_iter_preds = np.ones((prob_matrix.shape[0],)) - np.prod(P3, axis=1).flatten()
        return one_iter_preds

    # predictions = predictions.flatten()
    assert prob_matrix.shape[0] == prob_matrix.shape[1]
    assert predictions.shape[0] == prob_matrix.shape[0]
    import scipy.sparse as sp
    if sp.isspmatrix(prob_matrix):
        prob_matrix = prob_matrix.toarray()

    final_preds = predictions
    for i in range(piter):
        final_preds = one_iter(prob_matrix, final_preds)
        if substitute:
            final_preds[seed_idx] = 1

    return final_preds


def train_model(model_name: str, model, fea_constructor, graph, learning_rate: float, λ, γ, ckpt_dir, idx_split_args: dict={'ntraining':200,
 'nstopping':400,  'nval':10}, stopping_args: dict=stopping_args, test: bool=False, device: str='cuda', torch_seed: int=None, print_interval: int=10, batch_size=None) -> Tuple[(nn.Module, dict)]:
    if torch_seed is None:
        torch_seed = gen_seeds()
    torch.manual_seed(seed=torch_seed)
    logging.log(22, f"PyTorch seed: {torch_seed}")
    γ = torch.tensor(γ, device=device)
    optimizer = torch.optim.Adam((model.parameters()), lr=learning_rate)
    early_stopping = EarlyStopping(model, **stopping_args)
    epoch_stats = {'train':{},  'stopping':{}}
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    last_time = start_time
    temp_attr_mat_dict = {}
    for epoch in range(early_stopping.max_epochs):
        idx_np = {}
        idx_np['train'], idx_np['stopping'], idx_np['valtest'] = gen_splits_(np.arange(graph.prob_matrix.shape[0]), train_size=(idx_split_args['ntraining']),
          stopping_size=(idx_split_args['nstopping']),
          val_size=(idx_split_args['nval']))
        idx_all = {key:torch.LongTensor(val) for key, val in idx_np.items()}
        for phase in epoch_stats.keys():
            epoch_stats[phase]['loss'] = []
            epoch_stats[phase]['error'] = []

        for i, influ_mat in enumerate(graph.influ_mat_list):
            try:
                attr_mat = temp_attr_mat_dict[i]
            except KeyError:
                seed_vec = influ_mat[:, 0]
                attr_mat = fea_constructor(seed_vec)
                temp_attr_mat_dict[i] = attr_mat

            model = update_embedding(model, attr_mat)
            model = model.to(device)
            influ_vec = influ_mat[:, -1]
            labels_all = influ_vec
            dataloaders = get_dataloaders(idx_all, labels_all, batch_size=batch_size)
            for phase in epoch_stats.keys():
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                for idx, labels in dataloaders[phase]:
                    idx = idx.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        preds = model(idx)
                        loss = model.loss(preds, labels, λ, γ)
                        error = np.mean(np.abs(preds.cpu().detach().numpy() - labels.cpu().detach().numpy()))
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        epoch_stats[phase]['loss'].append(loss.item())
                        epoch_stats[phase]['error'].append(error)

        for phase in epoch_stats.keys():
            epoch_stats[phase]['loss'] = np.mean(epoch_stats[phase]['loss'])
            epoch_stats[phase]['error'] = np.mean(epoch_stats[phase]['error'])

        if epoch % print_interval == 0:
            duration = time.time() - last_time
            last_time = time.time()
            logging.info(f"Epoch {epoch}: Train loss = {epoch_stats['train']['loss']:.4f}, Train error = {epoch_stats['train']['error']:.4f}, early stopping loss = {epoch_stats['stopping']['loss']:.4f}, early stopping error = {epoch_stats['stopping']['error']:.4f}, ({duration:.3f} sec)")
        if len(early_stopping.stop_vars) > 0:
            stop_vars = [epoch_stats['stopping'][key] for key in early_stopping.stop_vars]
            if early_stopping.check(stop_vars, epoch):
                break

    runtime = time.time() - start_time
    runtime_perepoch = runtime / (epoch + 1)
    logging.log(22, f"Last epoch: {epoch}, best epoch: {early_stopping.best_epoch} ({runtime:.3f} sec)")
    model.load_state_dict(early_stopping.best_state)
    train_preds = get_predictions(model, idx_all['train'])
    train_error = np.abs(train_preds - labels_all[idx_all['train']]).mean()
    stopping_preds = get_predictions(model, idx_all['stopping'])
    stopping_error = np.abs(stopping_preds - labels_all[idx_all['stopping']]).mean()
    logging.log(21, f"Early stopping error: {stopping_error}")
    valtest_preds = get_predictions(model, idx_all['valtest'])
    valtest_error = np.abs(valtest_preds - labels_all[idx_all['valtest']]).mean()
    valtest_name = 'Test' if test else 'Validation'
    logging.log(22, f"{valtest_name} mean error: {valtest_error}")
    result = {}
    result['predictions'] = get_predictions(model, torch.arange(len(labels_all)))
    result['train'] = {'mean error': train_error}
    result['early_stopping'] = {'mean error': stopping_error}
    result['valtest'] = {'mean error': valtest_error}
    result['runtime'] = runtime
    result['runtime_perepoch'] = runtime_perepoch
    if model_name == 'deepis':
        model.gnn_model.features = None
    ckpt_name = '{}_{}_{}'.format(model_name, start_time_str, early_stopping.best_epoch)
    torch.save(model.state_dict(), ckpt_dir / ckpt_name)
    return (
     model, result)


def get_predictions(model, idx, batch_size=None):
    if batch_size is None:
        batch_size = idx.numel()
    dataset = TensorDataset(idx)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    preds = []
    for idx, in dataloader:
        with torch.set_grad_enabled(False):
            pred = model(idx)
            preds.append(pred)

    return torch.cat(preds, dim=0).cpu().numpy()


class GetPrediction:
    __module__ = __name__
    __qualname__ = 'GetPrediction'

    def __init__(self, model, fea_constructor):
        self.model = model
        self.fea_constructor = fea_constructor

    def __call__(self, seed_vec, prob_matrix, piter=2):
        assert len(seed_vec) == prob_matrix.shape[0], 'Illegal seed vector or prob_matrix'
        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()
        
        self.fea_constructor.prob_matrix = prob_matrix
        
        idx = np.arange(prob_matrix.shape[0])
        attr_mat = self.fea_constructor(seed_vec)
        self.model = update_embedding(self.model, attr_mat)
        preds = self.model(idx).detach().cpu().numpy()
        
        seed_idx = np.argwhere( seed_vec == 1 )
        preds = PIteration(prob_matrix, preds, seed_idx=seed_idx, piter=iter)
        return preds


def get_predictions_new_seeds(model, fea_constructor, seed_vec, idx):
    """Given a new seed set on the same graph, predict each node's probability
    Actually you can also put into a new graph's prob_matrix, then the model could predict results on new graphs.
    Parameters
    ----------
    model
        model for prediction
    prob_matrix
        the graph influence probability matrix
    seed_vec
        a vector indicates seeds
    idx
        nodes' influence probability that need to be predicted
    fea_iter_order
        probability matrix iteration order to construct nodes' feature matrix
    fea_propagate
        not used actually
    
    Return
    ------
        a vector indicates influence predictions of idx nodes
    """
    device = next(model.parameters()).device
    idx = torch.LongTensor(idx).to(device)
    # model = model.to('cpu')

    attr_mat = fea_constructor(seed_vec)
    model = update_embedding(model, attr_mat)
    # model = model.to(device)

    preds = model(idx)
    preds = preds.detach().cpu().numpy()
    return preds


def get_idx_new_seeds(model, prediction):
    """ given each node's probability,predict the seed set on the same graph
    Actually you can also put into a new graph's prob_matrix, then the model could predict results on new graphs.
    Parameters
    ----------
    model
        model for prediction
    prob_matrix
        the graph influence probability matrix
    seed_vec
        a vector indicates seeds
    idx
        nodes' influence probability that need to be predicted
    fea_iter_order
        probability matrix iteration order to construct nodes' feature matrix
    fea_propagate
        not used actually

    Return
    ------
        a vector indicates influence predictions of idx nodes
    """
    device = next(model.parameters()).device
    prediction = torch.tensor(prediction).to(device)
    result = model.backward(prediction)
    result = result.detach().cpu().numpy()
    return result