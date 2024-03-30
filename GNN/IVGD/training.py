from typing import  Tuple
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import scipy.sparse as sp
import sys
sys.path.append('../../')
from .preprocessing import gen_seeds, gen_splits_
from .earlystopping import EarlyStopping, stopping_args

class FeatureCons:
    """
    Initial feature constructor for different models.
    """

    def __init__(self, ndim=None):
        """
        Initialize FeatureCons object.

        Args:
        - ndim (int): Number of dimensions.
        """
        self.prob_matrix = None
        self.ndim = ndim

    def __deepis_fea(self, seed_vec):
        """
        Deeply compute features based on the given seed vector.

        Args:
        - seed_vec (torch.Tensor): Seed vector.

        Returns:
        - numpy.ndarray: Feature matrix.
        """
        seed_vec = seed_vec.reshape((-1, 1))
        import scipy.sparse as sp
        if sp.isspmatrix(self.prob_matrix):
            self.prob_matrix = self.prob_matrix.toarray()
        assert seed_vec.shape[0] == self.prob_matrix.shape[0], 'Seed vector is illegal'
        attr_mat = [seed_vec]
        for i in range(self.ndim - 1):
            attr_mat.append(self.prob_matrix.T @ attr_mat[(-1)])

        attr_mat = np.concatenate(attr_mat, axis=(-1))
        return attr_mat

    def __call__(self, seed_vec):
        """
        Call method to compute features based on the given seed vector.

        Args:
        - seed_vec (torch.Tensor): Seed vector.

        Returns:
        - numpy.ndarray: Feature matrix.
        """
        return self.__deepis_fea(seed_vec)


def get_dataloaders(idx, labels_np, batch_size=None):
    """
    Get data loaders for training, validation, and testing.

    Args:
    - idx (dict): Dictionary containing indices for different phases, training, early stopping, and validation/test.

    - labels_np (numpy.ndarray): Labels as numpy array.

    - batch_size (int): Batch size.

    Returns:
    - dataloaders (dict): Dictionary containing data loaders for different phases.
    """
    labels = torch.FloatTensor(labels_np)
    if batch_size is None:
        batch_size = max((val.numel() for val in idx.values()))
    datasets = {phase: TensorDataset(torch.LongTensor(ind), labels[ind]) for phase, ind in idx.items()}
    dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) for phase, dataset in
                   datasets.items()}
    return dataloaders


def construct_attr_mat(prob_matrix, seed_vec, order=5):
    """
    Construct attribute matrix based on the given probability matrix and seed vector.

    Args:
    - prob_matrix (numpy.ndarray or scipy.sparse.csr_matrix): Probability matrix.
    - seed_vec (numpy.ndarray): Seed vector.
    - order (int): Order of attribute matrix construction.

    Returns:
    - numpy.ndarray: Attribute matrix.
    """
    lanczos_flag = False
    if lanczos_flag:
        return lanczos_algo(prob_matrix, seed_vec, order)
    seed_vec = seed_vec.reshape((-1, 1))
    assert seed_vec.shape[0] == prob_matrix.shape[0]
    attr_mat = [seed_vec]
    if sp.isspmatrix(prob_matrix):
        prob_matrix = prob_matrix.toarray()
    for i in range(1, order + 1):
        attr_mat.append(prob_matrix.T @ attr_mat[(-1)])

    attr_mat = np.concatenate(attr_mat, axis=(-1))
    return attr_mat


def lanczos_algo(prob_matrix, seed_vec, order=5, epsilon=0.001):
    """
    Lanczos algorithm for computing attribute matrix.

    Args:
    - prob_matrix (numpy.ndarray or scipy.sparse.csr_matrix): Probability matrix.
    - seed_vec (numpy.ndarray): Seed vector.
    - order (int): Order of attribute matrix construction.
    - epsilon (float): Threshold for stopping the iteration.

    Returns:
    - numpy.ndarray: Attribute matrix.
    """
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
    """
    Update the embedding layer of the model with the new feature matrix.

    Args:
    - model (torch.nn.Module): Model.
    - feature_mat (numpy.ndarray): Feature matrix.

    Returns:
    - torch.nn.Module: Updated model.
    """
    assert getattr(model, 'gnn_model', None) is not None, 'Object model should have a submodule `gnn_model` '
    device = next(model.parameters()).device
    if model.gnn_model.features is None:
        new_embedding_layer = nn.Embedding(feature_mat.shape[0], feature_mat.shape[1])
        new_embedding_layer.weight = nn.Parameter(torch.FloatTensor(feature_mat), requires_grad=False)
        model.gnn_model.features = new_embedding_layer
    else:
        assert feature_mat.shape[1] == model.gnn_model.features.weight.shape[1], \
            'New dimension of new embedding weights is not consistent with the old dimension'
        model.gnn_model.features.weight = nn.Parameter(torch.FloatTensor(feature_mat), requires_grad=False)
        model.gnn_model.features.num_embeddings = feature_mat.shape[0]
        model.gnn_model.features.embedding_dim = feature_mat.shape[1]
    model = model.to(device)
    return model


def PIteration(prob_matrix, predictions, seed_idx, substitute=True, piter=10):
    """
    Perform final prediction iteration to fit the ideal equation system.

    Args:
    - prob_matrix (numpy.ndarray): Probability matrix.

    - predictions (numpy.ndarray): Predictions.

    - seed_idx (numpy.ndarray): Seed indices.

    - substitute (bool): Whether to substitute seed indices.

    - piter (int): Number of iterations.

    Returns:
    - numpy.ndarray: Final predictions.
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


def train_model(model, fea_constructor, prob_matrix, diff_mat, learning_rate: float, λ, γ,
                idx_split_args: dict = {'ntraining': 200, 'nstopping': 400, 'nval': 10},
                stopping_args: dict = stopping_args, test: bool = False, device: str = 'cuda', torch_seed: int = None,
                print_interval: int = 10, batch_size=None) -> Tuple[(nn.Module, dict)]:
    """
    Train the model using the specified parameters.

    Args:
    - model (nn.Module): Model to be trained.

    - fea_constructor (nn.Module): Feature constructor object.

    - prob_matrix (torch.Tensor): Probability matrix.

    - diff_mat (torch.utils.data.dataset.Subset): The diffusion matrix (number of simulations * number of graph nodes * 2 (the first column is seed vector and the second column is diffusion vector)).

    - learning_rate (float): Learning rate for optimizer.

    - λ: Lambda value.

    - γ: Gamma value.

    - idx_split_args (dict): Split of the dataset, ntraining is the number of training set, nstopping is the number of samples to determine the early stopping, and nval is the number of validation set.

    - stopping_args (dict): Stopping arguments.

    - test (bool): Whether to perform testing.

    - device (str): Device for training, cpu or cuda.

    - torch_seed (int): Seed for torch.

    - print_interval (int): Print interval.

    - batch_size (int): Batch size.

    Returns:
    - model (nn.Module): Trained model.

    - result (dict): Results of diffusion model,including predictions, train mean error, early_stopping mean error, val/test mean error, runtime, and runtime per epoch.
    """
    if torch_seed is None:
        torch_seed = gen_seeds()
    torch.manual_seed(seed=torch_seed)
    #logging.log(22, f"PyTorch seed: {torch_seed}")
    γ = torch.tensor(γ, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(model, **stopping_args)
    epoch_stats = {'train': {}, 'stopping': {}}
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    last_time = start_time
    temp_attr_mat_dict = {}

    # Loop through epochs
    for epoch in range(early_stopping.max_epochs):
        idx_np = {}
        idx_np['train'], idx_np['stopping'], idx_np['valtest'] = gen_splits_(np.arange(prob_matrix.shape[0]),
                                                                             train_size=idx_split_args['ntraining'],
                                                                             stopping_size=idx_split_args['nstopping'],
                                                                             val_size=idx_split_args['nval'])
        idx_all = {key: torch.LongTensor(val) for key, val in idx_np.items()}

        # Initialize epoch statistics
        for phase in epoch_stats.keys():
            epoch_stats[phase]['loss'] = []
            epoch_stats[phase]['error'] = []

        # Loop through different matrices
        for i, influ_mat in enumerate(diff_mat):
            try:
                attr_mat = temp_attr_mat_dict[i]
            except KeyError:
                seed_vec = influ_mat[:, 0]
                attr_mat = fea_constructor(seed_vec)
                temp_attr_mat_dict[i] = attr_mat

            model = update_embedding(model, attr_mat)
            model = model.to(device)
            influ_vec = influ_mat[:, -1]
            labels_all = influ_vec.numpy()
            dataloaders = get_dataloaders(idx_all, labels_all, batch_size=batch_size)

            # Iterate through training and validation/test phases
            for phase in epoch_stats.keys():
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                # Iterate through batches
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

        # Calculate epoch statistics
        for phase in epoch_stats.keys():
            epoch_stats[phase]['loss'] = np.mean(epoch_stats[phase]['loss'])
            epoch_stats[phase]['error'] = np.mean(epoch_stats[phase]['error'])

        # Log information
        if epoch % print_interval == 0:
            duration = time.time() - last_time
            last_time = time.time()
            #logging.info(
            print(f"Epoch {epoch}: Train loss = {epoch_stats['train']['loss']:.4f}, Train error = {epoch_stats['train']['error']:.4f}, early stopping loss = {epoch_stats['stopping']['loss']:.4f}, early stopping error = {epoch_stats['stopping']['error']:.4f}, ({duration:.3f} sec)")

        # Check early stopping
        if len(early_stopping.stop_vars) > 0:
            stop_vars = [epoch_stats['stopping'][key] for key in early_stopping.stop_vars]
            if early_stopping.check(stop_vars, epoch):
                break

    # Calculate runtime statistics
    runtime = time.time() - start_time
    runtime_perepoch = runtime / (epoch + 1)
    #logging.log(22, f"Last epoch: {epoch}, best epoch: {early_stopping.best_epoch} ({runtime:.3f} sec)")

    # Load best model state
    model.load_state_dict(early_stopping.best_state)

    # Calculate errors
    train_preds = get_predictions(model, idx_all['train'])
    train_error = np.abs(train_preds - labels_all[idx_all['train']]).mean()
    stopping_preds = get_predictions(model, idx_all['stopping'])
    stopping_error = np.abs(stopping_preds - labels_all[idx_all['stopping']]).mean()
    #logging.log(21, f"Early stopping error: {stopping_error}")
    valtest_preds = get_predictions(model, idx_all['valtest'])
    valtest_error = np.abs(valtest_preds - labels_all[idx_all['valtest']]).mean()
    valtest_name = 'Test' if test else 'Validation'
    #logging.log(22, f"{valtest_name} mean error: {valtest_error}")

    # Prepare results
    result = {}
    result['predictions'] = get_predictions(model, torch.arange(len(labels_all)))
    result['train'] = {'mean error': train_error}
    result['early_stopping'] = {'mean error': stopping_error}
    result['valtest'] = {'mean error': valtest_error}
    result['runtime'] = runtime
    result['runtime_perepoch'] = runtime_perepoch

    return model, result


def get_predictions(model, idx, batch_size=None):
    """
    Get predictions from the model.

    Args:
    - model (nn.Module): Model.
    - idx (Tensor): Indices.
    - batch_size (int): Batch size.

    Returns:
    - numpy.ndarray: Predictions.
    """
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
    """
    Callable class for getting predictions.
    """

    def __init__(self, model, fea_constructor):
        """
        Initialize GetPrediction object.

        Args:
        - model (nn.Module): Model for prediction.
        - fea_constructor: Feature constructor object.
        """
        self.model = model
        self.fea_constructor = fea_constructor

    def __call__(self, seed_vec, prob_matrix, piter=2):
        """
        Call method to get predictions.

        Args:
        - seed_vec (array-like): Seed vector.
        - prob_matrix (numpy.ndarray): Probability matrix.
        - piter (int): Iteration number.

        Returns:
        - numpy.ndarray: Predictions.
        """
        assert len(seed_vec) == prob_matrix.shape[0], 'Illegal seed vector or prob_matrix'
        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()

        self.fea_constructor.prob_matrix = prob_matrix

        idx = np.arange(prob_matrix.shape[0])
        attr_mat = self.fea_constructor(seed_vec)
        self.model = update_embedding(self.model, attr_mat)
        preds = self.model(idx).detach().cpu().numpy()

        seed_idx = np.argwhere(seed_vec == 1)
        preds = PIteration(prob_matrix, preds, seed_idx=seed_idx, piter=iter)
        return preds


def get_predictions_new_seeds(model, fea_constructor, seed_vec, idx):
    """
    Get predictions for new seeds.

    Given a new seed set on the same graph, predict each node's probability.

    Args:
    - model: Model for prediction.
    - fea_constructor: Feature constructor.
    - seed_vec: Seed vector.
    - idx: Indices.

    Returns:
    - numpy.ndarray: Predictions.
    """
    device = next(model.parameters()).device
    idx = torch.LongTensor(idx).to(device)

    attr_mat = fea_constructor(seed_vec)
    model = update_embedding(model, attr_mat)

    preds = model(idx)
    preds = preds.detach().cpu().numpy()
    return preds


def get_idx_new_seeds(model, prediction):
    """
    Get indices for new seeds.

    Given each node's probability, predict the seed set on the same graph.

    Args:
    - model: Model for prediction.
    - prediction: Predictions.

    Returns:
    - Tensor: Result.
    """
    device = next(model.parameters()).device
    prediction = torch.tensor(prediction).to(device)
    result = model.backward(prediction)
    return result
