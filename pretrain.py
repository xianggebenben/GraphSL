
import logging
import numpy as np
from pathlib import Path
import copy
import torch
from learning.IVGD.i_deepis import i_DeepIS, DiffusionPropagate
from learning.IVGD.models.MLP import MLPTransform
from learning.IVGD.training import train_model, FeatureCons,get_predictions_new_seeds
from data.utils import load_dataset
logging.basicConfig(
    format='%(asctime)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
me_op = lambda x, y: np.mean(np.abs(x - y))
te_op = lambda x, y: np.abs(np.sum(x) - np.sum(y))
# key parameters
dataset = 'karate' # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid'
graph = load_dataset(dataset)
print(graph)
influ_mat_list = copy.copy(graph.influ_mat_list)
num_node=influ_mat_list.shape[1]
num_training= int(len(graph.influ_mat_list)*0.8)
graph.influ_mat_list = graph.influ_mat_list[:num_training]
print(graph.influ_mat_list.shape), print(influ_mat_list.shape)
# training parameters
ndim = 5
propagate_model = DiffusionPropagate(graph.prob_matrix, niter=2)
fea_constructor = FeatureCons(ndim=ndim)
fea_constructor.prob_matrix = graph.prob_matrix
device = 'cpu'  # 'cpu', 'cuda'
args_dict = {
    'learning_rate': 1e-4,
    'λ': 0,
    'γ': 0,
    'ckpt_dir': Path('.'),
    'idx_split_args': {'ntraining': int(num_node/3), 'nstopping': int(num_node/3), 'nval': int(num_node/3), 'seed': 2413340114},
    'test': False,
    'device': device,
    'print_interval': 10
}
gnn_model = MLPTransform(input_dim=ndim, hiddenunits=[ndim, ndim], num_classes=1,device=device)
model = i_DeepIS(gnn_model=gnn_model, propagate=propagate_model)
model, result = train_model(dataset, model, fea_constructor, graph, **args_dict)
influ_pred=get_predictions_new_seeds(model,fea_constructor,graph.influ_mat_list[0,:,0],np.arange(len(graph.influ_mat_list[0,:,0])))
print("diffusion mae:"+str(me_op(influ_pred,graph.influ_mat_list[0,:,1])))
torch.save(model,"i-deepis_"+dataset+".pt")