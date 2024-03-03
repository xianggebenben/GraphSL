import logging
import numpy as np
from pathlib import Path
import copy
from learning.IVGD.i_deepis import i_DeepIS, DiffusionPropagate
from learning.IVGD.models.MLP import MLPTransform
from learning.IVGD.training import train_model
import torch.nn as nn
from learning.IVGD.training import FeatureCons, get_idx_new_seeds,get_predictions_new_seeds
from data.utils import load_dataset
from learning.IVGD.validity_net import validity_net
import torch
from sklearn.metrics import f1_score, accuracy_score,roc_auc_score
import torch.optim as optim
from data.utils import data_generation
logging.basicConfig(
    format='%(asctime)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
me_op = lambda x, y: np.mean(np.abs(x - y))
te_op = lambda x, y: np.abs(np.sum(x) - np.sum(y))
# key parameters
infect_prob =0.1
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid','meme7000','digg16000'
if data_name in ['meme7000', 'digg16000']:
    dataset = load_dataset(data_name)
else:
    dataset = data_generation(data_name=data_name,infect_prob=infect_prob)
diff_mat = dataset['diff_mat']
num_node = dataset['adj_mat'].shape[0]
prob_matrix =torch.ones(size=(num_node,num_node))*infect_prob
# training parameters
train_ratio = 0.6
diff_mat = copy.deepcopy(dataset['diff_mat'])
all_num = len(diff_mat)
train_num = int(all_num*train_ratio)
test_num = all_num-train_num
train_diff_mat, test_diff_mat = torch.utils.data.random_split(diff_mat, [train_num, test_num])
ndim = 5
propagate_model = DiffusionPropagate(prob_matrix, niter=2)
fea_constructor = FeatureCons(ndim=ndim)
fea_constructor.prob_matrix = prob_matrix
device = 'cpu'  # 'cpu', 'cuda'
args_dict = {
    'learning_rate': 1e-4,
    'λ': 0,
    'γ': 0,
    'ckpt_dir': Path('.'),
    'idx_split_args': {'ntraining': int(num_node/3), 'nstopping': int(num_node/3), 'nval': int(num_node/3)},
    'test': False,
    'device': device,
    'print_interval': 10
}
gnn_model = MLPTransform(input_dim=ndim, hiddenunits=[ndim, ndim], num_classes=1,device=device)
model = i_DeepIS(gnn_model=gnn_model, propagate=propagate_model)
model,_ = train_model(dataset, model, fea_constructor, prob_matrix,train_diff_mat, **args_dict)
influ_pred=get_predictions_new_seeds(model,fea_constructor,train_diff_mat[0,:,0],np.arange(len(train_diff_mat[0,:,0])))
print("diffusion mae:"+str(me_op(influ_pred,train_diff_mat[0,:,1])))

criterion = nn.CrossEntropyLoss()
alpha = 1
tau = 10
rho = 1e-3
lamda = 0
threshold=0.5
nu = torch.zeros(size=(train_diff_mat.shape[1], 1))
validity_net = validity_net(alpha=alpha, tau=tau, rho=rho)
optimizer = optim.SGD(validity_net.parameters(), lr=1e-2)
validity_net.train()
for i, influ_mat in enumerate(train_diff_mat):
    print("i={:d}".format(i))
    seed_vec = influ_mat[:, 0]
    seed_idx = np.argwhere(seed_vec == 1)  # used by PIteration
    influ_vec = influ_mat[:, -1]
    fea_constructor.prob_matrix = prob_matrix
    seed_preds = get_idx_new_seeds(model, influ_vec)
    seed_preds = torch.tensor(seed_preds).unsqueeze(-1).float()
    influ_vec = torch.tensor(influ_vec).unsqueeze(-1).float()
    seed_vec = torch.tensor(seed_vec).unsqueeze(-1).float()
    for epoch in range(10):
        print("epoch:" + str(epoch))
        optimizer.zero_grad()
        seed_correction = validity_net(seed_preds, seed_vec, lamda)
        loss = criterion(seed_correction, seed_vec.squeeze(-1).long())
        print("loss:{:0.6f}".format(loss))
        loss.backward(retain_graph=True)
        optimizer.step()
validity_net.eval()
test_auc= 0
for i, influ_mat in enumerate(test_diff_mat):
    print("i={:d}".format(i))
    seed_vec = influ_mat[:, 0]
    seed_idx = np.argwhere(seed_vec == 1)  # used by PIteration
    influ_vec = influ_mat[:, -1]
    positive = np.where(seed_vec == 1)
    seed_preds = get_idx_new_seeds(model, influ_vec)
    seed_preds = torch.tensor(seed_preds).unsqueeze(-1).float()
    influ_vec = torch.tensor(influ_vec).unsqueeze(-1).float()
    seed_vec = torch.tensor(seed_vec).unsqueeze(-1).float()
    seed_correction =validity_net(seed_preds, seed_preds,lamda)
    seed_correction =torch.softmax(seed_correction,dim=1)
    seed_preds = seed_preds.squeeze(-1).detach().numpy()
    seed_correction = seed_correction[:,1].squeeze(-1).detach().numpy()
    seed_vec = seed_vec.squeeze(-1).detach().numpy()
    test_auc += roc_auc_score(seed_vec, seed_preds)
test_auc = test_auc/test_num

print('test auc:', test_auc)
