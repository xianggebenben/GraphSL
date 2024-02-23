import logging
import numpy as np
import copy
import torch.nn as nn
import matplotlib.pyplot as plt

from main.training import FeatureCons, get_idx_new_seeds,get_predictions_new_seeds
from main.utils import load_dataset
from main.alm_net import alm_net
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,roc_auc_score,mean_squared_error

logging.basicConfig(
    format='%(asctime)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
plt.style.use('seaborn')
import torch.optim as optim

# key parameters
dataset = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid'
model_name = 'deepis'  # 'deepis', ''
graph = load_dataset(dataset)
print(graph)
influ_mat_list = copy.copy(graph.influ_mat_list)
num_training= int(len(graph.influ_mat_list)*0.8)
graph.influ_mat_list = graph.influ_mat_list[:num_training]
print(graph.influ_mat_list.shape), print(influ_mat_list.shape)
# training parameters
ndim =5 #for simulated datasets
fea_constructor = FeatureCons(model_name, ndim=ndim)
fea_constructor.prob_matrix = graph.prob_matrix
device = 'cuda'  # 'cpu', 'cuda'
model = torch.load("i-deepis_"+dataset+".pt")
influ_pred=get_predictions_new_seeds(model,fea_constructor,graph.influ_mat_list[0,:,0],np.arange(len(graph.influ_mat_list[0,:,0])))
criterion = nn.CrossEntropyLoss()
alpha = 1
tau = 10
rho = 1e-3
lamda = 0
threshold=0.5
nu = torch.zeros(size=(graph.influ_mat_list.shape[1], 1))
net = alm_net(alpha=alpha, tau=tau, rho=rho)
optimizer = optim.SGD(net.parameters(), lr=1e-2)
net.train()
for i, influ_mat in enumerate(graph.influ_mat_list):
    print("i={:d}".format(i))
    seed_vec = influ_mat[:, 0]
    seed_idx = np.argwhere(seed_vec == 1)  # used by PIteration
    influ_vec = influ_mat[:, -1]
    fea_constructor.prob_matrix = graph.prob_matrix
    seed_preds = get_idx_new_seeds(model, influ_vec)
    seed_preds = torch.tensor(seed_preds).unsqueeze(-1).float()
    influ_vec = torch.tensor(influ_vec).unsqueeze(-1).float()
    seed_vec = torch.tensor(seed_vec).unsqueeze(-1).float()
    for epoch in range(10):
        print("epoch:" + str(epoch))
        optimizer.zero_grad()
        seed_correction = net(seed_preds, seed_vec, lamda)
        loss = criterion(seed_correction, seed_vec.squeeze(-1).long())
        print("loss:{:0.6f}".format(loss))
        loss.backward(retain_graph=True)
        optimizer.step()
net.eval()
graph = load_dataset(dataset)
influ_mat_list = copy.copy(graph.influ_mat_list)
print(graph)
train_acc = 0
test_acc = 0
train_pr = 0
test_pr = 0
train_re = 0
test_re = 0
train_fs = 0
test_fs = 0
train_auc = 0
test_auc= 0
for i, influ_mat in enumerate(influ_mat_list):
    print("i={:d}".format(i))
    seed_vec = influ_mat[:, 0]
    seed_idx = np.argwhere(seed_vec == 1)  # used by PIteration
    influ_vec = influ_mat[:, -1]
    positive = np.where(seed_vec == 1)
    fea_constructor.prob_matrix = graph.prob_matrix
    seed_preds = get_idx_new_seeds(model, influ_vec)
    seed_preds = torch.tensor(seed_preds).unsqueeze(-1).float()
    influ_vec = torch.tensor(influ_vec).unsqueeze(-1).float()
    seed_vec = torch.tensor(seed_vec).unsqueeze(-1).float()
    seed_correction =net(seed_preds, seed_preds,lamda)
    seed_correction =torch.softmax(seed_correction,dim=1)
    seed_preds = seed_preds.squeeze(-1).detach().numpy()
    seed_correction = seed_correction[:,1].squeeze(-1).detach().numpy()
    seed_vec = seed_vec.squeeze(-1).detach().numpy()
    if i < num_training:
        train_acc += accuracy_score(seed_vec, seed_correction>=threshold)
        train_pr += precision_score(seed_vec, seed_correction>=threshold,zero_division=1)
        train_re += recall_score(seed_vec, seed_correction>=threshold)
        train_fs += f1_score(seed_vec, seed_correction>=threshold)
        train_auc += roc_auc_score(seed_vec, seed_correction)
    else:
        #print(accuracy_score(seed_vec, seed_correction>=threshold))
        test_acc += accuracy_score(seed_vec, seed_correction>=threshold)
        test_pr += precision_score(seed_vec, seed_correction>=threshold,zero_division=1)
        test_re += recall_score(seed_vec, seed_correction>=threshold)
        test_fs += f1_score(seed_vec, seed_correction>=threshold)
        test_auc += roc_auc_score(seed_vec, seed_preds)

print('training acc:', train_acc / num_training)
print('training pr:', train_pr / num_training)
print('training re:', train_re / num_training)
print('training fs:', train_fs / num_training)
print('training auc:', train_auc / num_training)
print('test acc:', test_acc / (len(influ_mat_list) - num_training))
print('test pr:', test_pr / (len(influ_mat_list) - num_training))
print('test re:', test_re / (len(influ_mat_list) - num_training))
print('test fs:', test_fs / (len(influ_mat_list) - num_training))
print('test auc:', test_auc / (len(influ_mat_list) - num_training))
