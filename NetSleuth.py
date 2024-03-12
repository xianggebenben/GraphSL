import networkx as nx
import copy
import torch
from data.utils import load_dataset
from sklearn.metrics import roc_auc_score
from data.utils import data_generation
from Prescribed import NetSleuth

data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme8000','digg16000'
if data_name in ['meme8000', 'digg16000']:
    dataset = load_dataset(data_name)
else:
    dataset = data_generation(data_name=data_name)
adj = dataset['adj_mat'].todense()
train_ratio = 0.6
diff_mat = copy.deepcopy(dataset['diff_mat'])
all_num = len(diff_mat)
train_num = int(all_num*train_ratio)
test_num = all_num-train_num
train_diff_mat, test_diff_mat = torch.utils.data.random_split(diff_mat, [train_num, test_num])
G = nx.from_numpy_array(adj)
netsleuth = NetSleuth(G)
opt_auc = 0
opt_k = 0
for k in (5,10,50,100):
    train_auc = 0
    for influ_mat in train_diff_mat:
        seed_vec = influ_mat[:, 0]
        influ_vec = influ_mat[:, -1]
        x = netsleuth(k,influ_vec)
        train_auc += roc_auc_score(seed_vec, x)
    train_auc = train_auc / train_num
    if train_auc > opt_auc:
        opt_auc = train_auc
        opt_k = k
print('the best training auc:', opt_auc)

test_auc = 0
for influ_mat in test_diff_mat:
    seed_vec = influ_mat[:, 0]
    influ_vec = influ_mat[:, -1]
    x = netsleuth(opt_k,influ_vec)
    test_auc += roc_auc_score(seed_vec, x)
test_auc = test_auc / test_num
print('test auc:', test_auc)