from scipy.sparse import csgraph
from data.utils import load_dataset
from scipy.sparse import coo_matrix
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from data.utils import data_generation
from Prescribed import LPSI
import torch
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme7000','digg16000'
if data_name in ['meme7000', 'digg16000']:
    dataset = load_dataset(data_name)
else:
    dataset = data_generation(data_name=data_name)
S = csgraph.laplacian(dataset["adj_mat"], normed=False)
S = np.array(coo_matrix.todense(S))
num_node = dataset["adj_mat"].shape[0]
train_ratio = 0.6
diff_mat = copy.deepcopy(dataset['diff_mat'])
all_num = len(diff_mat)
train_num = int(all_num*train_ratio)
test_num = all_num-train_num
train_diff_mat, test_diff_mat = torch.utils.data.random_split(diff_mat, [train_num, test_num])
lpsi = LPSI(S, num_node)
opt_auc = 0
opt_alpha = 0
for alpha in (0.01,0.1,1):
    train_auc = 0
    for influ_mat in train_diff_mat:
        seed_vec = influ_mat[:, 0]
        influ_vec = influ_mat[:, -1]
        x = lpsi(alpha,influ_vec)
        train_auc += roc_auc_score(seed_vec, x)
    train_auc=train_auc/train_num
    if train_auc > opt_auc:
        opt_auc = train_auc
        opt_alpha = alpha
print('the best training auc:', opt_auc)

test_auc = 0
for influ_mat in test_diff_mat:
    seed_vec = influ_mat[:, 0]
    influ_vec = influ_mat[:, -1]
    x = lpsi(opt_alpha, influ_vec)
    test_auc += roc_auc_score(seed_vec, x)
test_auc = test_auc / test_num
print('test auc:', test_auc)
