from scipy.sparse import csgraph
from data.utils import load_dataset
from scipy.sparse import coo_matrix
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from data.utils import data_generation
from handcraft import LPSI

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
train_num = int(len(diff_mat)*train_ratio)
train_auc = 0
test_auc = 0
alpha = 0.01
lpsi = LPSI(alpha, S, num_node)
for i, influ_mat in enumerate(diff_mat):
    print("simulation {:d}".format(i))
    seed_vec = influ_mat[:, 0]
    influ_vec = influ_mat[:, -1]
    x = lpsi(influ_vec)
    if i < train_num:
        train_auc += roc_auc_score(seed_vec, x)

    else:
        test_auc += roc_auc_score(seed_vec, x)

print('training auc:', train_auc / train_num)
print('test auc:', test_auc / (len(diff_mat)-train_num))
