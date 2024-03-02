import networkx as nx
import copy

from data.utils import load_dataset
from sklearn.metrics import roc_auc_score
from data.utils import data_generation
from handcraft import NetSleuth

data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme7000','digg16000'
if data_name in ['meme7000', 'digg16000']:
    dataset = load_dataset(data_name)
else:
    dataset = data_generation(data_name=data_name)
adj = dataset['adj_mat'].todense()
train_ratio = 0.6
diff_mat = copy.deepcopy(dataset['diff_mat'])
train_num = int(len(diff_mat)*train_ratio)
G = nx.from_numpy_array(adj)
train_auc = 0
test_auc = 0
netsleuth = NetSleuth(G)
for i, influ_mat in enumerate(diff_mat):
    print("simulation {:d}".format(i))
    seed_vec = influ_mat[:, 0]
    influ_vec = influ_mat[:, -1]
    x = netsleuth(10,influ_vec)
    if i < train_num:
        train_auc += roc_auc_score(seed_vec, x)

    else:
        test_auc += roc_auc_score(seed_vec, x)

print('training auc:', train_auc / train_num)
print('test auc:', test_auc / (len(diff_mat)-train_num))