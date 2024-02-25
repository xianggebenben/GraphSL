import networkx as nx
import numpy as np
import time

from data.utils import load_dataset, InverseProblemDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score

dataset = 'meme7000'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid','meme7000','digg16000'

graph = load_dataset(dataset)
adj = graph.adj_matrix.todense()

inverse_pairs = InverseProblemDataset(dataset)
num_training= int(len(graph.influ_mat_list)*0.8)
train_set =inverse_pairs.data[0:num_training,:]
test_set =inverse_pairs.data[num_training:,:]
G = nx.from_numpy_array(adj)


def NetSleuth(G, k, target_vector):
    G.remove_nodes_from([n for n in G if n not in np.where(target_vector == 1)[0]])
    lap = nx.laplacian_matrix(G).toarray()

    seed = []
    while len(seed) < k:
        value, vector = np.linalg.eig(lap)
        index = np.argmax(vector[np.argmin(value)])
        seed_index = list(G.nodes)[index]
        seed.append(seed_index)
        G.remove_node(seed_index)
        if len(G.nodes) ==0:
            break
        lap = nx.laplacian_matrix(G).toarray()
    return np.array(seed)


data = train_set
max_acc = 0
train_acc = 0
train_pr = 0
train_re = 0
train_fs = 0
pre=time.time()
data = train_set
for pair in data:
    G = nx.from_scipy_sparse_array(graph.adj_matrix)
    seed, target = pair[:, 0], pair[:, 1]
    # seed, target = data[0, :, 0], data[0, :, 1]
    k = len(seed[seed == 1])

    target[target > 0.1] = 1
    target[target != 1] = 0
    #print(k, len(target[target == 1]))
    pred_index = NetSleuth(G, k * 1.1, target)
    pred_seed = np.zeros(data.shape[1])
    pred_seed[pred_index] = 1
    train_acc += accuracy_score(seed, pred_seed)
    train_pr += precision_score(seed, pred_seed)
    train_re += recall_score(seed, pred_seed)
    train_fs += f1_score(seed, pred_seed)
after=time.time()
print("training_time:",after-pre)
test_acc = 0
test_pr = 0
test_re = 0
test_fs = 0

data = test_set
count = 0
for pair in data:
    count+=1
    print("count=",count)
    G = nx.from_scipy_sparse_array(graph.adj_matrix)
    seed, target = pair[:, 0], pair[:, 1]
    # seed, target = data[0, :, 0], data[0, :, 1]
    k = len(seed[seed == 1])

    target[target > 0.1] = 1
    target[target != 1] = 0
    #print(k, len(target[target == 1]))
    pred_index = NetSleuth(G, k * 1.1, target)
    pred_seed = np.zeros(data.shape[1])
    pred_seed[pred_index] = 1
    #print(accuracy_score(seed, pred_seed))
    test_acc += accuracy_score(seed, pred_seed)
    test_pr += precision_score(seed, pred_seed)
    test_re += recall_score(seed, pred_seed)
    test_fs += f1_score(seed, pred_seed)
    #print(roc_auc_score(seed,pred_seed))
    # if count==3:
    #     with open('NetSleuth_meme.npy', 'wb') as f:
    #         np.save(f, seed)
    #         np.save(f, pred_seed)
print("train:" + '{:.4f}'.format(train_acc / num_training) + "&" + '{:.4f}'.format(train_pr / num_training) + "&" + '{:.4f}'.format(
    train_re / num_training) + "&" + '{:.4f}'.format(train_fs / num_training))
print("test:" + '{:.4f}'.format(test_acc / len(test_set)) + "&" + '{:.4f}'.format(test_pr / len(test_set)) + "&" + '{:.4f}'.format(
    test_re / len(test_set)) + "&" + '{:.4f}'.format(test_fs / len(test_set)))

print('training acc:', train_acc / num_training)
print('training pr:', train_pr / num_training)
print('training re:', train_re / num_training)
print('training fs:', train_fs / num_training)
print('test acc:', test_acc / len(test_set))
print('test pr:', test_pr / len(test_set))
print('test re:', test_re / len(test_set))
print('test fs:', test_fs / len(test_set))
