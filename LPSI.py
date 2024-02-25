from scipy.sparse import csgraph
from data.utils import load_dataset
from scipy.sparse import coo_matrix
import numpy as np
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,roc_auc_score
import time
dataset = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme7000','digg16000'
graph = load_dataset(dataset)
S = csgraph.laplacian(graph.adj_matrix, normed=False)
S = np.array(coo_matrix.todense(S))
num_node = graph.adj_matrix.shape[0]
max_fs = 0
opt_alpha = 0.01
num_training= int(len(graph.influ_mat_list)*0.8)
influ_mat_list = graph.influ_mat_list[:num_training]
pre=time.time()
for alpha in [0.01, 0.1, 0.3, 0.6, 0.7, 0.9]:
    train_fs = 0
    for i, influ_mat in enumerate(influ_mat_list):
        print("i={:d}".format(i))
        seed_vec = influ_mat[:, 0]
        influ_vec = influ_mat[:, -1]
        x = (1 - alpha) * np.matmul(np.linalg.inv(np.eye(N=num_node) - alpha * S), influ_vec)
        train_fs += f1_score(seed_vec, x >= 0.5)
    train_fs = train_fs / num_training
    if train_fs > max_fs:
        max_fs = train_fs
        opt_alpha = alpha
after=time.time()
print(max_fs)
print(opt_alpha)
print("training_time:",after-pre)
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
graph = load_dataset(dataset)
influ_mat_list = copy.copy(graph.influ_mat_list)
train_count = 0
test_count = 0
for i, influ_mat in enumerate(influ_mat_list):
    print("i={:d}".format(i))
    seed_vec = influ_mat[:, 0]
    influ_vec = influ_mat[:, -1]
    x = (1 - opt_alpha) * np.matmul(np.linalg.inv(np.eye(N=num_node) - opt_alpha * S), influ_vec)

    if i < num_training:
        train_acc += accuracy_score(seed_vec, x >= 0.5)
        train_pr += precision_score(seed_vec, x >= 0.5)
        train_re += recall_score(seed_vec, x >= 0.5)
        train_fs += f1_score(seed_vec, x >= 0.5)
    else:
        test_acc += accuracy_score(seed_vec, x >= 0.5)
        test_pr += precision_score(seed_vec, x >= 0.5)
        test_re += recall_score(seed_vec, x >= 0.5)
        test_fs += f1_score(seed_vec, x >= 0.5)
        # print(roc_auc_score(seed_vec,x))
        # if i==num_training+2:
        #     with open('LPSI_digg.npy', 'wb') as f:
        #         np.save(f, seed_vec)
        #         np.save(f, x)
print("train:" + '{:.4f}'.format(train_acc / num_training) + "&" + '{:.4f}'.format(train_pr / num_training) + "&" + '{:.4f}'.format(
    train_re / num_training) + "&" + '{:.4f}'.format(train_fs / num_training))
print("test:" + '{:.4f}'.format(test_acc / (len(influ_mat_list)-num_training)) + "&" + '{:.4f}'.format(test_pr / (len(influ_mat_list)-num_training)) + "&" + '{:.4f}'.format(
    test_re / (len(influ_mat_list)-num_training)) + "&" + '{:.4f}'.format(test_fs / (len(influ_mat_list)-num_training)))

print('training acc:', train_acc / num_training)
print('training pr:', train_pr / num_training)
print('training re:', train_re / num_training)
print('training fs:', train_fs / num_training)
print('test acc:', test_acc / (len(influ_mat_list)-num_training))
print('test pr:', test_pr / (len(influ_mat_list)-num_training))
print('test re:', test_re / (len(influ_mat_list)-num_training))
print('test fs:', test_fs / (len(influ_mat_list)-num_training))
