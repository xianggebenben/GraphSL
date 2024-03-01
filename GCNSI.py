import time
from scipy.sparse import csgraph
from data.utils import load_dataset
import torch.nn.functional as F
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,roc_auc_score
import torch
from scipy.sparse import coo_matrix
import numpy as np
from data.utils import data_generation
from handcraft import GCNSI
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid','meme7000','digg16000'
if data_name in ['meme7000', 'digg16000']:
    dataset = load_dataset(data_name)
else:
    dataset = data_generation(data_name=data_name)
adj = dataset['adj_mat']
S = csgraph.laplacian(adj, normed=False)
S = np.array(coo_matrix.todense(S))
num_node = dataset['adj_mat'].shape[0]
alpha = 0.4
train_ratio = 0.6
diff_mat = dataset['diff_mat']
train_num= int(len(diff_mat)*train_ratio)
train_diff_mat = diff_mat[:train_num]
coo = adj.tocoo()
row = torch.from_numpy(coo.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(coo.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)
model=GCNSI()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,weight_decay=1e-4)
criterion= torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0])) #[2 3]
for epoch in range(10):
    print("epoch:" + str(epoch))
    optimizer.zero_grad()
    total_loss=0
    for i, influ_mat in enumerate(train_diff_mat):
        seed_vec = influ_mat[:, 0]
        influ_vec = influ_mat[:, -1]
        V3 = copy.deepcopy(influ_vec)
        V4 = copy.deepcopy(influ_vec)
        V3[influ_vec < 0.5] =  0.5
        V4[influ_vec >= 0.5] =  0.5
        d1 = influ_vec
        d1 = d1[:, np.newaxis]
        d2 = (1 - alpha) * np.matmul(np.linalg.inv(np.eye(N=num_node) - alpha * S), influ_vec)
        d2 = d2[:, np.newaxis]
        d3 = (1 - alpha) * np.matmul(np.linalg.inv(np.eye(N=num_node) - alpha * S), V3)
        d3 = d3[:, np.newaxis]
        d4 = (1 - alpha) * np.matmul(np.linalg.inv(np.eye(N=num_node) - alpha * S), V4)
        d4 = d4[:, np.newaxis]
        x = np.concatenate((d1, d2, d3, d4), axis=1)
        x = torch.tensor(x,dtype=torch.float)
        seed_vec = torch.tensor(seed_vec).squeeze(-1).long()
        pred=model(x,edge_index)
        loss = criterion(pred, seed_vec)
        total_loss+=loss
        loss.backward()
        optimizer.step()
    print("loss:{:0.6f}".format(total_loss/train_num))
train_acc = 0
test_acc = 0
train_pr = 0
test_pr = 0
train_re = 0
test_re = 0
train_fs = 0
test_fs = 0
for i, influ_mat in enumerate(influ_mat_list):
    print("i={:d}".format(i))
    seed_vec = influ_mat[:, 0]
    influ_vec = influ_mat[:, -1]
    V3 = influ_vec
    V4 = influ_vec
    V3[influ_vec < 0.5] =  0.5
    V4[influ_vec >= 0.5] =  0.5
    d1 = influ_vec
    d1 = d1[:, np.newaxis]
    d2 = (1 - alpha) * np.matmul(np.linalg.inv(np.eye(N=num_node) - alpha * S), influ_vec)
    d2 = d2[:, np.newaxis]
    d3 = (1 - alpha) * np.matmul(np.linalg.inv(np.eye(N=num_node) - alpha * S), V3)
    d3 = d3[:, np.newaxis]
    d4 = (1 - alpha) * np.matmul(np.linalg.inv(np.eye(N=num_node) - alpha * S), V4)
    d4 = d4[:, np.newaxis]
    x = np.concatenate((d1, d2, d3, d4), axis=1)
    x = torch.tensor(x,dtype=torch.float)
    pred = model(x, edge_index)
    pred = torch.softmax(pred,dim=1)
    pred = pred[:,1].squeeze(-1).detach().numpy()
    #print(pred)
    if i < train_num:
        train_acc += accuracy_score(seed_vec, pred >= 0.5)
        train_pr += precision_score(seed_vec, pred >= 0.5, zero_division=1)
        train_re += recall_score(seed_vec, pred >= 0.5)
        train_fs += f1_score(seed_vec, pred >= 0.5)
    else:
        test_acc += accuracy_score(seed_vec, pred >= 0.5)
        test_pr += precision_score(seed_vec, pred >= 0.5, zero_division=1)
        test_re += recall_score(seed_vec, pred >= 0.5)
        test_fs += f1_score(seed_vec, pred >= 0.5)
        print(roc_auc_score(seed_vec,pred))
print("train:" + '{:.4f}'.format(train_acc / train_num) + "&" + '{:.4f}'.format(train_pr / train_num) + "&" + '{:.4f}'.format(
    train_re / train_num) + "&" + '{:.4f}'.format(train_fs / train_num))
print("test:" + '{:.4f}'.format(test_acc / (len(influ_mat_list)-train_num)) + "&" + '{:.4f}'.format(test_pr / (len(influ_mat_list)-train_num)) + "&" + '{:.4f}'.format(
    test_re / (len(influ_mat_list)-train_num)) + "&" + '{:.4f}'.format(test_fs / (len(influ_mat_list)-train_num)))

print('training acc:', train_acc / train_num)
print('training fs:', train_fs / train_num)
print('test acc:', test_acc / (len(influ_mat_list)-train_num))
print('test fs:', test_fs / (len(influ_mat_list)-train_num))
