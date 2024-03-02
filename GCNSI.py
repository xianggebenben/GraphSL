import time
from scipy.sparse import csgraph
from data.utils import load_dataset
import torch.nn.functional as F
import copy
from sklearn.metrics import roc_auc_score
import torch
from scipy.sparse import coo_matrix
import numpy as np
from data.utils import data_generation
from handcraft import GCNSI, LPSI
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
gcnsi=GCNSI()
optimizer = torch.optim.SGD(gcnsi.parameters(), lr=1e-3,weight_decay=1e-4)
criterion= torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))
threshold =0.5
for epoch in range(10):
    print("epoch:" + str(epoch))
    optimizer.zero_grad()
    total_loss=0
    for i, influ_mat in enumerate(train_diff_mat):
        seed_vec = influ_mat[:, 0]
        diff_vec = influ_mat[:, -1]
        seed_vec = torch.tensor(seed_vec).squeeze(-1).long()
        pred=gcnsi(alpha,S,num_node,diff_vec,edge_index)
        loss = criterion(pred, seed_vec)
        total_loss+=loss
        loss.backward()
        optimizer.step()
    print("loss:{:0.6f}".format(total_loss/train_num))
train_auc = 0
test_auc = 0
for i, influ_mat in enumerate(diff_mat):
    print("i={:d}".format(i))
    seed_vec = influ_mat[:, 0]
    diff_vec = influ_mat[:, -1]
    pred = gcnsi(alpha,S,num_node,diff_vec,edge_index)
    pred = torch.softmax(pred,dim=1)
    pred = pred[:,1].squeeze(-1).detach().numpy()
    #print(pred)
    if i < train_num:
        train_auc += roc_auc_score(seed_vec, pred)
    else:
        test_auc += roc_auc_score(seed_vec, pred)

print('training acc:', train_auc / train_num)
print('test acc:', test_auc / (len(diff_mat)-train_num))
