from scipy.sparse import csgraph
from data.utils import load_dataset
from sklearn.metrics import roc_auc_score
import torch
from scipy.sparse import coo_matrix
import numpy as np
from data.utils import data_generation
from handcraft import GCNSI
import copy
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid','meme7000','digg16000'
if data_name in ['meme7000', 'digg16000']:
    dataset = load_dataset(data_name)
else:
    dataset = data_generation(data_name=data_name)
adj = dataset['adj_mat']
S = csgraph.laplacian(adj, normed=False)
S = np.array(coo_matrix.todense(S))
num_node = adj.shape[0]
alpha = 0.4
train_ratio = 0.6
diff_mat = copy.deepcopy(dataset['diff_mat'])
all_num = len(diff_mat)
train_num = int(all_num*train_ratio)
test_num = all_num-train_num
train_diff_mat, test_diff_mat = torch.utils.data.random_split(diff_mat, [train_num, test_num])
coo = adj.tocoo()
row = torch.from_numpy(coo.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(coo.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)
gcnsi=GCNSI()
optimizer = torch.optim.SGD(gcnsi.parameters(), lr=1e-3,weight_decay=1e-4)
criterion= torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]))
threshold =0.5
for epoch in range(500):
    print("epoch:" + str(epoch))
    optimizer.zero_grad()
    total_loss=0
    for influ_mat in train_diff_mat:
        seed_vec = influ_mat[:, 0]
        diff_vec = influ_mat[:, -1]
        seed_vec = torch.tensor(seed_vec).squeeze(-1).long()
        pred=gcnsi(alpha,S,num_node,threshold,diff_vec,edge_index)
        loss = criterion(pred, seed_vec)
        total_loss+=loss
        loss.backward()
        optimizer.step()
    print("loss:{:0.6f}".format(total_loss/train_num))
test_auc = 0
for influ_mat in test_diff_mat:
    seed_vec = influ_mat[:, 0]
    diff_vec = influ_mat[:, -1]
    pred = gcnsi(alpha,S,num_node,threshold,diff_vec,edge_index)
    pred = torch.softmax(pred,dim=1)
    pred = pred[:,1].squeeze(-1).detach().numpy()
    test_auc += roc_auc_score(seed_vec, pred)
test_auc = test_auc / test_num
print('test auc:', test_auc)
