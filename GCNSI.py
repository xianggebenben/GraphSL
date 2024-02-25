import time
from scipy.sparse import csgraph
from data.utils import load_dataset
import torch.nn.functional as F
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,roc_auc_score
import torch
from scipy.sparse import coo_matrix
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(4, 128)
        self.conv2 = GCNConv(128, 128)
        self.fc =torch.nn.Linear(128,2)

    def forward(self, x,edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.fc(x)

        return x
dataset = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid','meme7000','digg16000'
graph = load_dataset(dataset)
S = csgraph.laplacian(graph.adj_matrix, normed=False)
S = np.array(coo_matrix.todense(S))
num_node = graph.adj_matrix.shape[0]
alpha = 0.4
num_training= int(len(graph.influ_mat_list)*0.8)
influ_mat_list = graph.influ_mat_list[:num_training]
coo = graph.adj_matrix.tocoo()
row = torch.from_numpy(coo.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(coo.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)
model=Net()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,weight_decay=1e-4)
criterion= torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0])) #[2 3]
# the results may be random
# karate weight=torch.tensor([1.0,2.0])
# dolphins weight=torch.tensor([1.0,3.0])
# jazz weight=torch.tensor([1.0,2.0])
# netscience weight=torch.tensor([1.0,1.2])
# cora_ml weight=torch.tensor([1.0,9.0])
# power grid weight=torch.tensor([1.0, 2.5])
# meme7000
pre=time.time()
for epoch in range(10):
    print("epoch:" + str(epoch))
    optimizer.zero_grad()
    total_loss=0
    for i, influ_mat in enumerate(influ_mat_list):
        seed_vec = influ_mat[:, 0]
        influ_vec = influ_mat[:, -1]
        V3 = copy.copy(influ_vec)
        V4 = copy.copy(influ_vec)
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
    print("loss:{:0.6f}".format(total_loss/num_training))
after=time.time()
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
    if i < num_training:
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
print("train:" + '{:.4f}'.format(train_acc / num_training) + "&" + '{:.4f}'.format(train_pr / num_training) + "&" + '{:.4f}'.format(
    train_re / num_training) + "&" + '{:.4f}'.format(train_fs / num_training))
print("test:" + '{:.4f}'.format(test_acc / (len(influ_mat_list)-num_training)) + "&" + '{:.4f}'.format(test_pr / (len(influ_mat_list)-num_training)) + "&" + '{:.4f}'.format(
    test_re / (len(influ_mat_list)-num_training)) + "&" + '{:.4f}'.format(test_fs / (len(influ_mat_list)-num_training)))

print('training acc:', train_acc / num_training)
print('training fs:', train_fs / num_training)
print('test acc:', test_acc / (len(influ_mat_list)-num_training))
print('test fs:', test_fs / (len(influ_mat_list)-num_training))
