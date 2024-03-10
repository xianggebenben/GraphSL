import torch
from torch.optim import Adam
from data.utils import data_generation,load_dataset
import copy
from learning.SLVAE.model import VAE,GNN,DiffusionPropagate
import numpy as np
from learning.SLVAE.main import slvae
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
infect_prob =0.1
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid','meme7000','digg16000'
if data_name in ['meme7000', 'digg16000']:
    dataset = load_dataset(data_name)
else:
    dataset = data_generation(data_name=data_name,infect_prob=infect_prob)
diff_mat = dataset['diff_mat']
num_node = dataset['adj_mat'].shape[0]
prob_matrix = torch.ones(size=(num_node,num_node))*infect_prob
# training parameters
train_ratio = 0.6
diff_mat = copy.deepcopy(dataset['diff_mat'])
all_num = len(diff_mat)
train_num = int(all_num*train_ratio)
test_num = all_num-train_num
train_diff_mat, test_diff_mat = torch.utils.data.random_split(diff_mat, [train_num, test_num])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE().to(device)
gnn = GNN(prob_matrix=prob_matrix)
propagate = DiffusionPropagate(prob_matrix, niter=2)

model = slvae (vae,gnn,propagate).to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

# train slvae
num_epoch = 50
model.train()
for epoch in range(num_epoch):
        overall_loss = 0
        for influ_mat in train_diff_mat:
            seed_vec = influ_mat[:, 0]
            seed_idx = np.argwhere(seed_vec == 1)  # used by PIteration
            influ_vec = influ_mat[:, -1]
            influ_vec = influ_vec.unsqueeze(-1).float()
            seed_vec = seed_vec.unsqueeze(-1).float()
            optimizer.zero_grad()
            seed_vec_hat, mean, log_var,influ_vec_hat = model(seed_vec,True)
            loss = model.train_loss(seed_vec, seed_vec_hat, mean, log_var,influ_vec,influ_vec_hat)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tOverall Loss: ", overall_loss)

# infer source
model.eval()
for param in model.parameters():
    param.requires_grad = False
# init
seed_train = torch.zeros(size=(train_num,num_node))
for i, influ_mat in enumerate(train_diff_mat):
    seed_vec = influ_mat[:, 0].unsqueeze(-1).float()
    seed_train[i,:] = model.vae(seed_vec)[0].squeeze(-1)
seed_infer =[]
seed_mean = torch.mean(seed_train,0).unsqueeze(-1).to(device)
for i in range(test_num):
    seed_vec_hat, _, _, influ_vec_hat = model(seed_mean, False)
    seed_infer.append(seed_vec_hat)

for seed in seed_infer:
    seed.requires_grad = True

optimizer = Adam(seed_infer, lr=1e-3)

for epoch in range(num_epoch):
        overall_loss = 0
        for i, influ_mat in enumerate(test_diff_mat):
            seed_vec = influ_mat[:, 0]
            seed_idx = np.argwhere(seed_vec == 1)  # used by PIteration
            influ_vec = influ_mat[:, -1]
            influ_vec = influ_vec.unsqueeze(-1).float()
            seed_vec = seed_vec.unsqueeze(-1).float()
            optimizer.zero_grad()
            seed_vec_hat, _, _,influ_vec_hat = model(seed_infer[i], False)
            loss = model.infer_loss(influ_vec, influ_vec_hat, seed_vec_hat, seed_train)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tOverall Loss: ", overall_loss)

test_auc= 0

for i, influ_mat in enumerate(test_diff_mat):
    seed_vec = influ_mat[:, 0]
    seed_vec = seed_vec.squeeze(-1).detach().numpy()
    seed_pred = seed_infer[i].detach().numpy()
    test_auc += roc_auc_score(seed_vec, seed_pred)
test_auc = test_auc / test_num

print('test auc:', test_auc)










