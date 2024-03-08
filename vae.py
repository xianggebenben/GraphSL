import torch
from torch.optim import Adam
from data.utils import data_generation,load_dataset
import copy
from learning.SLVAE.model import VAE,GNN,DiffusionPropagate
import numpy as np
import torch.nn.functional as F
from learning.SLVAE.main import slvae
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
            seed_vec_hat, mean, log_var,influ_vec_hat = model(seed_vec)
            loss = model.loss(seed_vec, seed_vec_hat, mean, log_var,influ_vec,influ_vec_hat)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tOverall Loss: ", overall_loss)






