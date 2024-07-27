
from GraphSL.GNN.SLVAE.main import SLVAE
from GraphSL.GNN.IVGD.main import IVGD
from GraphSL.GNN.GCNSI.main import GCNSI
from GraphSL.Prescribed import LPSI, NetSleuth, OJC
from GraphSL.utils import load_dataset, diffusion_generation, split_dataset,download_dataset
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_name", type=str, help="data name",default='karate')
parser.add_argument("seed", type=int, help="random seed",default=0)

args = parser.parse_args()

curr_dir = os.getcwd()
# download datasets
download_dataset(curr_dir)
# load datasets ('karate', 'dolphins', 'jazz', 'netscience', 'cora_ml', 'power_grid')
data_name = args.data_name
graph = load_dataset(data_name, data_dir=curr_dir)
# generate diffusion
dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=1000, repeat_step=120, seed_ratio=0.1,random_seed=args.seed)
# split into training and test sets
adj, train_dataset, test_dataset = split_dataset(dataset)

#LPSI
print("LPSI:")
lpsi = LPSI()

# #train LPSI
alpha, thres, auc, f1, pred = lpsi.train(adj, train_dataset)
print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

# #test LPSI
metric = lpsi.test(adj, test_dataset, alpha, thres)
print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")

# #NetSleuth
print("NetSleuth:")
netSleuth = NetSleuth()

# #train NetSleuth
k, auc, f1 = netSleuth.train(adj, train_dataset)
print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

# #test NetSleuth
metric = netSleuth.test(adj, test_dataset, k)
print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")

# #OJC
print("OJC:")
ojc = OJC()

# #train OJC
Y, auc, f1 = ojc.train(adj, train_dataset)
print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

# #test OJC
metric = ojc.test(adj, test_dataset, Y)
print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")

# # GCNSI
print("GCNSI:")
gcnsi = GCNSI()

# #train GCNSI
gcnsi_model, thres, auc, f1, pred = gcnsi.train(adj, train_dataset)
print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")



# #test GCNSI
metric = gcnsi.test(adj, test_dataset, gcnsi_model, thres)
print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")

#IVGD
#print("IVGD:")
#ivgd = IVGD()

# # # # train IVGD diffusion
#diffusion_model = ivgd.train_diffusion(adj, train_dataset)

# # # # train IVGD
#ivgd_model, thres, auc, f1, pred = ivgd.train(
#       adj, train_dataset, diffusion_model)
# print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")


# # # test IVGD
#metric = ivgd.test(test_dataset, diffusion_model, ivgd_model, thres)
#print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")

#SLVAE
print("SLVAE:")
slave = SLVAE()

#train SLVAE
slvae_model, seed_vae_train, thres, auc, f1, pred = slave.train(
     adj, train_dataset)
print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")


# test SLVAE
metric = slave.infer(test_dataset, slvae_model, seed_vae_train, thres)
print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")