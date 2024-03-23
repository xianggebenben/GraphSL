from data.utils import diffusion_generation,load_dataset,split_dataset
from gnn.SLVAE.main import SLVAE
import warnings
warnings.filterwarnings("ignore")
infect_prob =0.1
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme8000','digg16000'
graph = load_dataset('data/' + data_name)
if data_name not in ['meme8000', 'digg16000']:
    dataset = diffusion_generation(graph=graph,infect_prob=infect_prob)
else:
    dataset = graph
adj,train_dataset,test_dataset =split_dataset(dataset)
slave=SLVAE()
slvae_model,seed_vae_train,thres,train_auc,f1 = slave.train(adj,train_dataset)
print(train_auc,f1)
metric = slave.infer(test_dataset,slvae_model,seed_vae_train,thres)
print(metric.acc,metric.pr,metric.re,metric.fs,metric.auc)










