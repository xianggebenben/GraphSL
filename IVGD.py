
from data.utils import load_dataset
from gnn.IVGD.main import IVGD
from data.utils import diffusion_generation,split_dataset

# key parameters
infect_prob =0.1
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme8000','digg16000'
graph = load_dataset('data/' + data_name)
if data_name not in ['meme8000', 'digg16000']:
    dataset = diffusion_generation(graph=graph,infect_prob=infect_prob)
else:
    dataset = graph
adj,train_dataset,test_dataset =split_dataset(dataset)

ivgd=IVGD()
diffusion_model = ivgd.train_diffusion(adj,train_dataset)
ivgd_model,lamda,thres,auc,f1 =ivgd.train(train_dataset,diffusion_model)
print(auc,f1)
metric = ivgd.test(test_dataset,diffusion_model,ivgd_model,lamda,thres)
print(metric.acc,metric.pr,metric.re,metric.fs,metric.auc)






