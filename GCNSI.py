from data.utils import load_dataset,split_dataset,diffusion_generation
from gnn.GCNSI.main import GCNSI
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme8000','digg16000'
graph = load_dataset('data/' + data_name)
if data_name not in ['meme8000', 'digg16000']:
    dataset = diffusion_generation(graph=graph)
else:
    dataset = graph
adj,train_dataset,test_dataset =split_dataset(dataset)
gcnsi = GCNSI()
gcnsi_model,alpha,thres,auc,f1 =gcnsi.train(adj,train_dataset)
print(auc,f1)
metric = gcnsi.test(adj,test_dataset,gcnsi_model,alpha,thres)
print(metric.acc,metric.pr,metric.re,metric.fs,metric.auc)


