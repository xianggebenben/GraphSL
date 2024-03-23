from data.utils import load_dataset,diffusion_generation,split_dataset
from Prescribed import LPSI
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme8000','digg16000'
graph = load_dataset('data/' + data_name)
if data_name not in ['meme8000', 'digg16000']:
    dataset = diffusion_generation(graph=graph)
else:
    dataset = graph
lpsi = LPSI()
adj,train_dataset,test_dataset =split_dataset(dataset)
alpha,thres,auc,f1 =lpsi.train(adj,train_dataset)
print(auc,f1)
metric=lpsi.test(adj,test_dataset,alpha,thres)
print(metric.acc,metric.pr,metric.re,metric.fs,metric.auc)