from data.utils import load_dataset,diffusion_generation,split_dataset
from Prescribed import OJC
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme8000','digg16000'
graph = load_dataset('data/' + data_name)
if data_name not in ['meme8000', 'digg16000']:
    dataset = diffusion_generation(graph=graph)
else:
    dataset = graph
ojc = OJC()
adj,train_dataset,test_dataset =split_dataset(dataset)
Y,thres,auc,f1 =ojc.train(adj,train_dataset)
print(auc,f1)
metric=ojc.test(adj,test_dataset,Y,thres)
print(metric.acc,metric.pr,metric.re,metric.fs,metric.auc)
