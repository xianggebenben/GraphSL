from data.utils import load_dataset
from data.utils import diffusion_generation
from Prescribed import LPSI
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme8000','digg16000'
graph = load_dataset('data/' + data_name)
if data_name not in ['meme8000', 'digg16000']:
    dataset = diffusion_generation(graph=graph)
else:
    dataset = graph
lpsi = LPSI()
metric=lpsi.run(dataset)
print(metric.acc,metric.pr,metric.re,metric.fs,metric.auc)