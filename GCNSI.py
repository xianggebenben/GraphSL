from scipy.sparse import csgraph
from data.utils import load_dataset,split_dataset,diffusion_generation
from sklearn.metrics import roc_auc_score
import torch
from scipy.sparse import coo_matrix
import numpy as np
from Prescribed import GCNSI
import copy
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme8000','digg16000'
graph = load_dataset('data/' + data_name)
if data_name not in ['meme8000', 'digg16000']:
    dataset = diffusion_generation(graph=graph)
else:
    dataset = graph
adj,train_dataset,test_dataset =split_dataset(dataset)


