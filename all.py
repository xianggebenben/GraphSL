from data.utils import load_dataset,diffusion_generation,split_dataset
from Prescribed import LPSI,NetSleuth,OJC
from GNN.GCNSI.main import GCNSI
from GNN.IVGD.main import IVGD
from GNN.SLVAE.main import SLVAE
data_name = 'karate'  # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid',,'meme8000','digg16000'
graph = load_dataset(data_name)
if data_name not in ['meme8000', 'digg16000']:
    dataset = diffusion_generation(graph=graph,infect_prob=0.1,diff_type='IC')
else:
    dataset = graph
lpsi = LPSI()
adj,train_dataset,test_dataset =split_dataset(dataset)
alpha,thres,auc,f1 =lpsi.train(adj,train_dataset)
print("LPSI:")
print(f"train auc: {auc:.4f}, train f1: {f1:.4f}")
metric=lpsi.test(adj,test_dataset,alpha,thres)
print(f"test acc: {metric.acc:.4f},test pr: {metric.pr:.4f},test re: {metric.re:.4f},test f1: {metric.f1:.4f},test auc: {metric.auc:.4f}")
netSleuth = NetSleuth()
k,thres,auc,f1=netSleuth.train(adj,train_dataset)
print("NetSleuth:")
metric = netSleuth.test(adj,test_dataset,k,thres)
print(f"test acc: {metric.acc:.4f},test pr: {metric.pr:.4f},test re: {metric.re:.4f},test f1: {metric.f1:.4f},test auc: {metric.auc:.4f}")
ojc = OJC()
adj,train_dataset,test_dataset =split_dataset(dataset)
Y,thres,auc,f1 =ojc.train(adj,train_dataset)
print("OJC:")
print(f"train auc: {auc:.4f}, train f1: {f1:.4f}")
metric=ojc.test(adj,test_dataset,Y,thres)
print(f"test acc: {metric.acc:.4f},test pr: {metric.pr:.4f},test re: {metric.re:.4f},test f1: {metric.f1:.4f},test auc: {metric.auc:.4f}")
gcnsi = GCNSI()
gcnsi_model,alpha,thres,auc,f1 =gcnsi.train(adj,train_dataset)
print("GCNSI:")
print(f"train auc: {auc:.4f}, train f1: {f1:.4f}")
metric = gcnsi.test(adj,test_dataset,gcnsi_model,alpha,thres)
print(f"test acc: {metric.acc:.4f},test pr: {metric.pr:.4f},test re: {metric.re:.4f},test f1: {metric.f1:.4f},test auc: {metric.auc:.4f}")
ivgd=IVGD()
diffusion_model = ivgd.train_diffusion(adj,train_dataset)
ivgd_model,lamda,thres,auc,f1 =ivgd.train(train_dataset,diffusion_model)
print("IVGD:")
print(f"train auc: {auc:.4f}, train f1: {f1:.4f}")
metric = ivgd.test(test_dataset,diffusion_model,ivgd_model,lamda,thres)
print(f"test acc: {metric.acc:.4f},test pr: {metric.pr:.4f},test re: {metric.re:.4f},test f1: {metric.f1:.4f},test auc: {metric.auc:.4f}")
slave=SLVAE()
slvae_model,seed_vae_train,thres,auc,f1 = slave.train(adj,train_dataset)
print("SLVAE:")
print(f"train auc: {auc:.4f}, train f1: {f1:.4f}")
metric = slave.infer(test_dataset,slvae_model,seed_vae_train,thres)
print(f"test acc: {metric.acc:.4f},test pr: {metric.pr:.4f},test re: {metric.re:.4f},test f1: {metric.f1:.4f},test auc: {metric.auc:.4f}")
