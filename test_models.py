import pytest
from GraphSL.GNN.SLVAE.main import SLVAE
from GraphSL.GNN.IVGD.main import IVGD
from GraphSL.GNN.GCNSI.main import GCNSI
from GraphSL.Prescribed import LPSI, NetSleuth, OJC
from GraphSL.utils import load_dataset, diffusion_generation, split_dataset, download_dataset
import os

curr_dir = os.getcwd()
data_name = 'karate'
graph = load_dataset(data_name, data_dir=curr_dir)
dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.2)
adj, train_dataset, test_dataset = split_dataset(dataset)

def test_lpsi():
    lpsi = LPSI()
    alpha, thres, auc, f1, pred = lpsi.train(adj, train_dataset)
    assert auc >=0
    assert f1 >=0
    metric = lpsi.test(adj, test_dataset, alpha, thres)
    assert metric.acc >= 0
    assert metric.pr >= 0
    assert metric.re >= 0
    assert metric.f1 >= 0
    assert metric.auc >= 0

def test_netsleuth():
    netSleuth = NetSleuth()
    k, auc, f1 = netSleuth.train(adj, train_dataset)
    assert auc >=0
    assert f1 >=0
    metric = netSleuth.test(adj, test_dataset, k)
    assert metric.acc >= 0
    assert metric.pr >= 0
    assert metric.re >= 0
    assert metric.f1 >= 0
    assert metric.auc >= 0

def test_ojc():
    ojc = OJC()
    Y, auc, f1 = ojc.train(adj, train_dataset)
    assert auc >=0
    assert f1 >=0
    metric = ojc.test(adj, test_dataset, Y)
    assert metric.acc >= 0
    assert metric.pr >= 0
    assert metric.re >= 0
    assert metric.f1 >= 0
    assert metric.auc >= 0

def test_gcnsi():
    gcnsi = GCNSI()
    gcnsi_model, thres, auc, f1, pred = gcnsi.train(adj, train_dataset)
    assert auc >=0
    assert f1 >=0
    metric = gcnsi.test(adj, test_dataset, gcnsi_model, thres)
    assert metric.acc >= 0
    assert metric.pr >= 0
    assert metric.re >= 0
    assert metric.f1 >= 0
    assert metric.auc >= 0

def test_ivgd():
    ivgd = IVGD()
    diffusion_model = ivgd.train_diffusion(adj, train_dataset)
    ivgd_model, thres, auc, f1, pred = ivgd.train(adj, train_dataset, diffusion_model)
    assert auc >=0
    assert f1 >=0
    metric = ivgd.test(test_dataset, diffusion_model, ivgd_model, thres)
    assert metric.acc >= 0
    assert metric.pr >= 0
    assert metric.re >= 0
    assert metric.f1 >= 0
    assert metric.auc >= 0

def test_slvae():
    slave = SLVAE()
    slvae_model, seed_vae_train, thres, auc, f1, pred = slave.train(adj, train_dataset)
    assert auc >=0
    assert f1 >=0
    metric = slave.infer(test_dataset, slvae_model, seed_vae_train, thres)
    assert metric.acc >= 0
    assert metric.pr >= 0
    assert metric.re >= 0
    assert metric.f1 >= 0
    assert metric.auc >= 0
