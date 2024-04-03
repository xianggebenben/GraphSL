.. GraphSL documentation master file, created by
   sphinx-quickstart on Sun Mar 24 14:11:33 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GraphSL's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
.. _quickstart:

Quickstart Guide
================

Installation
------------

First, install GraphSL using pip:

.. code-block:: bash

    pip install GraphSL

Or, clone the repo from https://github.com/xianggebenben/GraphSL.

Install requirements:

.. code-block:: bash

    pip install -r requirements.txt


Usage
-----

Now, you can import and use GraphSL in your Python code:

.. code-block:: python

   from GraphSL.data.utils import load_dataset, diffusion_generation, split_dataset
   from GraphSL.Prescribed import LPSI, NetSleuth, OJC
   from GraphSL.GNN.GCNSI.main import GCNSI
   from GraphSL.GNN.IVGD.main import IVGD
   from GraphSL.GNN.SLVAE.main import SLVAE
   data_name = 'karate'  # 'karate', 'dolphins', 'jazz', 'netscience', 'cora_ml', 'power_grid', , 'meme8000', 'digg16000'
   graph = load_dataset(data_name)
   if data_name not in ['meme8000', 'digg16000']:
       dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.3)
   else:
       dataset = graph
   adj, train_dataset, test_dataset =split_dataset(dataset)
   lpsi = LPSI()
   alpha, thres, auc, f1, pred =lpsi.train(adj, train_dataset)
   print("LPSI:")
   print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")
   metric=lpsi.test(adj, test_dataset, alpha, thres)
   print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")
   netSleuth = NetSleuth()
   k, auc, f1=netSleuth.train(adj, train_dataset)
   print("NetSleuth:")
   print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")
   metric = netSleuth.test(adj, test_dataset, k)
   print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")
   ojc = OJC()
   Y, auc, f1 =ojc.train(adj, train_dataset)
   print("OJC:")
   print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")
   metric=ojc.test(adj, test_dataset, Y)
   print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")
   gcnsi = GCNSI()
   gcnsi_model, thres, auc, f1, pred =gcnsi.train(adj, train_dataset)
   print("GCNSI:")
   print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")
   metric = gcnsi.test(adj, test_dataset, gcnsi_model, thres)
   print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")
   ivgd = IVGD()
   diffusion_model = ivgd.train_diffusion(adj, train_dataset)
   ivgd_model, thres, auc, f1, pred =ivgd.train(adj, train_dataset, diffusion_model)
   print("IVGD:")
   print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")
   metric = ivgd.test(test_dataset, diffusion_model, ivgd_model, thres)
   print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")
   slave = SLVAE()
   slvae_model, seed_vae_train, thres, auc, f1, pred = slave.train(adj, train_dataset)
   print("SLVAE:")
   print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")
   metric = slave.infer(test_dataset, slvae_model, seed_vae_train, thres)
   print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")


That's it! You're ready to start using GraphSL.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
