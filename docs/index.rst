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

Install GraphSL using pip:

.. code-block:: bash

    pip install GraphSL

Or, clone the repo (https://github.com/xianggebenben/GraphSL), and install requirements:

.. code-block:: bash

    pip install -r requirements.txt

Usage
------------

Now, you can import and use GraphSL in your Python code.
.. code-block:: python

    from GraphSL.GNN.SLVAE.main import SLVAE
    from GraphSL.GNN.IVGD.main import IVGD
    from GraphSL.GNN.GCNSI.main import GCNSI
    from GraphSL.Prescribed import LPSI, NetSleuth, OJC
    from GraphSL.utils import load_dataset, diffusion_generation, split_dataset
    import os
    curr_dir = os.getcwd()
    # download datasets
    download_dataset(curr_dir)
    # load datasets ('karate', 'dolphins', 'jazz', 'netscience', 'cora_ml', 'power_grid')
    data_name = 'karate'
    graph = load_dataset(data_name, data_dir=curr_dir)
    # generate diffusion
    dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.2)
    # split into training and test sets
    adj, train_dataset, test_dataset = split_dataset(dataset)

    # LPSI
    print("LPSI:")
    lpsi = LPSI()

    # train LPSI
    alpha, thres, auc, f1, pred = lpsi.train(adj, train_dataset)
    print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

    # test LPSI
    metric = lpsi.test(adj, test_dataset, alpha, thres)
    print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")

    # NetSleuth
    print("NetSleuth:")
    netSleuth = NetSleuth()

    # train NetSleuth
    k, auc, f1 = netSleuth.train(adj, train_dataset)
    print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

    # test NetSleuth
    metric = netSleuth.test(adj, test_dataset, k)
    print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")

    # OJC
    print("OJC:")
    ojc = OJC()

    # train OJC
    Y, auc, f1 = ojc.train(adj, train_dataset)
    print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

    # test OJC
    metric = ojc.test(adj, test_dataset, Y)
    print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")

    # GCNSI
    print("GCNSI:")
    gcnsi = GCNSI()

    # train GCNSI
    gcnsi_model, thres, auc, f1, pred = gcnsi.train(adj, train_dataset)
    print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

    # test GCNSI
    metric = gcnsi.test(adj, test_dataset, gcnsi_model, thres)
    print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")

    # IVGD
    print("IVGD:")
    ivgd = IVGD()

    # train IVGD diffusion
    diffusion_model = ivgd.train_diffusion(adj, train_dataset)

    # train IVGD
    ivgd_model, thres, auc, f1, pred = ivgd.train(adj, train_dataset, diffusion_model)
    print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

    # test IVGD
    metric = ivgd.test(test_dataset, diffusion_model, ivgd_model, thres)
    print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")

    # SLVAE
    print("SLVAE:")
    slave = SLVAE()

    # train SLVAE
    slvae_model, seed_vae_train, thres, auc, f1, pred = slave.train(adj, train_dataset)
    print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

    # test SLVAE
    metric = slave.infer(test_dataset, slvae_model, seed_vae_train, thres)
    print(f"test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}, test auc: {metric.auc:.3f}")

  


That's it! You're ready to start using GraphSL. You can check results on the Jupyter notebook tutorial.ipynb from the repo (https://github.com/xianggebenben/GraphSL).

If you use this package in your research, please consider citing our work as follows:

.. code-block:: bash

   @article{wang2024joss,
  year = {2024},
  author = {Wang Junxiang, Zhao Liang},
  title = {GraphSL: A Open-Source Library for Graph Source Localization Approaches and Benchmark Datasets},
  journal = {preprint, 	arXiv:2405.03724}
   }

Contact
------------

We welcome your contributions! If you'd like to contribute your datasets or algorithms, please submit a pull request consisting of an atomic commit and a brief message describing your contribution.

For a new dataset, please upload it to the data folder of the repo. The file should be a dictionary object saved by pickle. It contains a key "adj_mat" with the value of a graph adjacency matrix (sprase numpy array with the CSR format).

For a new algorithm, please determine whether it belongs to presribed methods or GNN-based methods: if it belongs to the prescribed methods,  add your algorithm as a new class in the GraphSL/Prescribed.py. Otherwises, please upload it as a folder under the GraphSL/GNN folder. Typically, the algorithm should include a "train" function and a "test" function, and the "test" function should return a Metric object.

Feel free to Email me(junxiang.wang@alumni.emory.edu) if you have any questions. Bug reports and feedback can be directed to the Github issues page.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
