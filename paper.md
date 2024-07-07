---
title: 'GraphSL: An Open-Source Library for Graph Source Localization Approaches and Benchmark Datasets'
tags:
  - Python
  - Graph Diffusion
  - Graph Source Localization
  - Prescribed Methods
  - GNN Methods
  - Benchmark
authors:
  - name: Junxiang Wang
    orcid: 0000-0002-6635-4296
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Liang Zhao
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name:  Emory University, United States
   index: 1
date: 25 Feb 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

We introduce GraphSL, a new library for studying the graph source localization problem. graph diffusion and graph source localization are inverse problems in nature: graph diffusion predicts information diffusions from information sources, while graph source localization predicts information sources from information diffusions. GraphSL facilitates the exploration of various graph diffusion models for simulating information diffusions and enables the evaluation of cutting-edge source localization approaches on established benchmark datasets. The source code of GraphSL is made available at [Github Repository](https://github.com/xianggebenben/GraphSL). Bug reports and feedback can be directed to the [Github issues page](https://github.com/xianggebenben/GraphSL/issues).


# Statement of Need

![An example of graph source localization.\label{fig:example}](SL_example.png)

Graph diffusion is a fundamental task in graph learning, which aims to predict future information diffusions given information sources. Conversely, its inverse problem, graph source localization, though rarely explored, stands as an extremely important topic: it focuses on the detection of information sources given their future information diffusions. As illustrated in \autoref{fig:example}, graph diffusion seeks to predict the information diffusion $\{b,c,d,e\}$ from a source node $b$, whereas graph source localization aims to identify the source node $b$ from the information diffusion $\{b,c,d,e\}$. Graph source localization spans a broad spectrum of promising research and real-world applications such as rumor detection [@evanega2020coronavirus], tracking of sources for computer viruses, [@kephart1993measuring], and failures detection in smart grids [@amin2007preventing]. Please refer to the survey paper [@jiang2016identifying] for more information. Hence, the graph source localization problem demands attention and extensive investigations from machine learning researchers.

Some open-source tools have been developed to support the research of the graph source localization problem due to its importance. Two recent examples are cosasi [@McCabe2022joss] and RPaSDT [@frkaszczak2022rpasdt]. However, they missed comprehensive simulations of information diffusion, real-world benchmark datasets, and source localization approaches. To fill this gap, we propose a new library GraphSL: the first one to include both real-world benchmark datasets and recent source localization methods to our knowledge, which enables researchers and practitioners to easily evaluate novel techniques against appropriate baselines. These methods do not require prior knowledge (e.g. single source or multiple sources), and can handle graph source localization based on various diffusion simulation models such as Independent Cascade (IC) and Linear Threshold (LT) [@shakarian2015independent]. Our GraphSL library is standardized: for instance, tests of all source inference methods return a Metric object, which provides five performance metrics (accuracy, precision, recall, F-score, and area under ROC curve) for performance evaluation.

Our proposed GraphSL library targets at both developers and practical users: they are free to add algorithms and datasets for personal needs by following the guidelines of in the "Contributions to GraphSL" section of [README.md](https://github.com/xianggebenben/GraphSL/blob/main/README.md).

# Methods and Benchmark Datasets

![The hierarchical structure of the GraphSL library: in total 6 algorithms are implemented, which can be devided into two categories: prescribed methods that rely on hand-crafted rules and GNN-based methods which learn rules from graph data. \label{fig:overview}](overview.png)

The structure of our GraphSL library is depicted in \autoref{fig:overview}. Existing methods can be categorized into two groups: Prescribed methods and Graph Neural Networks (GNN)-based methods.

Prescribed methods rely on hand-crafted rules and heuristics. For instance, LPSI propagates infection in networks and labels local peaks as source nodes [@wang2017multiple]. NetSleuth employs the Minimum Description Length principle to identify the optimal set of source nodes and virus propagation ripple [@prakash2012spotting]. OJC identifies a set of nodes (Jordan cover) that cover all observed infected nodes with the minimum radius [@zhu2017catch].

GNN-based methods learn rules from graph data in an end-to-end manner by capturing graph topology and neighboring information. For example, GCNSI utilizes LPSI to enhance input and then applies Graph Convolutional Networks (GCN) for source identification [@dong2019multiple]; IVGD introduces a graph residual scenario to make existing graph diffusion models invertible, and it devises a new set of validity-aware layers to project inferred sources to feasible regions [@IVGD_www22]. SLVAE uses forward diffusion estimation and deep generative models to approximate source distribution, leveraging prior knowledge for generalization under arbitrary diffusion patterns [@ling2022source].

|       Dataset      |  #Node |  #Edge
|:------------------:|:------:|:------:|
|       Karate [@lusseau2003bottlenose]       |   34   |   78
|      Dolphins [@lusseau2003bottlenose]     |   62   |   159
|         Jazz [@gleiser2003community]       |   198  |  2,742 
| Network   Science [@newman2006finding] |  1,589 |  2,742 |
|       Cora-ML [@mccallum2000automating]     |  2,810 |  7,981 |
|    Power   Grid [@watts1998collective]    |  4,941 |  6,594 |

Table: \label{tab:statistics}   Six benchmark graph datasets: their numbers of nodes and edges.

Aside from methods, we also release six benchmark graph datasets to facilitate the research of graph source localization, whose statistics are shown in \autoref{tab:statistics}. Information sources and diffusions can be generated by the function [diffusion_generation](https://graphsl.readthedocs.io/en/latest/GraphSL.html#GraphSL.utils.diffusion_generation).

# Availability and Documentation

GraphSL is available under the MIT License. The library may be cloned from the [GitHub repository](https://github.com/xianggebenben/GraphSL), or can be installed by pip: pip install GraphSL. Documentation is provided via [Read the Docs](https://graphsl.readthedocs.io/en/latest/index.html), including a quickstart introducing major functionality and a detailed API reference. Extensive unit testing is employed throughout the library.

# References
