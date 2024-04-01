---
title: 'GraphSL: A Open-Source Library for Graph Source Localization Algorithms and Benchmark Datasets'
tags:
  - Python
  - graph diffusion
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

We present GraphSL, an extensible library designed for investigating the graph source localization problem. Our library facilitates the exploration of various graph diffusion models for simulating information spread and enables the evaluation of cutting-edge source localization approaches on established benchmark datasets. Ongoing development efforts are underway, and we enthusiastically invite contributions from both academic and industrial sectors to further enhance GraphSL's capabilities.


# Statement of need

![An example of graph source localization.\label{fig:example}](SL_example.png)

Graph diffusion is a fundamental task in graph learning, which aims to predict future graph cascade patterns given source nodes. Conversely, its inverse problem, graph source localization, though rarely explored, stands as an extremely important topic: it focuses on detection of source nodes given their future graph cascade patterns. As illustrated in \autoref{fig:example}, graph diffusion seeks to predict the cascade pattern $\{b,c,d,e\}$ from a source node $b$, whereas graph source localization aims to identify source nodes $b$ from the cascade pattern $\{b,c,d,e\}$. Graph source localization spans a broad spectrum of promising research and real-world applications. For instance, online social media platforms like Twitter and Facebook have been instrumental in disseminating rumors and misinformation with significant repercussions [@evanega2020coronavirus]. Additionally, the rapid propagation of computer viruses across the Internet, infecting millions of computers, underscores the critical need for tracking their sources [@kephart1993measuring]. Moreover, in smart grids, where isolated failures can trigger rolling blackouts leading to substantial financial losses [@amin2007preventing], graph source localization plays a pivotal role. Hence, the graph source localization problem demands attention and extensive investigations from machine learning researchers.

Some open-source tools have been developed to support the research of graph source localization problem due to its importance. Two recent examples are cosasi [@McCabe2022joss] and RPaSDT [@frkaszczak2022rpasdt]. However, they missed comprehensive simulations of information diffusion, real-world benchmark datasets, and up-to-date state-of-the-art source localization approaches. To fill this gap, we propose a new library GraphSL: the first one to include both real-world benchmark datasets and recent source localization methods to our knowledge, which enables researchers and practitioners to easily evaluate novel techniques against appropriate baselines. These methods do not require prior knowledge (e.g. single source or multiple sources), and can handle graph source localization based on various diffusion simulation models such as Independent Cascade (IC) and Linear Threshold (LT). Our GraphSL library is standardized: for instance, tests of all source inference methods return an Evaluation object, which provides five metrics (accuracy, precision, recall, F-score, and area under ROC curve) for performance comparison.


# Problem Definition
Consider a graph $G=(V,E)$, where $V=\{v_1,\cdots,v_n\}$ and $E$ are the node set and the edge set respectively, $\vert V\vert=n$ is the number of nodes. 
$Y_t\in \{0,1\}^{n}$ is a diffusion vector at time $t$. $Y_{t,i}=1$ means that node $i$ is diffused, while $Y_{t,i}=0$ means that node $i$ is not diffused.   
$S$ is a set of source nodes. $x\in \{0,1\}^n$ is a source vector, $x_i=1$ if $v_i\in S$ and $x_i=0$ otherwise. 
The diffusion process begins at timestamp 0 and terminates at timestamp $T$. The graph diffusion model is denoted as $\theta$, and its inverse problem, 
graph source localization, is to infer $x$ from $Y_{T}$:
\begin{align}
    \theta^{-1}: Y_T \rightarrow x. \label{eq:source localization}
\end{align}

# Methods and Benchmark Datasets

![The hierarchical structure of our GraphSL library version 0.1.\label{fig:overview}](overview.png)

The structure of our GraphSL library is depicted in \autoref{fig:overview]. Existing graph source localization methods can be categorized into two groups: Prescribed methods and Graph Neural Networks (GNN)-based methods.

Prescribed methods rely on hand-crafted rules and heuristics. For instance, LPSI propagated infection in networks and labels local peaks as source nodes [@wang2017multiple]. NetSleuth employed the Minimum Description Length principle to identify the optimal set of source nodes and virus propagation ripple [@prakash2012spotting]. OJC identified a set of nodes (Jordan cover) that cover all observed infected nodes with the minimum radius [@zhu2017catch]. 

GNN-based methods learn rules from graph data in an end-to-end manner by capturing graph topology and neighboring information. For example, GCNSI utilized LPSI to enhance input and then applies Graph Convolutional Networks (GCN) for source identification [@dong2019multiple]; IVGD introduced a graph residual scenario to make existing graph diffusion models invertible, and it devises a new set of validity-aware layers to project inferred sources to feasible regions [@IVGD_www22]. SLVAE used forward diffusion estimation and deep generative models to approximate source distribution, leveraging prior knowledge for generalization under arbitrary diffusion patterns [@ling2022source].

|       Dataset      |  #Node |  #Edge | Average Degree | Has Seed-Diffusion Vector Pairs |
|:------------------:|:------:|:------:|:--------------:|:--------------------------:|
|       Karate       |   34   |   78   |      4.588     |             No             |
|      Dolphins      |   62   |   159  |      5.129     |             No             |
|         Jazz       |   198  |  2,742 |     27.697     |             No             |
| Network   Science  |  1,589 |  2,742 |      3.451     |             No             |
|       Cora-ML      |  2,810 |  7,981 |      5.68      |             No             |
|    Power   Grid    |  4,941 |  6,594 |      2.669     |             No             |
|     Memetracker    |  7,884 | 47,911 |     12.154     |            Yes             |
|        Digg        | 15,912 | 78,649 |      9.885     |            Yes             |

Table: \label{tab:statistics} The statistics of eight datasets.

 Aside from methods, we also provide eight benchmark datasets to facilitate the  graph SL research, whose statistics are shown in \autoref{tab:statistics}.
 Memetracker and Digg provide Source-Diffusion pairs $(x,Y_{T})$, while others do not. All datasets are introduced as follows:

 1. Karate [@lusseau2003bottlenose]. Karate contains the social ties among the members of a university karate club.

 2. Dolphins [@lusseau2003bottlenose]. Dolphins is a social network of bottlenose dolphins, where edges represent frequent associations between dolphins.

 3. Jazz [@gleiser2003community]. Jazz is a collaboration network between Jazz musicians. Each edge represents that two musicians have played together in a band.

 4. Network Science [@newman2006finding]. Network Science is a coauthorship network of scientists working on network theory and experiment. Each edge represents two scientists who have co-authored a paper.

 5. Cora-ML [@mccallum2000automating]. Cora-ML is a portal network of computer science research papers crawled by machine learning techniques.

 6. Power Grid [@watts1998collective]. Power Grid is a topology network of the Western States Power Grid of the United States.

 7. Memetracker [@leskovec2009meme]. The Memetracker keeps track of frequently used phrases on news social media. Only a small subset of the Memetracker network is used here.

 8. Digg [@hogg2012social]. Digg is a reply network of the social news. Only a small subset of the Memetracker network is used here.

# Availability and Documentation

GraphSL is available under the MIT License. The library may be cloned from the GitHub repository or via PyPI: pip install GraphSL. Documentation is provided via Read the Docs, including a quickstart introducing major functionality and a detailed API reference. Extensive unit testing is employed throughout the library. The source code of GraphSL is made available at (https://github.com/xianggebenben/GraphSL). Bug reports and feedback can be directed to the Github issues page (https://github.com/xianggebenben/GraphSL/issues).

# Acknowledgements

We knowledge the support from Junji Jiang during the development of the GraphSL library.

# References
