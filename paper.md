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


# Statement of need

![An example of graph source localization.\label{fig:example}](SL_example.png)

Graph diffusion is a fundamental task in graph learning, which aims to predict future graph cascade patterns given source nodes. Conversely, its inverse problem, 
graph source localization, though rarely explored, stands as an extremely important topic: it focuses on detection of source nodes given their future graph cascade 
patterns. As illustrated in \autoref{fig:example}, graph diffusion seeks to predict the cascade pattern $\{b,c,d,e\}$ from a source node $b$, whereas graph source 
localization aims to identify source nodes $b$ from the cascade pattern $\{b,c,d,e\}$. Graph source localization spans a broad spectrum
of promising research and real-world applications. For instance, online social media platforms like Twitter and Facebook have been instrumental in disseminating rumors
and misinformation with significant repercussions [@evanega2020coronavirus]. Additionally, the rapid propagation of computer viruses across the Internet, infecting 
millions of computers, underscores the critical need for tracking their sources [@kephart1993measuring]. Moreover, in smart grids, where isolated failures can trigger 
rolling blackouts leading to substantial financial losses [@amin2007preventing], graph source localization plays a pivotal role. Hence, the graph source localization 
problem demands attention and extensive investigations from machine learning researchers.

Some open-source tools have been developed to support the research of graph source localization problem due to its importance. Two recent examples are cosasi [@McCabe2022joss] 
and RPaSDT [@frkaszczak2022rpasdt]. 
However, they missed real-world benchmark datasets and up-to-date state-of-the-art approaches. To fill this gap, we propose a new package GraphSL: 
the first one to include  both real-world benchmark datasets and a number of recent source localization methods to our knowledge,
which enables researchers and practitioners to easily evaluate novel techniques against appropriate baselines. These methods do not require prior knowledge 
(e.g. single source or multiple sources), and can handle graph source localization based on various graph diffusion models such as 
Independent Cascade (IC) and Linear Threshold (LT). 

# Problem Definition
Consider a graph $G=(V,E)$, where $V=\{v_1,\cdots,v_n\}$ and $E$ are the node set and the edge set respectively, $\vert V\vert=n$ is the number of nodes. 
$Y_t\in \{0,1\}^{n}$ is a diffusion vector at time $t$. $Y_{t,i}=1$ means that node $i$ is diffused, while $Y_{t,i}=0$ means that node $i$ is not diffused.   
$S$ is a set of source nodes. $x\in \{0,1\}^n$ is a vector of source nodes, $x_i=1$ if $v_i\in S$ and $x_i=0$ otherwise. 
The diffusion process begins at timestamp 0 and terminates at timestamp $T$. The graph diffusion model is denoted as $\theta$, and its inverse problem, 
graph source localization, is to infer $x$ from $Y_{T}$:
\begin{align}
    \theta^{-1}: Y_T \rightarrow x. \label{eq:source localization}
\end{align}

# Methods and Benchmark Datasets

![The hierarchical structure of our GraphSL library version 0.1.\label{fig:overview}](overview.png)

The structure of our GraphSL library is shown in \autoref{fig:overview}. Existing graph source localization methods fall in two categories: Prescribed methods and. 
Graph Neural Networks(GNN)-based methods. Prescribed methods based on hand-crafted rules and heuristics. For example, LPSI propogated infection in the networks and labeled 
local peaks as source nodes [@wang2017multiple]; NetSleuth employed the Minimum Description Length principle to identify the best set of source nodes 
and virus propagation ripple. [@prakash2012spotting]; OJC finds a set of nodes (i.e. Jordan cover) that “cover” all observed infected nodes with 
the minimum radius [@zhu2017catch]. GCNSI used the LPSI to augment input, and then applied the Graph Convolutional Networks(GCN) 
for source identification [@dong2019multiple]. GNN-based methods "learn" rules from graph data in an end-to-end fashion by capturing graph topology and
neighboring information. As an example, IVGD proposes a graph residual scenario to make existing graph diffusion models invertible, and a new set of validity-aware layers
have been devised to project inferred sources to feasible regions [@IVGD_www22]; SLVAE uses forward diffusion estimation and deep generative models to approximate source distribution, 
leveraging prior knowledge for generalization under arbitrary diffusion patterns [@ling2022source].


# Availability and Documentation



# Acknowledgements

We knowledge the support from Junji Jiang during the development of the GraphSL library.

# References
