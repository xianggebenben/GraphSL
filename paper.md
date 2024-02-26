---
title: 'GSL: A Open-Source Python Library for Graph Source Localization Algorithms and Benchmark Datasets'
tags:
  - Python
  - graph diffusion
  - Graph Source Localization
  - Prescribed Methods
  - GNN Methods
authors:
  - name: Junxiang Wang
    orcid: 0000-0002-6635-4296
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
![An example of source localization.](SL_example.png){label="fig:example"}
Graphs are prevalent data structures where nodes are connected by their relations, finding wide application across various domains. In graph mining, a fundamental task is graph diffusion, which aims to predict future graph cascade patterns given source nodes. Conversely, its inverse problem, graph source localization, though rarely explored, stands as an extremely important topic: it focuses on detection of source nodes given their future graph cascade patterns. As illustrated in Figure \ref{fig:example}, graph diffusion seeks to predict the cascade pattern $\{b,c,d,e\}$ from a source node $b$, whereas graph source localization aims to identify source nodes $b$ from the cascade pattern $\{b,c,d,e\}$. Graph source localization spans a broad spectrum of promising research and real-world applications. For instance, online social media platforms like Twitter and Facebook have been instrumental in disseminating rumors and misinformation with significant repercussions [@evanega2020coronavirus]. Additionally, the rapid propagation of computer viruses across the Internet, infecting millions of computers, underscores the critical need for tracking their sources [@kephart1993measuring]. Moreover, in smart grids, where isolated failures can trigger rolling blackouts leading to substantial financial losses [@amin2007preventing], graph source localization plays a pivotal role. Hence, the graph source localization problem demands attention and extensive investigations from machine learning researchers.

# Package Descriptions

## Algorithms

## Benchmark Datasets

# Availability and Documentation



# Acknowledgements

The support from Junji Jiang is acknowledged during the development of the GSL library.

# References
