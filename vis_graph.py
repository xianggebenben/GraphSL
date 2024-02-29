from pathlib import Path
import pickle
import sys
import copy
import matplotlib.pyplot as plt
import networkx as nx

# dataset = 'karate' # 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid'
# sys.path.append('data')  # for pickle.load
# data_dir='data'
# data_dir = Path(data_dir)
# suffix = '_25c.SG'
# graph_name = dataset + suffix
# path_to_file = data_dir / graph_name
# with open(path_to_file, 'rb') as f:
#     graph = pickle.load(f)
#     print(graph)
#     adj_matrix = copy.copy(graph.adj_matrix)
#     prob_matrix = copy.copy(graph.prob_matrix)
#     influ_mat_list = copy.copy(graph.influ_mat_list)
#     # print(influ_mat_list[0])
#     # G = nx.read_sparse6(f)


import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.bipartite as bipartite

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.bipartite as bipartite

G = nx.davis_southern_women_graph()
women = G.graph["top"]
clubs = G.graph["bottom"]

print("Biadjacency matrix")
print(bipartite.biadjacency_matrix(G, women, clubs))

# project bipartite graph onto women nodes
W = bipartite.projected_graph(G, women)
print()
print("#Friends, Member")
for w in women:
    print(f"{W.degree(w)} {w}")

# project bipartite graph onto women nodes keeping number of co-occurence
# the degree computed is weighted and counts the total number of shared contacts
W = bipartite.weighted_projected_graph(G, women)
print()
print("#Friend meetings, Member")
for w in women:
    print(f"{W.degree(w, weight='weight')} {w}")

pos = nx.spring_layout(G, seed=648)  # Seed layout for reproducible node positions
nx.draw(G, pos)
plt.show()

G = nx.karate_club_graph()
# women = G.graph["top"]
# clubs = G.graph["bottom"]
#
# print("Biadjacency matrix")
# print(bipartite.biadjacency_matrix(G, women, clubs))
#
# # project bipartite graph onto women nodes
# W = bipartite.projected_graph(G, women)
# print()
# print("#Friends, Member")
# for w in women:
#     print(f"{W.degree(w)} {w}")
#
# # project bipartite graph onto women nodes keeping number of co-occurence
# # the degree computed is weighted and counts the total number of shared contacts
# W = bipartite.weighted_projected_graph(G, women)
# print()
# print("#Friend meetings, Member")
# for w in women:
#     print(f"{W.degree(w, weight='weight')} {w}")

pos = nx.spring_layout(G, seed=648)  # Seed layout for reproducible node positions
nx.draw(G, pos)
plt.show()

G = nx.krackhardt_kite_graph()

print("Betweenness")
b = nx.betweenness_centrality(G)
for v in G.nodes():
    print(f"{v:2} {b[v]:.3f}")

print("Degree centrality")
d = nx.degree_centrality(G)
for v in G.nodes():
    print(f"{v:2} {d[v]:.3f}")

print("Closeness centrality")
c = nx.closeness_centrality(G)
for v in G.nodes():
    print(f"{v:2} {c[v]:.3f}")

pos = nx.spring_layout(G, seed=367)  # Seed layout for reproducibility
nx.draw(G, pos)
plt.show()

n = 10  # 10 nodes
m = 20  # 20 edges
seed = 20160  # seed random number generators for reproducibility

# Use seed for reproducibility
G = nx.gnm_random_graph(n, m, seed=seed)

# some properties
print("node degree clustering")
for v in nx.nodes(G):
    print(f"{v} {nx.degree(G, v)} {nx.clustering(G, v)}")

print()
print("the adjacency list")
for line in nx.generate_adjlist(G):
    print(line)

pos = nx.spring_layout(G, seed=seed)  # Seed for reproducible layout
nx.draw(G, pos=pos)
plt.show()