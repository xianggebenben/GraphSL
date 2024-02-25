import numpy as np
import networkx as nx
import pickle
import random
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import torch


def generate_seed_vector(top_nodes, seed_num, G):
    seed_nodes = random.sample(top_nodes, seed_num)
    seed_vector = [1 if node in seed_nodes else 0 for node in G.nodes()]
    return seed_vector


def data_generation(adj_matrix, nums, percentage=10, diffusion='LT', dataset='cora_ml'):
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    node_num = len(G.nodes())
    seed_num = int(percentage * node_num / 100)
    samples = []

    degree_list = list(G.degree())
    degree_list.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [x[0] for x in degree_list[:int(len(degree_list) * 0.3)]]

    for i in range(nums):
        print('Sample {} generating'.format(i))
        seed_vector = generate_seed_vector(top_nodes, seed_num, G)
        inf_vec_all = torch.zeros(node_num)
        for j in range(10):
            if diffusion == 'LT':
                model = ep.ThresholdModel(G)
                config = mc.Configuration()
                for n in G.nodes():
                    config.add_node_configuration("threshold", n, 0.5)
            elif diffusion == 'IC':
                model = ep.IndependentCascadesModel(G)
                config = mc.Configuration()
                for e in G.edges():
                    config.add_edge_configuration("threshold", e, 1 / nx.degree(G)[e[1]])
            elif diffusion == 'SIS':
                model = ep.SISModel(G)
                config = mc.Configuration()
                config.add_model_parameter('beta', 0.001)
                config.add_model_parameter('lambda', 0.001)
            elif diffusion == 'SIR':
                model = ep.SIRModel(G)
                config = mc.Configuration()
                config.add_model_parameter('beta', 0.001)
                config.add_model_parameter('lambda', 0.001)
            elif diffusion == 'SI':
                model = ep.SIModel(G)
                config = mc.Configuration()
                config.add_model_parameter('beta', 0.01)
                config.add_model_parameter("fraction_infected", 0.05)
            else:
                raise ValueError('Only IC, LT and SIS are supported.')

            config.add_model_initial_configuration("Infected", seed_vector)

            model.set_initial_status(config)

            iterations = model.iteration_bunch(100)

            node_status = iterations[0]['status']

            for j in range(1, len(iterations)):
                node_status.update(iterations[j]['status'])

            inf_vec = np.array(list(node_status.values()))
            inf_vec[inf_vec == 2] = 1

            inf_vec_all += inf_vec

        inf_vec_all = inf_vec_all / 10
        samples.append([seed_vector, inf_vec_all])

    samples = torch.Tensor(samples).permute(0, 2, 1)
    f = open('{}_mean_{}{}.SG'.format(dataset, diffusion, percentage), 'wb')
    pickle.dump({'adj': adj_matrix, 'inverse_pairs': samples}, f)
    f.close()

    print('Data generation finished')

if __name__=="__main__":
    with open('adj/cora_ml.SG', 'rb') as f:
        adj_martix = pickle.load(f)

    data_generation(adj_martix, 10)