
import numpy as np
import networkx as nx
import pickle
import random
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import torch
import copy

def load_dataset(dataset, data_dir='data'):
    from pathlib import Path
    import pickle
    import sys

    sys.path.append('data') # for pickle.load

    data_dir = Path(data_dir)
    graph_name = dataset
    path_to_file = data_dir / graph_name
    with open(path_to_file, 'rb') as f:
        graph = pickle.load(f)
    return graph


def load_dataset(path:str):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph

def generate_seed_vector(top_nodes, seed_num, G):
    seed_nodes = random.sample(top_nodes, seed_num)
    seed_vector = [1 if node in seed_nodes else 0 for node in G.nodes()]
    return seed_vector

def diffusion_generation(graph, sim_num=10, diff_type='IC', time_step=100,repeat_step=10,seed_ratio=0.1, infect_prob=0.1, recover_prob=0.005, threshold=0.5):
    adj_mat = graph['adj_mat']
    G = nx.from_scipy_sparse_array(adj_mat)
    node_num = len(G.nodes())
    seed_num = int(seed_ratio * node_num)
    simulation = []

    degree_list = list(G.degree())
    degree_list.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [x[0] for x in degree_list[:int(len(degree_list) * 0.3)]]

    for i in range(sim_num):
        seed_vector = generate_seed_vector(top_nodes, seed_num, G)
        inf_vec_all = torch.zeros(node_num)
        config = mc.Configuration()
        for k in range(repeat_step):
            if diff_type == 'LT':
                model = ep.ThresholdModel(G)
                for n in G.nodes():
                    config.add_node_configuration("threshold", n, threshold)
            elif diff_type == 'IC':
                model = ep.IndependentCascadesModel(G)
                for e in G.edges():
                    config.add_edge_configuration("threshold", e, threshold)
            elif diff_type == 'SIS':
                model = ep.SISModel(G)
                config.add_model_parameter('beta', infect_prob)
                config.add_model_parameter('lambda', recover_prob)
            elif diff_type == 'SIR':
                model = ep.SIRModel(G)
                config.add_model_parameter('beta', infect_prob)
                config.add_model_parameter('lambda', recover_prob)
            elif diff_type == 'SI':
                model = ep.SIModel(G)
                config.add_model_parameter('beta', infect_prob)
            else:
                raise ValueError('Only IC, LT, SI, SIR and SIS are supported.')

            config.add_model_initial_configuration("Infected", seed_vector)

            model.set_initial_status(config)

            iterations = model.iteration_bunch(time_step)

            node_status = iterations[0]['status']

            for j in range(1, len(iterations)):
                node_status.update(iterations[j]['status'])

            inf_vec = np.array(list(node_status.values()))
            inf_vec[inf_vec == 2] = 1

            inf_vec_all += inf_vec

        inf_vec_all=inf_vec_all/repeat_step

        simulation.append([seed_vector, inf_vec_all])

    simulation = torch.Tensor(simulation).permute(0, 2, 1)

    dataset ={'adj_mat': adj_mat, 'diff_mat': simulation}
    return dataset

def split_dataset(dataset,train_ratio:float=0.6,seed:int=0):
    adj = dataset['adj_mat']
    diff_mat = copy.deepcopy(dataset['diff_mat'])
    all_num = len(diff_mat)
    train_num = int(all_num * train_ratio)
    test_num = all_num - train_num
    train_diff_mat, test_diff_mat = torch.utils.data.random_split(diff_mat, [train_num, test_num],
                                                                  generator=torch.Generator().manual_seed(seed))

    return adj, train_diff_mat, test_diff_mat

