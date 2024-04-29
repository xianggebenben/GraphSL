import numpy as np
import networkx as nx
import random
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import torch
import copy


def load_dataset(dataset, data_dir):
    """
    Load a dataset from a pickle file.

    Args:
        
    - dataset (str): The name of the dataset file, 'karate', 'dolphins', 'jazz', 'netscience', 'cora_ml', 'power_grid','meme8000', 'digg16000'.
        
    - data_dir (str): The directory where the dataset files are stored.

    Returns:
        
    - graph (dict): A dictionary containing the dataset.

    """
    import pickle

    data_dir = data_dir+"/data/"+dataset
    with open(data_dir, 'rb') as f:
        graph = pickle.load(f)
    return graph


def generate_seed_vector(top_nodes, seed_num, G):
    """
    Generate a seed vector for diffusion simulation.

    Args:

    - top_nodes (list): List of top nodes based on node degree.

    - seed_num (int): Number of seed nodes.

    - G (networkx.Graph): The graph object.

    Returns:

        seed_vector (list): Seed vector for diffusion simulation.
    """
    seed_nodes = random.sample(top_nodes, seed_num)
    seed_vector = [1 if node in seed_nodes else 0 for node in G.nodes()]
    return seed_vector


def diffusion_generation(graph, sim_num=10, diff_type='IC', time_step=10, repeat_step=10, seed_ratio=0.1,
                         infect_prob=0.1, recover_prob=0.005, threshold=0.5):
    """
    Generate diffusion matrices for a graph.

    Args:

    - graph (dict): Dictionary containing the graph information.

    - sim_num (int): Number of simulations.

    - diff_type (str): Type of diffusion model (IC, LT, SI, SIS, SIR). IC stands for Independent Cascade, LT stands for Linear Threshold, SI stands for Susceptible or Infective, SIS stands for Susceptible or Infective or Susceptible, SIR stands for Susceptible or Infective or Recovered.

    - time_step (int): Number of time steps in the simulation.

    - repeat_step (int): Number of repetitions for each simulation.

    - seed_ratio (float): Ratio of seed nodes.

    - infect_prob (float): Infection probability,  used in SIS, SIR or SI.

    - recover_prob (float): Recovery probability, used in SIS or SIR.

    - threshold (float): Threshold parameter for diffusion models, used in IC or LT.

    Returns:

    - dataset (dict): Dictionary containing ('adj_mat') adjacency matrix and ('diff_mat') diffusion matrices.

    Example:

    import os

    curr_dir = os.getcwd()

    from data.utils import load_dataset, diffusion_generation

    data_name = 'karate'

    graph = load_dataset(data_name, data_dir=curr_dir)

    dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)
    """
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
                config.add_model_parameter('gamma', recover_prob)
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

        inf_vec_all = inf_vec_all / repeat_step

        simulation.append([seed_vector, inf_vec_all])

    simulation = torch.Tensor(simulation).permute(0, 2, 1)

    dataset = {'adj_mat': adj_mat, 'diff_mat': simulation}
    return dataset


def split_dataset(dataset, train_ratio: float = 0.6, seed: int = 0):
    """
    Split the dataset into training and testing sets.

    Args:

    - dataset (dict): Dictionary containing the dataset.

    - train_ratio (float): Ratio of training data. Default is 0.6.

    - seed (int): Random seed for reproducibility. Default is 0.

    Returns:

    - adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.

    - train_dataset (torch.utils.data.dataset.Subset): The train dataset (number of simulations * number of graph nodes * 2(the first column is seed vector and the second column is diffusion vector)).

    - test_dataset (torch.utils.data.dataset.Subset): The test dataset (number of simulations * number of graph nodes * 2(the first column is seed vector and the second column is diffusion vector)).

    Example:

    import os

    curr_dir = os.getcwd()

    from data.utils import load_dataset, diffusion_generation, split_dataset

    data_name = 'karate'

    graph = load_dataset(data_name, data_dir = curr_dir)

    dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)

    adj, train_dataset, test_dataset =split_dataset(dataset)
    """
    adj = dataset['adj_mat']
    diff_mat = copy.deepcopy(dataset['diff_mat'])
    all_num = len(diff_mat)
    train_num = int(all_num * train_ratio)
    test_num = all_num - train_num
    train_diff_mat, test_diff_mat = torch.utils.data.random_split(diff_mat, [train_num, test_num],
                                                                  generator=torch.Generator().manual_seed(seed))

    return adj, train_diff_mat, test_diff_mat
