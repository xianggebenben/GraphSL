import numpy as np
import networkx as nx
import random
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import torch
import copy
import requests
import pickle
import os
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import random_split


def download_dataset(data_dir):
    """
    Download datasets from url.

    Args:

    - data_dir (str): The directory where the downloaded dataset files are stored.


    """

    api_url ="https://api.github.com/repos/xianggebenben/graphsl/contents/data?ref=main"

        # Send a request to fetch the folder contents
    response = requests.get(api_url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    data_dir = data_dir + "/data/"

    # Ensure the output directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check if response content type is JSON
    if 'application/json' in response.headers.get('Content-Type', ''):
        folder_contents = response.json()
    else:
        print(f"Response is not in JSON format. Response text:\n{response.text}")
        return

    # Process the contents of the folder
    for item in folder_contents:
        if item['type'] == 'file':
            # Download the file
            download_url = item['download_url']
            file_name = item['name']
            file_response = requests.get(download_url)
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'wb') as file:
                file.write(file_response.content)
            print(f"Downloaded {file_name}")


def load_dataset(dataset, data_dir):
    """
    Load a dataset from a pickle file.

    Args:

    - dataset (str): The name of the dataset file, 'karate', 'dolphins', 'jazz', 'netscience', 'cora_ml', 'power_grid'.

    - data_dir (str): The directory where the dataset files are stored.

    Returns:

    - graph (dict): A dictionary containing the dataset.

    """

    data_dir = data_dir + "/data/" + dataset
    with open(data_dir, 'rb') as f:
        graph = pickle.load(f)
    return graph


def generate_seed_vector(top_nodes, seed_num, G, random_seed):
    """
    Generate a seed vector for diffusion simulation.

    Args:

    - top_nodes (list): List of top nodes based on node degree.

    - seed_num (int): Number of seed nodes.

    - G (networkx.Graph): The graph object.

    - random_seed (int): Random Seed

    Returns:

        seed_vector (list): Seed vector for diffusion simulation.
    """
    random.seed(random_seed)
    seed_nodes = random.sample(top_nodes, seed_num)
    seed_vector = [1 if node in seed_nodes else 0 for node in G.nodes()]
    return seed_vector


def diffusion_generation(
        graph,
        sim_num=10,
        diff_type='IC',
        time_step=10,
        repeat_step=10,
        seed_ratio=0.1,
        infect_prob=0.1,
        recover_prob=0.005,
        threshold=0.5,
        random_seed=0):
    """
    Generate diffusion matrices for a graph.

    Args:

    - graph (dict): Dictionary containing the graph information.

    - sim_num (int): Number of simulations.

    - diff_type (str): Type of diffusion model (IC, LT, SI, SIS, SIR). IC stands for Independent Cascade, LT stands for Linear Threshold, SI stands for Susceptible or Infective, SIS stands for Susceptible or Infective or Susceptible, SIR stands for Susceptible or Infective or Recovered.

    - time_step (int): Number of time steps in the simulation.

    - repeat_step (int): Number of repetitions for each simulation.

    - seed_ratio (float): Ratio of seed nodes, should be between 0 and 0.3.

    - infect_prob (float): Infection probability,  used in SIS, SIR or SI.

    - recover_prob (float): Recovery probability, used in SIS or SIR.

    - threshold (float): Threshold parameter for diffusion models, used in IC or LT.

    - random_seed (int): Random seed.

    Returns:

    - dataset (dict): Dictionary containing ('adj_mat') adjacency matrix (the dimensionality is number of nodes * number of nodes) and ('diff_mat') diffusion matrices (the dimensionality is number of simulations * number of nodes * 2(the first column is the source vector, and the second column is the diffusion vector)).

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

    assert seed_ratio<=0.3 and seed_ratio>=0, "seed_ratio should be between 0 and 0.3"

    for i in range(sim_num):
        seed_vector = generate_seed_vector(top_nodes, seed_num, G, random_seed+i)
        inf_vec_all = torch.zeros(node_num)
        config = mc.Configuration()
        for k in range(repeat_step):
            if diff_type == 'LT':
                model = ep.ThresholdModel(G,random_seed)
                for n in G.nodes():
                    config.add_node_configuration("threshold", n, threshold)
            elif diff_type == 'IC':
                model = ep.IndependentCascadesModel(G,random_seed)
                for e in G.edges():
                    config.add_edge_configuration("threshold", e, threshold)
            elif diff_type == 'SIS':
                model = ep.SISModel(G,random_seed)
                config.add_model_parameter('beta', infect_prob)
                config.add_model_parameter('lambda', recover_prob)
            elif diff_type == 'SIR':
                model = ep.SIRModel(G,random_seed)
                config.add_model_parameter('beta', infect_prob)
                config.add_model_parameter('gamma', recover_prob)
            elif diff_type == 'SI':
                model = ep.SIModel(G,random_seed)
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
    train_diff_mat, test_diff_mat = random_split(
        diff_mat, [train_num, test_num], generator=torch.Generator().manual_seed(seed))

    return adj, train_diff_mat, test_diff_mat


def visualize_source_prediction(adj: csr_matrix, predictions: np.ndarray, labels: np.ndarray, save_dir: str, save_name: str):

    """
    Visualize source predictions.

    Args:

    - adj (csr_matrix): Dictionary containing the dataset.

    - predictions (numpy array): Predicted source vector, each entry should be either 0 or 1, where 1 means the source, and 0 means otherwise.

    - labels (numpy array): Labeled source vector, each entry should be either 0 or 1, where 1 means the source, and 0 means otherwise.

    - save_dir (str):  Dirctory of the saved figure.

    - save_name (str): Name of the saved figure.


    Example:

    from GraphSL.GNN.GCNSI.main import GCNSI

    from GraphSL.utils import load_dataset, diffusion_generation, split_dataset,download_dataset,visualize_source_prediction

    import os

    curr_dir = os.getcwd()

    download_dataset(curr_dir)

    data_name = 'karate'

    graph = load_dataset(data_name, data_dir=curr_dir)

    dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.2)

    adj, train_dataset, test_dataset = split_dataset(dataset)

    print("GCNSI:")

    gcnsi = GCNSI()

    gcnsi_model, thres, auc, f1, pred = gcnsi.train(adj, train_dataset)

    print(f"train auc: {auc:.3f}, train f1: {f1:.3f}")

    pred = (pred >= thres)

    visualize_source_prediction(adj,pred[:,0],train_dataset[0][:,0].numpy(),save_dir=curr_dir,save_name="GCNSI_source_prediction")

 """

    # Convert the adjacency matrix to a NetworkX graph
    graph = nx.from_scipy_sparse_array(adj)
    
    # Determine the number of nodes
    num_nodes = adj.shape[0]
    
    # Check that predictions and labels have the same length as the number of nodes
    if len(predictions) != num_nodes or len(labels) != num_nodes:
        raise ValueError("The length of predictions and labels must match the number of nodes in the graph.")
    
    # Set up the plot
    plt.figure(figsize=(10, 5))
    
    # Define the layout for the graph
    pos = nx.spring_layout(graph)

    # Plot the predictions
    plt.subplot(1, 2, 1)
    nx.draw(graph, pos, node_color=predictions, with_labels=True, cmap=plt.cm.coolwarm, node_size=500)
    plt.title("Predicted Sources")
    pred_patch_0 = mpatches.Patch(color=plt.cm.coolwarm(0.0), label='Not Source')
    pred_patch_1 = mpatches.Patch(color=plt.cm.coolwarm(1.0), label='Source')
    plt.legend(handles=[pred_patch_0, pred_patch_1], loc='best')
    
    
    # Plot the true labels
    plt.subplot(1, 2, 2)
    nx.draw(graph, pos, node_color=labels, with_labels=True, cmap=plt.cm.coolwarm, node_size=500)
    plt.title("True Sources")
    label_patch_0 = mpatches.Patch(color=plt.cm.coolwarm(0.0), label='Not Source')
    label_patch_1 = mpatches.Patch(color=plt.cm.coolwarm(1.0), label='Source')
    plt.legend(handles=[label_patch_0, label_patch_1], loc='best')
    

    
    # Show the plots
    plt.tight_layout()
    
    # Save the figure to the specified directory
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, save_name+".png")
    plt.savefig(file_path)
    plt.close()
    print(f"Figure saved to {file_path}")
