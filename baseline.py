import os
import csv
import argparse
from GraphSL.GNN.SLVAE.main import SLVAE
from GraphSL.GNN.IVGD.main import IVGD
from GraphSL.GNN.GCNSI.main import GCNSI
from GraphSL.Prescribed import LPSI, NetSleuth, OJC
from GraphSL.utils import load_dataset, diffusion_generation, split_dataset, download_dataset, visualize_source_prediction

# Set up argument parser
parser = argparse.ArgumentParser(description="Run graph source localization algorithms on a dataset.")

# Define arguments
parser.add_argument('--data_name', type=str, required=True, help="Name of the dataset (e.g., 'karate', 'dolphins', 'jazz', 'netscience', 'cora_ml', 'power_grid').")

# Parse the arguments
args = parser.parse_args()

data_name = args.data_name

curr_dir = os.getcwd()

# Download datasets
download_dataset(curr_dir)

# Load datasets
graph = load_dataset(data_name, data_dir=curr_dir)

# Prepare CSV file to store the metrics
output_file = f"{data_name}_metrics.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Random Seed", "Method", "Test ACC", "Test F1", "Test AUC"])

    for random_seed in range(10):
        print(f"Running with random seed {random_seed}...")
        
        # Generate diffusion
        dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1, random_seed=random_seed)

        # Split into training and test sets
        adj, train_dataset, test_dataset = split_dataset(dataset)

        # LPSI
        print("LPSI:")
        lpsi = LPSI()
        alpha, thres, auc, f1, pred = lpsi.train(adj, train_dataset)
        metric = lpsi.test(adj, test_dataset, alpha, thres)
        writer.writerow([random_seed, "LPSI", f"{metric.acc:.3f}", f"{metric.f1:.3f}", f"{metric.auc:.3f}"])

        # NetSleuth
        print("NetSleuth:")
        netSleuth = NetSleuth()
        k, auc, f1 = netSleuth.train(adj, train_dataset)
        metric = netSleuth.test(adj, test_dataset, k)
        writer.writerow([random_seed, "NetSleuth", f"{metric.acc:.3f}", f"{metric.f1:.3f}", f"{metric.auc:.3f}"])

        # OJC
        print("OJC:")
        ojc = OJC()
        Y, auc, f1 = ojc.train(adj, train_dataset)
        metric = ojc.test(adj, test_dataset, Y)
        writer.writerow([random_seed, "OJC", f"{metric.acc:.3f}", f"{metric.f1:.3f}", f"{metric.auc:.3f}"])

        # GCNSI
        print("GCNSI:")
        gcnsi = GCNSI()
        gcnsi_model, thres, auc, f1, pred = gcnsi.train(adj, train_dataset)
        metric = gcnsi.test(adj, test_dataset, gcnsi_model, thres)
        writer.writerow([random_seed, "GCNSI", f"{metric.acc:.3f}", f"{metric.f1:.3f}", f"{metric.auc:.3f}"])

        # IVGD
        print("IVGD:")
        ivgd = IVGD()
        diffusion_model = ivgd.train_diffusion(adj, train_dataset)
        ivgd_model, thres, auc, f1, pred = ivgd.train(adj, train_dataset, diffusion_model)
        metric = ivgd.test(adj, test_dataset, diffusion_model, ivgd_model, thres)
        writer.writerow([random_seed, "IVGD", f"{metric.acc:.3f}", f"{metric.f1:.3f}", f"{metric.auc:.3f}"])

        # SLVAE
        print("SLVAE:")
        slave = SLVAE()
        slvae_model, seed_vae_train, thres, auc, f1, pred = slave.train(adj, train_dataset)
        metric = slave.infer(test_dataset, slvae_model, seed_vae_train, thres)
        writer.writerow([random_seed, "SLVAE", f"{metric.acc:.3f}", f"{metric.f1:.3f}", f"{metric.auc:.3f}"])

print(f"Metrics saved to {output_file}")
