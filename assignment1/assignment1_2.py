###################################################
####                  IMPORTS                  ####
###################################################
import os
import pdb
import math
import random
import logging
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, degree
from itertools import combinations
###################################################
####                  PARSER                   ####
###################################################
parser = argparse.ArgumentParser(description = "Input Parameters")
parser.add_argument('--log_level', help = 'Log Level For Shell Logger.', choices = ["0", "10", "20", "30", "40", "50"], default = "20")
parser.add_argument('--csv_path', help = 'Log Level For Shell Logger.', default = "data")
parser.add_argument('--number_of_samples', help = 'Log Level For Shell Logger.', default = 0, type = int)
args = parser.parse_args()
console_log_level = int(args.log_level)
number_of_samples = args.number_of_samples
###################################################
####                   LOGGER                  ####
###################################################
logger = logging.getLogger("assignment1")
logger.setLevel(logging.DEBUG)
shell_handler = logging.StreamHandler()
shell_formatter = logging.Formatter('%(asctime)s %(filename)s %(levelname)s %(message)s')
shell_handler.setFormatter(shell_formatter)
logger.addHandler(shell_handler)
shell_handler.setLevel(console_log_level)
###################################################
####            FUNCTION DEFINITIONS           ####
###################################################
def parse_csv(csv_path):
    logger.info("Parsing CSV Files for train/test data")
    with open("{0}\\{1}".format(csv_path, "train.csv"), "rt") as file:
        entries = file.readlines()
    train_edge_list = []
    node_list = []
    for entry in entries:
        current_nodes = [int(item) for item in entry.split(",")]
        node_list += current_nodes
        for node in current_nodes[1:]:
            train_edge_list.append((current_nodes[0], node))
    node_list = list(set(node_list))
    with open("{0}\\{1}".format(csv_path, "test.csv"), "rt") as file:
        entries = file.readlines()
    test_edge_list = []
    for entry in entries:
        test_edge_list.append((entry.split(",")[1], entry.split(",")[2]))
    return node_list, train_edge_list, test_edge_list

def construct_graph(node_list, train_edge_list):
    logger.info("Constructing Graph from CSV Data")
    data_graph = nx.DiGraph()
    data_graph.add_nodes_from(node_list)
    data_graph.add_edges_from(train_edge_list)
    return data_graph

def extract_negative_samples(node_list, adjacency_matrix):
    logger.info("Extracting All Negative Samples")
    all_negative_samples = []
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if i != j:
                if adjacency_matrix[i,j] == 0:
                    all_negative_samples.append([node_list[i], node_list[j]])
    return all_negative_samples

def extract_positive_samples(node_list, train_edge_list):
    logger.info("Extracting All Positive Samples")
    all_positive_samples = []
    temp_graph = nx.DiGraph()
    temp_graph.add_nodes_from(node_list)
    temp_graph.add_edges_from(train_edge_list)
    for edge in range(train_edge_list):
        temp_graph.remove_edge(edge)
        if (nx.number_connected_components(temp_graph) == 1):
            all_positive_samples.append(edge)
        temp_graph.add_edge(edge)
    return all_positive_samples

def compute_vertex_features(data):
    logger.info("Computing Vertex Features from Graph")
    """
    Compute neighborhoods, subgraphs, and degrees for each node in a directed graph.

    Parameters:
    - data: PyTorch Geometric Data object representing a directed graph.

    Returns:
    A dictionary containing neighborhood information, subgraphs, and degrees for each node.
    """
    device = data.edge_index.device
    num_nodes = data.num_nodes
    results = {}
    pdb.set_trace()
    for node in range(num_nodes):
        # Compute in-neighborhood and out-neighborhood
        in_neighbors = (data.edge_index[1] == node).nonzero(as_tuple=True)[0]
        out_neighbors = (data.edge_index[0] == node).nonzero(as_tuple=True)[0]
        
        # Convert to actual node indices
        in_neighbor_nodes = data.edge_index[0][in_neighbors]
        out_neighbor_nodes = data.edge_index[1][out_neighbors]
        
        # Compute full neighborhood (union of in and out, excluding duplicates)
        full_neighbors = torch.cat((in_neighbor_nodes, out_neighbor_nodes)).unique()
        
        # Compute node-inclusive neighborhood
        node_inclusive_neighbors = torch.cat((full_neighbors, torch.tensor([node], device=device))).unique()
        
        # # Subgraphs - For simplicity, we'll compute subgraphs for full_neighbors and node_inclusive_neighbors only
        # _, sub_edge_index = subgraph(node_inclusive_neighbors, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        
        # # Degrees
        # in_degree = in_neighbors.size(0)
        # out_degree = out_neighbors.size(0)
        # full_degree = full_neighbors.size(0)
        # node_inclusive_degree = node_inclusive_neighbors.size(0)
        
        results[node] = {
            'in_neighbors': in_neighbor_nodes.tolist(),
            'out_neighbors': out_neighbor_nodes.tolist(),
            'full_neighbors': full_neighbors.tolist(),
            'node_inclusive_neighbors': node_inclusive_neighbors.tolist(),
            # 'subgraph_edge_index': sub_edge_index.cpu().numpy(),
            # 'degrees': {
            #     'in_degree': in_degree,
            #     'out_degree': out_degree,
            #     'full_degree': full_degree,
            #     'node_inclusive_degree': node_inclusive_degree,
            # }
        }
        
    return results

def compute_link_features(node_list, data_graph):
    logger.info("Computing Link Features from Graph")
    common_friends = {}
    total_friends = {}
    friends_measure = {}
    for pair in combinations(node_list, 2):
        pdb.set_trace()
        pass
    return None

logger.info("CSV Path: {0}".format(os.path.realpath(args.csv_path)))
node_list, train_edge_list, test_edge_list = parse_csv(os.path.realpath(args.csv_path))
train_edge_index = torch.tensor(train_edge_list, dtype=torch.long).t().contiguous()
data = Data(edge_index = train_edge_index)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
results = compute_vertex_features(data)
pdb.set_trace()
#data_graph = construct_graph(node_list, train_edge_list)
temp = compute_link_features(node_list, data_graph)
logger.info("Check")
all_negative_samples = extract_negative_samples(node_list, adjacency_matrix)
all_positive_samples = extract_positive_samples(node_list, adjacency_matrix)
