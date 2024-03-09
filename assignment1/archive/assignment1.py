###################################################
####                  IMPORTS                  ####
###################################################
import os
import pdb
import logging
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.utils as utils
from torch_geometric.utils import subgraph
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

logger.info("CSV Path: {0}".format(os.path.realpath(args.csv_path)))
node_list, train_edge_list, test_edge_list = parse_csv(os.path.realpath(args.csv_path))
pdb.set_trace()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_index = torch.tensor(train_edge_list, dtype=torch.long).t().contiguous()
edge_index_coo = torch.stack((edge_index[0], edge_index[1]))
num_nodes = edge_index.max().item() + 1
adj_matrix = torch.sparse_coo_tensor(edge_index_coo, torch.ones(edge_index.size(1)), size=(num_nodes, num_nodes))
#adj_matrix = adj_matrix.to(device)
neighborhood = utils.to_dense_adj(adj_matrix).bool()
in_neighborhood = torch.matmul(neighborhood.transpose(1, 2), torch.eye(num_nodes).bool().to(neighborhood.device))
out_neighborhood = torch.matmul(neighborhood, torch.eye(num_nodes).bool().to(neighborhood.device))
node_inclusive_neighborhood = in_neighborhood | out_neighborhood