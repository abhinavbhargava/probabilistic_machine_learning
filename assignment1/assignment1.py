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
import networkx as nx
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
shell_formatter = logging.Formatter('%(filename)s %(levelname)s %(message)s')
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

def compute_adjacency_matrix(node_list, train_edge_list):
	logger.info("Computing Adjacency Matrix")
	data_graph = nx.DiGraph()
	data_graph.add_nodes_from(node_list)
	data_graph.add_edges_from(train_edge_list)
	adjacency_matrix = nx.adjacency_matrix(data_graph)
	return data_graph, adjacency_matrix

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
		if (nx.number_connected_components(G_temp) == 1):
			all_positive_samples.append(edge)
		temp_graph.add_edge(edge)
	return all_positive_samples

logger.info("CSV Path: {0}".format(os.path.realpath(args.csv_path)))
node_list, train_edge_list, test_edge_list = parse_csv(os.path.realpath(args.csv_path))
data_graph, adjacency_matrix = compute_adjacency_matrix(node_list, train_edge_list)
pdb.set_trace()
all_negative_samples = extract_negative_samples(node_list, adjacency_matrix)
all_positive_samples = extract_positive_samples(node_list, adjacency_matrix)
