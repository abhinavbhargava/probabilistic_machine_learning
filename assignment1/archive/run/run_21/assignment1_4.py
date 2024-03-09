###################################################
####                  IMPORTS                  ####
###################################################
import os
import pdb
import math
import random
import logging
import numpy as np
import networkx as nx
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
###################################################
####                  PARSER                   ####
###################################################
console_log_level = 20
number_of_samples = 1000
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
    node_list = sorted(list(set(node_list)))
    with open("{0}\\{1}".format(csv_path, "test.csv"), "rt") as file:
        entries = file.readlines()[1:]
    test_edge_list = []
    for entry in entries:
        test_edge_list.append((int(entry.split(",")[1]), int(entry.split(",")[2].split("\n")[0])))
    test_node_list = []
    for edge in test_edge_list:
        for node in edge:
            test_node_list.append(node)
    test_node_list = list(set(test_node_list))
    return node_list, train_edge_list, test_edge_list, test_node_list

def construct_graph(node_list, train_edge_list):
    logger.info("Constructing Graph from CSV Data")
    data_graph = nx.DiGraph()
    data_graph.add_nodes_from(node_list)
    data_graph.add_edges_from(train_edge_list)
    return data_graph

if __name__ == "__main__":
    logger.info("CSV Path: {0}".format(os.path.realpath("data")))
    node_list, train_edge_list, test_edge_list, test_node_list = parse_csv(os.path.realpath("data"))
    data_graph = construct_graph(node_list, train_edge_list)
    with open("vertex_features.csv", "w") as file:
        # node_list = node_list[0:187198]
        # node_list = node_list[187198:374396]
        # node_list = node_list[374396:561594]
        # node_list = node_list[561594:748792]
        # node_list = node_list[748792:935990]
        # node_list = node_list[935990:1123188]
        # node_list = node_list[1123188:1310386]
        # node_list = node_list[1310386:1497584]
        # node_list = node_list[1497584:1684782]
        # node_list = node_list[1684782:1871980]
        # node_list = node_list[1871980:2059178]
        # node_list = node_list[2059178:2246376]
        # node_list = node_list[2246376:2433574]
        # node_list = node_list[2433574:2620772]
        # node_list = node_list[2620772:2807970]
        # node_list = node_list[2807970:2995168]
        # node_list = node_list[2995168:3182366]
        # node_list = node_list[3182366:3369564]
        # node_list = node_list[3369564:3556762]
        # node_list = node_list[3556762:3743960]
        # node_list = node_list[3743960:3931158]
        node_list = node_list[3931158:4118356]
        # node_list = node_list[4118356:4305554]
        # node_list = node_list[4305554:4492752]
        # node_list = node_list[4492752:4679950]
        # node_list = node_list[4679950:]
        for node in node_list:
            logger.info(node)
            neighborhood_in = list(data_graph.predecessors(node))
            neighborhood_out = list(data_graph.successors(node))
            neighborhood = set(neighborhood_in + neighborhood_out)
            neighborhood_inclusive = neighborhood.union({node})
            degree = len(neighborhood)
            in_degree = len(neighborhood_in)
            out_degree = len(neighborhood_out)
            bi_degree = len(set(neighborhood_in).intersection(set(neighborhood_out)))
            in_degree_densities = in_degree / len(neighborhood)
            out_degree_densities = out_degree / len(neighborhood)
            bi_degree_densities = bi_degree / len(neighborhood)
            neighborhood_inclusive_sub_graphs_link_number = sum(1 for n1, n2 in data_graph.edges(neighborhood_inclusive) if n1 in neighborhood_inclusive and n2 in neighborhood_inclusive)
            neighborhood_sub_graphs_link_number = sum(1 for n1, n2 in data_graph.edges(neighborhood) if n1 in neighborhood and n2 in neighborhood)
            try:
                neighborhood_sub_graphs_densities = degree / neighborhood_sub_graphs_link_number
                neighborhood_inclusive_sub_graphs_densities = degree / neighborhood_inclusive_sub_graphs_link_number
            except:
                neighborhood_sub_graphs_densities = 0
                neighborhood_inclusive_sub_graphs_densities = 0
            try:
                avg_scc = degree / nx.number_strongly_connected_components(neighborhood_sub_graphs)
            except:
                avg_scc = 0
            try:
                avg_wcc = degree / nx.number_weakly_connected_components(neighborhood_sub_graphs)
            except:
                avg_wcc = 0
            try:
                avg_scc_inclusive = degree / nx.number_strongly_connected_components(neighborhood_inclusive_sub_graphs)
            except:
                avg_scc_inclusive = 0
            file.write("{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11};{12};{13};{14};{15};{16};{17};{18}\n".format(str(node), str(neighborhood_in), str(neighborhood_out), str(neighborhood), str(neighborhood_inclusive), str(degree), str(in_degree), str(out_degree), str(bi_degree), str(in_degree_densities), str(out_degree_densities), str(bi_degree_densities), str(neighborhood_inclusive_sub_graphs_link_number), str(neighborhood_sub_graphs_link_number), str(neighborhood_sub_graphs_densities), str(neighborhood_inclusive_sub_graphs_densities), str(avg_scc), str(avg_wcc), str(avg_scc_inclusive)))
