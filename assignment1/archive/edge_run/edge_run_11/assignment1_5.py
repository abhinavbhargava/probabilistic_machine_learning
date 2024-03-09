###################################################
####                  IMPORTS                  ####
###################################################
import os
import pdb
import math
import random
import logging
import UsrIntel.R1
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
    with open("{0}/{1}".format(csv_path, "train.csv"), "rt") as file:
        entries = file.readlines()
    train_edge_list = []
    node_list = []
    for entry in entries:
        current_nodes = [int(item) for item in entry.split(",")]
        node_list += current_nodes
        for node in current_nodes[1:]:
            train_edge_list.append((current_nodes[0], node))
    node_list = sorted(list(set(node_list)))
    with open("{0}/{1}".format(csv_path, "test.csv"), "rt") as file:
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

def extract_negative_samples(node_list, train_edge_list, samples_to_extract):
    logger.info("Extracting All Negative Samples")
    negative_edge_list = []
    negative_node_set = set()
    num_nodes = len(node_list)
    existing_edges = set(train_edge_list)
    sampled_negative_edges = set()
    while len(negative_edge_list) < samples_to_extract:
        i, j = random.sample(range(num_nodes), 2)
        edge = (node_list[i], node_list[j])
        if edge not in existing_edges and edge not in sampled_negative_edges:
            negative_edge_list.append(edge)
            negative_node_set.update(edge)
            sampled_negative_edges.add(edge)
    return negative_edge_list, list(negative_node_set)

if __name__ == "__main__":
    logger.info("CSV Path: {0}".format(os.path.realpath("data")))
    node_list, train_edge_list, test_edge_list, test_node_list = parse_csv(os.path.realpath("data"))
    data_graph = construct_graph(node_list, train_edge_list)
    with open("link_features.csv", "w") as file:
        neighborhood_in = {}
        neighborhood_out = {}
        neighborhood = {}
        for node in node_list:
            neighborhood_in[node] = set(data_graph.predecessors(node))
            neighborhood_out[node] = set(data_graph.successors(node))
            neighborhood[node] = neighborhood_in[node].union(neighborhood_out[node])
        negative_edge_list, negative_node_list = extract_negative_samples(node_list, train_edge_list, 6000000)
        edge_list = train_edge_list + test_edge_list + negative_edge_list
        # edge_list = edge_list[0:1000178]
        # edge_list = edge_list[1000178:2000356]
        # edge_list = edge_list[2000356:3000534]
        # edge_list = edge_list[3000534:4000712]
        # edge_list = edge_list[4000712:5000890]
        # edge_list = edge_list[5000890:6001068]
        # edge_list = edge_list[6001068:7001246]
        # edge_list = edge_list[7001246:8001424]
        # edge_list = edge_list[8001424:9001602]
        # edge_list = edge_list[9001602:10001780]
        # edge_list = edge_list[10001780:11001958]
        edge_list = edge_list[11001958:12002136]
        # edge_list = edge_list[12002136:13002314]
        # edge_list = edge_list[13002314:14002492]
        # edge_list = edge_list[14002492:15002670]
        # edge_list = edge_list[15002670:16002848]
        # edge_list = edge_list[16002848:17003026]
        # edge_list = edge_list[17003026:18003204]
        # edge_list = edge_list[18003204:19003382]
        # edge_list = edge_list[19003382:20003560]
        # edge_list = edge_list[20003560:21003738]
        # edge_list = edge_list[21003738:22003916]
        # edge_list = edge_list[22003916:23004094]
        # edge_list = edge_list[23004094:24004272]
        # edge_list = edge_list[24004272:25004450]
        # edge_list = edge_list[25004450:26004628]
        # edge_list = edge_list[26004628:27004806]
        # edge_list = edge_list[27004806:28004984]
        # edge_list = edge_list[28004984:29005162]
        # edge_list = edge_list[29005162:30005340]
        count = 0
        #pdb.set_trace()
        for pair in edge_list:
            logger.info(count)
            common_friends = len(neighborhood[pair[0]].intersection(neighborhood[pair[1]]))
            common_friends_in = len(neighborhood_in[pair[0]].intersection(neighborhood_in[pair[1]]))
            common_friends_out = len(neighborhood_out[pair[0]].intersection(neighborhood_out[pair[1]]))
            common_friends_bi = len(neighborhood_in[pair[0]].intersection(neighborhood_out[pair[1]]))
            total_friends = len(neighborhood[pair[0]].union(neighborhood[pair[1]]))
            jaccards_coefficient = common_friends / total_friends if total_friends > 0 else 0
            transient_friends = len(neighborhood_out[pair[0]].intersection(neighborhood_in[pair[1]]))
            pas = len(neighborhood[pair[0]]) * len(neighborhood[pair[1]])
            #opposite_direction_friends = 1 if (pair[1], pair[0]) in train_edge_list else 0
            file.write("{0};{1};{2};{3};{4};{5};{6};{7};{8}\n".format(str(pair), str(common_friends), str(common_friends_in), str(common_friends_out), str(common_friends_bi), str(total_friends), str(jaccards_coefficient), str(transient_friends), str(pas)))
            count = count + 1
