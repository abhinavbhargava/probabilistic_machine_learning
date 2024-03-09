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
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import multiprocessing
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

def runner_compute_vertex_features(arguments):
    node, data_graph = arguments
    logger.info("Computing Vertex Features for Node {0}".format(node))
    neighborhood_in = set(data_graph.predecessors(node))
    neighborhood_out = set(data_graph.successors(node))
    neighborhood = neighborhood_in.union(neighborhood_out)
    neighborhood_inclusive = neighborhood.union({node})
    degree = len(neighborhood)
    in_degree = len(neighborhood_in)
    out_degree = len(neighborhood_out)
    bi_degree = len(neighborhood_in.intersection(neighborhood_out))
    if len(neighborhood) > 0:
        in_degree_densities = in_degree / degree
        out_degree_densities = out_degree / degree
        bi_degree_densities = bi_degree / degree
    else:
        in_degree_densities = out_degree_densities = bi_degree_densities = 0
    return [node, neighborhood_in, neighborhood_out, neighborhood, neighborhood_inclusive, degree, in_degree, out_degree, bi_degree, in_degree_densities, out_degree_densities, bi_degree_densities]

def compute_vertex_features(node_list, data_graph):
    logger.info("Computing Vertex Features from Graph")
    neighborhood = {}
    neighborhood_in = {}
    neighborhood_out = {}
    neighborhood_inclusive = {}
    degree = {}
    in_degree = {}
    out_degree = {}
    bi_degree = {}
    in_degree_densities = {}
    out_degree_densities = {}
    bi_degree_densities = {}
    command_list = []
    for node in node_list:
        command_list.append([node, data_graph])
    logger.info("Multiprocessing Started")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(runner_compute_vertex_features, command_list)
    logger.info("Multiprocessing Finished")
    for result in results:
        neighborhood_in[result[0]] = result[1]
        neighborhood_out[result[0]] = result[2]
        neighborhood[result[0]] = result[3]
        neighborhood_inclusive[result[0]] = result[4]
        degree[result[0]] = result[5]
        in_degree[result[0]] = result[6]
        out_degree[result[0]] = result[7]
        bi_degree[result[0]] = result[8]
        in_degree_densities[result[0]] = result[9]
        out_degree_densities[result[0]] = result[10]
        bi_degree_densities[result[0]] = result[11]
    return neighborhood, neighborhood_in, neighborhood_out, neighborhood_inclusive, in_degree, out_degree, bi_degree, in_degree_densities, out_degree_densities, bi_degree_densities

def extract_positive_samples(data_graph, train_edge_list, samples_to_extract):
    logger.info("Extracting All Positive Samples")
    positive_edge_list = []
    positive_node_set = set()
    sampled_edges = random.sample(train_edge_list, samples_to_extract)    
    for edge in sampled_edges:
        positive_edge_list.append(edge)
        positive_node_set.update(edge)
        
    return positive_edge_list, list(positive_node_set)

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

def extract_negative_samples(node_list, adjacency_matrix, samples_to_extract):
    logger.info("Extracting All Negative Samples")
    negative_edge_list = []
    negative_node_set = set()
    available_pairs = [(i, j) for i in range(adjacency_matrix.shape[0]) for j in range(i + 1, adjacency_matrix.shape[1]) if adjacency_matrix[i, j] == 0]
    if len(available_pairs) > samples_to_extract:
        sampled_pairs = random.sample(available_pairs, samples_to_extract)
    else:
        sampled_pairs = available_pairs
    for i, j in sampled_pairs:
        negative_edge_list.append((node_list[i], node_list[j]))
        negative_node_set.update([node_list[i], node_list[j]])
    return negative_edge_list, list(negative_node_set)

def compute_complex_vertex_features(node_list, data_graph, neighborhood, neighborhood_in, neighborhood_out, neighborhood_inclusive):
    logger.info("Computing Complex Vertex Features from Graph")
    neighborhood_sub_graphs = {}
    neighborhood_inclusive_sub_graphs = {}
    neighborhood_sub_graphs_link_number = {}
    neighborhood_inclusive_sub_graphs_link_number = {}
    neighborhood_sub_graphs_densities = {}
    neighborhood_inclusive_sub_graphs_densities = {}
    avg_scc = {}
    avg_wcc = {}
    avg_scc_inclusive = {}
    count = 0
    logger.info("Total Complex Node Feature Count Computations: {0}".format(len(node_list)))
    for node in node_list:
        if count % 100 == 0:
            logger.debug("Current Link Feature Count Computations: {0}".format(count))
        neighborhood_inclusive_sub_graphs[node] = data_graph.subgraph(neighborhood_inclusive[node])
        neighborhood_sub_graphs[node] = neighborhood_inclusive_sub_graphs[node].subgraph(neighborhood[node])
        neighborhood_sub_graphs_link_number[node] = len(neighborhood_sub_graphs[node].edges())
        neighborhood_inclusive_sub_graphs_link_number[node] = len(neighborhood_inclusive_sub_graphs[node].edges())
        try:
            neighborhood_sub_graphs_densities[node] = len(neighborhood[node]) / neighborhood_sub_graphs_link_number[node]
            neighborhood_inclusive_sub_graphs_densities[node] = len(neighborhood[node]) / neighborhood_inclusive_sub_graphs_link_number[node]
        except:
            neighborhood_sub_graphs_densities[node] = 0
            neighborhood_inclusive_sub_graphs_densities[node] = 0
        try:
            avg_scc[node] = len(neighborhood[node]) / nx.number_strongly_connected_components(neighborhood_sub_graphs[node])
        except:
            avg_scc[node] = 0
        try:
            avg_wcc[node] = len(neighborhood[node]) / nx.number_weakly_connected_components(neighborhood_sub_graphs[node])
        except:
            avg_scc[node] = 0
        try:
            avg_scc_inclusive[node] = len(neighborhood[node]) / nx.number_strongly_connected_components(neighborhood_inclusive_sub_graphs[node])
        except:
            avg_scc[node] = 0
        count = count + 1
    return neighborhood_sub_graphs, neighborhood_inclusive_sub_graphs, neighborhood_sub_graphs_link_number, neighborhood_inclusive_sub_graphs_link_number, neighborhood_sub_graphs_densities, neighborhood_inclusive_sub_graphs_densities, avg_scc, avg_wcc, avg_scc_inclusive

def compute_link_features(sample_edge_list, train_edge_list, neighborhood, neighborhood_in, neighborhood_out):
    logger.info("Computing Link Features from Graph")
    common_friends = {}
    common_friends_in = {}
    common_friends_out = {}
    common_friends_bi = {}
    total_friends = {}
    jaccards_coefficient = {}
    transient_friends = {}
    pas = {}
    friends_measure = {}
    opposite_direction_friends = {}
    count = 0
    logger.info("Total Link Feature Count Computations: {0}".format(len(sample_edge_list)))
    for pair in sample_edge_list:
        if count % 100 == 0:
            logger.debug("Current Link Feature Count Computations: {0}".format(count))
        common_friends[pair] = len(neighborhood[pair[0]]).intersection(neighborhood[pair[1]])
        common_friends_in[pair] = len(neighborhood_in[pair[0]]).intersection(neighborhood_in[pair[1]])
        common_friends_out[pair] = len(neighborhood_out[pair[0]]).intersection(neighborhood_out[pair[1]])
        common_friends_bi[pair] = len(neighborhood_in[pair[0]]).intersection(neighborhood_out[pair[1]])
        total_friends[pair] = len(neighborhood[pair[0]]).union(neighborhood[pair[1]])
        jaccards_coefficient[pair] = common_friends[pair] / total_friends[pair] if total_friends[pair] > 0 else 0
        transient_friends[pair] = len(neighborhood_out[pair[0]]).intersection(neighborhood_in[pair[1]])
        pas[pair] = len(neighborhood[pair[0]]) * len(neighborhood[pair[1]])
        # friends_measure[pair] = 0
        # for node_u in neighborhood[pair[0]]:
        #   for node_v in neighborhood[pair[1]]:
        #       if node_u == node_v or (node_u, node_v) in train_edge_list or (node_v, node_u) in train_edge_list:
        #           friends_measure[pair] += 1
        opposite_direction_friends[pair] = 1 if (pair[1], pair[0]) in train_edge_list else 0
        count = count + 1
    return common_friends, common_friends_in, common_friends_out, common_friends_bi, total_friends, jaccards_coefficient, transient_friends, pas, friends_measure, opposite_direction_friends

if __name__ == "__main__":
    logger.info("CSV Path: {0}".format(os.path.realpath(args.csv_path)))
    node_list, train_edge_list, test_edge_list, test_node_list = parse_csv(os.path.realpath(args.csv_path))
    data_graph = construct_graph(node_list, train_edge_list)
    neighborhood, neighborhood_in, neighborhood_out, neighborhood_inclusive, in_degree, out_degree, bi_degree, in_degree_densities, out_degree_densities, bi_degree_densities = compute_vertex_features(node_list, data_graph)
    positive_edge_list, positive_node_list = extract_positive_samples(data_graph, train_edge_list, number_of_samples)
    negative_edge_list, negative_node_list = extract_negative_samples(node_list, train_edge_list, number_of_samples)
    common_friends, common_friends_in, common_friends_out, common_friends_bi, total_friends, jaccards_coefficient, transient_friends, pas, friends_measure, opposite_direction_friends = compute_link_features(positive_edge_list + negative_edge_list, train_edge_list, neighborhood, neighborhood_in, neighborhood_out)
    neighborhood_sub_graphs, neighborhood_inclusive_sub_graphs, neighborhood_sub_graphs_link_number, neighborhood_inclusive_sub_graphs_link_number, neighborhood_sub_graphs_densities, neighborhood_inclusive_sub_graphs_densities, avg_scc, avg_wcc, avg_scc_inclusive = compute_complex_vertex_features(list(set(positive_node_list + negative_node_list + test_node_list)), data_graph, neighborhood, neighborhood_in, neighborhood_out, neighborhood_inclusive)
    features = []
    labels = []
    for edge in positive_edge_list:
        edge0_features = [in_degree[edge[0]], out_degree[edge[0]], bi_degree[edge[0]], in_degree_densities[edge[0]], out_degree_densities[edge[0]], bi_degree_densities[edge[0]], neighborhood_sub_graphs_link_number[edge[0]], neighborhood_inclusive_sub_graphs_link_number[edge[0]], neighborhood_sub_graphs_densities[edge[0]], neighborhood_inclusive_sub_graphs_densities[edge[0]], avg_scc[edge[0]], avg_wcc[edge[0]], avg_scc_inclusive[edge[0]]]
        edge1_features = [in_degree[edge[1]], out_degree[edge[1]], bi_degree[edge[1]], in_degree_densities[edge[1]], out_degree_densities[edge[1]], bi_degree_densities[edge[1]], neighborhood_sub_graphs_link_number[edge[1]], neighborhood_inclusive_sub_graphs_link_number[edge[1]], neighborhood_sub_graphs_densities[edge[1]], neighborhood_inclusive_sub_graphs_densities[edge[1]], avg_scc[edge[1]], avg_wcc[edge[1]], avg_scc_inclusive[edge[1]]]
        link_features = [common_friends[edge], common_friends_in[edge], common_friends_out[edge], common_friends_bi[edge], total_friends[edge], jaccards_coefficient[edge], transient_friends[edge], pas[edge], opposite_direction_friends[edge]]
        features.append(edge0_features + edge1_features + link_features)
        labels.append(1)
    for edge in negative_edge_list:
        edge0_features = [in_degree[edge[0]], out_degree[edge[0]], bi_degree[edge[0]], in_degree_densities[edge[0]], out_degree_densities[edge[0]], bi_degree_densities[edge[0]], neighborhood_sub_graphs_link_number[edge[0]], neighborhood_inclusive_sub_graphs_link_number[edge[0]], neighborhood_sub_graphs_densities[edge[0]], neighborhood_inclusive_sub_graphs_densities[edge[0]], avg_scc[edge[0]], avg_wcc[edge[0]], avg_scc_inclusive[edge[0]]]
        edge1_features = [in_degree[edge[1]], out_degree[edge[1]], bi_degree[edge[1]], in_degree_densities[edge[1]], out_degree_densities[edge[1]], bi_degree_densities[edge[1]], neighborhood_sub_graphs_link_number[edge[1]], neighborhood_inclusive_sub_graphs_link_number[edge[1]], neighborhood_sub_graphs_densities[edge[1]], neighborhood_inclusive_sub_graphs_densities[edge[1]], avg_scc[edge[1]], avg_wcc[edge[1]], avg_scc_inclusive[edge[1]]]
        link_features = [common_friends[edge], common_friends_in[edge], common_friends_out[edge], common_friends_bi[edge], total_friends[edge], jaccards_coefficient[edge], transient_friends[edge], pas[edge], opposite_direction_friends[edge]]
        features.append(edge0_features + edge1_features + link_features)
        labels.append(0)
    features = np.array(features)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)
    pdb.set_trace()
    bagging_classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    voting_clf = VotingClassifier(estimators=[('bagging', bagging_classifier), ('rf', rf_classifier)], voting='soft')
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    common_friends, common_friends_in, common_friends_out, common_friends_bi, total_friends, jaccards_coefficient, transient_friends, pas, friends_measure, opposite_direction_friends = compute_link_features(test_edge_list, train_edge_list, neighborhood, neighborhood_in, neighborhood_out)
    test_features = []
    for edge in test_edge_list:
        edge0_features = [in_degree[edge[0]], out_degree[edge[0]], bi_degree[edge[0]], in_degree_densities[edge[0]], out_degree_densities[edge[0]], bi_degree_densities[edge[0]], neighborhood_sub_graphs_link_number[edge[0]], neighborhood_inclusive_sub_graphs_link_number[edge[0]], neighborhood_sub_graphs_densities[edge[0]], neighborhood_inclusive_sub_graphs_densities[edge[0]], avg_scc[edge[0]], avg_wcc[edge[0]], avg_scc_inclusive[edge[0]]]
        edge1_features = [in_degree[edge[1]], out_degree[edge[1]], bi_degree[edge[1]], in_degree_densities[edge[1]], out_degree_densities[edge[1]], bi_degree_densities[edge[1]], neighborhood_sub_graphs_link_number[edge[1]], neighborhood_inclusive_sub_graphs_link_number[edge[1]], neighborhood_sub_graphs_densities[edge[1]], neighborhood_inclusive_sub_graphs_densities[edge[1]], avg_scc[edge[1]], avg_wcc[edge[1]], avg_scc_inclusive[edge[1]]]
        link_features = [common_friends[edge], common_friends_in[edge], common_friends_out[edge], common_friends_bi[edge], total_friends[edge], jaccards_coefficient[edge], transient_friends[edge], pas[edge], opposite_direction_friends[edge]]
        test_features.append(edge0_features + edge1_features + link_features)
    test_features = np.array(test_features)
    y_pred = rf.predict(test_features)
    with open("submissions.csv", "w") as file:
        count = 1
        file.write("Id,Predictions\n")
        for item in y_pred:
            file.write("{0},{1}\n".format(count, item))
            count = count + 1
