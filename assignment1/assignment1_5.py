###################################################
####                  IMPORTS                  ####
###################################################
import os
import ast
import pdb
import pickle
import random
import logging
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
###################################################
####                   LOGGER                  ####
###################################################
logger = logging.getLogger("assignment1")
logger.setLevel(logging.DEBUG)
shell_handler = logging.StreamHandler()
shell_formatter = logging.Formatter('%(asctime)s %(filename)s %(levelname)s %(message)s')
shell_handler.setFormatter(shell_formatter)
logger.addHandler(shell_handler)
shell_handler.setLevel(20)
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

def parse_node_features(node_list, data_graph):
    logger.info("Parsing Extracted Node Features")
    if os.path.exists("vertex_features.pickle"):
        with open("vertex_features.pickle", "rb") as file:
            vertex_features = pickle.load(file)
    else:
        if os.path.exists("vertex_features.csv"):
            vertex_features = {}
            with open("vertex_features.csv", "r") as file:
                lines = file.readlines()
                for line in lines:
                    items = [ast.literal_eval(item) for item in line.split(";")]
                    vertex_features[items[0]] = items[1:]
            with open("vertex_features.pickle", "wb") as file:
                pickle.dump(vertex_features, file)
        else:
            with open("vertex_features.csv", "w") as file:
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
            vertex_features = parse_node_features(node_list, data_graph)
    return vertex_features

def parse_link_features(node_list, data_graph, train_edge_list, test_edge_list):
    logger.info("Parsing Extracted Link Features")
    if os.path.exists("link_features.pickle"):
        with open("link_features.pickle", "rb") as file:
            link_features = pickle.load(file)
        # for link in link_features.keys():
        #     if link not in train_edge_list and link not in test_edge_list:
        #         negative_edge_list.append(link)
        negative_edge_list = list(set([key for key in link_features.keys()]) - set(train_edge_list).union(set(test_edge_list)))
        negative_edge_list = list(set([key for key in link_features.keys()]) - set(train_edge_list).union(set(test_edge_list)))
    else:
        if os.path.exists("link_features.csv"):
            link_features = {}
            negative_edge_list = []
            with open("link_features.csv", "r") as file:
                lines = file.readlines()
                for line in lines:
                    items = [ast.literal_eval(item) for item in line.split(";")]
                    link_features[items[0]] = items[1:]
                    # if items[0] not in train_edge_list and items[0] not in test_edge_list:
                    #     negative_edge_list.append(link)
            with open("link_features.pickle", "wb") as file:
                pickle.dump(link_features, file)
            negative_edge_list = list(set([key for key in link_features.keys()]) - set(train_edge_list).union(set(test_edge_list)))
        else:
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
                count = 0
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
            link_features = parse_link_features(node_list, data_graph, train_edge_list, test_edge_list)
    positive_edge_list = random.sample(train_edge_list, round(len(negative_edge_list) * 0.5))
    return link_features, negative_edge_list, positive_edge_list

if __name__ == "__main__":
    logger.info("CSV Path: {0}".format(os.path.realpath("data")))
    node_list, train_edge_list, test_edge_list, test_node_list = parse_csv(os.path.realpath("data"))
    data_graph = construct_graph(node_list, train_edge_list)
    vertex_features = parse_node_features(node_list, data_graph)
    link_features, negative_edge_list, positive_edge_list = parse_link_features(node_list, data_graph, train_edge_list, test_edge_list)
    logger.info("Parsed Node and Link Features")
    features = []
    labels = []
    for edge in positive_edge_list[:2000000]:
        features.append(link_features[edge])
        labels.append(1)
    for edge in negative_edge_list[:4000000]:
        features.append(link_features[edge])
        labels.append(0)
    features = np.array(features)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logistic_model = LogisticRegression(random_state=42, max_iter=10000, C=1.0)
    logistic_model.fit(X_train_scaled, y_train)
    y_pred = logistic_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    test_features = []
    for edge in test_edge_list:
        test_features.append(link_features[edge])
    test_features = np.array(test_features)
    test_features_scaled = scaler.transform(test_features)
    y_pred = logistic_model.predict(test_features_scaled)
    with open("submissions_5.csv", "w") as file:
        count = 1
        file.write("Id,Predictions\n")
        for item in y_pred:
            file.write("{0},{1}\n".format(count, item))
            count = count + 1
