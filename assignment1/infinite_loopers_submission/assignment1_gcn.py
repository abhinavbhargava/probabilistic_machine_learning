###################################################
####                  IMPORTS                  ####
###################################################
import os
import torch
import pickle
import logging
import networkx as nx
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import train_test_split
###################################################
####                  IMPORTS                  ####
###################################################
logger = logging.getLogger("assignment1")
logger.setLevel(logging.DEBUG)
shell_handler = logging.StreamHandler()
shell_formatter = logging.Formatter('%(asctime)s %(filename)s %(levelname)s %(message)s')
shell_handler.setFormatter(shell_formatter)
logger.addHandler(shell_handler)
shell_handler.setLevel(20)
###################################################
####             MODEL DEFINITIONS             ####
###################################################
class LinkPredictionModel(torch.nn.Module):
    def __init__(self, num_features):
        super(LinkPredictionModel, self).__init__()
        self.conv1 = GCNConv(num_features, 8)
        self.conv2 = GCNConv(8, 16)
        self.conv3 = GCNConv(16, 32)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, samples):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x_i = x[samples[:, 0]]
        x_j = x[samples[:, 1]]
        x_ij = torch.cat([x_i, x_j], dim=1)
        return torch.sigmoid(self.fc(x_ij)).squeeze()

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

def create_link_samples(edge_index, num_nodes, num_neg_samples=None):
    logger.info("Computing Link Samples")
    pos_edge_index = edge_index
    num_neg_samples = num_neg_samples or pos_edge_index.size(1)
    neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples)
    pos_edge_index = pos_edge_index.to(device)
    neg_edge_index = neg_edge_index.to(device)
    pos_labels = torch.ones(pos_edge_index.size(1), dtype=torch.float, device=device)
    neg_labels = torch.zeros(neg_edge_index.size(1), dtype=torch.float, device=device)
    samples = torch.cat([pos_edge_index.t(), neg_edge_index.t()], dim=0)  # Transpose to match shapes
    labels = torch.cat([pos_labels, neg_labels], dim=0)
    return samples, labels

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, samples_train)
    loss = criterion(out, labels_train)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(samples, labels):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, samples)
        return predictions

def predict_probabilities(model, data, test_edge_index_tensor):
    model.eval()
    with torch.no_grad():
        probabilities = model(data.x, data.edge_index, test_edge_index_tensor)
        labels = (probabilities >= 0.5).long()
    return labels.cpu()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("Computing Device: {0}".format(device))
node_list, train_edge_list, test_edge_list, test_node_list = parse_csv(os.path.realpath("data"))
data_graph = construct_graph(node_list, train_edge_list)
vertex_features = parse_node_features(node_list, data_graph)
logger.info("Converting to Tensors")
features_list = [vertex_features[node][4:] for node in vertex_features.keys()]
features_tensor = torch.tensor(features_list, dtype=torch.float).to(device)
edge_index_list = list(zip(*train_edge_list))
edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).to(device)
test_edge_index_list = list(zip(*test_edge_list))
test_edge_index_tensor = torch.tensor(test_edge_index_list, dtype=torch.long).to(device)
data = Data(x=features_tensor, edge_index=edge_index_tensor).to(device)
samples, labels = create_link_samples(data.edge_index, data.num_nodes)
logger.info("Spliting Train/Val")
samples_train, samples_validation, labels_train, labels_validation = train_test_split(samples.cpu().numpy(), labels.cpu().numpy(), test_size=0.1, random_state=42)
samples_train = torch.tensor(samples_train, dtype=torch.long).to(device)
samples_validation = torch.tensor(samples_validation, dtype=torch.long).to(device)
labels_train = torch.tensor(labels_train, dtype=torch.float).to(device)
labels_validation = torch.tensor(labels_validation, dtype=torch.float).to(device)
model = LinkPredictionModel(features_tensor.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()
logger.info("Starting Training Loop")
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        train_predictions = evaluate(samples_train, labels_train)
        train_auc = roc_auc_score(labels_train.cpu().numpy(), train_predictions.cpu().numpy())
        test_predictions = evaluate(samples_validation, labels_validation)
        test_auc = roc_auc_score(labels_validation.cpu().numpy(), test_predictions.cpu().numpy())
        logger.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}')
logger.info("Saving Model")
torch.save(model, 'gcn1.pth')
# model = torch.load('gcn1.pth')
# model.eval()
logger.info("Final Predictions")
test_labels = predict_probabilities(model, data, test_edge_index_tensor.t()).tolist()
with open("submissions_gcn1.csv", "w") as file:
    count = 1
    file.write("Id,Predictions\n")
    for item in test_labels:
        file.write("{0},{1}\n".format(count, item))
        count = count + 1