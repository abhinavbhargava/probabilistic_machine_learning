import torch
import os
import pdb
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def parse_csv(csv_path):
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

node_list, train_edge_list, test_edge_list, test_node_list = parse_csv(os.path.realpath("data"))


# Step 1: Data Preparation
with open("vertex_features.pickle", "rb") as file:
    vertex_features = pickle.load(file)
with open("link_features.pickle", "rb") as file:
    link_features = pickle.load(file)
features_list = [vertex_features[node][4:] for node in vertex_features.keys()]
features_tensor = torch.tensor(features_list, dtype=torch.float).to(device)
edge_index_list = list(zip(*train_edge_list))
edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).to(device)
final_edge_index_list = list(zip(*test_edge_list))
final_edge_index_tensor = torch.tensor(final_edge_index_list, dtype=torch.long).to(device)
data = Data(x=features_tensor, edge_index=edge_index_tensor).to(device)


def create_link_examples(edge_index, num_nodes, num_neg_samples=None):
    # Positive examples
    pos_edge_index = edge_index  # Assuming this is already a 2 x E tensor

    # Ensure num_neg_samples is set and reasonable
    num_neg_samples = num_neg_samples or pos_edge_index.size(1)

    # Generate negative examples
    neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples)

    # No need for reshaping pos_edge_index as it should already be in the correct shape
    # But ensure both are tensors and on the same device
    pos_edge_index = pos_edge_index.to(device)
    neg_edge_index = neg_edge_index.to(device)

    # Labels: 1s for positive, 0s for negative
    pos_labels = torch.ones(pos_edge_index.size(1), dtype=torch.float, device=device)
    neg_labels = torch.zeros(neg_edge_index.size(1), dtype=torch.float, device=device)

    # Combine
    examples = torch.cat([pos_edge_index.t(), neg_edge_index.t()], dim=0)  # Transpose to match shapes
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    return examples, labels

examples, labels = create_link_examples(data.edge_index, data.num_nodes)

# Split data for training and testing
examples_train, examples_test, labels_train, labels_test = train_test_split(
    examples.cpu().numpy(), labels.cpu().numpy(), test_size=0.1, random_state=42)
examples_train = torch.tensor(examples_train, dtype=torch.long).to(device)
examples_test = torch.tensor(examples_test, dtype=torch.long).to(device)
labels_train = torch.tensor(labels_train, dtype=torch.float).to(device)
labels_test = torch.tensor(labels_test, dtype=torch.float).to(device)

# Step 2: Model Definition
class LinkPredictionModel(torch.nn.Module):
    def __init__(self, num_features):
        super(LinkPredictionModel, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, examples):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x_i = x[examples[:, 0]]
        x_j = x[examples[:, 1]]
        x_ij = torch.cat([x_i, x_j], dim=1)
        return torch.sigmoid(self.fc(x_ij)).squeeze()

model = LinkPredictionModel(features_tensor.shape[1]).to(device)

# Step 3: Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, examples_train)
    loss = criterion(out, labels_train)
    loss.backward()
    optimizer.step()
    return loss.item()

# Step 4: Evaluation
def evaluate(examples, labels):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, examples)
        return predictions

def predict_probabilities(model, data, final_edge_index_tensor):
    model.eval()
    with torch.no_grad():    
        # If your model definition requires more inputs, adjust this call accordingly
        probabilities = model(data.x, data.edge_index, final_edge_index_tensor)
        #pdb.set_trace()
        # Assuming the model's output is a probability after a sigmoid layer
        # If the output is logits, you would apply a sigmoid here:
        # probabilities = torch.sigmoid(logits)
        labels = (probabilities >= 0.5).long()
    return labels.cpu()

#Training loop
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        train_predictions = evaluate(examples_train, labels_train)
        train_auc = roc_auc_score(labels_train.cpu().numpy(), train_predictions.cpu().numpy())
        test_predictions = evaluate(examples_test, labels_test)
        test_auc = roc_auc_score(labels_test.cpu().numpy(), test_predictions.cpu().numpy())
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}')
torch.save(model, 'gcn1.pth')
#model = torch.load('gcn1.pth')
model.eval()
#pdb.set_trace()
final_labels = predict_probabilities(model, data, final_edge_index_tensor.t()).tolist()
with open("submissions_gcn.csv", "w") as file:
    count = 1
    file.write("Id,Predictions\n")
    for item in final_labels:
        file.write("{0},{1}\n".format(count, item))
        count = count + 1