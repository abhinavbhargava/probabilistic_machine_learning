import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

nodes = set()
edges = []

# Read the file line by line
def read_graph(path):
    try:
        with open(path, 'r') as file:
            count_single_nodes = 0
            i = 0
            for line in file:
                # Split the line into elements
                elements = line.strip().split(',')

                # Extract source node (convert to int)
                source_node = elements[0]
                
                # Extract neighbors (convert to int)
                neighbors = [neighbor for neighbor in elements[1:]] 
                
                nodes.add(source_node)
                nodes.update(source_node)
                
                # Extract neighbors (convert to int)
                edge_list = [(source_node, neighbor) for neighbor in neighbors]
                edges.extend(edge_list)
                
                #if (i < 10):
                    #print(elements[0])
                    #print(edge_list)
                
                #G.add_node(source_node)
                #G.add_nodes_from(neighbors)
                #G.add_edges_from(edge_list)

    except FileNotFoundError:
        print(f"File not found: {path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def generate_random_non_edges(num_non_edges):
    non_edges = []
 
    while num_non_edges > 0:
        # Select two random nodes
        node1, node2 = random.sample(nodes, 2)
        # Check if there is no edge between the selected nodes
        if (node1, node2) not in edges:
            non_edges.append((node1, node2))
            num_non_edges -= 1
 
    return non_edges
 
non_edges = generate_random_non_edges(20000)

def get_edge_index_data_vec(feature_length, simplified):
  # Create a mapping from node to index
  node_to_idx = {node: idx for idx, node in enumerate(nodes)}

  if (simplified):    
    source_nodes = []
    target_nodes = []
    edge_labels = []
    #edge index
    for source_node, target_node, label in edges:
        source_idx = node_to_idx[source_node]
        target_idx = node_to_idx[source_node]
        source_node.append(source_idx)
        target_node.append(target_idx)
        edge_labels.append(label)
                
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    # Create the edge_label tensor
    edge_label = torch.tensor(edge_labels, dtype=torch.float)
    
    # Create a PyTorch Geometric Data object
    data = Data(edge_index=edge_index, edge_attr=edge_label)
  else:
    source_nodes = []
    target_nodes = []
    edge_labels = []
    #edge index
    for source_node, target_node, label in edges:
        source_idx = node_to_idx[source_node]
        target_idx = node_to_idx[source_node]
        source_node.append(source_idx)
        target_node.append(target_idx)
        edge_labels.append(label)
                
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    # Create the edge_label tensor
    edge_label = torch.tensor(edge_labels, dtype=torch.float)

    node_features = torch.randn(len(nodes), feature_length)  # Node features
    for node in nodes:
        node_idx = node_to_idx[node]
        node_features[node_idx] = get_vertex_feature(node)
    
    # Create a PyTorch Geometric Data object
    data = Data(X=node_features, edge_index=edge_index, edge_attr=edge_label)
      
  return data

data = get_edge_index_data_vec()
train_data_directed, val_data_directed, labels_train, labels_val  = train_test_split(data, ground_truth_labels, test_size=0.2, random_state=42)


class GCNModelWithRegularization(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_prob=0.5, weight_decay=0.0):
        super(GCNModelWithRegularization, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
model = GCNModelWithRegularization(feature_length, hidden_channels=64, out_channels=2, dropout_prob=0.5, weight_decay=0.0001)
# Set up other training-related components (optimizer, criterion, etc.)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training loop
for epoch in range(200):
    # Forward pass
    output = model(data)

    # Compute loss
    edge_label = data.edge_attr
    loss = criterion(output.flatten(), edge_label)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

model.eval
