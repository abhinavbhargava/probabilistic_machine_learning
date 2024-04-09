import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

def get_edge_index_data_vec(feature_length):
  # Create a mapping from node to index
  node_to_idx = {node: idx for idx, node in enumerate(nodes)}

  #edge index
  source_nodes = []
  target_nodes = []
  for source_node, target_node_list in edges:
      source_idx = node_to_idx[source_node]
      target_indices = [node_to_idx[target_node] for target_node in target_node_list]
      
      source_nodes.extend([source_idx] * len(target_indices))
      target_nodes.extend(target_indices)

  edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

  # data
  node_features = torch.randn(len(nodes), feature_length)  # Node features
  for node in nodes:
    node_idx = node_to_idx[node]
    node_features[node_idx] = get_vertex_feature(node)

  return edge_index, node_features

edge_index_data, node_features = get_edge_index_data_vec()
data = Data(x=node_features, edge_index=edge_index_data).to(device)
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
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = criterion(out, labels_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(G, G.ndata['feat'])
        val_loss = criterion(val_out, labels_val)
        # Optionally, evaluate other metrics on the validation set

    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')
    
