import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# Example graph creation
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)  # Directed graph

data = Data(edge_index=edge_index)

# Calculate degrees
deg = degree(data.edge_index[0], dtype=torch.long)

print("Node degrees:", deg)

# Convert to NetworkX graph for visualization and easy subgraph creation (for example)
G = to_networkx(data, to_undirected=True)

# Assuming you want a subgraph around node 1
sub_nodes = [0, 1, 2]  # Manually selected for this example; in practice, determine programmatically
subgraph = G.subgraph(sub_nodes)

# Plotting for illustration
nx.draw(subgraph, with_labels=True, node_color='lightblue')
plt.show()
