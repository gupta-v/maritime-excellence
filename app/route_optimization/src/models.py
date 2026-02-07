# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GNNImitator(nn.Module):
    """Enhanced GNN with attention and residual connections"""
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.input_layer = nn.Linear(in_channels, hidden_dim)
        self.dropout = nn.Dropout(0.1)

        # Use GAT (Graph Attention) with fallback to GCN
        try:
            self.conv1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=0.1)
            self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=0.1)
            self.use_gat = True
        except:
            print("GAT not available, using GCN")
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.use_gat = False

        # Residual connection
        self.residual = nn.Linear(hidden_dim, hidden_dim)

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, out_channels)
        )

    def forward(self, x, current_node_idx, edge_index):
        # Input processing
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)

        # Store for residual connection
        residual = self.residual(x)

        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        if self.use_gat:
            x = F.relu(self.conv2(x, edge_index))
        else:
            x = F.relu(self.conv2(x, edge_index))

        # Add residual connection
        x = x + residual

        # Get current node embedding
        if isinstance(current_node_idx, torch.Tensor):
            if current_node_idx.dim() == 0:  # Scalar tensor
                current_node_embedding = x[current_node_idx.item()]
            else:  # Batch of indices
                current_node_embedding = x[current_node_idx]
        else:
            current_node_embedding = x[current_node_idx]

        return self.output_layer(current_node_embedding)


class GNN_QNetwork(nn.Module):
    """Enhanced Q-Network with better architecture"""
    def __init__(self, in_channels, hidden_dim, max_actions):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_actions = max_actions

        # Input processing
        self.input_layer = nn.Linear(in_channels, hidden_dim)
        self.dropout = nn.Dropout(0.1)

        # Graph layers with fallback
        try:
            self.conv1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=0.1)
            self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=0.1)
            self.use_gat = True
        except:
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.use_gat = False

        # State representation
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Action value computation
        self.action_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_actions)
        )

    def forward(self, x, edge_index, current_node_idx=None):
        # Process node features
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)

        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # State encoding
        x = self.state_encoder(x)

        if current_node_idx is not None:
            # Return Q-values for specific node
            if isinstance(current_node_idx, torch.Tensor):
                current_state = x[current_node_idx.item()]
            else:
                current_state = x[current_node_idx]
            return self.action_value(current_state)
        else:
            # Return Q-values for all nodes
            return self.action_value(x)
