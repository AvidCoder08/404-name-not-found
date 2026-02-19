import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, out_channels=2):
        super(GNNModel, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def prepare_graph_data(transactions_df, labels_df=None):
    """
    Converts transactions DataFrame to PyG Data object.
    
    Node Features (heuristic):
    - In-degree, Out-degree
    - Total Amount Sent, Total Amount Received
    - Avg Amount Sent, Avg Amount Received
    """
    
    # 1. Map Account IDs to Integers
    senders = transactions_df['sender_id'].unique()
    receivers = transactions_df['receiver_id'].unique()
    all_accounts = list(set(senders) | set(receivers))
    account_map = {acc: i for i, acc in enumerate(all_accounts)}
    num_nodes = len(all_accounts)
    
    # 2. Build Edge Index
    src = transactions_df['sender_id'].map(account_map).values
    dst = transactions_df['receiver_id'].map(account_map).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    
    # 3. Construct Node Features
    # This is a bit slow in pandas for large graphs, but ok for 10k txs.
    
    # Group by sender
    sent_stats = transactions_df.groupby('sender_id')['amount'].agg(['count', 'sum', 'mean'])
    sent_stats.columns = ['out_degree', 'total_sent', 'avg_sent']
    
    # Group by receiver
    recv_stats = transactions_df.groupby('receiver_id')['amount'].agg(['count', 'sum', 'mean'])
    recv_stats.columns = ['in_degree', 'total_received', 'avg_received']
    
    # Merge into a features DataFrame indexed by account_id
    features = pd.DataFrame(index=all_accounts)
    features = features.join(sent_stats).join(recv_stats).fillna(0)
    
    # Normalize features
    scaler = StandardScaler()
    x_features = scaler.fit_transform(features.values)
    x = torch.tensor(x_features, dtype=torch.float)
    
    # 4. Handle Labels if provided (for training)
    y = None
    if labels_df is not None:
        # labels_df should have 'account_id' and 'is_suspicious'
        # We need to map labels to the node index order
        # Create a series mapped by account index
        
        # Initialize all as 0 (Benign)
        node_labels = np.zeros(num_nodes, dtype=int)
        
        # Set suspicious ones to 1
        suspicious_accounts = labels_df[labels_df['is_suspicious'] == 1]['account_id']
        suspicious_indices = [account_map[acc] for acc in suspicious_accounts if acc in account_map]
        node_labels[suspicious_indices] = 1
        
        y = torch.tensor(node_labels, dtype=torch.long)
        
    
    # 5. Create Train/Val/Test Masks
    # Random split: 80% train, 10% val, 10% test
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * 0.8)
    val_size = int(num_nodes * 0.1)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[indices[train_size:train_size+val_size]] = True
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[indices[train_size+val_size:]] = True
    
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.account_map = account_map # Store map to reverse later
    data.num_nodes = num_nodes
    
    return data

def train_model(data, epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(num_node_features=data.num_features).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Compute class weights to handle severe imbalance (few suspicious vs many benign)
    if data.y is not None:
        labels = data.y[data.train_mask]
        class_counts = torch.bincount(labels, minlength=2).float()
        # Inverse frequency weighting; clamp to avoid div-by-zero
        class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).to(device)
    else:
        class_weights = None

    model.train()
    for __ in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()
        
    return model

def predict(model, data):
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    with torch.no_grad():
        out = model(data)
        probs = F.softmax(out, dim=1)
        # Return probability of class 1 (Suspicious)
        return probs[:, 1].cpu().numpy()
