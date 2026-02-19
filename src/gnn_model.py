import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import os


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gnn_model.pt")
METADATA_PATH = os.path.join(MODEL_DIR, "gnn_metadata.json")


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


def _compute_node_features(transactions_df, all_accounts):
    """Compute the 6 node features from transaction data."""
    # Group by sender
    sent_stats = transactions_df.groupby('sender_id')['amount'].agg(['count', 'sum', 'mean'])
    sent_stats.columns = ['out_degree', 'total_sent', 'avg_sent']

    # Group by receiver
    recv_stats = transactions_df.groupby('receiver_id')['amount'].agg(['count', 'sum', 'mean'])
    recv_stats.columns = ['in_degree', 'total_received', 'avg_received']

    # Merge into a features DataFrame indexed by account_id
    features = pd.DataFrame(index=all_accounts)
    features = features.join(sent_stats).join(recv_stats).fillna(0)

    return features


def prepare_graph_data(transactions_df, labels_df=None):
    """
    Converts transactions DataFrame to PyG Data object for TRAINING.
    Fits a new StandardScaler on the data.
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
    features = _compute_node_features(transactions_df, all_accounts)

    # Normalize features — fit a NEW scaler
    scaler = StandardScaler()
    x_features = scaler.fit_transform(features.values)
    x = torch.tensor(x_features, dtype=torch.float)

    # 4. Handle Labels if provided (for training)
    y = None
    if labels_df is not None:
        node_labels = np.zeros(num_nodes, dtype=int)
        suspicious_accounts = labels_df[labels_df['is_suspicious'] == 1]['account_id']
        suspicious_indices = [account_map[acc] for acc in suspicious_accounts if acc in account_map]
        node_labels[suspicious_indices] = 1
        y = torch.tensor(node_labels, dtype=torch.long)

    # 5. Create Train/Val/Test Masks (80/10/10)
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * 0.8)
    val_size = int(num_nodes * 0.1)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[indices[train_size:train_size + val_size]] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[indices[train_size + val_size:]] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.account_map = account_map
    data.num_nodes = num_nodes
    data.scaler_mean = scaler.mean_.tolist()
    data.scaler_scale = scaler.scale_.tolist()

    return data


def prepare_inference_data(transactions_df, scaler_mean, scaler_scale):
    """
    Converts transactions DataFrame to PyG Data object for INFERENCE.
    Uses the saved scaler params from training (prevents data leakage).
    """
    senders = transactions_df['sender_id'].unique()
    receivers = transactions_df['receiver_id'].unique()
    all_accounts = list(set(senders) | set(receivers))
    account_map = {acc: i for i, acc in enumerate(all_accounts)}
    num_nodes = len(all_accounts)

    # Build Edge Index
    src = transactions_df['sender_id'].map(account_map).values
    dst = transactions_df['receiver_id'].map(account_map).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    # Compute features
    features = _compute_node_features(transactions_df, all_accounts)

    # Normalize using SAVED scaler params (not fitting new ones)
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_mean)
    scaler.scale_ = np.array(scaler_scale)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler_mean)
    x_features = scaler.transform(features.values)
    x = torch.tensor(x_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.account_map = account_map
    data.num_nodes = num_nodes

    return data


def train_model(data, epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(num_node_features=data.num_features).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Compute class weights to handle severe imbalance
    if data.y is not None:
        labels = data.y[data.train_mask]
        class_counts = torch.bincount(labels, minlength=2).float()
        class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).to(device)
    else:
        class_weights = None

    model.train()
    for epoch in range(epochs):
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
        return probs[:, 1].cpu().numpy()


def save_model(model, data):
    """Save trained model weights and metadata for later inference."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_node_features': data.num_features,
    }, MODEL_PATH)

    # Save metadata (scaler params for consistent normalization)
    metadata = {
        'scaler_mean': data.scaler_mean,
        'scaler_scale': data.scaler_scale,
        'num_node_features': int(data.num_features),
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_model():
    """Load a pre-trained model and metadata. Returns (model, metadata) or None."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return None

    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(num_node_features=metadata['num_node_features']).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, metadata


def model_exists():
    """Check if a pre-trained model exists."""
    return os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH)
