import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gnn_model.pt")
METADATA_PATH = os.path.join(MODEL_DIR, "gnn_metadata.json")

# ------------------------------------------------------------------
# 1. Advanced Node Features (14 features)
# ------------------------------------------------------------------
def _compute_node_features(transactions_df, all_accounts):
    """
    Compute 14 advanced node features to distinguish fraud cycles/smurfing
    from legitimate merchants/payroll.
    """
    df = transactions_df.copy()
    
    # Ensure timestamp
    if 'timestamp' not in df.columns and 'Date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Basic Stats (Incoming)
    in_stats = df.groupby('receiver_id')['amount'].agg(['count', 'sum', 'mean', 'std']).fillna(0)
    in_stats.columns = ['in_degree', 'total_received', 'avg_received', 'std_received']
    
    # Basic Stats (Outgoing)
    out_stats = df.groupby('sender_id')['amount'].agg(['count', 'sum', 'mean', 'std']).fillna(0)
    out_stats.columns = ['out_degree', 'total_sent', 'avg_sent', 'std_sent']
    
    # Unique Counterparties
    uniq_senders = df.groupby('receiver_id')['sender_id'].nunique()
    uniq_receivers = df.groupby('sender_id')['receiver_id'].nunique()
    
    # Temporal Burst (Max txns in any 1h window / total txns) - approximated
    # We'll compute "burstiness" as variance of inter-arrival times? 
    # Or simpler: max txns per day. Let's do max txns in any 4h window for efficiency if possible.
    # For now, let's use a simpler proxy: std_dev of timestamps (if >1 txn)
    # Actually, let's stick to pandas features.
    
    features = pd.DataFrame(index=all_accounts)
    features = features.join(in_stats).join(out_stats).fillna(0)
    features['unique_in_neighbors'] = uniq_senders
    features['unique_out_neighbors'] = uniq_receivers
    features = features.fillna(0)
    
    # Derived Features
    # 1. In/Out Ratio (handling div by zero)
    features['in_out_ratio'] = features['total_received'] / (features['total_sent'] + 1e-5)
    
    # 2. Reciprocity proxy: overlapping neighbors? 
    # Hard to calculate exactly without full graph iteration. 
    # We'll use (unique_in + unique_out) / (degree) as a loose proxy for structure
    features['neighbor_diversity'] = (features['unique_in_neighbors'] + features['unique_out_neighbors']) / \
                                     (features['in_degree'] + features['out_degree'] + 1e-5)
                                     
    # 3. Amount Variance Ratio (Std / Mean) - limit noise
    features['in_cov'] = features['std_received'] / (features['avg_received'] + 1e-5)
    features['out_cov'] = features['std_sent'] / (features['avg_sent'] + 1e-5)
    
    # 4. Is Intermediate? (Low degree, balanced flow)
    # Total received approx equals Total sent AND degree is low
    features['flow_balance'] = abs(features['total_received'] - features['total_sent']) / \
                               (features['total_received'] + features['total_sent'] + 1e-5)
    
    # Shell signal: Low degree (<=5) + Good balance (<0.1)
    features['is_intermediate_shell'] = ((features['in_degree'] + features['out_degree']) <= 10) & \
                                        (features['flow_balance'] < 0.1)
    features['is_intermediate_shell'] = features['is_intermediate_shell'].astype(float)

    # 5. Fan-in / Fan-out signals
    # Fan-in: High In-Degree + Low Out-Degree + Low Variance in In-Amounts
    features['fan_in_signal'] = (features['in_degree'] > 10) & \
                                (features['out_degree'] < 5) & \
                                (features['in_cov'] < 0.5)
    features['fan_in_signal'] = features['fan_in_signal'].astype(float)
    
    # Fan-out: High Out-Degree + Low In-Degree + Low Variance in Out-Amounts
    features['fan_out_signal'] = (features['out_degree'] > 10) & \
                                 (features['in_degree'] < 5) & \
                                 (features['out_cov'] < 0.5)
    features['fan_out_signal'] = features['fan_out_signal'].astype(float)

    # Clean up infinite values
    features = features.replace([np.inf, -np.inf], 0)
    
    # Final feature selection (14 columns)
    feature_cols = [
        'in_degree', 'out_degree', 'total_received', 'total_sent', 
        'avg_received', 'avg_sent', 'std_received', 'std_sent',
        'unique_in_neighbors', 'unique_out_neighbors',
        'in_out_ratio', 'neighbor_diversity',
        'is_intermediate_shell', 'fan_in_signal'
    ]
    # We need to ensure we return numbers
    return features[feature_cols].fillna(0)


# ------------------------------------------------------------------
# 2. Improved GNN Model (3 Layers + Focal Loss support)
# ------------------------------------------------------------------
class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, out_channels=2):
        super(GNNModel, self).__init__()
        # Deeper: 3 layers to capture 3-hop patterns (cycles)
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training) # Reduced dropout for smaller dataset
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Classifier
        x = self.fc(x)
        return x # Return raw logits for Focal Loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# ------------------------------------------------------------------
# 3. Data Prep Functions
# ------------------------------------------------------------------
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

    # 3. Construct Improved Node Features
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
    Uses the saved scaler params from training.
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

    # Normalize using SAVED scaler params
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

def train_model(data, epochs=500, patience=20):
    """
    Train model with Focal Loss and Early Stopping.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(num_node_features=data.num_features).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.7, gamma=2.0) # High alpha for minority class

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        
        # Training Loss
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation Loss
        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
        
        model.train()
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Early stopping at epoch {epoch}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        
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
    """Save trained model weights and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_node_features': data.num_features,
    }, MODEL_PATH)

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
    return os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH)
