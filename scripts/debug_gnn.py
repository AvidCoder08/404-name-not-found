
import pandas as pd
import json
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from src.gnn_model import prepare_graph_data, train_model, predict, GNNModel

def debug_gnn():
    print("Loading data...")
    if not os.path.exists("data/transactions.csv") or not os.path.exists("data/ground_truth.json"):
        print("Data files not found!")
        return

    df = pd.read_csv("data/transactions.csv")
    with open("data/ground_truth.json", "r") as f:
        gt = json.load(f)

    # Prepare labels
    sus_accs = {item['account_id']: 1 for item in gt['suspicious_accounts']}
    data_temp = prepare_graph_data(df)
    acc_list = list(data_temp.account_map.keys())
    labels = [sus_accs.get(acc, 0) for acc in acc_list]
    labels_df = pd.DataFrame({'account_id': acc_list, 'is_suspicious': labels})

    print(f"Total accounts: {len(acc_list)}")
    print(f"Suspicious accounts: {sum(labels)}")
    print(f"Suspicious ratio: {sum(labels)/len(acc_list):.4f}")

    print("Preparing graph data...")
    data = prepare_graph_data(df, labels_df)
    
    # Check class balance in train mask
    train_labels = data.y[data.train_mask]
    num_train = len(train_labels)
    num_suspicious_train = train_labels.sum().item()
    print(f"Training set size: {num_train}")
    print(f"Suspicious in training: {num_suspicious_train}")
    print(f"Training suspicious ratio: {num_suspicious_train/num_train:.4f}")

    print("Training model...")
    # Custom training loop to print loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(num_node_features=data.num_features).to(device)
    data = data.to(device)
    
    # Calculate class weights
    # Weight for class 1 = (Number of class 0) / (Number of class 1)
    num_benign = num_train - num_suspicious_train
    if num_suspicious_train > 0:
        weight_factor = num_benign / num_suspicious_train
    else:
        weight_factor = 1.0
    
    print(f"Calculated suggested weight for class 1: {weight_factor:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(51): # 50 epochs matching app.py
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            pred = out[data.train_mask].argmax(dim=1)
            correct = (pred == data.y[data.train_mask]).sum()
            acc = int(correct) / int(data.train_mask.sum())
            
            # Additional check: how many predicted as 1?
            predicted_ones = (pred == 1).sum().item()
            
            print(f"Epoch {epoch:03d}: Loss {loss.item():.4f}, Train Acc: {acc:.4f}, Predicted Suspicious: {predicted_ones}/{int(data.train_mask.sum())}")

    print("Predicting...")
    # Use the predict function from src.gnn_model
    probs = predict(model, data)
    
    print("Probability stats:")
    print(f"Min: {probs.min():.4f}")
    print(f"Max: {probs.max():.4f}")
    print(f"Mean: {probs.mean():.4f}")
    
    top_indices = probs.argsort()[::-1][:10]
    idx_to_acc = {v: k for k, v in data.account_map.items()}
    
    print("\nTop 10 Suspicious Candidates:")
    for idx in top_indices:
        acc = idx_to_acc[idx]
        score = probs[idx]
        is_actually_suspicious = sus_accs.get(acc, 0)
        print(f"{acc}: {score:.4f} (Ground Truth: {is_actually_suspicious})")

if __name__ == "__main__":
    debug_gnn()
