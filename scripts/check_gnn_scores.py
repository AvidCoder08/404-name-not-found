
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import torch
import numpy as np
from src.gnn_model import load_model, prepare_inference_data, predict

def check_scores():
    print("Loading model...")
    model_data = load_model()
    if model_data is None:
        print("ERROR: No model found at models/gnn_model.pt")
        return
    
    model, metadata = model_data
    print("Model loaded.")

    data_path = "money_muling_dataset_12k (1).csv"
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return

    print(f"Loading dataset {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocessing same as backend
    if 'timestamp' not in df.columns:
        if 'Date' in df.columns and 'Time' in df.columns:
             df['timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
    
    df['sender_id'] = df['sender_id'].astype(str)
    df['receiver_id'] = df['receiver_id'].astype(str)

    print("Preparing inference data...")
    data = prepare_inference_data(
        df,
        scaler_mean=metadata['scaler_mean'],
        scaler_scale=metadata['scaler_scale']
    )

    print("Running prediction...")
    probs = predict(model, data)
    scores = probs * 100
    
    print("\n--- Score Statistics ---")
    print(f"Total Accounts: {len(scores)}")
    print(f"Min Score: {scores.min():.2f}")
    print(f"Max Score: {scores.max():.2f}")
    print(f"Mean Score: {scores.mean():.2f}")
    print(f"Median Score: {np.median(scores):.2f}")
    
    print("\n--- Threshold Analysis ---")
    suspicious_nodes = []
    for thresh in [10, 30, 50, 70, 90]:
        nodes_idx = np.where(scores >= thresh)[0]
        count = len(nodes_idx)
        pct = (count / len(scores)) * 100
        print(f"Nodes >= {thresh}%: {count} ({pct:.2f}%)")
        if thresh == 50:
            idx_to_acc = {v: k for k, v in data.account_map.items()}
            suspicious_nodes = [idx_to_acc[i] for i in nodes_idx]

    print("\n--- Pattern Detection Comparison (Full vs Subgraph) ---")
    from src.graph_algo import build_graph, detect_cycles, detect_smurfing, detect_shells, extract_suspicious_subgraph
    
    print("Building full graph...")
    G = build_graph(df)
    
    print(f"Full Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Run on Full Graph
    print("Running detection on FULL GRAPH...")
    cycles_full = detect_cycles(G)
    smurfs_full = detect_smurfing(df)
    shells_full = detect_shells(G)
    print(f"  Cycles: {len(cycles_full)}")
    print(f"  Smurfs: {len(smurfs_full)}")
    print(f"  Shells: {len(shells_full)}")

    # Analyze scores of fraud nodes
    print("\n--- Fraud Node Score Analysis ---")
    fraud_nodes = set()
    for c in cycles_full:
        fraud_nodes.update(c)
    for s in smurfs_full:
        fraud_nodes.add(s['center'])
        fraud_nodes.update(s['members'])
    for sh in shells_full:
        fraud_nodes.update(sh)
    
    print(f"Total Unique Fraud Nodes (from Full Graph patterns): {len(fraud_nodes)}")
    
    if fraud_nodes:
        fraud_scores = []
        node_scores_dict = {k: v for k, v in zip(data.account_map.keys(), scores)}
        
        for node in fraud_nodes:
            if node in node_scores_dict:
                fraud_scores.append(node_scores_dict[node])
            else:
                print(f"Warning: Fraud node {node} not in GNN scores!")
                
        fraud_scores = np.array(fraud_scores)
        print(f"Fraud Scores - Mean: {fraud_scores.mean():.2f}% | Median: {np.median(fraud_scores):.2f}%")
        print(f"Fraud Scores - Min: {fraud_scores.min():.2f}% | Max: {fraud_scores.max():.2f}%")
        
        for thresh in [10, 30, 50]:
            caught = (fraud_scores >= thresh).sum()
            print(f"Fraud Nodes caught at threshold {thresh}%: {caught}/{len(fraud_scores)} ({caught/len(fraud_scores)*100:.1f}%)")

    # Run on Subgraph
    print("\nRunning detection on SUBGRAPH (Threshold=50, Hops=1)...")
    if not suspicious_nodes:
        print("  No suspicious nodes found! Subgraph is empty.")
    else:
        subG = extract_suspicious_subgraph(G, suspicious_nodes, hops=1)
        print(f"  Subgraph: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")
        
        # Create sub_df for smurfing (as per backend logic)
        sub_nodes = set(subG.nodes())
        sub_df = df[df['sender_id'].isin(sub_nodes) & df['receiver_id'].isin(sub_nodes)].copy()

        cycles_sub = detect_cycles(subG)
        smurfs_sub = detect_smurfing(sub_df)
        shells_sub = detect_shells(subG)
        
        print(f"  Cycles: {len(cycles_sub)} (Match: {len(cycles_sub)}/{len(cycles_full)})")
        print(f"  Smurfs: {len(smurfs_sub)} (Match: {len(smurfs_sub)}/{len(smurfs_full)})")
        print(f"  Shells: {len(shells_sub)} (Match: {len(shells_sub)}/{len(shells_full)})")
        
    print("\nDone.")

if __name__ == "__main__":
    check_scores()
