
import sys
import os
import pandas as pd
import asyncio
import functools
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph_algo import build_graph, detect_cycles, detect_smurfing, detect_shells
from src.gnn_model import load_model, predict, prepare_inference_data
from backend.main import extract_suspicious_subgraph, _build_graph_nodes_links

# Mock FastAPI state logic/utils for standalone run
def get_sus_scores(df, model, data):
    # This mimics backend.main logic
    # predict(model, data) returns a list/tensor of probabilities
    probs = predict(model, data)
    
    # Map back to account IDs
    # data.account_map maps {account_id: index}
    # We need {index: account_id}
    idx_to_acc = {v: k for k, v in data.account_map.items()}
    
    score_map = {}
    for idx, prob in enumerate(probs):
        if idx in idx_to_acc:
            score_map[idx_to_acc[idx]] = float(prob) * 100
            
    return score_map

async def run_analysis():
    dataset_path = "money_muling_dataset_12k (1).csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    print(f"Loading {dataset_path}...")
    df = pd.read_csv(dataset_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. GROUND TRUTH (Full Graph)
    print("\n--- 1. CALCULATING GROUND TRUTH (Full Graph) ---")
    G_full = build_graph(df)
    print(f"Full Graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")
    
    cycles_gt = detect_cycles(G_full)
    smurfs_gt = detect_smurfing(df)
    shells_gt = detect_shells(G_full)
    
    gt_total = len(cycles_gt) + len(smurfs_gt) + len(shells_gt)
    print(f"Ground Truth Detected Patterns:")
    print(f"  Cycles: {len(cycles_gt)}")
    print(f"  Smurfing: {len(smurfs_gt)}")
    print(f"  Shells: {len(shells_gt)}")
    print(f"  TOTAL: {gt_total}")

    # 2. ALGO (GNN Filtered - Simulating User's Logic)
    print("\n--- 2. SIMULATING ALGORITHM (Strict GNN Filter) ---")
    print("Loading GNN model...")
    gnn_model, metadata = load_model()
    
    # Check if metadata exists (it should if load_model returned a tuple)
    scaler_mean = metadata.get('scaler_mean') if metadata else None
    scaler_scale = metadata.get('scaler_scale') if metadata else None
    
    data = prepare_inference_data(df, scaler_mean=scaler_mean, scaler_scale=scaler_scale)
    
    print("Predicting scores...")
    score_map = get_sus_scores(df, gnn_model, data)
    
    # User's Logic from backend/main.py:
    # if len(suspicious_nodes) < 10: try threshold 10, else threshold 50.
    
    sus_threshold = 50
    suspicious_nodes = [n for n, s in score_map.items() if s > sus_threshold]
    
    log_msg = f"Standard GNN Filter (Threshold {sus_threshold}%)"
    hops = 1
    
    if len(suspicious_nodes) < 10:
        sus_threshold = 10
        suspicious_nodes = [n for n, s in score_map.items() if s > sus_threshold]
        log_msg = f"Fallback GNN Filter (Threshold {sus_threshold}%) - Low Confidence Mode"
        hops = 2
        
    print(f"Logic Triggered: {log_msg}")
    print(f"Suspicious Nodes selected: {len(suspicious_nodes)}")
    
    if len(suspicious_nodes) == 0:
        print("No suspicious nodes found. Subgraph is empty.")
        algo_total = 0
    else:
        # Extract subgraph
        print(f"Extracting subgraph ({hops} hops)...")
        subG = extract_suspicious_subgraph(G_full, suspicious_nodes, hops=hops)
        print(f"Subgraph: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")
        
        cycles_algo = detect_cycles(subG)
        
        # Note: Smurfing in backend (with Strict GNN) filters the DF to the subgraph nodes.
        # See backend/main.py: sub_df = df[df['sender_id'].isin(sub_nodes) & df['receiver_id'].isin(sub_nodes)]
        
        sub_nodes = set(subG.nodes())
        sub_df = df[df['sender_id'].isin(sub_nodes) & df['receiver_id'].isin(sub_nodes)].copy()
        
        smurfs_algo = detect_smurfing(sub_df)
        
        shells_algo = detect_shells(subG)
        
        algo_total = len(cycles_algo) + len(smurfs_algo) + len(shells_algo)
        
        print(f"Algorithm Detected Patterns (GNN Filtered):")
        print(f"  Cycles: {len(cycles_algo)}")
        print(f"  Smurfing: {len(smurfs_algo)}")
        print(f"  Shells: {len(shells_algo)}")
        print(f"  TOTAL: {algo_total}")

    print("\n--- COMPARISON RESULTS ---")
    print(f"Ground Truth (Hypothetical Max): {gt_total}")
    print(f"Current Algorithm Detection:    {algo_total}")
    if gt_total > 0:
        recall = (algo_total / gt_total) * 100
        print(f"Recall: {recall:.2f}%")
        print(f"Missed: {gt_total - algo_total}")
    else:
        print("No patterns in ground truth.")

if __name__ == "__main__":
    asyncio.run(run_analysis())
