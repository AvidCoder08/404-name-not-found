"""
Benchmark Speed: GNN Pipeline vs. Brute Force.

Compares:
1. GNN Pipeline: Score -> Filter top 10% -> Subgraph -> Pattern Algo
2. Brute Force: Pattern Algo on FULL graph
"""
import time
import pandas as pd
import networkx as nx
import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.gnn_model import load_model, prepare_inference_data, predict
from src.graph_algo import build_graph, detect_cycles, detect_smurfing, detect_shells
from scripts.generate_training_data import generate_synthetic_dataset

def run_benchmark():
    print("="*60)
    print("BENCHMARKING SPEED: GNN vs BRUTE FORCE")
    print("="*60)

    # 1. Generate Dataset (20k txns to make differences visible)
    print("Generating 5,000 transactions...")
    output_dir = "data_benchmark"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    df, _, _ = generate_synthetic_dataset(
        seed=123, 
        output_dir=output_dir,
        n_normal_accounts=500, 
        n_normal_txns=4000,
        n_cycle_rings=5,
        n_smurf_rings=3,
        n_shell_chains=2
    )
    print(f"Dataset: {len(df):,} txns, {len(set(df['sender_id'])|set(df['receiver_id'])):,} accounts")

    # Load Model
    model, metadata = load_model()
    
    # ---------------------------------------------------------
    # METHOD A: GNN Pipeline (The "Smart" Way)
    # ---------------------------------------------------------
    print("\n--- Method A: GNN Pipeline ---")
    start_a = time.time()
    
    # 1. Score
    if 'timestamp' not in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'])
    data_test = prepare_inference_data(
        df[['sender_id', 'receiver_id', 'amount', 'timestamp']],
        scaler_mean=metadata['scaler_mean'],
        scaler_scale=metadata['scaler_scale'],
    )
    probs = predict(model, data_test)
    
    # 2. Filter (Top 10% suspicious or threshold > 50%)
    # Let's use threshold > 50%
    idx_to_acc = {v: k for k, v in data_test.account_map.items()}
    sus_nodes = [idx_to_acc[i] for i, p in enumerate(probs) if p > 0.5]
    
    # 3. Subgraph Extraction (1-hop)
    sus_set = set(sus_nodes)
    neighbor_set = set(sus_nodes)
    # Fast vectorized filter
    mask = df['sender_id'].isin(sus_set) | df['receiver_id'].isin(sus_set)
    df_sub = df[mask]
    
    # 4. Pattern Detection on Subgraph
    G_sub = build_graph(df_sub)
    cycles = detect_cycles(G_sub)
    smurfs = detect_smurfing(G_sub)
    shells = detect_shells(G_sub)
    
    time_a = time.time() - start_a
    print(f"Time: {time_a:.4f} seconds")
    print(f"Result: {len(cycles)} cycles, {len(smurfs)} smurfs, {len(shells)} shells")
    print(f"Processed: {len(df_sub)} txns (filtered from {len(df)})")

    # ---------------------------------------------------------
    # METHOD B: Brute Force (The "Slow" Way)
    # ---------------------------------------------------------
    print("\n--- Method B: Brute Force (Full Graph) ---")
    start_b = time.time()
    
    # 1. Build Full Graph
    G_full = build_graph(df)
    
    # 2. Run Patterns on Everything
    cycles_b = detect_cycles(G_full)
    smurfs_b = detect_smurfing(G_full)
    shells_b = detect_shells(G_full)
    
    time_b = time.time() - start_b
    print(f"Time: {time_b:.4f} seconds")
    print(f"Result: {len(cycles_b)} cycles, {len(smurfs_b)} smurfs, {len(shells_b)} shells")
    
    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"SPEEDUP FACTOR: {time_b / time_a:.1f}x FASTER")
    print("="*60)
    print(f"GNN Pipeline: {time_a:.4f}s")
    print(f"Brute Force:  {time_b:.4f}s")

if __name__ == "__main__":
    run_benchmark()
