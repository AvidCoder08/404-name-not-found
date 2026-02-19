import sys
import os
import pandas as pd
import networkx as nx
import time
from sklearn.metrics import precision_score, recall_score, f1_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_algo import build_graph, detect_cycles, extract_suspicious_subgraph

def load_data():
    print("Loading data...")
    df = pd.read_csv("data_test/synthetic_train.csv")
    labels = pd.read_csv("data_test/synthetic_labels.csv")
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df, labels

def evaluate(detected_accounts, true_labels):
    y_true = []
    y_pred = []
    
    # Only evaluate accounts present in labels file
    valid_accounts = set(true_labels['account_id'].values)
    
    for _, row in true_labels.iterrows():
        acc_id = row['account_id']
        y_true.append(row['is_suspicious'])
        y_pred.append(1 if acc_id in detected_accounts else 0)
        
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return p, r, f1

def run_current_algo(G, suspicious_nodes):
    print("\nRunning CURRENT algo (Subgraph + Bailout)...")
    start = time.time()
    # Simulate main.py logic: subgraph hops=1
    subG = extract_suspicious_subgraph(G, suspicious_nodes, hops=1)
    cycles = detect_cycles(subG, max_length=5)
    end = time.time()
    print(f"Time: {end - start:.4f}s")
    print(f"Cycles found: {len(cycles)}")
    return set([node for cycle in cycles for node in cycle])

def run_native_algo(G):
    print("\nRunning NATIVE algo (Full Graph + length_bound=5)...")
    start = time.time()
    # Using simple_cycles with length_bound directly on simple DiGraph
    simple_G = nx.DiGraph(G)
    # Remove reciprocal edges? Or keeping them? Let's keep them for native test 
    # to see if length_bound handles them gracefully.
    cycles = list(nx.simple_cycles(simple_G, length_bound=5))
    # Filter length >= 3
    cycles = [c for c in cycles if len(c) >= 3]
    end = time.time()
    print(f"Time: {end - start:.4f}s")
    print(f"Cycles found: {len(cycles)}")
    return set([node for cycle in cycles for node in cycle])

if __name__ == "__main__":
    df, labels = load_data()
    G = build_graph(df)
    
    # Identify true suspicious nodes to simulate GNN input for current algo
    # We cheat slightly by using ground truth labels as "suspicion scores > 50" proxy
    # or we can assume GNN is perfect for this benchmark.
    suspicious_nodes = labels[labels['is_suspicious'] == 1]['account_id'].tolist()
    
    # Current
    detected_current = run_current_algo(G, suspicious_nodes)
    p, r, f1 = evaluate(detected_current, labels)
    print(f"Current Metrics: Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")
    
    # Native
    try:
        detected_native = run_native_algo(G)
        p, r, f1 = evaluate(detected_native, labels)
        print(f"Native Metrics: Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")
    except Exception as e:
        print(f"Native algo failed: {e}")
