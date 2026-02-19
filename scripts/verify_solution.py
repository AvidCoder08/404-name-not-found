import pandas as pd
import torch
import os
from src.gnn_model import prepare_graph_data, train_model, predict
from src.graph_algo import build_graph, detect_cycles, detect_smurfing

def test_pipeline():
    print("Loading data...")
    if not os.path.exists("data/transactions.csv"):
        print("Data not found!")
        return
        
    df = pd.read_csv("data/transactions.csv")
    
    print("Testing Graph Algo...")
    G = build_graph(df)
    cycles = detect_cycles(G)
    print(f"Cycles found: {len(cycles)}")
    smurfs = detect_smurfing(G)
    print(f"Smurfs found: {len(smurfs)}")
    
    print("Testing GNN Pipeline...")
    # Create dummy labels for testing
    senders = df['sender_id'].unique()
    receivers = df['receiver_id'].unique()
    all_accounts = list(set(senders) | set(receivers))
    labels_df = pd.DataFrame({'account_id': all_accounts, 'is_suspicious': 0})
    
    data = prepare_graph_data(df, labels_df)
    print(f"Graph Data: {data}")
    
    print("Training Model (1 epoch)...")
    model = train_model(data, epochs=1)
    
    print("Predicting...")
    probs = predict(model, data)
    print(f"Predictions shape: {probs.shape}")
    
    assert probs.shape[0] == data.num_nodes
    print("Verification Successful!")

if __name__ == "__main__":
    test_pipeline()
