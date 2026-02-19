"""
Script to train GNN on synthetic data and evaluate on both synthetic test and LI-Small_Trans.
"""
import pandas as pd
import numpy as np
import torch
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.gnn_model import prepare_graph_data, prepare_inference_data, train_model, predict, save_model

def evaluate(model, df_test, metadata, sus_accounts_set, name="Test Set"):
    print(f"\nEvaluating on {name}...")
    
    # Needs timestamp column!
    if 'timestamp' not in df_test.columns:
        if 'Date' in df_test.columns and 'Time' in df_test.columns:
             df_test['timestamp'] = pd.to_datetime(df_test['Date'].astype(str) + ' ' + df_test['Time'].astype(str), errors='coerce')
    
    # Ensure string IDs
    df_test['sender_id'] = df_test['sender_id'].astype(str)
    df_test['receiver_id'] = df_test['receiver_id'].astype(str)

    data_test = prepare_inference_data(
        df_test,
        scaler_mean=metadata['scaler_mean'],
        scaler_scale=metadata['scaler_scale'],
    )
    
    probs = predict(model, data_test)
    idx_to_acc = {v: k for k, v in data_test.account_map.items()}
    scores = {idx_to_acc[i]: float(p)*100 for i, p in enumerate(probs)}
    
    # Calculate metrics at thresholds
    print(f"{'Thresh':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 60)
    for t in [50, 60, 70, 80, 90]:
        sus_pred = {acc for acc, s in scores.items() if s >= t}
        tp = len(sus_pred & sus_accounts_set)
        fp = len(sus_pred - sus_accounts_set)
        fn = len(sus_accounts_set - sus_pred)
        
        prec = tp/(tp+fp) if tp+fp > 0 else 0
        rec = tp/(tp+fn) if tp+fn > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
        
        print(f"{t:>7}% {prec:>7.1%} {rec:>7.1%} {f1:>7.3f} {tp:>6} {fp:>6} {fn:>6}")

def main():
    print("="*60)
    print("GNN TRAINING ON SYNTHETIC DATA")
    print("="*60)
    
    # 1. Load Training Data
    train_path = "data/synthetic_train.csv"
    labels_path = "data/synthetic_labels.csv"
    
    print(f"Loading {train_path}...")
    df_train = pd.read_csv(train_path)
    # Ensure correct columns
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    suspicious_accounts = set(labels_df[labels_df['is_suspicious']==1]['account_id'])
    print(f"  {len(df_train)} txns, {len(suspicious_accounts)} suspicious accounts")
    
    # 2. Train
    print("\nPreparing graph data...")
    df_graph = df_train[['sender_id', 'receiver_id', 'amount', 'timestamp']].copy()
    data = prepare_graph_data(df_graph)
    acc_list = list(data.account_map.keys())
    
    # Prepare node labels
    node_labels = [1 if acc in suspicious_accounts else 0 for acc in acc_list]
    labels_df_graph = pd.DataFrame({'account_id': acc_list, 'is_suspicious': node_labels})
    data = prepare_graph_data(df_graph, labels_df_graph)
    
    print("\nTraining GNN (500 epochs, early stopping)...")
    t0 = time.time()
    model = train_model(data, epochs=500)
    print(f"  Training done in {time.time()-t0:.1f}s")
    
    save_model(model, data)
    metadata = {
        'scaler_mean': data.scaler_mean,
        'scaler_scale': data.scaler_scale,
        'num_node_features': int(data.num_features),
    }

    # 3. Evaluate on Synthetic Test Set (Seed 43)
    test_path = "data_test/synthetic_train.csv"
    test_labels_path = "data_test/synthetic_labels.csv"
    if os.path.exists(test_path):
        df_test_syn = pd.read_csv(test_path)
        labels_test_syn = pd.read_csv(test_labels_path)
        sus_test_syn = set(labels_test_syn[labels_test_syn['is_suspicious']==1]['account_id'])
        
        evaluate(model, df_test_syn, metadata, sus_test_syn, name="Synthetic Test (Seed 43)")
        
    # 4. Evaluate on LI-Small_Trans (test_transactions.csv)
    li_path = "data/test_transactions.csv"
    if os.path.exists(li_path):
        # We need ground truth for LI-Small
        # The ground truth is embedded in LI-Small_Trans.csv logic... 
        # Wait, the script that converted it didn't save labels separately.
        # But `LI-Small_Trans.csv` had `Is Laundering`.
        # I'll re-read `LI-Small_Trans.csv` to get labels.
        
        print("\nLoading LI-Small_Trans ground truth...")
        try:
            df_li_orig = pd.read_csv("data/LI-Small_Trans.csv", nrows=100000) # Use sample for evaluation speed
            # Determine suspicious accounts
            # Columns: Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
            # Account ID format: {Bank}_{Account}
            df_li_orig['sender_id'] = df_li_orig['From Bank'].astype(str) + '_' + df_li_orig['Account'].astype(str)
            df_li_orig['receiver_id'] = df_li_orig['To Bank'].astype(str) + '_' + df_li_orig['Account.1'].astype(str)
            
            fraud_txns = df_li_orig[df_li_orig['Is Laundering'] == 1]
            sus_li = set(fraud_txns['sender_id'].unique()) | set(fraud_txns['receiver_id'].unique())
            
            # Now load the converted test set (which has same transactions)
            df_li_test = pd.read_csv(li_path, nrows=100000)
            
            evaluate(model, df_li_test, metadata, sus_li, name="LI-Small Test (100k sample)")
            
        except Exception as e:
            print(f"Could not evaluate on LI-Small: {e}")

if __name__ == "__main__":
    main()
