"""
Evaluate GNN model accuracy against SAML-D ground truth labels.
Trains on first 100k rows, evaluates on held-out rows 100k-200k.
"""
import pandas as pd
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.gnn_model import (
    prepare_graph_data, prepare_inference_data, 
    train_model, predict, save_model, GNNModel
)

print("=" * 60)
print("GNN MODEL — TRAIN & EVALUATE")
print("=" * 60)

saml_path = "SAML-D.csv"

# ── STEP 1: Train on first 100k rows ──
print("\n[1/4] Loading training data (first 100k rows)...")
df_train = pd.read_csv(
    saml_path,
    usecols=['Sender_account', 'Receiver_account', 'Amount', 'Is_laundering'],
    nrows=100_000,
)
df_train_graph = df_train.rename(columns={
    'Sender_account': 'sender_id', 'Receiver_account': 'receiver_id', 'Amount': 'amount',
})
df_train_graph['sender_id'] = df_train_graph['sender_id'].astype(str)
df_train_graph['receiver_id'] = df_train_graph['receiver_id'].astype(str)

# Derive per-account labels
fraud_train = df_train[df_train['Is_laundering'] == 1]
sus_train = set(fraud_train['Sender_account'].astype(str)) | set(fraud_train['Receiver_account'].astype(str))
print(f"  Train transactions: {len(df_train):,}")
print(f"  Fraudulent train txns: {len(fraud_train):,} ({len(fraud_train)/len(df_train)*100:.2f}%)")
print(f"  Suspicious train accounts: {len(sus_train):,}")

# Build labels
data_train = prepare_graph_data(df_train_graph.drop(columns=['Is_laundering']))
acc_list = list(data_train.account_map.keys())
labels = [1 if acc in sus_train else 0 for acc in acc_list]
labels_df = pd.DataFrame({'account_id': acc_list, 'is_suspicious': labels})
data_train = prepare_graph_data(df_train_graph.drop(columns=['Is_laundering']), labels_df)

sus_count = sum(labels)
print(f"  Label split: {sus_count} suspicious / {len(labels)-sus_count} benign")

# Train
print("\n[2/4] Training GNN (300 epochs)...")
model = train_model(data_train, epochs=300)
save_model(model, data_train)
print("  ✅ Model trained and saved.")

# ── STEP 2: Evaluate on held-out rows 100k-200k ──
print("\n[3/4] Loading test data (rows 100k-200k)...")
df_test = pd.read_csv(
    saml_path,
    usecols=['Sender_account', 'Receiver_account', 'Amount', 'Is_laundering'],
    skiprows=range(1, 100_001),
    nrows=100_000,
)
df_test.columns = ['Sender_account', 'Receiver_account', 'Amount', 'Is_laundering']

df_test_graph = df_test.rename(columns={
    'Sender_account': 'sender_id', 'Receiver_account': 'receiver_id', 'Amount': 'amount',
})
df_test_graph['sender_id'] = df_test_graph['sender_id'].astype(str)
df_test_graph['receiver_id'] = df_test_graph['receiver_id'].astype(str)

# Ground truth for test
fraud_test = df_test[df_test['Is_laundering'] == 1]
sus_test = set(fraud_test['Sender_account'].astype(str)) | set(fraud_test['Receiver_account'].astype(str))
all_test_accounts = set(df_test_graph['sender_id']) | set(df_test_graph['receiver_id'])
print(f"  Test transactions: {len(df_test):,}")
print(f"  Fraudulent test txns: {len(fraud_test):,} ({len(fraud_test)/len(df_test)*100:.2f}%)")
print(f"  Suspicious test accounts: {len(sus_test):,} / {len(all_test_accounts):,}")

# Inference using saved scaler
data_test = prepare_inference_data(
    df_test_graph.drop(columns=['Is_laundering']),
    scaler_mean=data_train.scaler_mean,
    scaler_scale=data_train.scaler_scale,
)

print("\n[4/4] Running predictions on test set...")
probs = predict(model, data_test)
idx_to_acc = {v: k for k, v in data_test.account_map.items()}
scores = {idx_to_acc[i]: float(p) * 100 for i, p in enumerate(probs)}

# ── RESULTS ──
print("\n" + "=" * 60)
print("ACCURACY AT DIFFERENT THRESHOLDS")
print("=" * 60)
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>7} {'FP':>7} {'TN':>7} {'FN':>7}")
print("-" * 75)

for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    tp = fp = tn = fn = 0
    for acc, score in scores.items():
        actual = acc in sus_test
        predicted = score >= threshold
        if predicted and actual: tp += 1
        elif predicted and not actual: fp += 1
        elif not predicted and actual: fn += 1
        else: tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"{threshold:>9}% {precision:>9.1%} {recall:>9.1%} {f1:>9.3f} {tp:>7} {fp:>7} {tn:>7} {fn:>7}")

# Score distributions
sus_scores = [scores[a] for a in sus_test if a in scores]
ben_scores = [scores[a] for a in (all_test_accounts - sus_test) if a in scores]
print(f"\n  Suspicious accounts avg score: {np.mean(sus_scores):.2f}%" if sus_scores else "")
print(f"  Benign accounts avg score:     {np.mean(ben_scores):.2f}%" if ben_scores else "")

# Top 20
print("\n" + "=" * 60)
print("TOP 20 HIGHEST-SCORED ACCOUNTS")
print("=" * 60)
top20 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
print(f"{'Account':>15} {'Score':>10} {'Fraud?':>10}")
for acc, score in top20:
    label = "YES ✓" if acc in sus_test else "NO ✗"
    print(f"{acc:>15} {score:>9.2f}% {label:>10}")
