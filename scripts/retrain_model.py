
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.gnn_model import prepare_graph_data, train_model, save_model

def retrain():
    print("Starting GNN Model Retraining...")
    
    # Paths
    train_path = "data_benchmark/synthetic_train.csv"
    labels_path = "data_benchmark/synthetic_labels.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(labels_path):
        print(f"ERROR: Training data not found at {train_path} or {labels_path}")
        return

    # Load Data
    print("Loading data...")
    df_train = pd.read_csv(train_path)
    df_labels = pd.read_csv(labels_path)
    
    # Preprocess Timestamps (Crucial for features)
    if 'timestamp' in df_train.columns:
        df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    
    print(f"Transactions: {len(df_train)}")
    print(f"Labels: {len(df_labels)} (Suspicious: {df_labels['is_suspicious'].sum()})")

    # Prepare Graph Data
    print("Preparing graph data (feature engineering)...")
    data = prepare_graph_data(df_train, df_labels)
    
    print(f"Graph created: {data.num_nodes} nodes, {data.num_edges} edges.")
    print(f"Features dimension: {data.num_features}")

    # Train Model
    print("Training model (this may take a minute)...")
    # Increase epochs slightly to ensure convergence on small data
    model = train_model(data, epochs=400, patience=30)
    
    # Save Model
    print("Saving model...")
    save_model(model, data)
    print("Model saved to models/gnn_model.pt")
    print("Done.")

if __name__ == "__main__":
    retrain()
