import json
import os
import sys

# Ensure root directory is in sys.path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Modified imports to match new src structure
from src.gnn_model import predict, prepare_inference_data, load_model
from src.graph_algo import build_graph, detect_cycles, detect_smurfing, detect_shells

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model, metadata = None, None

@app.on_event("startup")
def startup_event():
    global model, metadata
    try:
        loaded = load_model()
        if loaded:
            model, metadata = loaded
            print("Model loaded successfully.")
        else:
            print("No pre-trained model found.")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    global model, metadata
    
    # Read CSV
    df = pd.read_csv(file.file)

    # Preprocessing: Ensure timestamp
    if 'timestamp' not in df.columns:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['timestamp'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        else:
            df['timestamp'] = pd.to_datetime(0, unit='s') # Default if missing
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
    df['sender_id'] = df['sender_id'].astype(str)
    df['receiver_id'] = df['receiver_id'].astype(str)
    
    # Ensure transaction_id exists
    if 'transaction_id' not in df.columns:
        df['transaction_id'] = range(len(df))
        df['transaction_id'] = df['transaction_id'].astype(str)

    # ---- GNN Scoring ----
    suspicion_scores = {}
    
    if model and metadata:
        try:
            # Use prepare_inference_data with saved scaler params
            data = prepare_inference_data(
                df, 
                scaler_mean=metadata['scaler_mean'], 
                scaler_scale=metadata['scaler_scale']
            )
            
            probs = predict(model, data)
            
            idx_to_acc = {v: k for k, v in data.account_map.items()}
            for idx, prob in enumerate(probs):
                suspicion_scores[idx_to_acc[idx]] = float(prob) * 100
        except Exception as e:
            print(f"GNN Prediction failed: {e}")
            # Fallback or empty scores
            pass
    else:
        print("Model not loaded, skipping GNN scoring.")

    # ---- Graph Algorithms ----
    G = build_graph(df)
    
    # Detect patterns
    cycles = detect_cycles(G)
    smurfs = detect_smurfing(G)
    shells = detect_shells(G)

    rings = []
    ring_counter = 1

    for cycle in cycles:
        rings.append({
            "ring_id": f"CYCLE_{ring_counter}",
            "pattern_type": "cycle",
            "member_accounts": cycle,
            "risk_score": 90.0,
        })
        ring_counter += 1

    for smurf in smurfs:
        rings.append({
            "ring_id": f"SMURF_{ring_counter}",
            "pattern_type": smurf["type"],
            "member_accounts": smurf["members"] + [smurf["center"]],
            "risk_score": 80.0,
        })
        ring_counter += 1
        
    for shell in shells:
        rings.append({
            "ring_id": f"SHELL_{ring_counter}",
            "pattern_type": "layered_shell",
            "member_accounts": shell,
            "risk_score": 85.0
        })
        ring_counter += 1

    summary = {
        "total_transactions": len(df),
        "accounts_flagged": len([s for s in suspicion_scores.values() if s > 50]),
        "fraud_rings_detected": len(rings),
    }

    return {
        "suspicion_scores": suspicion_scores,
        "fraud_rings": rings,
        "summary": summary,
    }
