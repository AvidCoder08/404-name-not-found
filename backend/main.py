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

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Money Muling Detection API is running"}

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


from fastapi.responses import StreamingResponse
import asyncio

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    async def process_and_stream():
        # Buffer the file content since we can't seek on UploadFile in some async contexts easily,
        # but for parsing we need it. 
        # Better: Read file content first (blocking I/O but okay for this scale) or use pd.read_csv directly.
        
        yield json.dumps({"progress": 10, "log": "Reading CSV file..."}) + "\n"
        await asyncio.sleep(0.1) # Small render yield
        
        try:
            # Read CSV
            df = pd.read_csv(file.file)
            yield json.dumps({"progress": 20, "log": f"Loaded {len(df)} transactions."}) + "\n"
            
            # Preprocessing
            yield json.dumps({"progress": 30, "log": "Preprocessing timestamps and IDs..."}) + "\n"
            if 'timestamp' not in df.columns:
                if 'Date' in df.columns and 'Time' in df.columns:
                    df['timestamp'] = pd.to_datetime(
                        df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
                else:
                    df['timestamp'] = pd.to_datetime(0, unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
            df['sender_id'] = df['sender_id'].astype(str)
            df['receiver_id'] = df['receiver_id'].astype(str)
            
            if 'transaction_id' not in df.columns:
                df['transaction_id'] = range(len(df))
                df['transaction_id'] = df['transaction_id'].astype(str)
                
            await asyncio.sleep(0.5)

            # GNN Scoring
            yield json.dumps({"progress": 50, "log": "Initializing GNN model for suspicious activity scoring..."}) + "\n"
            suspicion_scores = {}
            if model and metadata:
                try:
                    data = prepare_inference_data(
                        df, 
                        scaler_mean=metadata['scaler_mean'], 
                        scaler_scale=metadata['scaler_scale']
                    )
                    probs = predict(model, data)
                    
                    idx_to_acc = {v: k for k, v in data.account_map.items()}
                    for idx, prob in enumerate(probs):
                        suspicion_scores[idx_to_acc[idx]] = float(prob) * 100
                    yield json.dumps({"progress": 70, "log": "GNN inference complete."}) + "\n"
                except Exception as e:
                    print(f"GNN Prediction failed: {e}")
                    yield json.dumps({"progress": 70, "log": f"GNN Prediction failed: {e}"}) + "\n"
            else:
                yield json.dumps({"progress": 70, "log": "GNN model not loaded. Skipping scoring."}) + "\n"
            
            await asyncio.sleep(0.5)

            # Graph Algorithms
            yield json.dumps({"progress": 80, "log": "Building transaction graph..."}) + "\n"
            G = build_graph(df)
            
            yield json.dumps({"progress": 90, "log": "Detecting fraud rings (Cycles, Smurfing, Shells)..."}) + "\n"
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
            
            result_data = {
                "suspicion_scores": suspicion_scores,
                "fraud_rings": rings,
                "summary": summary,
            }
            
            yield json.dumps({"progress": 100, "log": "Analysis complete!", "result": result_data}) + "\n"
            
        except Exception as e:
            yield json.dumps({"progress": 0, "log": f"Error: {str(e)}", "error": True}) + "\n"

    return StreamingResponse(process_and_stream(), media_type="application/x-ndjson")
