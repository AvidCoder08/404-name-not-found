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
from concurrent.futures import ThreadPoolExecutor
import functools

_executor = ThreadPoolExecutor(max_workers=2)

def _build_graph_nodes_links(suspicion_scores, fraud_rings):
    """Helper to build graph_update payload from scores and rings."""
    nodes = {}
    links = []

    # Add scored nodes
    for acc_id, score in suspicion_scores.items():
        color = '#F2B8B5' if score >= 90 else '#FFB74D' if score >= 70 else '#D0BCFF' if score >= 50 else '#CAC4D0'
        nodes[acc_id] = {
            "id": acc_id,
            "val": max(score / 10, 2),
            "color": color,
            "score": score,
        }

    # Add ring nodes/links
    for ring in fraud_rings:
        members = ring["member_accounts"]
        pattern = ring["pattern_type"]
        for m in members:
            if m not in nodes:
                nodes[m] = {"id": m, "val": 3, "color": "#CAC4D0", "score": 0}
        if pattern in ('cycle', 'layered_shell'):
            for i in range(len(members)):
                links.append({"source": members[i], "target": members[(i + 1) % len(members)], "type": pattern})
        elif pattern in ('fan_out', 'fan_in'):
            center = members[-1]
            for i in range(len(members) - 1):
                links.append({"source": center, "target": members[i], "type": pattern})
        else:
            for i in range(len(members)):
                links.append({"source": members[i], "target": members[(i + 1) % len(members)], "type": "other"})

    return list(nodes.values()), links


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    loop = asyncio.get_event_loop()

    async def process_and_stream():
        yield json.dumps({"progress": 5, "step": "upload", "log": "Reading CSV file..."}) + "\n"
        await asyncio.sleep(0.05)

        try:
            # ── Read CSV ──────────────────────────────────────────────
            df = await loop.run_in_executor(_executor, functools.partial(pd.read_csv, file.file))
            yield json.dumps({"progress": 15, "step": "csv", "log": f"Loaded {len(df)} transactions."}) + "\n"

            # ── Preprocessing ─────────────────────────────────────────
            yield json.dumps({"progress": 20, "step": "preprocess", "log": "Preprocessing timestamps and IDs..."}) + "\n"
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

            # ── GNN Scoring ───────────────────────────────────────────
            yield json.dumps({"progress": 30, "step": "gnn", "log": "Running GNN model..."}) + "\n"
            suspicion_scores = {}
            if model and metadata:
                try:
                    data = await loop.run_in_executor(
                        _executor,
                        functools.partial(prepare_inference_data, df,
                                          scaler_mean=metadata['scaler_mean'],
                                          scaler_scale=metadata['scaler_scale'])
                    )
                    probs = await loop.run_in_executor(_executor, functools.partial(predict, model, data))

                    idx_to_acc = {v: k for k, v in data.account_map.items()}
                    for idx, prob in enumerate(probs):
                        suspicion_scores[idx_to_acc[idx]] = float(prob) * 100
                    yield json.dumps({"progress": 50, "step": "gnn", "log": "GNN inference complete."}) + "\n"
                except Exception as e:
                    print(f"GNN Prediction failed: {e}")
                    yield json.dumps({"progress": 50, "step": "gnn", "log": f"GNN scoring skipped: {e}"}) + "\n"
            else:
                yield json.dumps({"progress": 50, "step": "gnn", "log": "GNN model not loaded. Skipping."}) + "\n"

            # ── Send initial graph with scored nodes ──────────────────
            initial_nodes, _ = _build_graph_nodes_links(suspicion_scores, [])
            yield json.dumps({
                "progress": 55, "step": "graph",
                "log": "Building transaction graph...",
                "graph_update": {"nodes": initial_nodes, "links": []}
            }) + "\n"

            # ── Build Graph ───────────────────────────────────────────
            G = await loop.run_in_executor(_executor, functools.partial(build_graph, df))
            yield json.dumps({"progress": 65, "step": "graph", "log": f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."}) + "\n"

            # ── Detect Cycles ─────────────────────────────────────────
            yield json.dumps({"progress": 70, "step": "cycles", "log": "Detecting cycles..."}) + "\n"
            cycles = await loop.run_in_executor(_executor, functools.partial(detect_cycles, G))

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

            # Stream graph with cycles
            nodes_so_far, links_so_far = _build_graph_nodes_links(suspicion_scores, rings)
            yield json.dumps({
                "progress": 78, "step": "cycles",
                "log": f"Found {len(cycles)} cycles.",
                "graph_update": {"nodes": nodes_so_far, "links": links_so_far}
            }) + "\n"

            # ── Detect Smurfing ───────────────────────────────────────
            yield json.dumps({"progress": 82, "step": "smurfing", "log": "Detecting smurfing patterns..."}) + "\n"
            smurfs = await loop.run_in_executor(_executor, functools.partial(detect_smurfing, G))

            for smurf in smurfs:
                rings.append({
                    "ring_id": f"SMURF_{ring_counter}",
                    "pattern_type": smurf["type"],
                    "member_accounts": smurf["members"] + [smurf["center"]],
                    "risk_score": 80.0,
                })
                ring_counter += 1

            # Stream graph with smurfing
            nodes_so_far, links_so_far = _build_graph_nodes_links(suspicion_scores, rings)
            yield json.dumps({
                "progress": 88, "step": "smurfing",
                "log": f"Found {len(smurfs)} smurfing patterns.",
                "graph_update": {"nodes": nodes_so_far, "links": links_so_far}
            }) + "\n"

            # ── Detect Shells ─────────────────────────────────────────
            yield json.dumps({"progress": 92, "step": "shells", "log": "Detecting shell patterns..."}) + "\n"
            shells = await loop.run_in_executor(_executor, functools.partial(detect_shells, G))

            for shell in shells:
                rings.append({
                    "ring_id": f"SHELL_{ring_counter}",
                    "pattern_type": "layered_shell",
                    "member_accounts": shell,
                    "risk_score": 85.0,
                })
                ring_counter += 1

            # ── Final result ──────────────────────────────────────────
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

            # Final graph update
            final_nodes, final_links = _build_graph_nodes_links(suspicion_scores, rings)
            yield json.dumps({
                "progress": 100, "step": "done",
                "log": "Analysis complete!",
                "graph_update": {"nodes": final_nodes, "links": final_links},
                "result": result_data
            }) + "\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({"progress": 0, "log": f"Error: {str(e)}", "error": True}) + "\n"

    return StreamingResponse(process_and_stream(), media_type="application/x-ndjson")
