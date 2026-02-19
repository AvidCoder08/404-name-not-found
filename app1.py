import streamlit as st
import pandas as pd
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import json
import os
import time

from src.gnn_model import prepare_graph_data, train_model, predict

st.set_page_config(layout="wide", page_title="Money Muling Detection Engine")

st.title("💸 Money Muling Detection Engine")
st.markdown("### Graph-Based Financial Crime Detection")

# ==========================================================
# PDF-COMPLIANT GRAPH ALGORITHMS
# ==========================================================

def get_bounded_cycles(G, min_len=3, max_len=5):
    cycles = []
    for start_node in G.nodes():
        stack = [(start_node, [start_node])]
        while stack:
            curr, path = stack.pop()
            if len(path) > max_len:
                continue
            for neighbor in G.successors(curr):
                if neighbor == start_node and len(path) >= min_len:
                    if min(path) == start_node:
                        cycles.append(path)
                elif neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
    return cycles


def get_temporal_smurfs(df):
    smurfs = []

    potential_fan_in = df.groupby('receiver_id')['sender_id'].nunique()
    for recv in potential_fan_in[potential_fan_in >= 10].index:
        txns = df[df['receiver_id'] == recv].sort_values('timestamp').drop_duplicates('sender_id')
        if len(txns) >= 10:
            txns['time_diff'] = txns['timestamp'].diff(periods=9)
            if (txns['time_diff'] <= pd.Timedelta(hours=72)).any():
                senders = txns['sender_id'].unique().tolist()
                smurfs.append({"type": "fan_in_smurfing", "center": recv, "members": senders})

    potential_fan_out = df.groupby('sender_id')['receiver_id'].nunique()
    for sender in potential_fan_out[potential_fan_out >= 10].index:
        txns = df[df['sender_id'] == sender].sort_values('timestamp').drop_duplicates('receiver_id')
        if len(txns) >= 10:
            txns['time_diff'] = txns['timestamp'].diff(periods=9)
            if (txns['time_diff'] <= pd.Timedelta(hours=72)).any():
                receivers = txns['receiver_id'].unique().tolist()
                smurfs.append({"type": "fan_out_smurfing", "center": sender, "members": receivers})

    return smurfs


def get_layered_shells(simple_G, in_degrees, out_degrees):
    shells = []
    shell_candidates = set(
        n for n in simple_G.nodes()
        if 2 <= (in_degrees.get(n, 0) + out_degrees.get(n, 0)) <= 3
        and in_degrees.get(n, 0) >= 1 and out_degrees.get(n, 0) >= 1
    )

    for u in shell_candidates:
        for v in simple_G.successors(u):
            if v in shell_candidates and u != v:
                for start in simple_G.predecessors(u):
                    if start == v:
                        continue
                    for end in simple_G.successors(v):
                        if end == u or end == start:
                            continue
                        shells.append([start, u, v, end])
    return shells


# ==========================================================
# DATA LOADING
# ==========================================================

st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload Transactions CSV", type=["csv"])

if uploaded_file is None:
    st.sidebar.info("Using default dataset: data/transactions.csv")
    if os.path.exists("data/transactions.csv"):
        df = pd.read_csv("data/transactions.csv")
    else:
        st.error("Default dataset not found.")
        st.stop()
else:
    df = pd.read_csv(uploaded_file)

col_mapping = {
    'sourceid': 'sender_id',
    'destinationid': 'receiver_id',
    'amountofmoney': 'amount',
    'date': 'timestamp'
}
df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

df['sender_id'] = df['sender_id'].astype(str)
df['receiver_id'] = df['receiver_id'].astype(str)

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

st.sidebar.success(f"Loaded {len(df)} transactions")

# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3 = st.tabs(
    ["🕵️‍♂️ Detection Dashboard", "🕸️ Graph Visualization", "📥 Export Results"]
)

# ==========================================================
# TAB 1 – DETECTION
# ==========================================================

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        train_btn = st.button("Train GNN Model")

    with col2:
        detect_btn = st.button("Run Graph Algorithms")

    if "suspicion_scores" not in st.session_state:
        st.session_state.suspicion_scores = {}

    if "detected_rings" not in st.session_state:
        st.session_state.detected_rings = []

    if "processing_time" not in st.session_state:
        st.session_state.processing_time = 0.0

    # ---------------- GNN ----------------
    if train_btn:

        st.session_state.processing_time = 0.0
        start_time = time.time()

        data = prepare_graph_data(df)
        model = train_model(data, epochs=200)
        probs = predict(model, data)

        scores = {}
        idx_to_acc = {v: k for k, v in data.account_map.items()}
        for idx, prob in enumerate(probs):
            scores[idx_to_acc[idx]] = float(prob) * 100

        st.session_state.suspicion_scores = scores
        st.session_state.processing_time += (time.time() - start_time)

        st.success("GNN complete.")

    # ---------------- Heuristics ----------------
    if detect_btn:

        start_time = time.time()

        G = nx.from_pandas_edgelist(
            df, 'sender_id', 'receiver_id', ['amount'],
            create_using=nx.MultiDiGraph()
        )
        simple_G = nx.DiGraph(G)

        in_degrees = df.groupby('receiver_id').size().to_dict()
        out_degrees = df.groupby('sender_id').size().to_dict()

        cycles = get_bounded_cycles(simple_G)
        smurfs = get_temporal_smurfs(df)
        shells = get_layered_shells(simple_G, in_degrees, out_degrees)

        st.session_state.detected_rings = []
        ring_counter = 1

        for cycle in cycles:
            st.session_state.detected_rings.append({
                "ring_id": f"RING_{ring_counter:03d}",
                "pattern_type": "cycle",
                "member_accounts": cycle,
                "risk_score": 90.0
            })
            ring_counter += 1

        for smurf in smurfs:
            st.session_state.detected_rings.append({
                "ring_id": f"RING_{ring_counter:03d}",
                "pattern_type": smurf['type'],
                "member_accounts": smurf['members'] + [smurf['center']],
                "risk_score": 85.0
            })
            ring_counter += 1

        for shell in shells:
            st.session_state.detected_rings.append({
                "ring_id": f"RING_{ring_counter:03d}",
                "pattern_type": "layered_shell_network",
                "member_accounts": shell,
                "risk_score": 80.0
            })
            ring_counter += 1

        st.session_state.processing_time += (time.time() - start_time)

        st.success("Heuristic detection complete.")

# ==========================================================
# TAB 3 – EXPORT (FINAL PDF-COMPLIANT)
# ==========================================================

with tab3:

    st.subheader("Download Results")

    if st.session_state.suspicion_scores and st.session_state.detected_rings:

        suspicious_list = []

        for acc, score in st.session_state.suspicion_scores.items():

            ring_id = ""
            patterns = []

            for ring in st.session_state.detected_rings:
                if acc in ring['member_accounts']:
                    ring_id = ring['ring_id']
                    patterns.append(ring['pattern_type'])

            if score > 50 or patterns:

                if not patterns and score > 75:
                    patterns = ["gnn_high_risk_anomaly"]

                suspicious_list.append({
                    "account_id": str(acc),
                    "suspicion_score": round(float(score), 2),
                    "detected_patterns": list(set(patterns)),
                    "ring_id": ring_id
                })

        suspicious_list.sort(
            key=lambda x: x["suspicion_score"],
            reverse=True
        )

        fraud_rings_output = []

        for ring in st.session_state.detected_rings:
            fraud_rings_output.append({
                "ring_id": ring["ring_id"],
                "member_accounts": [str(x) for x in ring["member_accounts"]],
                "pattern_type": ring["pattern_type"],
                "risk_score": float(ring["risk_score"])
            })

        total_accounts = len(
            pd.concat([df['sender_id'], df['receiver_id']]).unique()
        )

        output_json = {
            "suspicious_accounts": suspicious_list,
            "fraud_rings": fraud_rings_output,
            "summary": {
                "total_accounts_analyzed": total_accounts,
                "suspicious_accounts_flagged": len(suspicious_list),
                "fraud_rings_detected": len(fraud_rings_output),
                "processing_time_seconds": round(st.session_state.processing_time, 2)
            }
        }

        json_str = json.dumps(output_json, indent=2)

        st.download_button(
            label="📥 Download Results JSON",
            data=json_str,
            file_name="money_muling_results.json",
            mime="application/json"
        )

        st.json(output_json)

    else:
        st.info("Run detection first.")
