import streamlit as st
import pandas as pd
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import json
import os
import time

# Your existing GNN imports
from src.gnn_model import prepare_graph_data, train_model, predict

st.set_page_config(layout="wide", page_title="Money Muling Detection Engine")

st.title("💸 Money Muling Detection Engine")
st.markdown("### Graph-Based Financial Crime Detection")

# ==========================================
# ADVANCED PDF-COMPLIANT GRAPH ALGORITHMS
# (Inlined to guarantee exact test-case matching)
# ==========================================

def get_bounded_cycles(G, min_len=3, max_len=5):
    """PDF Rule 1: Detect cycles of length 3 to 5"""
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
    """PDF Rule 2: 10+ transactions Fan-in/Fan-out within a 72-hour window"""
    smurfs = []

    # Fan-in (Aggregator)
    potential_fan_in = df.groupby('receiver_id')['sender_id'].nunique()
    for recv in potential_fan_in[potential_fan_in >= 10].index:
        txns = df[df['receiver_id'] == recv].sort_values('timestamp').drop_duplicates('sender_id')
        if len(txns) >= 10:
            txns['time_diff'] = txns['timestamp'].diff(periods=9)
            if (txns['time_diff'] <= pd.Timedelta(hours=72)).any():
                end_time = txns[txns['time_diff'] <= pd.Timedelta(hours=72)].iloc[0]['timestamp']
                start_time = end_time - pd.Timedelta(hours=72)
                senders = df[(df['receiver_id'] == recv) & (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]['sender_id'].unique().tolist()
                smurfs.append({"type": "fan_in_smurfing", "center": recv, "members": senders})

    # Fan-out (Disperser)
    potential_fan_out = df.groupby('sender_id')['receiver_id'].nunique()
    for sender in potential_fan_out[potential_fan_out >= 10].index:
        txns = df[df['sender_id'] == sender].sort_values('timestamp').drop_duplicates('receiver_id')
        if len(txns) >= 10:
            txns['time_diff'] = txns['timestamp'].diff(periods=9)
            if (txns['time_diff'] <= pd.Timedelta(hours=72)).any():
                end_time = txns[txns['time_diff'] <= pd.Timedelta(hours=72)].iloc[0]['timestamp']
                start_time = end_time - pd.Timedelta(hours=72)
                receivers = df[(df['sender_id'] == sender) & (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]['receiver_id'].unique().tolist()
                smurfs.append({"type": "fan_out_smurfing", "center": sender, "members": receivers})

    return smurfs

def get_layered_shells(simple_G, in_degrees, out_degrees):
    """PDF Rule 3: Chains of 3+ hops where intermediate accounts have 2-3 total transactions"""
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
                    if start == v: continue
                    for end in simple_G.successors(v):
                        if end == u or end == start: continue
                        shells.append([start, u, v, end])
    return shells

# ==========================================
# STREAMLIT SIDEBAR & DATA PREP
# ==========================================
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload Transactions CSV", type=["csv"])

if uploaded_file is None:
    st.sidebar.info("Using default dataset: `data/transactions.csv`")
    if os.path.exists("data/transactions.csv"):
        df = pd.read_csv("data/transactions.csv")
    else:
        st.error("Default dataset not found. Please generate it first.")
        st.stop()
else:
    df = pd.read_csv(uploaded_file)

# CRITICAL: Map ML dataset columns to the exact PDF requirements
col_mapping = {
    'sourceid': 'sender_id',
    'destinationid': 'receiver_id',
    'amountofmoney': 'amount',
    'date': 'timestamp'
}
df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

# Preprocessing
df['sender_id'] = df['sender_id'].astype(str)
df['receiver_id'] = df['receiver_id'].astype(str)
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

st.sidebar.success(f"Loaded {len(df)} transactions")

# Tabs
tab1, tab2, tab3 = st.tabs(["🕵️‍♂️ Detection Dashboard", "🕸️ Graph Visualization", "📥 Export Results"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("GNN Suspicion Scoring")
        train_btn = st.button("Train GNN Model & Analyze")

    with col2:
        st.subheader("Heuristic Pattern Detection")
        detect_btn = st.button("Run Graph Algorithms")

    if train_btn or detect_btn:
        st.divider()

    # Shared State for Results
    if "suspicion_scores" not in st.session_state:
        st.session_state.suspicion_scores = {}
    if "detected_rings" not in st.session_state:
        st.session_state.detected_rings = []
    if "processing_time" not in st.session_state:
        st.session_state.processing_time = 0.0

    # ---------------------------------------------------------
    # GNN Execution (Your exact original code block)
    # ---------------------------------------------------------
    if train_btn:
        start_time_gnn = time.time()
        with st.spinner("Loading labels from SAML-D.csv (this may take a moment for 9.5M rows)..."):
            # ── Derive per-account labels from the real SAML-D dataset ──
            # SAML-D.csv has: Sender_account, Receiver_account, Is_laundering (0/1)
            # Account IDs in formatted_transactions.csv have ACC_ prefix
            labels_df = None
            saml_path = "SAML-D.csv"
            if os.path.exists(saml_path):
                suspicious_accounts = set()
                # Read in chunks to handle the 9.5M row file
                for chunk in pd.read_csv(
                    saml_path,
                    usecols=['Sender_account', 'Receiver_account', 'Is_laundering'],
                    chunksize=500_000,
                ):
                    laundering = chunk[chunk['Is_laundering'] == 1]
                    for _, row in laundering.iterrows():
                        suspicious_accounts.add(f"ACC_{row['Sender_account']}")
                        suspicious_accounts.add(f"ACC_{row['Receiver_account']}")

                st.info(f"Found **{len(suspicious_accounts)}** unique suspicious accounts from SAML-D labels.")
            else:
                st.warning("SAML-D.csv not found. Falling back to ground_truth.json if available.")
                if os.path.exists("data/ground_truth.json"):
                    with open("data/ground_truth.json", "r") as f:
                        gt = json.load(f)
                    suspicious_accounts = {item['account_id'] for item in gt['suspicious_accounts']}
                else:
                    suspicious_accounts = set()
                    st.warning("No label source found. GNN will run untrained.")

        with st.spinner("Preparing Graph Data..."):
            data = prepare_graph_data(df)
            acc_list = list(data.account_map.keys())

            if suspicious_accounts:
                labels = [1 if acc in suspicious_accounts else 0 for acc in acc_list]
                labels_df = pd.DataFrame({'account_id': acc_list, 'is_suspicious': labels})
                data = prepare_graph_data(df, labels_df)
                sus_count = sum(labels)
                st.info(f"Label split: **{sus_count}** suspicious / **{len(labels) - sus_count}** benign nodes in graph.")

        with st.spinner("Training GNN Model (GraphSAGE, 200 epochs, class-weighted)..."):
             model = train_model(data, epochs=200)
             probs = predict(model, data)

             scores = {}
             idx_to_acc = {v: k for k, v in data.account_map.items()}
             for idx, prob in enumerate(probs):
                 scores[idx_to_acc[idx]] = float(prob) * 100

             st.session_state.suspicion_scores = scores
             st.session_state.processing_time += (time.time() - start_time_gnn)
             st.success(f"GNN Training Complete in {time.time() - start_time_gnn:.2f}s")

             st.markdown("#### Top High-Risk Accounts (GNN)")
             top_risky = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
             st.table(pd.DataFrame(top_risky, columns=["Account ID", "Suspicion Score"]))

    # ---------------------------------------------------------
    # Graph Algo Execution (Upgraded for PDF Specs)
    # ---------------------------------------------------------
    if detect_btn:
        start_time_algo = time.time()
        with st.spinner("Building NetworkX Graph..."):
            G = nx.from_pandas_edgelist(df, 'sender_id', 'receiver_id', ['amount'], create_using=nx.MultiDiGraph())
            simple_G = nx.DiGraph(G)

            in_degrees = df.groupby('receiver_id').size().to_dict()
            out_degrees = df.groupby('sender_id').size().to_dict()

        with st.spinner("Detecting Cycles (Ring Muling)..."):
            cycle_candidates = [n for n in simple_G.nodes() if in_degrees.get(n, 0) > 0 and out_degrees.get(n, 0) > 0 and (in_degrees.get(n, 0) + out_degrees.get(n, 0)) < 50]
            cycles = get_bounded_cycles(simple_G.subgraph(cycle_candidates), min_len=3, max_len=5)

        with st.spinner("Detecting Smurfing (72-Hour Fan-in/Fan-out)..."):
            smurfs = get_temporal_smurfs(df)

        with st.spinner("Detecting Layered Shell Networks..."):
            shells = get_layered_shells(simple_G, in_degrees, out_degrees)

        st.session_state.detected_rings = []
        ring_counter = 1

        for cycle in cycles:
            st.session_state.detected_rings.append({
                "ring_id": f"RING_{ring_counter:03d}", "pattern_type": "cycle",
                "member_accounts": cycle, "risk_score": 90.0
            })
            ring_counter += 1

        for smurf in smurfs:
             st.session_state.detected_rings.append({
                "ring_id": f"RING_{ring_counter:03d}", "pattern_type": smurf['type'],
                "member_accounts": smurf['members'] + [smurf['center']], "risk_score": 85.0
            })
             ring_counter += 1

        for shell in shells:
             st.session_state.detected_rings.append({
                "ring_id": f"RING_{ring_counter:03d}", "pattern_type": "layered_shell_network",
                "member_accounts": shell, "risk_score": 80.0
            })
             ring_counter += 1

        st.session_state.processing_time += (time.time() - start_time_algo)
        st.success(f"Detected {len(cycles)} cycles, {len(smurfs)} smurfing rings, and {len(shells)} shell networks.")

        st.markdown("#### Fraud Ring Summary")
        if st.session_state.detected_rings:
            summary_data = []
            for ring in st.session_state.detected_rings:
                summary_data.append({
                    "Ring ID": ring['ring_id'], "Type": ring['pattern_type'],
                    "Members": len(ring['member_accounts']), "Risk Score": ring['risk_score'],
                    "Account IDs": ", ".join([str(x) for x in ring['member_accounts']]) # PDF Requirement
                })
            st.dataframe(pd.DataFrame(summary_data))
        else:
            st.info("No rings detected.")

# ==========================================
# GRAPH VISUALIZATION (Untouched)
# ==========================================
with tab2:
    st.markdown("### Interactive Graph Visualization")
    st.info("Visualizing a subgraph of high-risk nodes (Top 50) to maintain performance.")

    if st.session_state.suspicion_scores:
        sorted_accs = sorted(st.session_state.suspicion_scores.items(), key=lambda x: x[1], reverse=True)
        top_accs = [acc for acc, score in sorted_accs[:50]]

        ring_nodes = set()
        for ring in st.session_state.detected_rings:
            for member in ring['member_accounts']:
                ring_nodes.add(member)

        display_nodes = list(set(top_accs) | ring_nodes)
        if len(display_nodes) > 100:
            display_nodes = display_nodes[:100]

        G_full = nx.from_pandas_edgelist(df, 'sender_id', 'receiver_id', ['amount'], create_using=nx.MultiDiGraph())
        subG = G_full.subgraph(display_nodes)

        nodes, edges = [], []
        for node_id in subG.nodes():
            score = st.session_state.suspicion_scores.get(node_id, 0)
            color = "#00ff00"
            if score > 50: color = "#ffff00"
            if score > 80: color = "#ff0000"
            nodes.append(Node(id=node_id, label=node_id, size=15, color=color))

        for u, v, data in subG.edges(data=True):
            edges.append(Edge(source=u, target=v, label=f"${data['amount']:.0f}"))

        config = Config(width=800, height=600, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
        return_value = agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.warning("Please run the Analysis first to generate scores and visualize high-risk nodes.")

# ==========================================
# EXPORT RESULTS (Upgraded for Exact JSON Match)
# ==========================================
with tab3:
    st.subheader("Download Results")

    if st.session_state.suspicion_scores and st.session_state.detected_rings:
        suspicious_list = []

        for acc, score in st.session_state.suspicion_scores.items():
            ring_id = None
            patterns = []

            # Match account to detected rings
            for ring in st.session_state.detected_rings:
                if acc in ring['member_accounts']:
                    ring_id = ring['ring_id']
                    patterns.append(ring['pattern_type'])

            # PDF Constraint: Include accounts if score > threshold OR caught in a heuristic pattern
            if score > 50 or patterns:
                if not patterns and score > 75:
                    patterns = ["gnn_high_risk_anomaly"]

                suspicious_list.append({
                    "account_id": str(acc),
                    "suspicion_score": float(score),
                    "detected_patterns": list(set(patterns)),
                    "ring_id": str(ring_id) if ring_id else None
                })

        # Sort descending per PDF spec
        suspicious_list.sort(key=lambda x: x['suspicion_score'], reverse=True)

        output_json = {
            "suspicious_accounts": suspicious_list,
            "fraud_rings": st.session_state.detected_rings,
            "summary": {
                "total_accounts_analyzed": len(pd.concat([df['sender_id'], df['receiver_id']]).unique()),
                "suspicious_accounts_flagged": len(suspicious_list),
                "fraud_rings_detected": len(st.session_state.detected_rings),
                "processing_time_seconds": round(st.session_state.processing_time, 2) # PDF Requirement
            }
        }

        json_str = json.dumps(output_json, indent=2)

        st.download_button(
            label="Download Results JSON",
            data=json_str,
            file_name="results.json",
            mime="application/json"
        )
        st.json(output_json)
    else:
        st.info("Run Analysis to generate results.")