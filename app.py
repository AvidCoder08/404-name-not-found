import streamlit as st
import pandas as pd
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import json
import os
import time
import io

from src.gnn_model import (
    prepare_graph_data, prepare_inference_data,
    train_model, predict,
    save_model, load_model, model_exists,
)
from src.graph_algo import extract_suspicious_subgraph

st.set_page_config(layout="wide", page_title="Money Muling Detection Engine")

st.title("💸 Money Muling Detection Engine")
st.markdown("### Hybrid GNN Pipeline — Score → Extract → Detect")

# ==========================================
# PATTERN DETECTION ALGORITHMS
# (Run on GNN-flagged subgraphs only)
# ==========================================

def get_bounded_cycles(G, min_len=3, max_len=5):
    """Detect cycles of length 3 to 5 in the subgraph."""
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
    """Detect 10+ transactions Fan-in/Fan-out within a 72-hour window."""
    smurfs = []

    # Fan-in (Aggregator)
    potential_fan_in = df.groupby('receiver_id')['sender_id'].nunique()
    for recv in potential_fan_in[potential_fan_in >= 10].index:
        txns = df[df['receiver_id'] == recv].sort_values('timestamp').drop_duplicates('sender_id')
        if len(txns) >= 10:
            txns = txns.copy()
            txns['time_diff'] = txns['timestamp'].diff(periods=9)
            if (txns['time_diff'] <= pd.Timedelta(hours=72)).any():
                end_time = txns[txns['time_diff'] <= pd.Timedelta(hours=72)].iloc[0]['timestamp']
                start_time = end_time - pd.Timedelta(hours=72)
                senders = df[
                    (df['receiver_id'] == recv) &
                    (df['timestamp'] >= start_time) &
                    (df['timestamp'] <= end_time)
                ]['sender_id'].unique().tolist()
                smurfs.append({"type": "fan_in_smurfing", "center": recv, "members": senders})

    # Fan-out (Disperser)
    potential_fan_out = df.groupby('sender_id')['receiver_id'].nunique()
    for sender in potential_fan_out[potential_fan_out >= 10].index:
        txns = df[df['sender_id'] == sender].sort_values('timestamp').drop_duplicates('receiver_id')
        if len(txns) >= 10:
            txns = txns.copy()
            txns['time_diff'] = txns['timestamp'].diff(periods=9)
            if (txns['time_diff'] <= pd.Timedelta(hours=72)).any():
                end_time = txns[txns['time_diff'] <= pd.Timedelta(hours=72)].iloc[0]['timestamp']
                start_time = end_time - pd.Timedelta(hours=72)
                receivers = df[
                    (df['sender_id'] == sender) &
                    (df['timestamp'] >= start_time) &
                    (df['timestamp'] <= end_time)
                ]['receiver_id'].unique().tolist()
                smurfs.append({"type": "fan_out_smurfing", "center": sender, "members": receivers})

    return smurfs


def get_layered_shells(simple_G, in_degrees, out_degrees):
    """Chains of 3+ hops where intermediate accounts have 2-3 total transactions."""
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


def run_pattern_detection(subgraph_G, sub_df):
    """Run all pattern detection algorithms on a subgraph."""
    simple_G = nx.DiGraph(subgraph_G)

    in_degrees = sub_df.groupby('receiver_id').size().to_dict() if len(sub_df) > 0 else {}
    out_degrees = sub_df.groupby('sender_id').size().to_dict() if len(sub_df) > 0 else {}

    # Filter nodes for cycle detection to keep it tractable
    cycle_candidates = [
        n for n in simple_G.nodes()
        if in_degrees.get(n, 0) > 0 and out_degrees.get(n, 0) > 0
        and (in_degrees.get(n, 0) + out_degrees.get(n, 0)) < 50
    ]
    cycles = get_bounded_cycles(simple_G.subgraph(cycle_candidates), min_len=3, max_len=5)

    smurfs = get_temporal_smurfs(sub_df) if 'timestamp' in sub_df.columns else []

    shells = get_layered_shells(simple_G, in_degrees, out_degrees)

    return cycles, smurfs, shells


# ==========================================
# SIDEBAR & DATA LOADING
# ==========================================
st.sidebar.header("⚙️ Configuration")

# Model status
if model_exists():
    st.sidebar.success("✅ Pre-trained model found")
else:
    st.sidebar.warning("⚠️ No pre-trained model — train one first")

st.sidebar.divider()

# Suspicion threshold
threshold = st.sidebar.slider(
    "GNN Suspicion Threshold (%)",
    min_value=10, max_value=90, value=50, step=5,
    help="Accounts above this threshold are flagged suspicious. Pattern detection runs on their subgraph."
)

# Subgraph hops
hops = st.sidebar.slider(
    "Subgraph Neighborhood (hops)",
    min_value=1, max_value=3, value=2,
    help="How many hops around each suspicious node to include in the subgraph for pattern detection."
)

st.sidebar.divider()
st.sidebar.header("📂 Input Data")
uploaded_file = st.sidebar.file_uploader("Upload Transactions CSV", type=["csv"])

# Sample size for analysis (SAML-D is 9.5M rows — we sample for speed)
sample_size = st.sidebar.number_input(
    "Max transactions to analyze",
    min_value=10_000, max_value=1_000_000, value=100_000, step=10_000,
    help="For large datasets, a random sample is used for analysis speed."
)

if uploaded_file is None:
    st.sidebar.info("Using default dataset: `SAML-D.csv`")
    saml_path = "SAML-D.csv"
    if os.path.exists(saml_path):
        # Read only needed columns + sample for analysis performance
        df = pd.read_csv(
            saml_path,
            usecols=['Time', 'Date', 'Sender_account', 'Receiver_account', 'Amount'],
            nrows=sample_size,
        )
        # Map SAML-D columns to standard names
        df = df.rename(columns={
            'Sender_account': 'sender_id',
            'Receiver_account': 'receiver_id',
            'Amount': 'amount',
        })
        # Combine Date + Time into timestamp
        if 'Date' in df.columns and 'Time' in df.columns:
            df['timestamp'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                errors='coerce'
            )
            df = df.drop(columns=['Date', 'Time'], errors='ignore')
    else:
        st.error("SAML-D.csv not found. Please upload a CSV.")
        st.stop()
else:
    df = pd.read_csv(uploaded_file, nrows=sample_size)
    # Map common alternative column names
    col_mapping = {
        'sourceid': 'sender_id',
        'destinationid': 'receiver_id',
        'amountofmoney': 'amount',
        'date': 'timestamp',
        'Sender_account': 'sender_id',
        'Receiver_account': 'receiver_id',
        'Amount': 'amount',
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    if 'Date' in df.columns and 'Time' in df.columns:
        df['timestamp'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            errors='coerce'
        )
        df = df.drop(columns=['Date', 'Time'], errors='ignore')

df['sender_id'] = df['sender_id'].astype(str)
df['receiver_id'] = df['receiver_id'].astype(str)
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

st.sidebar.success(f"Loaded **{len(df):,}** transactions")

# ==========================================
# SESSION STATE
# ==========================================
if "suspicion_scores" not in st.session_state:
    st.session_state.suspicion_scores = {}
if "detected_rings" not in st.session_state:
    st.session_state.detected_rings = []
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0.0
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "🕵️ Detection Dashboard",
    "🕸️ Graph Visualization",
    "📥 Export Results"
])

# ==========================================
# TAB 1 — DETECTION DASHBOARD
# ==========================================
with tab1:
    st.markdown("#### Pipeline: GNN Scoring → Subgraph Extraction → Pattern Detection")

    col_train, col_analyze = st.columns(2)

    with col_train:
        st.markdown("##### 1️⃣ Train & Save Model (one-time)")
        train_btn = st.button("🧠 Train & Save GNN Model", use_container_width=True)

    with col_analyze:
        st.markdown("##### 2️⃣ Analyze Dataset")
        analyze_btn = st.button(
            "🔍 Analyze Dataset" + (" (model ready)" if model_exists() else " (no model)"),
            use_container_width=True,
            disabled=not model_exists()
        )

    # -----------------------------------------------
    # TRAIN & SAVE (one-time on SAML-D.csv)
    # -----------------------------------------------
    if train_btn:
        start_time = time.time()

        with st.spinner("Building labels from the loaded dataset..."):
            # The SAML-D data we loaded already has Is_laundering column
            # Re-read with the label column to get ground truth
            saml_path = "SAML-D.csv"

            if os.path.exists(saml_path):
                df_with_labels = pd.read_csv(
                    saml_path,
                    usecols=['Sender_account', 'Receiver_account', 'Amount', 'Is_laundering'],
                    nrows=sample_size,
                )
                df_with_labels = df_with_labels.rename(columns={
                    'Sender_account': 'sender_id',
                    'Receiver_account': 'receiver_id',
                    'Amount': 'amount',
                })
                df_with_labels['sender_id'] = df_with_labels['sender_id'].astype(str)
                df_with_labels['receiver_id'] = df_with_labels['receiver_id'].astype(str)

                # Derive per-account labels: suspicious if involved in ANY fraudulent transaction
                fraud_txns = df_with_labels[df_with_labels['Is_laundering'] == 1]
                suspicious_accounts = set(fraud_txns['sender_id'].unique()) | set(fraud_txns['receiver_id'].unique())
                st.info(f"Found **{len(suspicious_accounts):,}** unique suspicious accounts from {len(fraud_txns):,} fraudulent transactions.")
            else:
                st.warning("SAML-D.csv not found. Falling back to ground_truth.json.")
                if os.path.exists("data/ground_truth.json"):
                    with open("data/ground_truth.json", "r") as f:
                        gt = json.load(f)
                    suspicious_accounts = {item['account_id'] for item in gt['suspicious_accounts']}
                else:
                    st.error("No label source found. Cannot train without labels.")
                    st.stop()

        with st.spinner("Preparing graph data for training..."):
            data = prepare_graph_data(df)
            acc_list = list(data.account_map.keys())

            labels = [1 if acc in suspicious_accounts else 0 for acc in acc_list]
            labels_df = pd.DataFrame({'account_id': acc_list, 'is_suspicious': labels})
            data = prepare_graph_data(df, labels_df)

            sus_count = sum(labels)
            st.info(f"Label split: **{sus_count:,}** suspicious / **{len(labels) - sus_count:,}** benign nodes.")

        with st.spinner("Training GNN Model (GraphSAGE, 200 epochs, class-weighted)..."):
            model = train_model(data, epochs=200)

        with st.spinner("Saving model..."):
            save_model(model, data)

        elapsed = time.time() - start_time
        st.success(f"✅ Model trained and saved in {elapsed:.1f}s — you won't need to retrain again!")
        st.rerun()  # Refresh so Analyze button becomes enabled

    # -----------------------------------------------
    # ANALYZE DATASET (load pretrained → score → subgraph → patterns)
    # -----------------------------------------------
    if analyze_btn:
        start_time = time.time()

        # Step 1: Load pre-trained model
        with st.spinner("Loading pre-trained GNN model..."):
            result = load_model()
            if result is None:
                st.error("No pre-trained model found. Please train first.")
                st.stop()
            model, metadata = result

        # Step 2: Prepare inference data (using saved scaler)
        with st.spinner("Preparing graph data for inference..."):
            data = prepare_inference_data(
                df,
                scaler_mean=metadata['scaler_mean'],
                scaler_scale=metadata['scaler_scale'],
            )

        # Step 3: GNN Scoring
        with st.spinner("Scoring all accounts with GNN..."):
            probs = predict(model, data)
            scores = {}
            idx_to_acc = {v: k for k, v in data.account_map.items()}
            for idx, prob in enumerate(probs):
                scores[idx_to_acc[idx]] = float(prob) * 100

            st.session_state.suspicion_scores = scores

        # Step 4: Filter high-risk nodes
        suspicious_nodes = [acc for acc, score in scores.items() if score >= threshold]

        st.info(f"GNN flagged **{len(suspicious_nodes):,}** accounts above {threshold}% threshold "
                f"(out of {len(scores):,} total).")

        # Step 5: Extract suspicious subgraph
        with st.spinner(f"Extracting {hops}-hop subgraph around {len(suspicious_nodes)} suspicious nodes..."):
            G_full = nx.from_pandas_edgelist(
                df, 'sender_id', 'receiver_id', ['amount'],
                create_using=nx.MultiDiGraph()
            )
            if suspicious_nodes:
                subgraph_G = extract_suspicious_subgraph(G_full, suspicious_nodes, hops=hops)
                subgraph_nodes = set(subgraph_G.nodes())
                sub_df = df[
                    df['sender_id'].isin(subgraph_nodes) &
                    df['receiver_id'].isin(subgraph_nodes)
                ]
                st.info(f"Subgraph: **{len(subgraph_nodes):,}** nodes, **{len(sub_df):,}** transactions.")
            else:
                subgraph_G = nx.MultiDiGraph()
                sub_df = pd.DataFrame(columns=df.columns)
                st.warning("No accounts above threshold. Lowering the threshold may help.")

        # Step 6: Run pattern detection on subgraph
        with st.spinner("Running pattern detection on suspicious subgraph..."):
            cycles, smurfs, shells = run_pattern_detection(subgraph_G, sub_df)

        # Build detected rings list
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

        elapsed = time.time() - start_time
        st.session_state.processing_time = elapsed
        st.session_state.analysis_complete = True

        st.success(f"✅ Analysis complete in {elapsed:.1f}s — "
                   f"{len(cycles)} cycles, {len(smurfs)} smurfing rings, {len(shells)} shell networks detected.")

    # -----------------------------------------------
    # RESULTS DISPLAY (always visible if data exists)
    # -----------------------------------------------
    if st.session_state.suspicion_scores:
        st.divider()
        st.markdown("#### 🔝 Top High-Risk Accounts (GNN)")
        top_risky = sorted(
            st.session_state.suspicion_scores.items(),
            key=lambda x: x[1], reverse=True
        )[:15]
        st.table(pd.DataFrame(top_risky, columns=["Account ID", "Suspicion Score (%)"]))

    if st.session_state.detected_rings:
        st.divider()
        st.markdown("#### 🔗 Detected Fraud Rings (from GNN-flagged subgraph)")
        summary_data = []
        for ring in st.session_state.detected_rings:
            summary_data.append({
                "Ring ID": ring['ring_id'],
                "Type": ring['pattern_type'],
                "Members": len(ring['member_accounts']),
                "Risk Score": ring['risk_score'],
                "Account IDs": ", ".join([str(x) for x in ring['member_accounts']])
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)


# ==========================================
# TAB 2 — GRAPH VISUALIZATION (GNN-only)
# ==========================================
with tab2:
    st.markdown("### Interactive Graph — GNN Suspicion Scores")

    if st.session_state.suspicion_scores:
        # Get top N nodes by GNN score for visualization
        sorted_accs = sorted(
            st.session_state.suspicion_scores.items(),
            key=lambda x: x[1], reverse=True
        )

        # Show suspicious accounts + their transaction partners
        top_accs = [acc for acc, score in sorted_accs[:50]]
        display_nodes = set(top_accs)

        # Add ring members to display
        for ring in st.session_state.detected_rings:
            for member in ring['member_accounts']:
                display_nodes.add(str(member))

        display_nodes = list(display_nodes)
        if len(display_nodes) > 150:
            display_nodes = display_nodes[:150]

        G_full = nx.from_pandas_edgelist(
            df, 'sender_id', 'receiver_id', ['amount'],
            create_using=nx.MultiDiGraph()
        )
        subG = G_full.subgraph(display_nodes)

        # Build ring membership lookup for highlights
        ring_membership = {}
        for ring in st.session_state.detected_rings:
            for member in ring['member_accounts']:
                ring_membership[str(member)] = ring['pattern_type']

        nodes, edges = [], []
        for node_id in subG.nodes():
            score = st.session_state.suspicion_scores.get(node_id, 0)

            # Color gradient: green → yellow → orange → red
            if score > 80:
                color = "#ff1744"  # Red — high risk
            elif score > 60:
                color = "#ff9100"  # Orange
            elif score > 40:
                color = "#ffd600"  # Yellow
            else:
                color = "#00e676"  # Green — low risk

            # Border highlight for ring members
            size = 20 if node_id in ring_membership else 15
            label = f"{node_id}\n({score:.0f}%)"

            nodes.append(Node(
                id=node_id, label=label, size=size, color=color,
            ))

        for u, v, data_edge in subG.edges(data=True):
            edges.append(Edge(
                source=u, target=v,
                label=f"${data_edge.get('amount', 0):.0f}"
            ))

        config = Config(
            width=900, height=650, directed=True,
            nodeHighlightBehavior=True, highlightColor="#F7A7A6",
            physics=True,
        )

        st.info(f"Showing **{len(nodes)}** nodes and **{len(edges)}** edges. "
                f"Node colors reflect GNN suspicion scores.")

        # Color legend
        leg1, leg2, leg3, leg4 = st.columns(4)
        leg1.markdown("🟢 Low (<40%)")
        leg2.markdown("🟡 Medium (40-60%)")
        leg3.markdown("🟠 High (60-80%)")
        leg4.markdown("🔴 Critical (>80%)")

        agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.warning("Run **Analyze Dataset** first to generate GNN scores and visualize the graph.")


# ==========================================
# TAB 3 — EXPORT RESULTS
# ==========================================
with tab3:
    st.subheader("📥 Export GNN Analysis Results")

    if st.session_state.suspicion_scores:
        # Build suspicious accounts list
        suspicious_list = []
        for acc, score in st.session_state.suspicion_scores.items():
            ring_id = None
            patterns = []

            for ring in st.session_state.detected_rings:
                if acc in [str(m) for m in ring['member_accounts']]:
                    ring_id = ring['ring_id']
                    patterns.append(ring['pattern_type'])

            if score >= threshold or patterns:
                if not patterns and score > 75:
                    patterns = ["gnn_high_risk_anomaly"]

                suspicious_list.append({
                    "account_id": str(acc),
                    "suspicion_score": round(float(score), 2),
                    "detected_patterns": list(set(patterns)),
                    "ring_id": str(ring_id) if ring_id else None
                })

        suspicious_list.sort(key=lambda x: x['suspicion_score'], reverse=True)

        # Fraud rings output
        fraud_rings_output = []
        for ring in st.session_state.detected_rings:
            fraud_rings_output.append({
                "ring_id": ring["ring_id"],
                "member_accounts": [str(x) for x in ring["member_accounts"]],
                "pattern_type": ring["pattern_type"],
                "risk_score": float(ring["risk_score"])
            })

        total_accounts = len(pd.concat([df['sender_id'], df['receiver_id']]).unique())

        output_json = {
            "suspicious_accounts": suspicious_list,
            "fraud_rings": fraud_rings_output,
            "summary": {
                "total_accounts_analyzed": total_accounts,
                "suspicious_accounts_flagged": len(suspicious_list),
                "fraud_rings_detected": len(fraud_rings_output),
                "gnn_threshold_used": threshold,
                "processing_time_seconds": round(st.session_state.processing_time, 2)
            }
        }

        json_str = json.dumps(output_json, indent=2)

        # --- Downloads ---
        st.markdown("#### Download Options")
        col_json, col_csv = st.columns(2)

        with col_json:
            st.download_button(
                label="📄 Download JSON",
                data=json_str,
                file_name="gnn_results.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_csv:
            # Build CSV from suspicious accounts
            if suspicious_list:
                csv_df = pd.DataFrame(suspicious_list)
                csv_df['detected_patterns'] = csv_df['detected_patterns'].apply(lambda x: "; ".join(x))
                csv_buffer = io.StringIO()
                csv_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
            else:
                csv_data = "account_id,suspicion_score,detected_patterns,ring_id\n"

            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name="gnn_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # --- Preview ---
        st.divider()
        st.markdown("#### Results Preview")

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Accounts", f"{total_accounts:,}")
        m2.metric("Flagged Suspicious", f"{len(suspicious_list):,}")
        m3.metric("Fraud Rings", f"{len(fraud_rings_output)}")
        m4.metric("Processing Time", f"{st.session_state.processing_time:.1f}s")

        st.divider()
        with st.expander("🔍 Full JSON Output", expanded=False):
            st.json(output_json)

    else:
        st.info("Run **Analyze Dataset** to generate results for export.")
