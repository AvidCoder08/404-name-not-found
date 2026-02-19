import streamlit as st
import pandas as pd
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import json
import torch
from src.gnn_model import prepare_graph_data, train_model, predict
from src.graph_algo import build_graph, detect_cycles, detect_smurfing, detect_shells, extract_suspicious_subgraph
import os
import time

st.set_page_config(layout="wide", page_title="Money Muling Detection Engine")

st.title("💸 Money Muling Detection Engine")
st.markdown("### Graph-Based Financial Crime Detection")

# Sidebar
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

# Preprocessing
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

st.sidebar.success(f"Loaded {len(df)} transactions")

# Tabs
tab1, tab2, tab3 = st.tabs(["🕵️‍♂️ Detection Dashboard", "🕸️ Graph Visualization", "📥 Export Results"])

with tab1:
    st.subheader("Hybrid Detection Pipeline")
    st.markdown(
        "**Funnel Approach:** "
        "Stage 1 (GNN Fast Filter) → Stage 2 (Subgraph Extraction) → Stage 3 (Deterministic Search)"
    )

    # Threshold slider for filtering high-risk nodes after GNN scoring
    risk_threshold = st.slider(
        "GNN Risk Threshold (accounts above this score proceed to Stage 2)",
        min_value=0, max_value=100, value=50, step=5,
    )
    hop_radius = st.slider(
        "Subgraph Extraction Hops (neighborhood radius around flagged nodes)",
        min_value=1, max_value=4, value=2,
    )

    run_pipeline = st.button("🚀 Run Hybrid Detection Pipeline")

    # Shared State for Results
    if "suspicion_scores" not in st.session_state:
        st.session_state.suspicion_scores = {}
    if "detected_rings" not in st.session_state:
        st.session_state.detected_rings = []
    if "subgraph_nodes" not in st.session_state:
        st.session_state.subgraph_nodes = []

    if run_pipeline:
        st.divider()
        pipeline_start = time.time()

        # ── Stage 1: GNN Scoring ──────────────────────────────────────────
        st.markdown("#### Stage 1 — GNN Fast Filter (GraphSAGE)")
        with st.spinner("Preparing graph data for GNN..."):
            labels_df = None
            if os.path.exists("data/ground_truth.json"):
                with open("data/ground_truth.json", "r") as f:
                    gt = json.load(f)
                sus_accs = {item['account_id']: 1 for item in gt['suspicious_accounts']}
                data = prepare_graph_data(df)
                acc_list = list(data.account_map.keys())
                labels = [sus_accs.get(acc, 0) for acc in acc_list]
                labels_df = pd.DataFrame({'account_id': acc_list, 'is_suspicious': labels})
                data = prepare_graph_data(df, labels_df)
            else:
                st.warning("No ground truth found. Using untrained weights for demo.")
                data = prepare_graph_data(df)

        with st.spinner("Training GNN model..."):
            model = train_model(data, epochs=200)
            probs = predict(model, data)

            scores = {}
            idx_to_acc = {v: k for k, v in data.account_map.items()}
            for idx, prob in enumerate(probs):
                scores[idx_to_acc[idx]] = float(prob) * 100
            st.session_state.suspicion_scores = scores

        stage1_time = time.time() - pipeline_start
        high_risk_accounts = [acc for acc, s in scores.items() if s >= risk_threshold]

        # Top-K fallback: always investigate at least the top 20 accounts
        if len(high_risk_accounts) < 20:
            sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            high_risk_accounts = [acc for acc, _ in sorted_all[:20]]
            st.info(
                f"Only {sum(1 for _, s in scores.items() if s >= risk_threshold)} accounts "
                f"above threshold — using **Top 20** accounts as fallback."
            )

        st.success(
            f"Stage 1 complete in {stage1_time:.2f}s — "
            f"{len(high_risk_accounts)} accounts proceeding to Stage 2"
        )

        st.markdown("**Top 10 High-Risk Accounts (GNN)**")
        top_risky = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        st.table(pd.DataFrame(top_risky, columns=["Account ID", "Suspicion Score"]))

        # ── Stage 2: Subgraph Extraction ──────────────────────────────────
        st.divider()
        st.markdown("#### Stage 2 — Subgraph Extraction")
        with st.spinner("Building full graph & extracting suspicious neighborhood..."):
            G_full = build_graph(df)
            subG = extract_suspicious_subgraph(G_full, high_risk_accounts, hops=hop_radius)
            st.session_state.subgraph_nodes = list(subG.nodes())

        st.success(
            f"Extracted subgraph: **{subG.number_of_nodes()}** nodes, "
            f"**{subG.number_of_edges()}** edges "
            f"(from full graph of {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges)"
        )

        # ── Stage 3: Deterministic Search on Subgraph ─────────────────────
        st.divider()
        st.markdown("#### Stage 3 — Deterministic Cycle & Smurfing Detection")
        st.caption("Running exact algorithms on the focused subgraph only.")

        with st.spinner("Detecting cycles (ring muling) in subgraph..."):
            cycles = detect_cycles(subG)
        with st.spinner("Detecting smurfing (fan-in/fan-out) in subgraph..."):
            smurfs = detect_smurfing(subG)

        st.session_state.detected_rings = []

        for i, cycle in enumerate(cycles):
            st.session_state.detected_rings.append({
                "ring_id": f"CYCLE_{i}",
                "pattern_type": "cycle",
                "member_accounts": cycle,
                "risk_score": 90.0,
            })
        for i, smurf in enumerate(smurfs):
            st.session_state.detected_rings.append({
                "ring_id": f"SMURF_{i}",
                "pattern_type": smurf['type'],
                "member_accounts": smurf['members'] + [smurf['center']],
                "risk_score": 80.0,
            })

        total_time = time.time() - pipeline_start
        st.success(
            f"Stage 3 complete — {len(cycles)} cycles, {len(smurfs)} smurfing patterns "
            f"| Total pipeline time: **{total_time:.2f}s**"
        )

        # Summary Table
        st.markdown("#### Fraud Ring Summary")
        if st.session_state.detected_rings:
            summary_data = []
            for ring in st.session_state.detected_rings:
                summary_data.append({
                    "Ring ID": ring['ring_id'],
                    "Type": ring['pattern_type'],
                    "Members": len(ring['member_accounts']),
                    "Risk Score": ring['risk_score'],
                })
            st.dataframe(pd.DataFrame(summary_data))
        else:
            st.info("No rings detected in the extracted subgraph.")

with tab2:
    st.markdown("### Interactive Graph Visualization")
    st.info("Visualizing a subgraph of high-risk nodes (Top 50) to maintain performance.")
    
    if st.session_state.suspicion_scores:
        # Filter top nodes
        sorted_accs = sorted(st.session_state.suspicion_scores.items(), key=lambda x: x[1], reverse=True)
        top_accs = [acc for acc, score in sorted_accs[:50]]
        
        # Also include nodes from detected rings
        ring_nodes = set()
        for ring in st.session_state.detected_rings:
            for member in ring['member_accounts']:
                ring_nodes.add(member)
        
        display_nodes = list(set(top_accs) | ring_nodes)
        if len(display_nodes) > 100:
            display_nodes = display_nodes[:100] # Cap at 100 for viz
            
        # Build Subgraph
        G_full = build_graph(df)
        subG = G_full.subgraph(display_nodes)
        
        # Convert to agraph nodes/edges
        nodes = []
        edges = []
        
        for node_id in subG.nodes():
            score = st.session_state.suspicion_scores.get(node_id, 0)
            color = "#00ff00" # Green
            if score > 50: color = "#ffff00" # Yellow
            if score > 80: color = "#ff0000" # Red
            
            nodes.append(Node(id=node_id, label=node_id, size=15, color=color))
            
        for u, v, data in subG.edges(data=True):
            edges.append(Edge(source=u, target=v, label=f"${data['amount']:.0f}"))
            
        config = Config(width=800, height=600, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
        
        return_value = agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.warning("Please run the Analysis first to generate scores and visualize high-risk nodes.")

with tab3:
    st.subheader("Download Results")
    
    if st.session_state.suspicion_scores and st.session_state.detected_rings:
        # Prepare JSON structure
        suspicious_list = []
        for acc, score in st.session_state.suspicion_scores.items():
            if score > 50:
                # Find associated ring if any
                ring_id = None
                patterns = []
                for ring in st.session_state.detected_rings:
                    if acc in ring['member_accounts']:
                        ring_id = ring['ring_id']
                        patterns.append(ring['pattern_type'])
                
                suspicious_list.append({
                    "account_id": acc,
                    "suspicion_score": score,
                    "detected_patterns": list(set(patterns)),
                    "ring_id": ring_id
                })
        
        # Sort by score
        suspicious_list.sort(key=lambda x: x['suspicion_score'], reverse=True)
        
        output_json = {
            "suspicious_accounts": suspicious_list,
            "fraud_rings": st.session_state.detected_rings,
            "summary": {
                "total_accounts_analyzed": len(pd.concat([df['sender_id'], df['receiver_id']]).unique()),
                "suspicious_accounts_flagged": len(suspicious_list),
                "fraud_rings_detected": len(st.session_state.detected_rings)
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
