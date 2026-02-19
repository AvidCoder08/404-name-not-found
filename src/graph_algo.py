import networkx as nx
from datetime import timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

def build_graph(df):
    """Builds a MultiDiGraph from transactions DataFrame (vectorized)."""
    G = nx.MultiDiGraph()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Vectorized: use column arrays directly instead of slow iterrows()
    senders = df['sender_id'].values
    receivers = df['receiver_id'].values
    amounts = df['amount'].values
    timestamps = df['timestamp'].values
    tx_ids = df['transaction_id'].values

    for s, r, a, t, tid in zip(senders, receivers, amounts, timestamps, tx_ids):
        G.add_edge(s, r, amount=a, timestamp=t, transaction_id=tid)
    return G

def _detect_cycles_inner(G, max_length=5, max_cycles=100, max_scc_size=50):
    """Inner function for cycle detection (runs in a thread with timeout)."""
    found_cycles = set()
    try:
        sccs = [scc for scc in nx.strongly_connected_components(G) if 3 <= len(scc) <= max_scc_size]
        # Sort smallest first — small SCCs are fast and most likely to be real fraud rings
        sccs.sort(key=len)
        for scc in sccs:
            subgraph = G.subgraph(scc)
            for cycle in nx.simple_cycles(subgraph, length_bound=max_length):
                if len(cycle) >= 3:
                    min_node_idx = cycle.index(min(cycle))
                    normalized = tuple(cycle[min_node_idx:] + cycle[:min_node_idx])
                    found_cycles.add(normalized)
                if len(found_cycles) >= max_cycles:
                    break
            if len(found_cycles) >= max_cycles:
                break
    except Exception as e:
        print(f"Cycle detection error: {e}")
    return [list(c) for c in found_cycles]


def detect_cycles(G, max_length=5, max_cycles=100, timeout=3):
    """
    Detects simple cycles with a hard timeout (default 3s) to prevent hanging.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_detect_cycles_inner, G, max_length, max_cycles)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            print(f"Cycle detection timed out after {timeout}s, returning partial results.")
            return []
        except Exception as e:
            print(f"Cycle detection failed: {e}")
            return []

def detect_smurfing(G, time_window_hours=72, min_fan=10):
    """
    Detects Fan-in (aggregators) and Fan-out (distributors) patterns.
    """
    smurfs = []
    
    # Iterate over all nodes
    for node in G.nodes():
        # Fan-in
        in_edges = list(G.in_edges(node, data=True))
        if len(in_edges) >= min_fan:
            # Check time window
            timestamps = sorted([d['timestamp'] for u, v, d in in_edges])
            if timestamps and (timestamps[-1] - timestamps[0]) <= timedelta(hours=time_window_hours):
                 smurfs.append({
                     "type": "fan_in",
                     "center": node,
                     "members": [u for u, v, d in in_edges],
                     "count": len(in_edges)
                 })

        # Fan-out
        out_edges = list(G.out_edges(node, data=True))
        if len(out_edges) >= min_fan:
            timestamps = sorted([d['timestamp'] for u, v, d in out_edges])
            if timestamps and (timestamps[-1] - timestamps[0]) <= timedelta(hours=time_window_hours):
                smurfs.append({
                     "type": "fan_out",
                     "center": node,
                     "members": [v for u, v, d in out_edges],
                     "count": len(out_edges)
                 })
                 
    return smurfs

def detect_shells(G, min_path_length=3, max_tx_limit=5):
    """
    Detects long chains where intermediate nodes have low distinct transaction counts.
    (Simplified Layered Shell detection)
    """
    # This is complex. Simplified heuristic:
    # Find paths u -> v -> w ...
    # where degree(v), degree(w) are low (mostly just passing money).
    
    shells = []
    low_activity_nodes = [n for n in G.nodes() if G.degree(n) <= max_tx_limit]
    
    # Construct subgraph of low activity nodes
    sub = G.subgraph(low_activity_nodes)
    
    # Find long paths in this subgraph
    # Since it's low degree, components should be small chains
    components = nx.weakly_connected_components(sub)
    for comp in components:
        if len(comp) >= min_path_length:
            shells.append(list(comp))
            
    return shells


def extract_suspicious_subgraph(G, suspicious_nodes, hops=2):
    """
    Extracts a subgraph containing the suspicious nodes and their k-hop
    neighborhood. This is the Stage 2 "Subgraph Extraction" step of the
    hybrid funnel approach.

    Args:
        G: The full NetworkX graph.
        suspicious_nodes: List of node IDs flagged by the GNN (Stage 1).
        hops: Number of hops to expand around each suspicious node.

    Returns:
        A NetworkX subgraph containing only the relevant neighborhood.
    """
    neighborhood = set(suspicious_nodes)

    for node in suspicious_nodes:
        if node not in G:
            continue
        # ego_graph returns the subgraph induced by the k-hop neighborhood
        ego = nx.ego_graph(G, node, radius=hops, undirected=False)
        neighborhood.update(ego.nodes())

    return G.subgraph(neighborhood).copy()
