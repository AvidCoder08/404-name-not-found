import networkx as nx
from datetime import timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import signal
import threading

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

# Shared stop event for cooperative cancellation
_stop_event = threading.Event()

def _detect_cycles_inner(G, max_length=5, max_cycles=500):
    """Inner function for cycle detection with cooperative cancellation."""
    _stop_event.clear()
    found_cycles = set()
    iterations = 0
    MAX_ITERATIONS = 1000000  # Increased limit for better accuracy
    try:
        # Convert to simple DiGraph to reduce edge count for cycle detection
        simple_G = nx.DiGraph(G)
        
        # Optimization: Remove reciprocal edges (A<->B) because we only care about cycles >= 3
        # Reciprocal edges create thousands of trivial 2-cycles that clog the search
        reciprocal_edges = []
        for u, v in simple_G.edges():
            if simple_G.has_edge(v, u):
                reciprocal_edges.append((u, v))
        simple_G.remove_edges_from(reciprocal_edges)

        sccs = [scc for scc in nx.strongly_connected_components(simple_G) if len(scc) >= 3]
        for scc in sccs:
            if _stop_event.is_set():
                break
            subgraph = simple_G.subgraph(scc)
            for cycle in nx.simple_cycles(subgraph):
                iterations += 1
                if iterations >= MAX_ITERATIONS or _stop_event.is_set():
                    print(f"Cycle detection bailed out after {iterations} iterations.")
                    return [list(c) for c in found_cycles]
                if len(cycle) > max_length:
                    continue
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


def detect_cycles(G, max_length=5, max_cycles=500, timeout=5):
    """
    Detects simple cycles with a hard timeout (default 5s) to prevent hanging.
    Uses cooperative cancellation with a stop event + iteration limit.
    """
    _stop_event.clear()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_detect_cycles_inner, G, max_length, max_cycles)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            _stop_event.set()  # Signal the inner function to stop
            print(f"Cycle detection timed out after {timeout}s, returning partial results.")
            # Give it a moment to check the stop event and return
            try:
                return future.result(timeout=1)
            except (FuturesTimeoutError, Exception):
                return []
        except Exception as e:
            print(f"Cycle detection failed: {e}")
            return []

def detect_smurfing(G, time_window_hours=72, min_fan=10):
    """
    Detects Fan-in (aggregators) and Fan-out (distributors) patterns.
    """
    smurfs = []
    time_window = pd.Timedelta(hours=time_window_hours)
    
    # Iterate over all nodes
    for node in G.nodes():
        # Fan-in
        in_edges = list(G.in_edges(node, data=True))
        if len(in_edges) >= min_fan:
            # Check time window
            timestamps = sorted([pd.Timestamp(d['timestamp']) for u, v, d in in_edges])
            if timestamps and (timestamps[-1] - timestamps[0]) <= time_window:
                 smurfs.append({
                     "type": "fan_in",
                     "center": node,
                     "members": [u for u, v, d in in_edges],
                     "count": len(in_edges)
                 })

        # Fan-out
        out_edges = list(G.out_edges(node, data=True))
        if len(out_edges) >= min_fan:
            timestamps = sorted([pd.Timestamp(d['timestamp']) for u, v, d in out_edges])
            if timestamps and (timestamps[-1] - timestamps[0]) <= time_window:
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
