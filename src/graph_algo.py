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


def find_cycles_dfs(G, max_length=5):
    """
    Finds elementary cycles of length <= max_length using DFS.
    This is much faster than nx.simple_cycles for dense graphs when we only care about short cycles.
    """
    cycles = set()
    
    # Use a sorted list of nodes for canonical ordering to avoid duplicates (e.g. A-B-C vs B-C-A)
    # But simple DFS with visited set in path is easier for just finding them.
    # To avoid duplicates, we can enforce that we only start DFS from node `start` and only 
    # visit neighbors `n` where `n >= start` (in some ordering).
    # However, standard DFS with path tracking is sufficient if we normalize the cycle.
    
    nodes = sorted(list(G.nodes()))
    
    def dfs(start_node, current_node, path, depth):
        if len(path) > max_length:
            return

        # Optimization: only visit neighbors that are >= start_node to reduce redundant searches
        # (This implies we look for the cycle starting at its smallest node)
        
        for neighbor in G.neighbors(current_node):
            if neighbor == start_node:
                if len(path) >= 3:
                     # Cycle found!
                     cycles.add(tuple(path))
            elif neighbor not in path:
                # Only continue if neighbor > start_node to ensure we find "canonical" rotation
                if str(neighbor) > str(start_node):
                    dfs(start_node, neighbor, path + [neighbor], depth + 1)

    for node in nodes:
        if _stop_event.is_set():
             break
        dfs(node, node, [node], 1)
        
    return [list(c) for c in cycles]


def _detect_cycles_inner(G, max_length=5, max_cycles=500):
    """Inner function for cycle detection with cooperative cancellation."""
    _stop_event.clear()
    
    try:
        # Convert to simple DiGraph
        simple_G = nx.DiGraph(G)
        
        # DO NOT remove reciprocal edges anymore, as they might be part of valid 3-cycles (A->B->C->A)
        # coupled with A<->B. But wait, A<->B is a 2-cycle. 
        # A->B->C->A is a 3-cycle. 
        # The DFS approach handles loose reciprocal edges naturally (cycle length 2 ignored).
        
        # Get SCCs to limit search space
        sccs = [scc for scc in nx.strongly_connected_components(simple_G) if len(scc) >= 3]
        
        all_cycles = []
        for scc in sccs:
            if _stop_event.is_set():
                break
            if len(all_cycles) >= max_cycles:
                break
                
            subgraph = simple_G.subgraph(scc)
            
            # Use our custom DFS instead of nx.simple_cycles
            found = find_cycles_dfs(subgraph, max_length)
            all_cycles.extend(found)
            
        return all_cycles[:max_cycles]

    except Exception as e:
        print(f"Cycle detection error: {e}")
        return []

def detect_cycles(G, max_length=5, max_cycles=500, timeout=10): # increased default timeout slightly
    """
    Detects simple cycles with a hard timeout (default 10s).
    """
    _stop_event.clear()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_detect_cycles_inner, G, max_length, max_cycles)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            _stop_event.set()
            print(f"Cycle detection timed out after {timeout}s.")
            try:
                # Wait briefly for thread to notice stop event
                return future.result(timeout=1)
            except (FuturesTimeoutError, Exception):
                return []
        except Exception as e:
            print(f"Cycle detection failed: {e}")
            return []

def detect_smurfing(G, time_window_hours=72, min_fan=10):
    """
    Detects Fan-in and Fan-out patterns with stricter checks to avoid Merchants/Payroll.
    """
    smurfs = []
    time_window = pd.Timedelta(hours=time_window_hours)
    
    for node in G.nodes():

        # 1. Fan-in (Aggregator)
        in_edges = list(G.in_edges(node, data=True))
        if len(in_edges) >= min_fan:
            # Check for sliding window of `min_fan` transactions within time_window
            # Sort edges by timestamp
            in_edges_sorted = sorted(in_edges, key=lambda x: pd.Timestamp(x[2]['timestamp']))
            
            found_window = False
            relevant_edges = []
            
            # Sliding window check
            for i in range(len(in_edges_sorted) - min_fan + 1):
                window = in_edges_sorted[i : i + min_fan]
                t_start = pd.Timestamp(window[0][2]['timestamp'])
                t_end = pd.Timestamp(window[-1][2]['timestamp'])
                if t_end - t_start <= time_window:
                    found_window = True
                    relevant_edges = window
                    break
            
            if found_window:
                # Check Flow Ratio on the RELEVANT edges (or maybe total? PDF implies the pattern IS the burst)
                # Let's check flow ratio on the burst input vs total output (or burst output)
                
                # If they receive a burst, do they send it out?
                # We should look at total out flow relative to this burst in.
                
                burst_in_amount = sum(float(d['amount']) for u,v,d in relevant_edges)
                
                out_edges = list(G.out_edges(node, data=True))
                total_out = sum(float(d['amount']) for u,v,d in out_edges)
                
                # Strict: Output should be comparable to burst input (or at least significant)
                if burst_in_amount > 0:
                    flow_ratio = total_out / burst_in_amount
                    # If they keep the money, it's not a mule.
                    if flow_ratio > 0.5: 
                         smurfs.append({
                             "type": "fan_in",
                             "center": node,
                             "members": [u for u, v, d in relevant_edges],
                             "count": len(relevant_edges)
                         })

        # 2. Fan-out (Distributor)
        out_edges = list(G.out_edges(node, data=True))
        if len(out_edges) >= min_fan:
            out_edges_sorted = sorted(out_edges, key=lambda x: pd.Timestamp(x[2]['timestamp']))
            
            found_window = False
            relevant_edges = []
            
            for i in range(len(out_edges_sorted) - min_fan + 1):
                window = out_edges_sorted[i : i + min_fan]
                t_start = pd.Timestamp(window[0][2]['timestamp'])
                t_end = pd.Timestamp(window[-1][2]['timestamp'])
                if t_end - t_start <= time_window:
                    found_window = True
                    relevant_edges = window
                    break

            if found_window:
                burst_out_amount = sum(float(d['amount']) for u,v,d in relevant_edges)
                
                in_edges = list(G.in_edges(node, data=True))
                total_in = sum(float(d['amount']) for u,v,d in in_edges)
                
                if burst_out_amount > 0:
                    flow_ratio = total_in / burst_out_amount
                    if flow_ratio > 0.5: 
                         smurfs.append({
                             "type": "fan_out",
                             "center": node,
                             "members": [v for u, v, d in relevant_edges],
                             "count": len(relevant_edges)
                         })
                 
    return smurfs

def detect_shells(G, min_path_length=3, max_tx_limit=3):
    shells = []
    # Strict degree check for intermediate nodes
    # They should basically only receive and send (degree 2 or 3)
    low_activity_nodes = [n for n in G.nodes() if G.degree(n) <= max_tx_limit and G.degree(n) >= 2]
    
    if not low_activity_nodes:
        return []

    # Construct subgraph
    # We only care about edges BETWEEN these low activity nodes
    sub = G.subgraph(low_activity_nodes).copy()
    
    # We want simple paths. 
    components = nx.weakly_connected_components(sub)
    
    for component_nodes in components:
        if len(component_nodes) >= min_path_length + 1:
            comp_sub = sub.subgraph(component_nodes)
            try:
                undir = comp_sub.to_undirected()
                if nx.is_connected(undir):
                    # Check diameter or path length?
                    # The component might be a star or tree. We want a chain.
                    # Simple check: max degree in component <= 2 implies lines/cycles.
                    # Since total degree <= 3, induced degree will be <= 3.
                    # If induced degree is <= 2 for all nodes, it is a collection of lines and cycles.
                    max_deg = max(dict(undir.degree()).values())
                    if max_deg <= 2:
                         shells.append(list(component_nodes))
            except:
                pass
            
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
