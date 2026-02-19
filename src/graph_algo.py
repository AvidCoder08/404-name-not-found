import networkx as nx
from datetime import timedelta
import pandas as pd

def build_graph(df):
    """Builds a MultiDiGraph from transactions DataFrame."""
    G = nx.MultiDiGraph()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    for _, row in df.iterrows():
        G.add_edge(
            row['sender_id'], 
            row['receiver_id'], 
            amount=row['amount'], 
            timestamp=row['timestamp'],
            transaction_id=row['transaction_id']
        )
    return G

def detect_cycles(G, max_length=5):
    """
    Detects simple cycles in the graph up to max_length.
    Returns a list of cycles (list of nodes).
    """
    # NetworkX simple_cycles is computationally expensive for large graphs.
    # We can limit the search or use a depth-limited DFS.
    # For this challenge size (10k txs), simple_cycles might be okay if graph is sparse,
    # but strictly finding short cycles is better.
    
    cycles = []
    
    # Use recursive DFS with depth limit
    def dfs(start_node, current_node, path, visited_edges):
        if len(path) > max_length:
            return
        
        for neighbor in G.neighbors(current_node):
            if neighbor == start_node and len(path) >= 3:
                cycles.append(path + [start_node])
                return
            
            if neighbor not in path:
                dfs(start_node, neighbor, path + [neighbor], visited_edges)

    # Note: The above naive DFS is still slow. 
    # Use nx.simple_cycles but filter by length? 
    # Or iterate over nodes and run BFS/DFS for limited depth.
    
    # Better approach for finding short cycles:
    try:
        # nx.simple_cycles returns an iterator. We can iterate and break/filter.
        # But it finds ALL elementary cycles, which is exponential.
        # Let's try finding cycles for each SCC or just use a limited search.
        
        # Optimization: Only look at strongly connected components
        sccs = list(nx.strongly_connected_components(G))
        for scc in sccs:
            if len(scc) < 3: continue
            
            subgraph = G.subgraph(scc)
            # This is still risky if SCC is large.
            # Let's stick to a robust simple approach:
            # For each edge (u, v), look for path v -> u of length 1 to max_length-1
            pass
            
    except Exception:
        pass

    # Correct approach for large graphs: 
    # 1. Iterate all nodes. 2. DFS limited to depth K.
    # To avoid duplicates, impose ordering or use a set of sorted tuples.
    
    found_cycles = set()
    
    nodes = list(G.nodes())
    for start_node in nodes:
        # BFS to find paths back to start_node
        queue = [(start_node, [start_node])]
        while queue:
            curr, path = queue.pop(0)
            if len(path) > max_length:
                continue
            
            # Get neighbors. Note: MultiDiGraph neighbors might have duplicates, use set
            neighbors = set(G.neighbors(curr))
            
            for neighbor in neighbors:
                 if neighbor == start_node:
                     if len(path) >= 3:
                        cycle = tuple(path)
                        # Normalize cycle to start with min node to avoid duplicates
                        min_node_idx = cycle.index(min(cycle))
                        normalized_cycle = cycle[min_node_idx:] + cycle[:min_node_idx]
                        found_cycles.add(normalized_cycle)
                 elif neighbor not in path:
                     queue.append((neighbor, path + [neighbor]))
                     
    return [list(c) for c in found_cycles]

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
