
import networkx as nx
import pandas as pd
from datetime import timedelta

def get_bounded_cycles(G, min_len=3, max_len=5):
    """Detect cycles of length 3 to 5 in the subgraph."""
    cycles = []
    # simple_cycles is too slow for dense graphs, but we need strictly bounded cycles.
    # The app.py iterative DFS approach:
    for start_node in G.nodes():
        # Stack stores (current_node, path_so_far)
        stack = [(start_node, [start_node])]
        while stack:
            curr, path = stack.pop()
            if len(path) > max_len:
                continue
            for neighbor in G.successors(curr):
                if neighbor == start_node and len(path) >= min_len:
                    # Found a cycle back to start
                    # Canonical check: only add if start_node is the smallest in the path
                    # This avoids adding the same cycle rotated (e.g. 1-2-3 vs 2-3-1)
                    if min(path) == start_node:
                        cycles.append(path)
                elif neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
    return cycles

def get_temporal_smurfs(df):
    """Detect 10+ transactions Fan-in/Fan-out within a 72-hour window using Pandas."""
    smurfs = []

    if 'timestamp' not in df.columns:
        return []

    # Fan-in (Aggregator)
    potential_fan_in = df.groupby('receiver_id')['sender_id'].nunique()
    # Filter for receivers with >= 10 unique senders
    for recv in potential_fan_in[potential_fan_in >= 10].index:
        # Get all txns for this receiver, sort by time
        txns = df[df['receiver_id'] == recv].sort_values('timestamp').drop_duplicates('sender_id')
        if len(txns) >= 10:
            txns = txns.copy()
            # Calculate time difference between transaction i and transaction i+9 (10th transaction)
            # This checks if ANY window of 10 transactions happened within 72h
            txns['time_diff'] = txns['timestamp'].diff(periods=9)
            
            if (txns['time_diff'] <= pd.Timedelta(hours=72)).any():
                # Get the end time of the first matching window
                match_row = txns[txns['time_diff'] <= pd.Timedelta(hours=72)].iloc[0]
                end_time = match_row['timestamp']
                start_time = end_time - pd.Timedelta(hours=72)
                
                # Retrieve all senders in that window
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
                match_row = txns[txns['time_diff'] <= pd.Timedelta(hours=72)].iloc[0]
                end_time = match_row['timestamp']
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
    # Identify candidates: nodes with total degree 2-3 (likely just pass-through)
    shell_candidates = set(
        n for n in simple_G.nodes()
        if 2 <= (in_degrees.get(n, 0) + out_degrees.get(n, 0)) <= 3
        and in_degrees.get(n, 0) >= 1 and out_degrees.get(n, 0) >= 1
    )
    
    # Search for pattern: start -> u -> v -> end
    # where u and v are shell_candidates
    for u in shell_candidates:
        for v in simple_G.successors(u):
            if v in shell_candidates and u != v:
                # We found a link between two shell candidates.
                # Now look for the 'start' (feeder) and 'end' (collector)
                for start in simple_G.predecessors(u):
                    if start == v: # Avoid immediate cycles u<->v
                        continue
                    for end in simple_G.successors(v):
                        if end == u or end == start: # Avoid loops
                            continue
                        
                        # Found the pattern
                        shells.append([start, u, v, end])
    return shells

def build_graph(df):
    """Builds a MultiDiGraph from transactions DataFrame (vectorized)."""
    G = nx.MultiDiGraph()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Vectorized: use column arrays directly
    senders = df['sender_id'].values
    receivers = df['receiver_id'].values
    amounts = df['amount'].values
    timestamps = df['timestamp'].values if 'timestamp' in df.columns else [None]*len(df)
    tx_ids = df['transaction_id'].values if 'transaction_id' in df.columns else range(len(df))

    for s, r, a, t, tid in zip(senders, receivers, amounts, timestamps, tx_ids):
        G.add_edge(s, r, amount=a, timestamp=t, transaction_id=tid)
    return G

def extract_suspicious_subgraph(G, suspicious_nodes, hops=2):
    """Extracts a subgraph around suspicious nodes."""
    neighborhood = set(suspicious_nodes)
    for node in suspicious_nodes:
        if node not in G:
            continue
        ego = nx.ego_graph(G, node, radius=hops, undirected=False)
        neighborhood.update(ego.nodes())
    return G.subgraph(neighborhood).copy()

# Wrapper functions to maintain some compatibility with main.py structure where possible,
# or simply aliases to match the new names.

def detect_cycles(G, max_length=5):
    """Alias for get_bounded_cycles conforming to expected output format if needed."""
    return get_bounded_cycles(G, min_len=3, max_len=max_length)

def detect_smurfing(df):
    """Alias for get_temporal_smurfs."""
    # NOTE: This now takes DF as input, not G!
    return get_temporal_smurfs(df)

def detect_shells(G):
    """Alias for get_layered_shells."""
    simple_G = nx.DiGraph(G)
    in_degrees = dict(simple_G.in_degree())
    out_degrees = dict(simple_G.out_degree())
    return get_layered_shells(simple_G, in_degrees, out_degrees)
