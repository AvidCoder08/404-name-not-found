
import networkx as nx
import pandas as pd
from datetime import timedelta

def get_bounded_cycles(G, min_len=3, max_len=5):
    """Detect cycles of length 3 to 5 in the subgraph."""
    raw_cycles = []
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
                        raw_cycles.append(path)
                elif neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
    
    # Filter for temporal consistency
    valid_cycles = []
    for cycle in raw_cycles:
        # Check if "real" (temporally consistent)
        # We need to find ONE valid flow sequence (edges strictly increasing in time)
        
        edge_timestamps = []
        possible = True
        
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if not G.has_edge(u, v):
                possible = False
                break
            
            # Get all timestamps for this edge (u, v)
            ts_list = []
            # G is likely a MultiDiGraph, so we iterate over keys
            if G.is_multigraph():
                 for k, data in G[u][v].items():
                    if 'timestamp' in data:
                        ts = data['timestamp']
                        # Ensure it's a pandas timestamp or datetime
                        if isinstance(ts, str):
                            ts = pd.to_datetime(ts)
                        ts_list.append(ts)
            else:
                data = G[u][v]
                if 'timestamp' in data:
                    ts = data['timestamp']
                    if isinstance(ts, str):
                        ts = pd.to_datetime(ts)
                    ts_list.append(ts)
            
            edge_timestamps.append(ts_list)
        
        if not possible:
            continue

        # DFS to find increasing path
        def find_increasing_path(idx, current_chain_len, last_ts, start_idx):
            if current_chain_len == len(cycle):
                return True
            
            next_edge_idx = (start_idx + idx) % len(cycle)
            candidates = edge_timestamps[next_edge_idx]
            
            for ts in candidates:
                if ts > last_ts:
                    if find_increasing_path(idx + 1, current_chain_len + 1, ts, start_idx):
                        return True
            return False

        # Try starting flow at each edge
        is_temporal = False
        L = len(cycle)
        for start_idx in range(L):
            for start_ts in edge_timestamps[start_idx]:
                if find_increasing_path(1, 1, start_ts, start_idx):
                    is_temporal = True
                    break
            if is_temporal:
                break
        
        if is_temporal:
            valid_cycles.append(cycle)
            
    return valid_cycles

def get_temporal_smurfs(df):
    """
    Detect 10+ transactions Fan-in/Fan-out within a 72-hour window.
    Improved with Merchant/Payroll Filtering:
    - Exclude if Amount Coefficient of Variation (CV) > 0.5 (Merchants have variable amounts, mules have structured scale).
    - Exclude if Flow Balance is normal for merchant (Fan-in should NOT just stay there, it should move out).
    """
    smurfs = []

    if 'timestamp' not in df.columns:
        return []

    # Helper for CV
    def is_structured_amounts(amounts):
        if len(amounts) < 2: return True
        return (amounts.std() / (amounts.mean() + 1e-5)) < 0.5

    # Fan-in (Aggregator)
    # Filter for receivers with >= 10 unique senders
    potential_fan_in = df.groupby('receiver_id')['sender_id'].nunique()
    for recv in potential_fan_in[potential_fan_in >= 10].index:
        # Get all txns for this receiver, sort by time
        txns = df[df['receiver_id'] == recv].sort_values('timestamp')
        
        # Sliding window check
        # We need to find ANY subset of 10+ transactions from different senders within 72h
        # Simplified: Check if 10th txn - 1st txn <= 72h (for any window of size 10)
        # But we need disjoint senders.
        
        # Let's iterate through windows of unique senders if possible, or just strict time window check
        # Improved approach: Use a rolling window on time
        
        txns = txns.reset_index(drop=True)
        if len(txns) < 10: continue

        found = False
        # Vectorized check for time window of at least 10 items
        # checking txns[i+9] - txns[i]
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=10)
        # This is tricky with unique senders constraint.
        # Let's stick to the heuristic: if raw count in 72h > 10, then check uniqueness.
        
        for i in range(len(txns) - 9):
            window = txns.iloc[i:i+100] # Look ahead enough to find 10 unique
            start_time = window.iloc[0]['timestamp']
            end_time = start_time + pd.Timedelta(hours=72)
            
            valid_window = window[window['timestamp'] <= end_time]
            unique_senders = valid_window['sender_id'].unique()
            
            if len(unique_senders) >= 10:
                # FOUND CANDIDATE pattern
                # NOW APPLY FILTERS (Merchant Filter)
                amounts = valid_window['amount']
                
                # Filter 1: Structure check (Mules usually receive similar amounts)
                # Merchants have high variance.
                if not is_structured_amounts(amounts):
                     # Likely a merchant with random purchases
                     continue
                
                # Filter 2: Flow Balance (Fan-in only)
                # Mules accumulate to PASS ON. Merchants accumulate to KEEP.
                # Check total out vs total in for this account in the whole DF
                total_in = df[df['receiver_id'] == recv]['amount'].sum()
                total_out = df[df['sender_id'] == recv]['amount'].sum()
                
                # If they keep > 80% of money, likely a merchant/sink.
                # Mule: total_out should be close to total_in (e.g. > 80% passed on)
                flow_passed_ratio = total_out / (total_in + 1e-5)
                if flow_passed_ratio < 0.1: # Keeps almost everything
                    continue

                smurfs.append({"type": "fan_in_smurfing", "center": recv, "members": unique_senders.tolist()})
                found = True
                break # Found one instance for this node, adequate for flagging
                
    # Fan-out (Disperser)
    potential_fan_out = df.groupby('sender_id')['receiver_id'].nunique()
    for sender in potential_fan_out[potential_fan_out >= 10].index:
        txns = df[df['sender_id'] == sender].sort_values('timestamp')
        
        txns = txns.reset_index(drop=True)
        if len(txns) < 10: continue

        for i in range(len(txns) - 9):
            window = txns.iloc[i:i+100]
            start_time = window.iloc[0]['timestamp']
            end_time = start_time + pd.Timedelta(hours=72)
            
            valid_window = window[window['timestamp'] <= end_time]
            unique_receivers = valid_window['receiver_id'].unique()
            
            if len(unique_receivers) >= 10:
                # FOUND CANDIDATE
                amounts = valid_window['amount']
                
                # Filter 1: Structure check (Payroll is structured, but so is muling)
                # So this filter is less useful for fan-out (Payroll has fixed salaries).
                # But legitimate payroll usually happens monthly/bi-weekly, not random 72h bursts?
                # Actually payroll IS a burst.
                # So we need to distinguish Payroll vs Mule Fan-out.
                # Payroll: Source is a business (High degree, leaves system).
                # Mule Fan-out: Source received funds recently (Cycle/Flow).
                
                # Check Source's Incoming Flow
                total_in = df[df['receiver_id'] == sender]['amount'].sum()
                total_out = df[df['sender_id'] == sender]['amount'].sum()
                
                # Payroll source usually generates money (Capital) or receives bulk.
                # Mule Source receives ~same amount just before.
                # Metric: Flow Balance.
                # If Total In ~= Total Out, likely Mule.
                # If Total Out >> Total In, likely Payroll/Source.
                
                balance_ratio = total_in / (total_out + 1e-5)
                
                if balance_ratio < 0.5: 
                    # Disperses much more than received -> Originator/Payroll
                    continue
                    
                smurfs.append({"type": "fan_out_smurfing", "center": sender, "members": unique_receivers.tolist()})
                break
                
    return smurfs

def get_layered_shells(simple_G, in_degrees, out_degrees):
    """
    Chains of 3+ hops where intermediate accounts have 2-3 total transactions.
    Intermediate Node Logic: 
    - Total Degree (In + Out) is exactly 2 or 3.
    - It basically just receives and sends.
    """
    shells = []
    # Identify candidates: nodes with total degree 2-3 (likely just pass-through)
    shell_candidates = set(
        n for n in simple_G.nodes()
        if 2 <= (in_degrees.get(n, 0) + out_degrees.get(n, 0)) <= 3
        # Must have at least 1 in and 1 out to be a bridge
        and in_degrees.get(n, 0) >= 1 and out_degrees.get(n, 0) >= 1
    )
    
    # Search for pattern: start -> u -> v -> end
    # where u and v are shell_candidates
    # We need a chain of at least 2 intermediates to form a "Layered" Shell network of 3+ hops?
    # PDF says: "chains of 3+ hops where intermediate accounts have only 2–3 total transactions"
    # 3 hops = A -> B -> C -> D. (Intermediate B, C).
    
    for u in shell_candidates:
        # Check neighbors
        for v in simple_G.successors(u):
            if v in shell_candidates and u != v:
                # u -> v is a shell-to-shell link
                # Now extend backwards to 'start' and forwards to 'end'
                
                # Backward extension
                predecessors = list(simple_G.predecessors(u))
                successors = list(simple_G.successors(v))
                
                for start in predecessors:
                    if start == v: continue 
                    for end in successors:
                        if end == u or end == start: continue
                        
                        # Valid Chain: start -> u -> v -> end
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
