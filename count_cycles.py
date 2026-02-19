import pandas as pd
import networkx as nx
from src.graph_algo import build_graph, get_bounded_cycles

def check_cycles():
    print("Loading dataset...")
    df = pd.read_csv("money_muling_dataset_12k (1).csv")
    
    print("Building graph...")
    G = build_graph(df)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    print("Detecting cycles (length 3-5)...")
    cycles = get_bounded_cycles(G, min_len=3, max_len=5)
    
    print(f"Total detected cycles (raw): {len(cycles)}")
    
    unique_cycles = set()
    real_cycles = 0
    
    for cycle in cycles:
        # Normalize
        min_node = min(cycle)
        min_idx = cycle.index(min_node)
        normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
        
        if normalized in unique_cycles:
            continue
        unique_cycles.add(normalized)
        
        # Check if "real" (temporally consistent)
        # We need to find ONE valid flow sequence (edges strictly increasing in time)
        # G is MultiDiGraph, G[u][v] is a dict of edges key->attr
        # For simplicity, extract all timestamps for each edge position in the cycle
        
        edge_timestamps = []
        possible = True
        
        for i in range(len(normalized)):
            u = normalized[i]
            v = normalized[(i + 1) % len(normalized)]
            if not G.has_edge(u, v):
                possible = False
                break
            
            # Get all timestamps for this edge (u, v)
            ts_list = []
            for k, data in G[u][v].items():
                if 'timestamp' in data:
                    ts_list.append(pd.to_datetime(data['timestamp']))
                else: 
                     # timestamp is in edge data as 'timestamp' key if built with build_graph
                     # In build_graph: "G.add_edge(s, r, amount=a, timestamp=t...)"
                     # Check if it's a Timestamp object or string
                    pass
            edge_timestamps.append(ts_list)
        
        if not possible:
            continue

        # Check for existence of an increasing sequence
        # We need to select one timestamp t_i from edge_timestamps[i] for each i
        # such that for some rotation k: t_k < t_{k+1} < ... < t_{k+L-1}
        # Since number of txns per edge is usually 1, we can simplify.
        # If >1, we can try to find ANY valid path.
        
        # Helper DFS to find increasing path
        def find_increasing_path(idx, current_chain_len, last_ts, start_idx):
            if current_chain_len == len(normalized):
                return True
            
            # Next edge in the cycle is at (start_idx + idx) % L
            # But wait, we iterate through the cycle length L.
            # Let's say we start at edge `start_idx`. This is the first edge of the flow.
            # Then we need edge `(start_idx + 1) % L` to have timestamp > last_ts.
            
            next_edge_idx = (start_idx + idx) % len(normalized)
            candidates = edge_timestamps[next_edge_idx]
            
            for ts in candidates:
                if ts > last_ts:
                    if find_increasing_path(idx + 1, current_chain_len + 1, ts, start_idx):
                        return True
            return False

        # Try starting flow at each edge
        is_temporal = False
        L = len(normalized)
        for start_idx in range(L):
            # Try each timestamp of the starting edge
            for start_ts in edge_timestamps[start_idx]:
                # We need to find the rest L-1 edges increasing
                if find_increasing_path(1, 1, start_ts, start_idx):
                    is_temporal = True
                    break
            if is_temporal:
                break
        
        if is_temporal:
            real_cycles += 1

    print(f"Unique topological cycles: {len(unique_cycles)}")
    print(f"Real cycles (temporally consistent): {real_cycles}")


if __name__ == "__main__":
    check_cycles()
