
import pandas as pd
import networkx as nx
from src.graph_algo import build_graph, detect_cycles

def check_cycles():
    print("Loading dataset...")
    df = pd.read_csv("money_muling_dataset_12k (1).csv")
    

    print("Building graph...")
    G = build_graph(df)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Debug: Check reciprocal edges
    simple_G = nx.DiGraph(G)
    reciprocal_pairs = 0
    for u, v in simple_G.edges():
        if simple_G.has_edge(v, u):
            reciprocal_pairs += 1
    # Each pair is counted twice
    print(f"Reciprocal edges count: {reciprocal_pairs} (pairs: {reciprocal_pairs//2})")
    
    # Check SCCs before cycle detection
    sccs = [scc for scc in nx.strongly_connected_components(simple_G) if len(scc) >= 3]
    print(f"Number of SCCs with size >= 3: {len(sccs)}")
    if sccs:
        print(f"Largest SCC size: {max(len(scc) for scc in sccs)}")
    

    print("Detecting cycles with new DFS method...")
    import time
    start_time = time.time()
    cycles = detect_cycles(G, max_length=5)
    end_time = time.time()
    
    print(f"Detected {len(cycles)} cycles in {end_time - start_time:.2f}s.")
    for i, cycle in enumerate(cycles[:5]):
        print(f"Cycle {i+1}: {cycle}")
        
    print("\nDetecting Smurfing (Fan-in/Fan-out)...")
    start_time = time.time()
    # Assuming smurfing detection logic is imported or available in graph_algo
    from src.graph_algo import detect_smurfing, detect_shells
    smurfs = detect_smurfing(df)
    end_time = time.time()
    print(f"Detected {len(smurfs)} smurfing patterns in {end_time - start_time:.2f}s.")
    for i, s in enumerate(smurfs[:3]):
        print(f"Smurf {i+1}: Type={s['type']}, Center={s['center']}, Count={len(s['members'])}")


    print("\nDetecting Shells...")
    
    # Debug Shell inner logic
    low_activity_nodes = [n for n in G.nodes() if G.degree(n) <= 3 and G.degree(n) >= 2]
    print(f"Debug: Found {len(low_activity_nodes)} low activity nodes (degree 2-3).")
    if low_activity_nodes:
        sub = G.subgraph(low_activity_nodes)
        comps = list(nx.weakly_connected_components(sub))
        if comps:
            print(f"Debug: Found {len(comps)} components in low-activity subgraph.")
            print(f"Debug: Max component size: {max(len(c) for c in comps)}")
            
    start_time = time.time()
    shells = detect_shells(G)
    end_time = time.time()
    print(f"Detected {len(shells)} shell chains in {end_time - start_time:.2f}s.")
    for i, s in enumerate(shells[:3]):
        print(f"Shell {i+1}: {s}")

    # Debug: Check degree stats
    print("\nDebug: Degree Stats")
    degrees = [d for n, d in G.degree()]
    max_deg = max(degrees)
    print(f"Max degree: {max_deg}")
    import numpy as np
    print(f"Mean degree: {np.mean(degrees):.2f}")
    
    # Check potential fan-in/out candidates
    fan_in_cand = [n for n, d in G.in_degree() if d >= 10]
    print(f"Nodes with in-degree >= 10: {len(fan_in_cand)}")
    if fan_in_cand:
        # Check flow for first candidate
        n = fan_in_cand[0]
        in_edges = G.in_edges(n, data=True)
        out_edges = G.out_edges(n, data=True)
        tin = sum(float(d['amount']) for u,v,d in in_edges)
        tout = sum(float(d['amount']) for u,v,d in out_edges)
        print(f"Example Candidate {n}: In={tin:.2f}, Out={tout:.2f}, Ratio={tout/tin if tin>0 else 0:.2f}")


if __name__ == "__main__":
    check_cycles()
