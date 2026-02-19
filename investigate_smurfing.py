
import pandas as pd
import networkx as nx
from src.graph_algo import build_graph

def investigate():
    print("Loading dataset...")
    df = pd.read_csv("money_muling_dataset_12k (1).csv")
    G = build_graph(df)
    
    print("Finding high degree nodes...")
    # potential aggregators (Fan-in)
    fan_ins = [n for n, d in G.in_degree() if d >= 10]
    print(f"Nodes with in-degree >= 10: {len(fan_ins)}")
    
    for n in fan_ins[:5]:
        in_edges = list(G.in_edges(n, data=True))
        timestamps = sorted([pd.Timestamp(d['timestamp']) for u, v, d in in_edges])
        duration = timestamps[-1] - timestamps[0]
        
        total_in = sum(float(d['amount']) for u,v,d in in_edges)
        out_edges = list(G.out_edges(n, data=True))
        total_out = sum(float(d['amount']) for u,v,d in out_edges)
        
        ratio = total_out / total_in if total_in > 0 else 0
        
        print(f"Node {n}: In-degree={len(in_edges)}, Duration={duration}, FlowRatio={ratio:.2f}")
        if duration > pd.Timedelta(hours=72):
            print(f"   -> Fails strict 72h window. (Duration > 72h)")
        if ratio <= 0.5:
             print(f"   -> Fails flow ratio check. (Retains/Sinks funds)")

if __name__ == "__main__":
    investigate()
