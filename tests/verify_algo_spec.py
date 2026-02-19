
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import networkx as nx
from src.graph_algo import detect_cycles, detect_smurfing, detect_shells, build_graph
from datetime import datetime, timedelta

def test_cycle_detection():
    print("Testing Cycle Detection...")
    # A -> B -> C -> A
    # Add dummy timestamps and transaction IDs
    base_time = datetime(2023, 1, 1, 10, 0, 0)
    data = [
        {'transaction_id': '1', 'sender_id': 'A', 'receiver_id': 'B', 'amount': 100, 'timestamp': base_time},
        {'transaction_id': '2', 'sender_id': 'B', 'receiver_id': 'C', 'amount': 100, 'timestamp': base_time},
        {'transaction_id': '3', 'sender_id': 'C', 'receiver_id': 'A', 'amount': 100, 'timestamp': base_time}
    ]
    df = pd.DataFrame(data)
    G = build_graph(df)
    cycles = detect_cycles(G)
    
    assert len(cycles) > 0, "Failed to detect A->B->C->A cycle"
    # Cycle format in app.py logic is list of nodes [A, B, C] (or rotated)
    # Actually it returns path [A, B, C] for A->B->C->A
    print(f"PASS: Detected cycles: {cycles}")

def test_smurfing_detection():
    print("\nTesting Smurfing Detection...")
    base_time = datetime(2023, 1, 1, 10, 0, 0)
    
    # Fan-in: 10 users -> Target
    data = []
    for i in range(10):
        data.append({
            'sender_id': f'User_{i}', 
            'receiver_id': 'Target_In', 
            'amount': 100, 
            'timestamp': base_time + timedelta(hours=i) # All within 10 hours (< 72h)
        })
        
    # Fan-out: Target -> 10 users
    for i in range(10):
        data.append({
            'sender_id': 'Target_Out', 
            'receiver_id': f'User_{i}', 
            'amount': 100, 
            'timestamp': base_time + timedelta(hours=i)
        })

    df = pd.DataFrame(data)
    # detect_smurfing now expects DF
    smurfs = detect_smurfing(df)
    
    fan_in = [s for s in smurfs if s['type'] == 'fan_in_smurfing']
    fan_out = [s for s in smurfs if s['type'] == 'fan_out_smurfing']
    
    assert len(fan_in) > 0, "Failed to detect Fan-In"
    assert len(fan_out) > 0, "Failed to detect Fan-Out"
    print(f"PASS: Detected {len(fan_in)} fan-in and {len(fan_out)} fan-out smurfs.")

def test_shell_detection():
    print("\nTesting Shell Detection...")
    # Start -> S1 -> S2 -> End
    # S1 and S2 must have low degree (2-3)
    base_time = datetime(2023, 1, 1, 10, 0, 0)
    data = [
        {'transaction_id': '1', 'sender_id': 'Start', 'receiver_id': 'S1', 'amount': 100, 'timestamp': base_time},
        {'transaction_id': '2', 'sender_id': 'S1', 'receiver_id': 'S2', 'amount': 100, 'timestamp': base_time},
        {'transaction_id': '3', 'sender_id': 'S2', 'receiver_id': 'End', 'amount': 100, 'timestamp': base_time}
    ]
    df = pd.DataFrame(data)
    G = build_graph(df)
    shells = detect_shells(G)
    
    assert len(shells) > 0, "Failed to detect Start->S1->S2->End shell chain"
    print(f"PASS: Detected shells: {shells}")

if __name__ == "__main__":
    test_cycle_detection()
    test_smurfing_detection()
    test_shell_detection()
    print("\nALL TESTS PASSED!")
