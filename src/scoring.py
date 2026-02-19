
import pandas as pd
import networkx as nx

def calculate_suspicion_scores(G, df, cycles, smurfs, shells, gnn_scores=None):
    """
    Calculate unified suspicion scores (0-100) for all accounts.
    
    Base Score:
    - GNN Inference Score (if available) -> 0-100 range.
    
    Pattern Bonuses (Additive):
    - Cycle Membership: +50
    - Smurfing (Fan-in/Fan-out): +40
    - Layered Shell: +30
    
    Constraints:
    - Max Score = 100
    - Scores are rounded to 1 decimal place.
    """
    
    # Initialize scores
    scores = {}
    
    # 1. Base Scores from GNN
    if gnn_scores:
        for acc, score in gnn_scores.items():
            scores[acc] = float(score) # Assumed 0-100 from main.py
    else:
        # Default base if GNN fails or not loaded
        for node in G.nodes():
            scores[node] = 0.0

    # Ensure all graph nodes are in scores
    for node in G.nodes():
        if node not in scores:
            scores[node] = 0.0

    # 2. Apply Pattern Bonuses
    
    # Cycles
    for cycle in cycles:
        for node in cycle:
            if node in scores:
                scores[node] += 50.0
    
    # Smurfing
    for smurf in smurfs:
        center = smurf['center']
        members = smurf['members']
        
        # Center gets full bonus
        if center in scores:
            scores[center] += 40.0
            
        # Members get partial? PDF doesn't specify diff, implies all involved are suspicious.
        # Let's give members full bonus too for now, as they are part of the ring.
        for member in members:
            if member in scores:
                scores[member] += 40.0

    # Shells
    for shell in shells:
        for node in shell:
            if node in scores:
                scores[node] += 30.0

    # 3. Normalization and Formatting
    final_scores = {}
    for acc, raw_score in scores.items():
        # Cap at 100
        final_score = min(100.0, raw_score)
        # Round
        final_scores[acc] = round(final_score, 1)
        
    return final_scores

def format_suspicious_accounts(scores, rings):
    """
    Format the output list for JSON:
    { "account_id": "...", "suspicion_score": 87.5, "detected_patterns": [...], "ring_id": "..." }
    """
    # 1. Map accounts to rings and patterns
    acc_info = {}
    
    for ring in rings:
        ring_id = ring['ring_id']
        pattern = ring['pattern_type']
        for member in ring['member_accounts']:
            if member not in acc_info:
                acc_info[member] = {'patterns': set(), 'ring_ids': []}
            acc_info[member]['patterns'].add(pattern)
            acc_info[member]['ring_ids'].append(ring_id)
            
    # 2. Build list
    result_list = []
    
    # Sort by score descending
    sorted_accounts = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    for acc, score in sorted_accounts:
        if score < 1: continue # Filter out zero/low scores to keep JSON clean? Or keep all?
        # PDF says "Suspicious nodes MUST be visually distinct... suspicious_accounts array"
        # Usually implies a threshold. Let's use > 10 as a loose threshold for "suspicious_accounts" list.
        if score < 10: continue

        info = acc_info.get(acc, {'patterns': set(), 'ring_ids': []})
        patterns = sorted(list(info['patterns']))
        
        # If no pattern but high GNN score
        if not patterns and score > 50:
            patterns.append("high_risk_model_score")
            
        # Ring ID: Join if multiple, or pick first. PDF implies string.
        ring_id_str = ",".join(info['ring_ids']) if info['ring_ids'] else None
        
        result_list.append({
            "account_id": acc,
            "suspicion_score": score,
            "detected_patterns": patterns,
            "ring_id": ring_id_str
        })
        
    return result_list
