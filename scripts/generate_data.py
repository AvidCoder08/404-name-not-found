import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
import faker
import json
import networkx as nx
import os

# Configuration
NUM_ACCOUNTS = 1000
NUM_HONEST_TRANSACTIONS = 5000
NUM_MULE_RINGS = 10
MULE_RING_SIZE_RANGE = (3, 5)
NUM_SMURFING = 5
SMURF_FAN_SIZE = (10, 20)
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 1, 31)

fake = faker.Faker()

def generate_accounts(n):
    return [f"ACC_{str(i).zfill(5)}" for i in range(n)]

def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def generate_honest_transactions(accounts, n):
    transactions = []
    for _ in range(n):
        sender = random.choice(accounts)
        receiver = random.choice(accounts)
        while sender == receiver:
            receiver = random.choice(accounts)
        
        amount = round(random.uniform(10.0, 5000.0), 2)
        timestamp = random_date(START_DATE, END_DATE)
        
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": amount,
            "timestamp": timestamp,
            "is_fraud": False,
            "pattern": None
        })
    return transactions

def inject_mule_rings(accounts, num_rings, size_range):
    fraud_txs = []
    fraud_rings = []
    
    available_accounts = list(accounts)
    
    for i in range(num_rings):
        ring_size = random.randint(*size_range)
        if len(available_accounts) < ring_size:
            break
            
        ring_members = random.sample(available_accounts, ring_size)
        # Remove used accounts to avoid overlap for simplicity in ground truth, 
        # though real fraud mimics overlap. Keeping it simple for now.
        for member in ring_members:
            available_accounts.remove(member)
            
        ring_id = f"RING_{str(i).zfill(3)}"
        base_amount = round(random.uniform(10000.0, 50000.0), 2)
        start_time = random_date(START_DATE, END_DATE - timedelta(hours=48))
        
        ring_txs = []
        # Create cycle A -> B -> C -> ... -> A
        for j in range(ring_size):
            sender = ring_members[j]
            receiver = ring_members[(j + 1) % ring_size]
            
            # Add some jitter to amount and time
            amount = round(base_amount * random.uniform(0.95, 0.99), 2) # Structuring fees
            timestamp = start_time + timedelta(hours=random.randint(1, 12) * (j+1))
            
            tx = {
                "transaction_id": str(uuid.uuid4()),
                "sender_id": sender,
                "receiver_id": receiver,
                "amount": amount,
                "timestamp": timestamp,
                "is_fraud": True,
                "pattern": "cycle",
                "ring_id": ring_id
            }
            fraud_txs.append(tx)
            ring_txs.append(tx)
            
        fraud_rings.append({
            "ring_id": ring_id,
            "member_accounts": ring_members,
            "pattern_type": "cycle",
            "risk_score": 95.0 + random.random() * 5
        })
        
    return fraud_txs, fraud_rings

def inject_smurfing(accounts, num_instances, fan_size_range):
    fraud_txs = []
    smurf_groups = []
    available_accounts = list(accounts) # Reuse full list or distinct? Let's overlap with general pop
    
    for i in range(num_instances):
        is_fan_out = random.choice([True, False])
        fan_size = random.randint(*fan_size_range)
        
        main_account = f"SMURF_MAIN_{i}" # Create distinct bad actor or use existing
        # Let's use existing for more realism, but ensure they aren't part of known valid ones?
        # Actually creating new IDs for specific roles helps debug.
        # But per requirements, nodes are sender/receiver IDs.
        
        main_account = random.choice(accounts)
        others = random.sample([a for a in accounts if a != main_account], fan_size)
        
        start_time = random_date(START_DATE, END_DATE - timedelta(hours=72))
        total_amount = round(random.uniform(50000.0, 100000.0), 2)
        chunk_amount = round(total_amount / fan_size, 2)
        
        smurf_txs = []
        for j, other in enumerate(others):
            timestamp = start_time + timedelta(minutes=random.randint(1, 120))
            
            if is_fan_out:
                sender = main_account
                receiver = other
            else: # Fan-in
                sender = other
                receiver = main_account
            
            tx = {
                "transaction_id": str(uuid.uuid4()),
                "sender_id": sender,
                "receiver_id": receiver,
                "amount": chunk_amount,
                "timestamp": timestamp,
                "is_fraud": True,
                "pattern": "fan_out" if is_fan_out else "fan_in",
                "ring_id": f"SMURF_{i}"
            }
            fraud_txs.append(tx)
            smurf_txs.append(tx)
            
        smurf_groups.append({
            "ring_id": f"SMURF_{i}",
            "member_accounts": [main_account] + others,
            "pattern_type": "smurfing",
            "risk_score": 85.0 + random.random() * 10
        })
        
    return fraud_txs, smurf_groups

def main():
    print("Generating accounts...")
    accounts = generate_accounts(NUM_ACCOUNTS)
    
    print("Generating honest transactions...")
    honest_txs = generate_honest_transactions(accounts, NUM_HONEST_TRANSACTIONS)
    
    print("Injecting Mule Rings...")
    ring_txs, rings = inject_mule_rings(accounts, NUM_MULE_RINGS, MULE_RING_SIZE_RANGE)
    
    print("Injecting Smurfing...")
    smurf_txs, smurfs = inject_smurfing(accounts, NUM_SMURFING, SMURF_FAN_SIZE)
    
    all_txs = honest_txs + ring_txs + smurf_txs
    random.shuffle(all_txs)
    
    df = pd.DataFrame(all_txs)
    
    # Save CSV
    os.makedirs("data", exist_ok=True)
    df[['transaction_id', 'sender_id', 'receiver_id', 'amount', 'timestamp']].to_csv("data/transactions.csv", index=False)
    print(f"Saved {len(df)} transactions to data/transactions.csv")
    
    # Generate Ground Truth JSON
    suspicious_accounts = []
    
    # Collect all suspicious accounts
    sus_acc_map = {}
    
    for ring in rings:
        for acc in ring['member_accounts']:
            if acc not in sus_acc_map:
                sus_acc_map[acc] = {
                    "account_id": acc,
                    "suspicion_score": ring['risk_score'],
                    "detected_patterns": [ring['pattern_type']],
                    "ring_id": ring['ring_id']
                }
            else:
                sus_acc_map[acc]["detected_patterns"].append(ring['pattern_type'])
                
    for smurf in smurfs:
        for acc in smurf['member_accounts']:
             if acc not in sus_acc_map:
                sus_acc_map[acc] = {
                    "account_id": acc,
                    "suspicion_score": smurf['risk_score'],
                    "detected_patterns": [smurf['pattern_type']],
                    "ring_id": smurf['ring_id']
                }
             else:
                sus_acc_map[acc]["detected_patterns"].append(smurf['pattern_type'])
                
    suspicious_accounts = list(sus_acc_map.values())
    
    ground_truth = {
        "suspicious_accounts": suspicious_accounts,
        "fraud_rings": rings + smurfs,
        "summary": {
            "total_accounts_analyzed": NUM_ACCOUNTS,
            "suspicious_accounts_flagged": len(suspicious_accounts),
            "fraud_rings_detected": len(rings) + len(smurfs)
        }
    }
    
    with open("data/ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    print("Saved ground truth to data/ground_truth.json")

if __name__ == "__main__":
    main()
