"""
Synthetic Training Data Generator for Money Muling Detection.

Generates a dataset with KNOWN fraud patterns (cycles, smurfing, shells)
and legitimate traps (merchants, payroll) that match the hackathon spec.
Output: spec-format CSV + ground truth labels.
"""
import random
import uuid
import json
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


def _rand_ts(base, window_hours=72):
    """Random timestamp within window_hours of base."""
    return base + timedelta(seconds=random.randint(0, int(window_hours * 3600)))


def _acc_id():
    """Generate a realistic account ID."""
    return f"ACC_{random.randint(10000, 99999)}"


def generate_synthetic_dataset(
    seed=42,
    n_normal_accounts=500,
    n_normal_txns=5000,
    n_cycle_rings=8,
    n_smurf_rings=5,
    n_shell_chains=4,
    n_merchant_traps=5,
    n_payroll_traps=3,
    output_dir="data",
):
    """
    Generate a synthetic dataset with planted fraud patterns and legitimate traps.

    Returns:
        df: DataFrame in spec format (transaction_id, sender_id, receiver_id, amount, timestamp)
        labels: dict mapping account_id → is_suspicious (0 or 1)
    """
    random.seed(seed)
    np.random.seed(seed)

    transactions = []
    suspicious_accounts = set()
    ring_assignments = {}  # account_id → ring_id
    pattern_assignments = {}  # account_id → list of patterns
    rings = []  # ring metadata

    base_time = datetime(2024, 1, 15, 8, 0, 0)
    txn_counter = [0]

    def add_txn(sender, receiver, amount, ts):
        txn_counter[0] += 1
        transactions.append({
            'transaction_id': f'TXN_{txn_counter[0]:07d}',
            'sender_id': sender,
            'receiver_id': receiver,
            'amount': round(amount, 2),
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        })

    # ────────────────────────────────────────────
    # 1. NORMAL ACCOUNTS (background noise)
    # ────────────────────────────────────────────
    normal_accs = [_acc_id() for _ in range(n_normal_accounts)]
    for _ in range(n_normal_txns):
        s, r = random.sample(normal_accs, 2)
        amt = round(np.random.lognormal(mean=5, sigma=2), 2)  # Wide range of amounts
        ts = _rand_ts(base_time, window_hours=720)  # Spread over 30 days
        add_txn(s, r, amt, ts)

    # ────────────────────────────────────────────
    # 2. MERCHANT TRAPS (look like fan-in but legit)
    # Many customers → 1 merchant, spread over time, varied amounts
    # ────────────────────────────────────────────
    for i in range(n_merchant_traps):
        merchant = f"MERCHANT_{i:03d}"
        n_customers = random.randint(15, 40)
        customers = random.sample(normal_accs, min(n_customers, len(normal_accs)))

        for cust in customers:
            # Merchants get payments spread across many days (not bursty)
            ts = _rand_ts(base_time, window_hours=720)
            # Varied amounts (different products/services)
            amt = round(np.random.lognormal(mean=4, sigma=1.5), 2)
            add_txn(cust, merchant, amt, ts)

        # Merchant also sends some payments (to suppliers, refunds)
        for _ in range(random.randint(3, 8)):
            ts = _rand_ts(base_time, window_hours=720)
            amt = round(np.random.lognormal(mean=6, sigma=1), 2)
            add_txn(merchant, random.choice(normal_accs), amt, ts)

    # ────────────────────────────────────────────
    # 3. PAYROLL TRAPS (look like fan-out but legit)
    # 1 company → many employees, regular amounts, spread over time
    # ────────────────────────────────────────────
    for i in range(n_payroll_traps):
        company = f"PAYROLL_{i:03d}"
        n_employees = random.randint(15, 35)
        employees = random.sample(normal_accs, min(n_employees, len(normal_accs)))

        # Payroll runs monthly — 2 pay periods
        for pay_period in range(2):
            pay_date = base_time + timedelta(days=pay_period * 14)
            base_salary = np.random.uniform(2000, 8000)
            for emp in employees:
                # Salaries are similar but not identical
                salary = round(base_salary + np.random.normal(0, 500), 2)
                ts = _rand_ts(pay_date, window_hours=2)  # Payroll within 2hrs
                add_txn(company, emp, salary, ts)

        # Company also receives revenue
        for _ in range(random.randint(5, 15)):
            ts = _rand_ts(base_time, window_hours=720)
            amt = round(np.random.lognormal(mean=8, sigma=1), 2)
            add_txn(random.choice(normal_accs), company, amt, ts)

    # ────────────────────────────────────────────
    # 4. CYCLE RINGS (A → B → C → A)
    # ────────────────────────────────────────────
    for i in range(n_cycle_rings):
        ring_id = f"RING_CYCLE_{i:03d}"
        cycle_len = random.randint(3, 5)
        cycle_accs = [_acc_id() for _ in range(cycle_len)]

        # Mark all as suspicious
        for acc in cycle_accs:
            suspicious_accounts.add(acc)
            ring_assignments[acc] = ring_id
            pattern_assignments.setdefault(acc, []).append(f"cycle_length_{cycle_len}")

        # Create cycle transactions (multiple rounds to make it realistic)
        n_rounds = random.randint(2, 4)
        base_amount = np.random.uniform(500, 5000)
        for round_num in range(n_rounds):
            cycle_start = _rand_ts(base_time, window_hours=480)
            for j in range(cycle_len):
                sender = cycle_accs[j]
                receiver = cycle_accs[(j + 1) % cycle_len]
                # Amount stays similar within a cycle (layering)
                amt = round(base_amount * np.random.uniform(0.9, 1.1), 2)
                ts = cycle_start + timedelta(hours=j * random.uniform(1, 12))
                add_txn(sender, receiver, amt, ts)

        # Add some noise transactions too
        for acc in cycle_accs:
            for _ in range(random.randint(1, 3)):
                ts = _rand_ts(base_time, window_hours=720)
                if random.random() < 0.5:
                    add_txn(acc, random.choice(normal_accs), round(np.random.uniform(50, 500), 2), ts)
                else:
                    add_txn(random.choice(normal_accs), acc, round(np.random.uniform(50, 500), 2), ts)

        rings.append({
            'ring_id': ring_id,
            'pattern_type': 'cycle',
            'member_accounts': cycle_accs,
            'risk_score': round(np.random.uniform(80, 98), 1),
        })

    # ────────────────────────────────────────────
    # 5. SMURFING RINGS (fan-in → aggregator → fan-out)
    # ────────────────────────────────────────────
    for i in range(n_smurf_rings):
        ring_id = f"RING_SMURF_{i:03d}"
        is_fan_in = random.random() < 0.5

        if is_fan_in:
            # Many senders → 1 aggregator (within 72hrs, similar small amounts)
            aggregator = _acc_id()
            n_senders = random.randint(10, 20)
            senders = [_acc_id() for _ in range(n_senders)]

            all_members = senders + [aggregator]
            for acc in all_members:
                suspicious_accounts.add(acc)
                ring_assignments[acc] = ring_id
                pattern_assignments.setdefault(acc, []).append("smurfing_fan_in")

            # All sends within 72hrs with small, similar amounts
            burst_start = _rand_ts(base_time, window_hours=200)
            smurf_amount = np.random.uniform(200, 900)  # Under reporting threshold
            for sender in senders:
                amt = round(smurf_amount * np.random.uniform(0.85, 1.15), 2)
                ts = _rand_ts(burst_start, window_hours=72)
                add_txn(sender, aggregator, amt, ts)

            # Aggregator then moves the money out
            total = smurf_amount * n_senders
            ts_out = burst_start + timedelta(hours=random.randint(24, 96))
            add_txn(aggregator, random.choice(normal_accs), round(total * 0.95, 2), ts_out)

            rings.append({
                'ring_id': ring_id,
                'pattern_type': 'smurfing_fan_in',
                'member_accounts': all_members,
                'risk_score': round(np.random.uniform(85, 98), 1),
            })
        else:
            # 1 distributor → many receivers (within 72hrs)
            distributor = _acc_id()
            n_receivers = random.randint(10, 20)
            receivers = [_acc_id() for _ in range(n_receivers)]

            all_members = [distributor] + receivers
            for acc in all_members:
                suspicious_accounts.add(acc)
                ring_assignments[acc] = ring_id
                pattern_assignments.setdefault(acc, []).append("smurfing_fan_out")

            # Distributor receives a large amount first
            ts_in = _rand_ts(base_time, window_hours=200)
            total = np.random.uniform(5000, 20000)
            add_txn(random.choice(normal_accs), distributor, round(total, 2), ts_in)

            # Then disperses in small amounts within 72hrs
            burst_start = ts_in + timedelta(hours=random.randint(1, 24))
            split_amount = total / n_receivers
            for recv in receivers:
                amt = round(split_amount * np.random.uniform(0.85, 1.15), 2)
                ts = _rand_ts(burst_start, window_hours=72)
                add_txn(distributor, recv, amt, ts)

            rings.append({
                'ring_id': ring_id,
                'pattern_type': 'smurfing_fan_out',
                'member_accounts': all_members,
                'risk_score': round(np.random.uniform(85, 98), 1),
            })

    # ────────────────────────────────────────────
    # 6. LAYERED SHELL CHAINS (A → B → C → D, intermediaries have 2-3 txns)
    # ────────────────────────────────────────────
    for i in range(n_shell_chains):
        ring_id = f"RING_SHELL_{i:03d}"
        chain_len = random.randint(3, 5)
        chain_accs = [_acc_id() for _ in range(chain_len)]

        for acc in chain_accs:
            suspicious_accounts.add(acc)
            ring_assignments[acc] = ring_id
            pattern_assignments.setdefault(acc, []).append("layered_shell")

        # Money flows through the chain
        n_flows = random.randint(1, 3)
        for flow in range(n_flows):
            flow_start = _rand_ts(base_time, window_hours=480)
            base_amt = np.random.uniform(1000, 10000)
            for j in range(chain_len - 1):
                # Each intermediary skims a small fee
                amt = round(base_amt * (0.98 ** j), 2)
                ts = flow_start + timedelta(hours=j * random.uniform(2, 24))
                add_txn(chain_accs[j], chain_accs[j + 1], amt, ts)

        # Intermediaries have VERY low activity (key signal: only 2-3 txns total)
        # Don't add extra noise txns for intermediaries — they should stay low-activity

        rings.append({
            'ring_id': ring_id,
            'pattern_type': 'layered_shell',
            'member_accounts': chain_accs,
            'risk_score': round(np.random.uniform(75, 95), 1),
        })

    # ── Build DataFrame ──
    df = pd.DataFrame(transactions)
    df = df.sort_values('timestamp').reset_index(drop=True)
    # Re-number transaction IDs after sorting
    df['transaction_id'] = [f'TXN_{i+1:07d}' for i in range(len(df))]

    # ── Build labels ──
    all_accounts = set(df['sender_id']) | set(df['receiver_id'])
    labels = {}
    for acc in all_accounts:
        labels[acc] = {
            'account_id': acc,
            'is_suspicious': 1 if acc in suspicious_accounts else 0,
            'ring_id': ring_assignments.get(acc, None),
            'detected_patterns': pattern_assignments.get(acc, []),
        }

    # ── Save ──
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'synthetic_train.csv')
    df.to_csv(csv_path, index=False)

    gt_path = os.path.join(output_dir, 'synthetic_ground_truth.json')
    ground_truth = {
        'suspicious_accounts': [
            labels[acc] for acc in sorted(suspicious_accounts)
        ],
        'fraud_rings': rings,
        'summary': {
            'total_accounts': len(all_accounts),
            'suspicious_accounts': len(suspicious_accounts),
            'normal_accounts': len(all_accounts) - len(suspicious_accounts),
            'total_transactions': len(df),
            'n_cycle_rings': n_cycle_rings,
            'n_smurf_rings': n_smurf_rings,
            'n_shell_chains': n_shell_chains,
            'n_merchant_traps': n_merchant_traps,
            'n_payroll_traps': n_payroll_traps,
        }
    }
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    # Also save a labels CSV for training
    labels_csv_path = os.path.join(output_dir, 'synthetic_labels.csv')
    labels_df = pd.DataFrame([
        {'account_id': acc, 'is_suspicious': 1 if acc in suspicious_accounts else 0}
        for acc in all_accounts
    ])
    labels_df.to_csv(labels_csv_path, index=False)

    print(f"Generated dataset:")
    print(f"  Transactions: {len(df):,}")
    print(f"  Accounts:     {len(all_accounts):,}")
    print(f"  Suspicious:   {len(suspicious_accounts):,}")
    print(f"  Normal:       {len(all_accounts) - len(suspicious_accounts):,}")
    print(f"  Rings:        {len(rings)}")
    print(f"  Saved to:     {csv_path}")
    print(f"  Ground truth: {gt_path}")
    print(f"  Labels:       {labels_csv_path}")

    return df, labels, ground_truth


if __name__ == '__main__':
    generate_synthetic_dataset()
