"""
Verify Model Generalization.
Generates a brand new dataset (Seed 9999) and evaluates the pre-trained model on it.
Use this to prove the model learns abstract patterns, not specific accounts.
"""
import pandas as pd
import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.gnn_model import load_model, prepare_inference_data, predict
from scripts.generate_training_data import generate_synthetic_dataset

def verify():
    # 1. Generate new data (Seed 9999 - completely unseen)
    print("="*60)
    print("GENERATING BRAND NEW DATA (Seed 9999)")
    print("="*60)
    output_dir = "data_verification"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    df, labels, _ = generate_synthetic_dataset(
        seed=9999, 
        output_dir=output_dir,
        n_normal_accounts=600,  # Slightly different size
        n_cycle_rings=10,       # More rings to test robustness
        n_smurf_rings=6,
        n_shell_chains=5,
        n_merchant_traps=6,
        n_payroll_traps=4
    )
    
    # Load ground truth
    labels_df = pd.read_csv(os.path.join(output_dir, "synthetic_labels.csv"))
    sus_accounts_set = set(labels_df[labels_df['is_suspicious']==1]['account_id'])
    print(f"\nGround Truth: {len(sus_accounts_set)} suspicious accounts out of {len(labels_df)} total.")

    # 2. Load Pre-Trained Model
    print("\nLoading Pre-Trained Model...")
    result = load_model()
    if result is None:
        print("Error: No model found! Please train first.")
        return
    model, metadata = result
    
    # 3. Predict on New Data
    print("Predicting on new universe...")
    
    # Ensure columns and types match app logic
    if 'timestamp' not in df.columns:
         df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    data_test = prepare_inference_data(
        df[['sender_id', 'receiver_id', 'amount', 'timestamp']],
        scaler_mean=metadata['scaler_mean'],
        scaler_scale=metadata['scaler_scale'],
    )
    
    probs = predict(model, data_test)
    idx_to_acc = {v: k for k, v in data_test.account_map.items()}
    scores = {idx_to_acc[i]: float(p)*100 for i, p in enumerate(probs)}
    
    # 4. Evaluate
    print(f"\n{'Thresh':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 60)
    for t in [50, 70, 90]:
        sus_pred = {acc for acc, s in scores.items() if s >= t}
        tp = len(sus_pred & sus_accounts_set)
        fp = len(sus_pred - sus_accounts_set)
        fn = len(sus_accounts_set - sus_pred)
        
        prec = tp/(tp+fp) if tp+fp > 0 else 0
        rec = tp/(tp+fn) if tp+fn > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
        
        print(f"{t:>7}% {prec:>7.1%} {rec:>7.1%} {f1:>7.3f} {tp:>6} {fp:>6} {fn:>6}")
        
    print("\nVERDICT:")
    if prec > 0.9 and rec > 0.9:
        print("✅ SUCCESS: The model generalized perfectly to unseen data!")
        print("   It learned the RULES of fraud (cycles, variance, burstiness),")
        print("   not the specific accounts or amounts.")
    else:
        print("❌ FAILURE: The model struggled on new data.")

if __name__ == "__main__":
    verify()
