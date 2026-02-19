
import requests
import io
import pandas as pd
import time

def test_api(url="http://127.0.0.1:8000"):
    print(f"Testing API at {url}...")
    
    # 1. Check Root/Docs
    try:
        resp = requests.get(f"{url}/docs")
        if resp.status_code == 200:
            print("✅ Server is reachable (Swagger UI available).")
        else:
            print(f"❌ Server reachable but returning {resp.status_code}.")
    except Exception as e:
        print(f"❌ Could not reach server: {e}")
        return

    # 2. Test /analyze endpoint
    print("\nSending sample transaction data...")
    
    # Sample data with missing transaction_id (to test auto-generation)
    csv_data = """sender_id,receiver_id,amount,timestamp
A,B,500,2024-01-01 10:00:00
B,C,500,2024-01-01 11:00:00
C,A,500,2024-01-01 12:00:00
X,Y,10,2024-01-02 09:00:00
"""
    
    try:
        files = {"file": ("test_sample.csv", io.BytesIO(csv_data.encode()), "text/csv")}
        response = requests.post(f"{url}/analyze", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Analysis Successful!")
            print(f"   - Suspicion Scores: {len(data.get('suspicion_scores', {}))} accounts")
            print(f"   - Fraud Rings Detected: {len(data.get('fraud_rings', []))}")
            
            rings = data.get('fraud_rings', [])
            if rings:
                print(f"   - Example Ring: {rings[0]['ring_id']} ({rings[0]['pattern_type']})")
        else:
            print(f"❌ Analysis Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_api()
