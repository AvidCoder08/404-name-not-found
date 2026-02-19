

import sys
import os
import json
import asyncio
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import analyze

class MockUploadFile:
    def __init__(self, content):
        import io
        self.file = io.BytesIO(content.encode())
        self.filename = "test.csv"

async def test_json_format():
    # Create a dummy CSV
    csv_content = "transaction_id,sender_id,receiver_id,amount,timestamp\n" \
                  "TX1,ACC1,ACC2,100,2023-01-01 10:00:00\n" \
                  "TX2,ACC2,ACC3,100,2023-01-01 11:00:00\n" \
                  "TX3,ACC3,ACC1,100,2023-01-01 12:00:00"
    
    mock_file = MockUploadFile(csv_content)
    
    print("Calling analyze directly...")
    response = await analyze(file=mock_file)
    
    final_result = None
    
    # Iterate over streaming response
    async for line in response.body_iterator:
        if line:
            try:
                data = json.loads(line)
                if "result" in data:
                    final_result = data["result"]
                    break
            except:
                pass
        
    if not final_result:
        print("FAILED: No result found in stream.")
        return

    print("Result received. Verifying format...")
    
    # 1. suspicious_accounts
    assert "suspicious_accounts" in final_result, "Missing 'suspicious_accounts' key"
    assert isinstance(final_result["suspicious_accounts"], list), "'suspicious_accounts' must be a list"
    if final_result["suspicious_accounts"]:
        acc = final_result["suspicious_accounts"][0]
        expected_keys = {"account_id", "suspicion_score", "detected_patterns", "ring_id"}
        assert expected_keys.issubset(acc.keys()), f"Missing keys in suspicious_accounts: {expected_keys - set(acc.keys())}"
        print("  [OK] suspicious_accounts format")
    
    # 2. fraud_rings
    assert "fraud_rings" in final_result, "Missing 'fraud_rings' key"
    assert isinstance(final_result["fraud_rings"], list), "'fraud_rings' must be a list"
    if final_result["fraud_rings"]:
        ring = final_result["fraud_rings"][0]
        expected_keys = {"ring_id", "member_accounts", "pattern_type", "risk_score"}
        assert expected_keys.issubset(ring.keys()), f"Missing keys in fraud_rings: {expected_keys - set(ring.keys())}"
        assert isinstance(ring["ring_id"], str), "ring_id must be a string"
        print("  [OK] fraud_rings format")
        
    # 3. summary
    assert "summary" in final_result, "Missing 'summary' key"
    summary = final_result["summary"]
    expected_keys = {"total_accounts_analyzed", "suspicious_accounts_flagged", "fraud_rings_detected", "processing_time_seconds"}
    assert expected_keys.issubset(summary.keys()), f"Missing keys in summary: {expected_keys - set(summary.keys())}"
    print("  [OK] summary format")
    
    print("\nSUCCESS: JSON output matches PDF requirements via direct call.")

if __name__ == "__main__":
    try:
        asyncio.run(test_json_format())
    except AssertionError as e:
        print(f"\nFAILED: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {e}")
