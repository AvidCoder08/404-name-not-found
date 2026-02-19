
import requests
import json
import os

def test_streaming():
    url = "http://127.0.0.1:8000/analyze"
    file_path = "money_muling_dataset_12k (1).csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Sending POST request to {url} with {file_path}...")
    
    try:
        with open(file_path, 'rb') as f:
            # Note: The key for the file in the form data must match the backend expectation ('file')
            response = requests.post(url, files={'file': f}, stream=True)
            
            print(f"Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error Content: {response.text}")
                return

            print("Streaming response:")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        data = json.loads(decoded_line)
                        # Pretty print important steps
                        if "step" in data:
                            print(f"[STEP: {data.get('step')}] {data.get('log', '')}")
                            if "graph_update" in data:
                                nodes = len(data['graph_update'].get('nodes', []))
                                links = len(data['graph_update'].get('links', []))
                                print(f"   -> Graph Update: {nodes} nodes, {links} links")
                            
                            # Check summary in result
                            if "result" in data and "summary" in data["result"]:
                                print(f"\n[SUMMARY RAW] {json.dumps(data['result']['summary'], indent=2)}")
                                rings_count = len(data['result'].get('fraud_rings', []))
                                print(f"[RINGS COUNT in RESULT] {rings_count}")
                        else:
                            print(f"[INFO] {decoded_line[:100]}...")
                    except json.JSONDecodeError:
                        print(f"[RAW] {decoded_line}")
                        
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_streaming()
