
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8002"

def debug_diagnose(code="600519"):
    url = f"{BASE_URL}/api/stock/diagnose"
    print(f"Requesting {url} for code={code}...")
    try:
        start = time.time()
        resp = requests.get(url, params={"code": code}, timeout=30)
        elapsed = time.time() - start
        
        print(f"Status Code: {resp.status_code}")
        print(f"Time Elapsed: {elapsed:.2f}s")
        
        if resp.status_code == 200:
            data = resp.json()
            # Print key sections summary
            print("\n--- Diagnosis Summary ---")
            print(f"Code: {data.get('code')}")
            print(f"Name: {data.get('name')}")
            print(f"Overall Score: {data.get('overall_score')}")
            print(f"Summary: {data.get('summary')}")
            
            print("\n--- Details ---")
            details = data.get("details", {})
            tech = details.get("technical", {})
            fund = details.get("fundamental", {})
            flow = details.get("fund_flow", {})
            
            print(f"Technical: Score={data.get('scores', {}).get('technical')} | Summary={tech.get('summary')}")
            print(f"Fundamental: Score={data.get('scores', {}).get('fundamental')} | Summary={fund.get('summary')}")
            print(f"Fund Flow: Score={data.get('scores', {}).get('fund_flow')} | Summary={flow.get('summary')}")
            
            print("\n--- Raw JSON Dump (Partial) ---")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:500] + "...")
        else:
            print(f"Error Response: {resp.text}")
            
    except Exception as e:
        print(f"Request Failed: {e}")

if __name__ == "__main__":
    debug_diagnose("600519")
