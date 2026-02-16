
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8002"
TIMEOUT = 30

def check_endpoint(name, url, method="GET", expected_status=200, check_json=True, payload=None):
    print(f"Checking {name}...", end=" ", flush=True)
    try:
        if method == "GET":
            response = requests.get(url, timeout=TIMEOUT)
        elif method == "POST":
            response = requests.post(url, json=payload, timeout=TIMEOUT)
        
        if response.status_code == expected_status:
            if check_json:
                try:
                    data = response.json()
                    # Basic validation that it's somewhat valid data
                    if isinstance(data, (dict, list)):
                         print(f"✅ PASS ({len(str(data))} bytes)")
                         return True
                    else:
                         print(f"⚠️  WARN (Invalid JSON Type)")
                         return False
                except:
                    print(f"❌ FAIL (Invalid JSON)")
                    return False
            else:
                print(f"✅ PASS (HTML/Text)")
                return True
        else:
            print(f"❌ FAIL (Status {response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ ERROR ({str(e)})")
        return False

print("=== ROX 3.0 Full System Verification ===\n")

# 1. Frontend Pages
print("--- Frontend Pages ---")
check_endpoint("Homepage (ROX 2.0)", f"{BASE_URL}/", check_json=False)
check_endpoint("Professional Ver (ROX 3.0)", f"{BASE_URL}/pro", check_json=False)
check_endpoint("Strategy Builder", f"{BASE_URL}/builder", check_json=False)
check_endpoint("Market Map", f"{BASE_URL}/map", check_json=False)

# 2. Market Data
print("\n--- Market Data APIs ---")
check_endpoint("Market Indices", f"{BASE_URL}/api/market/indices")
check_endpoint("K-Line (sh000001)", f"{BASE_URL}/api/market/kline?code=sh000001")
check_endpoint("Fenshi (sh000001)", f"{BASE_URL}/api/market/fenshi?code=sh000001")
check_endpoint("Dragon Tiger List", f"{BASE_URL}/api/market/dragon-tiger")
check_endpoint("Sector Rotation", f"{BASE_URL}/api/market/rotation")
check_endpoint("Macro Data (Legacy)", f"{BASE_URL}/api/market/macro")
check_endpoint("Macro Indicators (Phase 6)", f"{BASE_URL}/api/macro/indicators")

# 3. Stock Analysis
print("\n--- Stock Analysis APIs ---")
check_endpoint("Stock Diagnosis (600519)", f"{BASE_URL}/api/stock/diagnose?code=600519")
check_endpoint("AI Deep Analysis (600519)", f"{BASE_URL}/api/analysis/dashboard/600519")

# 4. Trading & Portfolio
print("\n--- Trading APIs ---")
check_endpoint("Portfolio Summary (Sim)", f"{BASE_URL}/api/portfolio/summary?mode=sim")
check_endpoint("Portfolio Summary (Real)", f"{BASE_URL}/api/portfolio/summary?mode=real")

# 5. System
print("\n--- System APIs ---")
check_endpoint("System Health", f"{BASE_URL}/health")
check_endpoint("AI Settings", f"{BASE_URL}/api/settings/ai")
check_endpoint("System Status", f"{BASE_URL}/api/system/status")

print("\n=== Verification Complete ===")
