
import asyncio
import os
import sys


# Add project root to sys.path
sys.path.append(os.getcwd())

# Load .env manually
try:
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value.strip()
    print("Loaded .env")
except Exception as e:
    print(f"Failed to load .env: {e}")

# Unset proxies to avoid local proxy errors
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

from app.services.dashboard_analyzer import dashboard_analyzer
import pandas as pd

async def test_analysis():
    symbol = "600519"
    stock_name = "贵州茅台"
    
    print(f"Testing analysis for {stock_name} ({symbol})...")
    
    # Mock DataFrame
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "close": [1700.0, 1750.0],
        "open": [1700.0, 1710.0],
        "high": [1760.0, 1780.0],
        "low": [1690.0, 1700.0],
        "volume": [10000, 20000]
    })
    
    realtime = {"price": 1750.0}
    
    try:
        result = await dashboard_analyzer.analyze(symbol, stock_name, df, realtime=realtime)
        print("Analysis Result:")
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # Verify keys
        if "dashboard" in result:
            print("✅ Dashboard structure present")
        else:
            print("❌ Dashboard structure missing")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_analysis())
