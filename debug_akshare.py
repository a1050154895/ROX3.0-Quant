
import akshare as ak
import time
import pandas as pd

def test_fenshi(symbol):
    print(f"Testing Fenshi for {symbol}...")
    start = time.time()
    try:
        df = ak.stock_zh_a_minute(symbol=symbol, period="1")
        elapsed = time.time() - start
        if df is not None and not df.empty:
            print(f"✅ Fenshi OK ({len(df)} rows) in {elapsed:.2f}s")
        else:
            print(f"❌ Fenshi Empty in {elapsed:.2f}s")
    except Exception as e:
        print(f"❌ Fenshi Error: {e}")

def test_info(symbol):
    print(f"Testing Info for {symbol}...")
    start = time.time()
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        elapsed = time.time() - start
        if df is not None and not df.empty:
            print(f"✅ Info OK ({len(df)} rows) in {elapsed:.2f}s")
        else:
            print(f"❌ Info Empty in {elapsed:.2f}s")
    except Exception as e:
        print(f"❌ Info Error: {e}")

def test_fund_flow(symbol):
    print(f"Testing Fund Flow for {symbol}...")
    start = time.time()
    try:
        market = "sh" if symbol.startswith("6") else "sz"
        df = ak.stock_individual_fund_flow(stock=symbol, market=market)
        elapsed = time.time() - start
        if df is not None and not df.empty:
            print(f"✅ Fund Flow OK ({len(df)} rows) in {elapsed:.2f}s")
        else:
            print(f"❌ Fund Flow Empty in {elapsed:.2f}s")
    except Exception as e:
        print(f"❌ Fund Flow Error: {e}")

def test_news(symbol):
    print(f"Testing News for {symbol}...")
    start = time.time()
    try:
        df = ak.stock_news_em(symbol=symbol)
        elapsed = time.time() - start
        if df is not None and not df.empty:
            print(f"✅ News OK ({len(df)} rows) in {elapsed:.2f}s")
        else:
            print(f"❌ News Empty in {elapsed:.2f}s")
    except Exception as e:
        print(f"❌ News Error: {e}")

if __name__ == "__main__":
    # test_fenshi("sh000001") # Already tested and confirmed OK logic issue was in API code
    # test_info("600519") # Fast
    test_fund_flow("600519")
    test_news("600519")
