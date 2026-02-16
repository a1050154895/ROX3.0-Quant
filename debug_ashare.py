
import time
from app.rox_quant.datasources import ashare_lite

def test_ashare():
    print("Testing Ashare Lite...")
    start = time.time()
    try:
        df = ashare_lite.get_price("600519", count=300, frequency='1d')
        elapsed = time.time() - start
        if df is not None and not df.empty:
            print(f"✅ Ashare OK ({len(df)} rows) in {elapsed:.2f}s")
        else:
            print(f"❌ Ashare Empty in {elapsed:.2f}s")
    except Exception as e:
        print(f"❌ Ashare Error: {e}")

if __name__ == "__main__":
    test_ashare()
