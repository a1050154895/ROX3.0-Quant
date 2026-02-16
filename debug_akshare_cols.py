
import akshare as ak
import pandas as pd

def check_cols(symbol):
    print(f"Checking columns for {symbol}...")
    try:
        df = ak.stock_zh_a_minute(symbol=symbol, period="1")
        if df is not None and not df.empty:
            print("Columns:", df.columns.tolist())
            print(df.tail(2))
        else:
            print("Empty DataFrame")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_cols("sh000001")
