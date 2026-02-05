import sys
import os
sys.path.append(os.getcwd())

from app.rox_quant.strategies.jq_10y_52x import JQTenYearFiftyTwoTimes
from app.rox_quant.strategies.jq_small_cap import JQSmallCap
from app.rox_quant.strategies.jq_etf_rotation import JQETFRotation
from app.rox_quant.strategies.jq_dragon import JQDragonTrend

def test_strategy(name, strategy_cls, pool=None):
    print(f"\n--- Testing {name} ---")
    try:
        strategy = strategy_cls(stock_pool=pool)
        results = strategy.run(
            graph_json="",
            start_date="2024-01-01",
            end_date="2024-03-01",
            initial_capital=100000.0
        )
        metrics = results.get("metrics", {})
        print(f"Success! Return: {metrics.get('total_return', 0):.2%}")
        print(f"Trades: {len(results.get('trades', []))}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 1. 10y 52x
    test_strategy("10y 52x", JQTenYearFiftyTwoTimes)
    
    # 2. Small Cap (Use a manual pool of small caps for test)
    test_strategy("Small Cap", JQSmallCap, pool=["002415.SZ", "300059.SZ", "603288.SH"])
    
    # 3. ETF Rotation
    test_strategy("ETF Rotation", JQETFRotation)
    
    # 4. Dragon Trend
    test_strategy("Dragon Trend", JQDragonTrend, pool=["600519.SH", "300750.SZ"])
