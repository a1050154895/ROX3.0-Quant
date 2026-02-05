
import sys
import os
import asyncio
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rox_quant.strategies import (
    BookSmallCapTiming,
    BookDualThrust,
    BookDoubleMA,
    BookTurtle
)

async def verify_strategies():
    strategies = [
        ("book_small_cap_timing", BookSmallCapTiming),
        ("book_dual_thrust", BookDualThrust),
        ("book_double_ma", BookDoubleMA),
        ("book_turtle", BookTurtle)
    ]
    
    print("Starting verification of Book Strategies...")
    
    for strategy_id, StrategyClass in strategies:
        print(f"\nTesting {strategy_id}...")
        try:
            strategy = StrategyClass()
            # Run a short backtest
            results = strategy.run(
                graph_json="",
                start_date="2023-01-01",
                end_date="2023-02-01",
                initial_capital=100000.0
            )
            
            metrics = results.get("metrics", {})
            returns = metrics.get("total_return", "N/A")
            print(f"✅ {strategy_id} Passed! Return: {returns}%")
            
        except Exception as e:
            print(f"❌ {strategy_id} Failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_strategies())
