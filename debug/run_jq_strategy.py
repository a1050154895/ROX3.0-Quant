import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rox_quant.strategies.jq_migration_demo import JQMigratedStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_demo():
    print("Starting JoinQuant Migration Demo...")
    print("Note: This requires a working internet connection to fetch data via AkShare.")
    print("If it hangs, it might be due to network issues or data provider timeouts.")
    
    # Initialize Strategy
    # We use a small pool of stocks for demonstration purposes
    strategy = JQMigratedStrategy(stock_pool=["600519", "000858", "601318"])
    
    # Run Backtest
    start_date = "2024-01-01"
    end_date = "2024-03-01"
    
    try:
        results = strategy.run(
            graph_json="", # Not used in this demo
            start_date=start_date,
            end_date=end_date,
            initial_capital=1000000
        )
        
        print("\nBacktest Completed!")
        print("-" * 30)
        metrics = results.get("metrics", {})
        print(f"Total Return: {metrics.get('total_return')}%")
        print(f"Max Drawdown: {metrics.get('max_drawdown')}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe')}")
        print("-" * 30)
        
        # Show trade log sample
        if len(results['equity_curve']) > 0:
            print(f"Final Value: {results['equity_curve'][-1]:.2f}")
            
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
