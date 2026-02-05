from typing import List, Dict, Optional
import numpy as np
from app.rox_quant.portfolio_backtest import PortfolioBacktestEngine
from app.rox_quant.context import Context

class BookDoubleMA(PortfolioBacktestEngine):
    """
    Strategy: Double Moving Average (Golden Cross/Death Cross)
    Source: "Python Quantitative Trading" (Book 1)
    
    Logic:
    1. MA Short (e.g., 5 days)
    2. MA Long (e.g., 20 days)
    3. Buy when Short > Long (Golden Cross)
    4. Sell when Short < Long (Death Cross)
    """
    def __init__(self, stock_pool: Optional[List[str]] = None):
        super().__init__()
        self.stock_pool = stock_pool or ["600519.SH", "000001.SZ"]
        self.g = {
            "short_window": 5,
            "long_window": 20
        }

    def initialize(self, context: Context):
        pass

    def handle_bar(self, context: Context, graph_json: str):
        target_weights = {}
        
        for symbol in self.stock_pool:
            # Need enough data for Long MA
            hist = self.provider.get_history(symbol, days=self.g["long_window"] + 2)
            if not hist or len(hist) < self.g["long_window"]:
                continue
                
            closes = [bar.close for bar in hist]
            
            # Calculate MAs for TODAY
            ma_short = np.mean(closes[-self.g["short_window"]:])
            ma_long = np.mean(closes[-self.g["long_window"]:])
            
            if ma_short > ma_long:
                # Bullish
                target_weights[symbol] = 1.0 / len(self.stock_pool)
            else:
                # Bearish
                target_weights[symbol] = 0.0
                
        self._rebalance(context, target_weights)
