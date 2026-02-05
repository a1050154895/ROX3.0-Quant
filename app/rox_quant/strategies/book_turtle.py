from typing import List, Dict, Optional
import numpy as np
from app.rox_quant.portfolio_backtest import PortfolioBacktestEngine
from app.rox_quant.context import Context

class BookTurtle(PortfolioBacktestEngine):
    """
    Strategy: Turtle Trading (Simplified)
    Source: "Python Quantitative Trading" (Book 1)
    
    Logic:
    1. Donchian Channel: 
       - Buy if Price > High of last 20 days.
       - Sell if Price < Low of last 10 days.
    2. ATR (Average True Range) for volatility measurement (skipped in simple version, used for position sizing in full version).
    """
    def __init__(self, stock_pool: Optional[List[str]] = None):
        super().__init__()
        self.stock_pool = stock_pool or ["600519.SH", "000001.SZ"]
        self.g = {
            "in_period": 20,
            "out_period": 10
        }

    def initialize(self, context: Context):
        pass

    def handle_bar(self, context: Context, graph_json: str):
        target_weights = {}
        
        for symbol in self.stock_pool:
            # Need max period + 1
            hist = self.provider.get_history(symbol, days=self.g["in_period"] + 2)
            if not hist or len(hist) < self.g["in_period"] + 1:
                continue
            
            current_close = hist[-1].close
            prev_hist = hist[:-1] # Data excluding today
            
            # Donchian High (Last 20 days)
            # Use max of Highs
            highs = [bar.high for bar in prev_hist[-self.g["in_period"]:]]
            donchian_high = max(highs)
            
            # Donchian Low (Last 10 days)
            # Use min of Lows
            lows = [bar.low for bar in prev_hist[-self.g["out_period"]:]]
            donchian_low = min(lows)
            
            # Current Position Status
            has_position = symbol in context.portfolio.positions and context.portfolio.positions[symbol].quantity > 0
            
            if current_close > donchian_high:
                # Breakout -> Buy
                target_weights[symbol] = 1.0 / len(self.stock_pool)
            elif current_close < donchian_low:
                # Breakdown -> Sell
                target_weights[symbol] = 0.0
            else:
                # Hold status quo
                if has_position:
                    target_weights[symbol] = 1.0 / len(self.stock_pool)
                else:
                    target_weights[symbol] = 0.0
                    
        self._rebalance(context, target_weights)
