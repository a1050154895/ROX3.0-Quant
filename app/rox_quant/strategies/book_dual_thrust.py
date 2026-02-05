from typing import List, Dict, Optional
import numpy as np
from app.rox_quant.portfolio_backtest import PortfolioBacktestEngine
from app.rox_quant.context import Context

class BookDualThrust(PortfolioBacktestEngine):
    """
    Strategy: Dual Thrust (Intraday/Daily Trend)
    Source: "Python Quantitative Trading" (Book 1)
    
    Logic:
    1. Calculate Range = Max(HH-LC, HC-LL) over last N days.
    2. Buy Line = Open + K1 * Range
    3. Sell Line = Open - K2 * Range
    4. If Price > Buy Line -> Long
    5. If Price < Sell Line -> Short (or Close Long in stocks)
    """
    def __init__(self, stock_pool: Optional[List[str]] = None):
        super().__init__()
        self.stock_pool = stock_pool or ["600519.SH"] # Usually single asset, but supports pool
        self.g = {
            "N": 5,
            "K1": 0.5,
            "K2": 0.5
        }

    def initialize(self, context: Context):
        pass

    def handle_bar(self, context: Context, graph_json: str):
        target_weights = {}
        
        for symbol in self.stock_pool:
            # Need N+1 days to calculate Range and have today's Open
            hist = self.provider.get_history(symbol, days=self.g["N"] + 1)
            if not hist or len(hist) < self.g["N"] + 1:
                continue
                
            # Previous N days data (excluding today)
            prev_hist = hist[:-1] 
            current_bar = hist[-1]
            
            # Calculate Range
            hh = max(bar.high for bar in prev_hist)
            hc = max(bar.close for bar in prev_hist)
            lc = min(bar.close for bar in prev_hist)
            ll = min(bar.low for bar in prev_hist)
            
            r_range = max(hh - lc, hc - ll)
            
            buy_line = current_bar.open + self.g["K1"] * r_range
            sell_line = current_bar.open - self.g["K2"] * r_range
            
            # Trading Logic (using Close as proxy for intraday breakout)
            # In real CTA, this would monitor tick/minute data.
            if current_bar.close > buy_line:
                # Buy Signal
                target_weights[symbol] = 1.0 / len(self.stock_pool)
            elif current_bar.close < sell_line:
                # Sell Signal
                target_weights[symbol] = 0.0
            else:
                # Hold previous position (simplified)
                # To implement "Hold", we need to check current portfolio. 
                # PortfolioBacktestEngine._rebalance is stateless target, 
                # so we need to know if we are already in.
                # However, for simplicity here, we assume NO POSITION if no signal,
                # unless we track state.
                # Better approach: Check if we have position in context.portfolio.positions
                if symbol in context.portfolio.positions and context.portfolio.positions[symbol].quantity > 0:
                     target_weights[symbol] = 1.0 / len(self.stock_pool) # Keep holding
                else:
                     target_weights[symbol] = 0.0

        self._rebalance(context, target_weights)
