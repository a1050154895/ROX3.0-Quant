from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from app.rox_quant.portfolio_backtest import PortfolioBacktestEngine
from app.rox_quant.context import Context

class BookSmallCapTiming(PortfolioBacktestEngine):
    """
    Strategy: Small Cap Rotation + 2-8 Timing
    Source: "Quantitative Investment Technical Analysis Combat" (Book 2)
    
    Logic:
    1. Timing (The "2-8" part): 
       - Calculate a market sentiment indicator (e.g., Avg Price of Pool vs MA20).
       - If Market > MA20: Bull Market -> Hold Stocks.
       - If Market < MA20: Bear Market -> Clear Positions (or hold Money Market Fund).
    2. Selection (The "Small Cap" part):
       - If Bull Market: Select Top N stocks with smallest Market Cap (Proxied by Price).
    3. Rebalance:
       - Check Timing Daily.
       - Check Rotation Weekly.
    """
    def __init__(self, stock_pool: Optional[List[str]] = None):
        super().__init__()
        # Default pool: A mix of stocks to simulate a market
        self.stock_pool = stock_pool or [
            "000001.SZ", "000002.SZ", "600519.SH", "601318.SH", # Big
            "002415.SZ", "002475.SZ", "300059.SZ", "000799.SZ", # Small
            "603259.SH", "603288.SH", "002304.SZ", "300750.SZ"  # Small
        ]
        self.g = {
            "stock_num": 3,
            "days_counter": 0,
            "rebalance_interval": 5, # Weekly rotation
            "market_trend": "bull"   # Current market state
        }

    def initialize(self, context: Context):
        context.benchmark = "000300.SH" 

    def _get_market_trend(self):
        """
        Calculate market trend based on the average close of the stock pool.
        (Using pool average as a proxy for 'The Market' if index data is unavailable)
        """
        closes = []
        for symbol in self.stock_pool:
            hist = self.provider.get_history(symbol, days=21)
            if hist and len(hist) > 0:
                closes.append(hist[-1].close)
        
        if not closes:
            return "neutral"
            
        # Create a simple equal-weighted index
        current_idx = np.mean(closes)
        
        # In a real implementation, we would track the index history properly.
        # Here we do a simplified check: Compare current average price to 
        # the average price of the pool (this is a weak proxy, but functional for demo).
        # Better: Check if > 50% of stocks are above their MA20.
        
        stocks_above_ma20 = 0
        valid_stocks = 0
        
        for symbol in self.stock_pool:
            hist = self.provider.get_history(symbol, days=21)
            if not hist or len(hist) < 21:
                continue
            
            closes = [bar.close for bar in hist]
            ma20 = np.mean(closes[:-1]) # Previous 20 days
            current = closes[-1]
            
            if current > ma20:
                stocks_above_ma20 += 1
            valid_stocks += 1
            
        if valid_stocks == 0:
            return "bear"
            
        # If more than 40% stocks are healthy, we assume Bull
        if (stocks_above_ma20 / valid_stocks) > 0.4:
            return "bull"
        else:
            return "bear"

    def handle_bar(self, context: Context, graph_json: str):
        self.g["days_counter"] += 1
        
        # 1. Check Timing (Daily)
        market_trend = self._get_market_trend()
        
        if market_trend == "bear":
            # Clear all positions
            self._rebalance(context, {})
            return

        # 2. Check Rotation (Weekly)
        if self.g["days_counter"] % self.g["rebalance_interval"] != 0:
            return

        # 3. Select Small Caps
        candidates = []
        for symbol in self.stock_pool:
            hist = self.provider.get_history(symbol, days=5)
            if not hist:
                continue
            last_close = hist[-1].close
            candidates.append({"symbol": symbol, "score": last_close})
            
        # Sort by Price (Small Cap Proxy) - Ascending
        candidates.sort(key=lambda x: x["score"])
        selected = candidates[:self.g["stock_num"]]
        
        target_weights = {}
        if selected:
            weight = 0.99 / len(selected) # Leave some cash
            for item in selected:
                target_weights[item["symbol"]] = weight
                
        self._rebalance(context, target_weights)
