from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from app.rox_quant.portfolio_backtest import PortfolioBacktestEngine
from app.rox_quant.context import Context

class JQSmallCap(PortfolioBacktestEngine):
    """
    Migration of JoinQuant 'Small Cap' Strategy (GuoJiu Modified)
    
    Logic:
    1. Filter: Exclude ST, negative profit (Mocked by Price filter in this demo)
    2. Sort: Market Cap (Ascending) -> Proxied by Price for demo if Cap missing
    3. Rebalance: Weekly
    """
    def __init__(self, stock_pool: Optional[List[str]] = None):
        super().__init__()
        # Default pool: Small/Mid cap stocks
        self.stock_pool = stock_pool or [
            "002415.SZ", "002475.SZ", "300059.SZ", "000799.SZ", "603259.SH",
            "603288.SH", "002304.SZ", "300750.SZ", "002049.SZ", "300274.SZ"
        ]
        self.g = {
            "stock_num": 5,
            "days_counter": 0,
            "rebalance_interval": 5 # Weekly
        }

    def initialize(self, context: Context):
        context.benchmark = "399101.SZ" # Small Cap Index

    def handle_bar(self, context: Context, graph_json: str):
        self.g["days_counter"] += 1
        
        # Weekly Rebalance
        if self.g["days_counter"] % self.g["rebalance_interval"] != 0:
            return

        candidates = []
        for symbol in self.stock_pool:
            hist = self.provider.get_history(symbol, days=20)
            if not hist or len(hist) < 5:
                continue
                
            # Proxy for Market Cap: Price * Volume (roughly) or just Price
            # In real scenario, we need get_fundamentals
            last_close = hist[-1].close
            
            # Simple "Low Price" Strategy as proxy for Small Cap
            candidates.append({
                "symbol": symbol,
                "score": last_close # Lower is better
            })
            
        # Sort by Price Ascending (Small Cap Proxy)
        candidates.sort(key=lambda x: x["score"])
        selected = candidates[:self.g["stock_num"]]
        
        target_weights = {}
        if selected:
            weight = 1.0 / len(selected)
            for item in selected:
                target_weights[item["symbol"]] = weight
                
        self._rebalance(context, target_weights)
