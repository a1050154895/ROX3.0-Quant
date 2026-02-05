from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import math
from app.rox_quant.portfolio_backtest import PortfolioBacktestEngine
from app.rox_quant.context import Context

class JQETFRotation(PortfolioBacktestEngine):
    """
    Migration of JoinQuant 'ETF Rotation' Strategy
    
    Logic:
    1. Pool: ETFs (Gold, Nasdaq, HS300, etc.)
    2. Factor: Momentum Score = Annualized Return * R-Squared (Linear Regression of log prices)
    3. Rebalance: Monthly (approx 20 days)
    """
    def __init__(self, stock_pool: Optional[List[str]] = None):
        super().__init__()
        self.stock_pool = stock_pool or [
            "518880.SH", # Gold
            "513100.SH", # Nasdaq
            "510300.SH", # HS300
            "159915.SZ", # ChiNext
            "512660.SH", # Military
            "512480.SH", # Semi
            "515030.SH", # New Energy
        ]
        self.g = {
            "stock_num": 2,
            "days_counter": 0,
            "rebalance_interval": 20, # Monthly
            "momentum_days": 25
        }

    def initialize(self, context: Context):
        context.benchmark = "000300.SH"

    def calculate_score(self, prices: List[float]) -> float:
        if len(prices) < self.g["momentum_days"]:
            return -999.0
            
        # Log prices
        y = np.log(prices)
        x = np.arange(len(y))
        
        # Linear Regression
        # Slope, Intercept
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Annualized Return
        annualized_return = math.pow(math.exp(m), 250) - 1
        
        # R-Squared
        y_pred = m * x + c
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)
            
        return annualized_return * r_squared

    def handle_bar(self, context: Context, graph_json: str):
        self.g["days_counter"] += 1
        
        if self.g["days_counter"] % self.g["rebalance_interval"] != 0:
            return

        candidates = []
        for symbol in self.stock_pool:
            hist = self.provider.get_history(symbol, days=self.g["momentum_days"] + 5)
            if not hist or len(hist) < self.g["momentum_days"]:
                continue
                
            closes = [p.close for p in hist[-self.g["momentum_days"]:]]
            score = self.calculate_score(closes)
            
            candidates.append({
                "symbol": symbol,
                "score": score
            })
            
        # Sort by Score Descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Filter negative scores
        selected = [c for c in candidates if c["score"] > 0][:self.g["stock_num"]]
        
        target_weights = {}
        if selected:
            weight = 1.0 / len(selected)
            for item in selected:
                target_weights[item["symbol"]] = weight
        else:
            # Empty portfolio if no good trends
            pass
                
        self._rebalance(context, target_weights)
