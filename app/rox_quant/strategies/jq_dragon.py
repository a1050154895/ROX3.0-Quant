from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from app.rox_quant.portfolio_backtest import PortfolioBacktestEngine
from app.rox_quant.context import Context

class JQDragonTrend(PortfolioBacktestEngine):
    """
    Migration of JoinQuant 'Dragon' Strategy (Limit Up Follow)
    
    Logic:
    1. Find stocks with 2 consecutive Limit Ups (approx > 9.5%)
    2. Buy on open
    3. Sell if not Limit Up anymore
    """
    def __init__(self, stock_pool: Optional[List[str]] = None):
        super().__init__()
        # Needs a volatile pool
        self.stock_pool = stock_pool or [
            "000001.SZ", "600519.SH", "300750.SZ", "601318.SH", 
            "002415.SZ", "601888.SH", "002304.SZ"
        ]
        self.g = {
            "stock_num": 5,
            "days_counter": 0,
            "holding_period": 3
        }
        self.hold_time = {} # {symbol: days_held}

    def initialize(self, context: Context):
        context.benchmark = "000905.SH"

    def handle_bar(self, context: Context, graph_json: str):
        self.g["days_counter"] += 1
        
        # 1. Sell Logic: Max holding period or trend break
        current_positions = list(context.portfolio.positions.keys())
        target_weights = {k: v.weight for k, v in context.portfolio.positions.items()}
        
        for symbol in current_positions:
            self.hold_time[symbol] = self.hold_time.get(symbol, 0) + 1
            
            # Simple Sell: Hold for 3 days then exit
            if self.hold_time[symbol] >= self.g["holding_period"]:
                target_weights[symbol] = 0.0
                del self.hold_time[symbol]
                
        # 2. Buy Logic: 2 consecutive limit ups
        candidates = []
        for symbol in self.stock_pool:
            if symbol in current_positions:
                continue
                
            hist = self.provider.get_history(symbol, days=5)
            if not hist or len(hist) < 3:
                continue
                
            # Check last 2 days returns
            p_today = hist[-1].close
            p_yest = hist[-2].close
            p_day_before = hist[-3].close
            
            ret_1 = (p_today - p_yest) / p_yest
            ret_2 = (p_yest - p_day_before) / p_day_before
            
            # Threshold for "Limit Up" approx > 9%
            if ret_1 > 0.09 and ret_2 > 0.09:
                candidates.append(symbol)
                
        # Allocate to new candidates
        available_slots = self.g["stock_num"] - len([k for k,v in target_weights.items() if v > 0])
        
        if available_slots > 0 and candidates:
            # Buy top N candidates
            to_buy = candidates[:available_slots]
            # Re-normalize weights? 
            # Simplification: Assume equal weight 1/N for all slots
            slot_weight = 1.0 / self.g["stock_num"]
            
            for sym in to_buy:
                target_weights[sym] = slot_weight
                self.hold_time[sym] = 0
                
        self._rebalance(context, target_weights)
