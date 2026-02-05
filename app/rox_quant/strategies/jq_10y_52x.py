from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from app.rox_quant.portfolio_backtest import PortfolioBacktestEngine
from app.rox_quant.context import Context

class JQTenYearFiftyTwoTimes(PortfolioBacktestEngine):
    """
    Migration of JoinQuant Strategy: '10年52倍，年化59，全新因子方法超稳定'
    
    Original Logic:
    1. Universe: All A-Shares
    2. Filter: Yesterday Limit Up (Close == High Limit)
    3. Factor: ARBR
    4. Rank: ARBR (Specific Range or Sort)
    5. Rebalance: Weekly
    
    ROX Adaptation:
    - Uses 'Volume Ratio' as a proxy for ARBR if high/low data is missing in simple history.
    - If OHLC is available, calculates ARBR.
    - Operates on a provided stock pool or default list.
    """
    
    def __init__(self, stock_pool: Optional[List[str]] = None):
        super().__init__()
        # Default pool if none provided (Mix of volatile and steady stocks for demo)
        self.stock_pool = stock_pool or [
            "600519", "000858", "601318", "300750", "000001", 
            "002415", "600036", "002475", "601888", "300059",
            "000799", "600809", "002304", "603259", "603288"
        ]
        self.g = {
            "stock_num": 3,
            "days_counter": 0,
            "rebalance_interval": 5 # Weekly
        }

    def _extract_symbols(self, graph_json: str):
        return self.stock_pool

    def initialize(self, context: Context):
        context.benchmark = "000905.SH"
        
    def calculate_arbr(self, df: pd.DataFrame) -> float:
        """
        Calculate ARBR Factor.
        AR = Sum(H - O) / Sum(O - L) * 100
        BR = Sum(H - C.shift(1)) / Sum(C.shift(1) - L) * 100
        Using last 26 days.
        """
        if len(df) < 26:
            return 0.0
            
        # Ensure we have required columns. DataProvider might return lower case.
        cols = {c.lower(): c for c in df.columns}
        if not all(k in cols for k in ['high', 'low', 'open', 'close']):
            # Fallback to Volume Momentum if OHLC not fully available
            if 'volume' in cols:
                return df[cols['volume']].iloc[-1] / (df[cols['volume']].mean() + 1e-6)
            return 0.0
            
        subset = df.iloc[-26:].copy()
        
        # AR
        h_minus_o = (subset[cols['high']] - subset[cols['open']]).sum()
        o_minus_l = (subset[cols['open']] - subset[cols['low']]).sum()
        ar = (h_minus_o / o_minus_l * 100) if o_minus_l != 0 else 0
        
        # BR
        # We need previous close for BR
        subset['prev_close'] = subset[cols['close']].shift(1)
        subset = subset.dropna()
        
        h_minus_pc = (subset[cols['high']] - subset['prev_close']).sum()
        pc_minus_l = (subset['prev_close'] - subset[cols['low']]).sum()
        br = (h_minus_pc / pc_minus_l * 100) if pc_minus_l != 0 else 0
        
        return ar + br # Simplified score
        
    def handle_bar(self, context: Context, graph_json: str):
        self.g["days_counter"] += 1
        
        # Weekly Rebalance
        if self.g["days_counter"] % self.g["rebalance_interval"] != 0:
            return

        candidates = []
        
        for symbol in self.stock_pool:
            # Fetch 30 days history
            hist = self.provider.get_history(symbol, days=40)
            if not hist or len(hist) < 30:
                continue
                
            # Convert to DataFrame
            # Note: DataProvider.get_history returns PricePoint which might only have close/volume
            # If we need OHLC, we might need to rely on the fact that 'vars(p)' might contain more if underlying provider gave it
            # Or assume PricePoint only has what's defined.
            # In rox3.0 DataProvider, PricePoint is: date, close, volume.
            # So ARBR calculation will fallback to Volume.
            
            df = pd.DataFrame([vars(p) for p in hist])
            df = df.sort_values("date")
            
            # Check Yesterday Trend (Proxy for Limit Up)
            # Limit Up is approx > 9.5%
            last_close = df.iloc[-1]["close"]
            prev_close = df.iloc[-2]["close"]
            pct_change = (last_close - prev_close) / prev_close
            
            # Strategy requires Limit Up. 
            # For robustness in this demo environment where we might not hit limit ups in the small pool:
            # We relax to > 3% gain.
            if pct_change < 0.03: 
                continue
            
            score = self.calculate_arbr(df)
            
            candidates.append({
                "symbol": symbol,
                "score": score
            })
            
        # Sort by Score (Descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:self.g["stock_num"]]
        
        # Generate Target Weights
        target_weights = {}
        if selected:
            weight = 1.0 / len(selected)
            for item in selected:
                target_weights[item["symbol"]] = weight
                
        self._rebalance(context, target_weights)
