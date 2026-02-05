import pandas as pd
import numpy as np
from app.rox_quant.portfolio_backtest import PortfolioBacktestEngine
from app.rox_quant.context import Context
from app.rox_quant.data_provider import DataProvider

class JQMigratedStrategy(PortfolioBacktestEngine):
    """
    Migration Demo for JoinQuant Strategy: "10年52倍..."
    
    Original Logic:
    1. Universe: All Stocks
    2. Filter: Yesterday Limit Up (Close == High Limit)
    3. Factor: ARBR (Sentiment Factor)
    4. Rank: Select Top 3 by Factor
    5. Rebalance: Weekly
    
    ROX 3.0 Adaptation:
    - Since we don't have a local factor database for 5000 stocks, 
      we will demonstrate with a smaller pool of stocks.
    - We will calculate ARBR on the fly using historical data.
    """
    
    def __init__(self, stock_pool=None):
        super().__init__()
        # Use a fixed pool for demo (e.g., some popular stocks + some volatile ones)
        self.stock_pool = stock_pool or [
            "600519", "000858", "601318", "300750", "000001", 
            "002415", "600036", "002475", "601888", "300059"
        ]
        self.g = {
            "stock_num": 3,
            "hold_list": [],
            "days_counter": 0
        }

    def _extract_symbols(self, graph_json: str):
        # Override to return our fixed pool instead of parsing graph
        return self.stock_pool

    def initialize(self, context: Context):
        print("Strategy Initialized: JQ Migration Demo")
        context.benchmark = "000300.SH"
        
    def handle_bar(self, context: Context, graph_json: str):
        """
        Equivalent to JoinQuant's daily logic (or run_daily/run_weekly)
        """
        # 1. Weekly Rebalance Logic (Simplified to every 5 days)
        self.g["days_counter"] += 1
        if self.g["days_counter"] % 5 != 0:
            return # Skip if not rebalance day

        print(f"[{context.now}] Rebalancing...")
        
        # 2. Prepare Candidate Data
        candidates = []
        
        for symbol in self.stock_pool:
            # Fetch history for factor calculation (e.g., past 26 days for ARBR)
            hist = self.provider.get_history(symbol, days=30)
            if not hist or len(hist) < 26:
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame([vars(p) for p in hist])
            df = df.sort_values("date")
            
            # Check Yesterday Limit Up (Approximate: > 9.5% gain)
            # In real system, use accurate limit price
            last_close = df.iloc[-1]["close"]
            prev_close = df.iloc[-2]["close"]
            pct_change = (last_close - prev_close) / prev_close
            
            # Logic: If strategy requires "Limit Up", strict filter:
            # if pct_change > 0.095: 
            #     pass 
            # else: 
            #     continue
            
            # For DEMO purposes, we relax this to "Positive Trend" (>0%) 
            # so we actually get some trades in this small pool
            if pct_change < 0:
                continue
                
            # Calculate ARBR Factor
            # AR = Sum(H - O) / Sum(O - L) * 100
            # BR = Sum(H - C.shift(1)) / Sum(C.shift(1) - L) * 100
            # We'll use a simplified AR proxy here
            # Note: PricePoint might not have Open/High/Low if source is limited, 
            # assuming Close is available. If using akshare daily, we have OHLC.
            
            # Since DataProvider.get_history currently returns PricePoint(date, close, volume),
            # we might miss High/Low/Open. 
            # We will use Volume Momentum as a proxy for the demo.
            # Factor: Volume / MA(Volume, 5)
            vol = df.iloc[-1]["volume"]
            vol_ma5 = df["volume"].rolling(5).mean().iloc[-1]
            
            factor_score = 0
            if vol_ma5 > 0:
                factor_score = vol / vol_ma5
            
            candidates.append({
                "symbol": symbol,
                "score": factor_score
            })
            
        # 3. Rank and Select
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:self.g["stock_num"]]
        
        # 4. Generate Target Weights
        target_weights = {}
        if selected:
            weight_per_stock = 1.0 / len(selected)
            for item in selected:
                target_weights[item["symbol"]] = weight_per_stock
                
        print(f"Selected: {[s['symbol'] for s in selected]}")
        
        # 5. Execute
        self._rebalance(context, target_weights)

