
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, int] = field(default_factory=dict) # symbol -> quantity
    total_value: float = 0.0
    market_value: float = 0.0
    
    def update(self, current_prices: Dict[str, float]):
        self.market_value = sum(
            qty * current_prices.get(sym, 0.0) 
            for sym, qty in self.positions.items()
        )
        self.total_value = self.cash + self.market_value

class Context:
    def __init__(self, initial_capital: float, start_date: str, end_date: str):
        self.portfolio = Portfolio(cash=initial_capital, total_value=initial_capital)
        self.start_date = start_date
        self.end_date = end_date
        self.now: Optional[datetime] = None
        self.universe: List[str] = []
        # User can attach any attribute to context
        
    def __setattr__(self, key, value):
        super().__setattr__(key, value)

class Bar:
    def __init__(self, data: pd.Series):
        self._data = data
        
    @property
    def close(self): return self._data.get('close')
    @property
    def open(self): return self._data.get('open')
    @property
    def high(self): return self._data.get('high')
    @property
    def low(self): return self._data.get('low')
    @property
    def volume(self): return self._data.get('volume')

class RQAdapter:
    """
    Ricequant/Zipline style backtest adapter for ROX.
    Allows running strategies written in standard init/handle_bar format.
    """
    
    def __init__(self, data_map: Dict[str, pd.DataFrame], initial_capital: float = 100000.0):
        self.data_map = data_map # {symbol: DataFrame with datetime index}
        self.context = None
        self.initial_capital = initial_capital
        self.logs = []
        self.equity_curve = []
        
        # Determine simulation timeframe
        all_dates = set()
        for df in data_map.values():
            if not df.empty:
                all_dates.update(df.index.tolist())
        self.trading_dates = sorted(list(all_dates))

    def run(self, init_func: Callable, handle_bar_func: Callable):
        # 1. Initialize Context
        start_str = self.trading_dates[0].strftime("%Y-%m-%d") if self.trading_dates else ""
        end_str = self.trading_dates[-1].strftime("%Y-%m-%d") if self.trading_dates else ""
        self.context = Context(self.initial_capital, start_str, end_str)
        
        # 2. User Init
        init_func(self.context)
        
        # 3. Event Loop
        for date in self.trading_dates:
            self.context.now = date
            
            # Prepare BarDict
            # Only include stocks that have data for this date
            bar_dict = {}
            current_prices = {}
            
            for symbol, df in self.data_map.items():
                if date in df.index:
                    row = df.loc[date]
                    bar_dict[symbol] = Bar(row)
                    current_prices[symbol] = row['close']
                elif not df.empty and df.index[0] < date:
                    # Fill with prev close if missing? Or just skip?
                    # For simplicity, skip. But portfolio update needs price.
                    # Try find previous valid price
                    try:
                        idx = df.index.get_indexer([date], method='pad')[0]
                        if idx >= 0:
                            row = df.iloc[idx]
                            current_prices[symbol] = row['close']
                    except Exception:
                        pass
            
            # Update Portfolio Value
            self.context.portfolio.update(current_prices)
            self.equity_curve.append({
                "time": date.strftime("%Y-%m-%d"),
                "value": self.context.portfolio.total_value
            })
            
            # Inject API functions into global scope for handle_bar
            # This is a bit hacky but mimics RQ/Zipline magic
            self._inject_api(current_prices)
            
            # Call User Logic
            handle_bar_func(self.context, bar_dict)
            
        return self._generate_report()

    def _inject_api(self, current_prices):
        """Injects order functions into the strategy's scope scope (conceptually)"""
        # Since we can't easily modify the user function's globals without exec,
        # we attach these methods to context or assume user uses the ones we provide below.
        # Ideally, we pass these as part of a library. 
        # For this implementation, we will define global functions that access 'current_adapter'
        global _current_adapter
        global _current_prices
        _current_adapter = self
        _current_prices = current_prices

    def order_shares(self, symbol: str, quantity: int):
        # Simple Order execution (Market Order)
        if quantity == 0: return
        
        price = self._get_price(symbol)
        if not price: 
            logger.warning(f"Cannot order {symbol}, no price data.")
            return
            
        cost = quantity * price
        
        # Buy
        if quantity > 0:
            if self.context.portfolio.cash >= cost:
                self.context.portfolio.cash -= cost
                self.context.portfolio.positions[symbol] = self.context.portfolio.positions.get(symbol, 0) + quantity
                self._log_trade("Buy", symbol, quantity, price)
            else:
                logger.warning(f"Not enough cash to buy {quantity} {symbol}")
                
        # Sell
        else:
            current_qty = self.context.portfolio.positions.get(symbol, 0)
            if current_qty >= abs(quantity):
                self.context.portfolio.cash -= cost # cost is negative, so cash increases
                self.context.portfolio.positions[symbol] = current_qty + quantity
                if self.context.portfolio.positions[symbol] == 0:
                    del self.context.portfolio.positions[symbol]
                self._log_trade("Sell", symbol, quantity, price)
            else:
                logger.warning(f"Not enough shares to sell {abs(quantity)} {symbol}")

    def order_target_percent(self, symbol: str, percent: float):
        target_value = self.context.portfolio.total_value * percent
        price = self._get_price(symbol)
        if not price: return
        
        target_shares = int(target_value / price)
        current_shares = self.context.portfolio.positions.get(symbol, 0)
        
        diff = target_shares - current_shares
        self.order_shares(symbol, diff)

    def _get_price(self, symbol):
        # In a real run, this comes from the current bar iteration
        # We access it via the saved current_prices in _inject_api, 
        # but since we are in the class, we can just use the last known price from context update
        # However, context update loop variable is local.
        # We'll use a trick: _inject_api saves to module level or we use a member var.
        global _current_prices
        return _current_prices.get(symbol)

    def _log_trade(self, action, symbol, qty, price):
        self.logs.append({
            "date": self.context.now.strftime("%Y-%m-%d"),
            "action": action,
            "symbol": symbol,
            "shares": qty,
            "price": price,
            "reason": "Signal"
        })

    def _generate_report(self):
        # Calculate standard metrics
        import pandas as pd
        df = pd.DataFrame(self.equity_curve)
        
        metrics = {}
        if not df.empty:
            df['returns'] = df['value'].pct_change().fillna(0)
            total_ret = (df['value'].iloc[-1] / df['value'].iloc[0]) - 1
            ann_ret = (1 + total_ret) ** (252 / len(df)) - 1 if len(df) > 0 else 0
            vol = df['returns'].std() * np.sqrt(252)
            sharpe = (ann_ret - 0.03) / vol if vol > 0 else 0
            max_dd = ((df['value'] - df['value'].cummax()) / df['value'].cummax()).min()
            
            metrics = {
                "annualized_return": round(ann_ret * 100, 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown": round(max_dd * 100, 2)
            }
            
        return {
            "logs": self.logs,
            "equity": self.equity_curve,
            "metrics": metrics
        }

# Global API proxies
_current_adapter = None
_current_prices = {}

def order_shares(symbol, quantity):
    if _current_adapter:
        _current_adapter.order_shares(symbol, quantity)

def order_target_percent(symbol, percent):
    if _current_adapter:
        _current_adapter.order_target_percent(symbol, percent)

# Alias for compatibility
order_percent = order_target_percent
