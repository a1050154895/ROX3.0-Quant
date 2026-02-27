import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
import datetime
import time
from app.utils.optimized_cache import LRUCache

logger = logging.getLogger(__name__)

class Context:
    def __init__(self):
        self.portfolio = {
            "cash": 100000.0,
            "total_value": 100000.0,
            "positions": {} # code -> qty
        }
        self.universe = []
        # Dynamic attributes allowed
        self.model = None
        self.model_weights = {}
        self.holding_period = 0
        self.max_position_pct = 1.0
        
        # Internal state for the engine to use during execution
        self._current_date = None
        self._data_provider = None
        self._engine = None
        self._current_prices = {}

    def order_target_value(self, code: str, target_value: float):
        if self._engine:
            self._engine.order_target_value(code, target_value)

    def order_target(self, code: str, amount: int):
        if self._engine:
            self._engine.order_target(code, amount)

class QuantEngine:
    def __init__(self, data_provider):
        self.provider = data_provider
        self.context = Context()
        self.context._data_provider = data_provider
        self.context._engine = self
        
        self.strategy_initialize = None
        self.strategy_handle_data = None
        
        self.history = [] # List of portfolio snapshots
        self.performance_metrics = {}
        self.execution_times = {}
        self.price_cache = LRUCache(maxsize=1000)  # 缓存价格数据

    def load_strategy(self, strategy_module):
        """
        加载策略模块
        """
        start_time = time.time()
        try:
            if hasattr(strategy_module, 'initialize'):
                self.strategy_initialize = strategy_module.initialize
            if hasattr(strategy_module, 'handle_data'):
                self.strategy_handle_data = strategy_module.handle_data
                
            # Run initialize immediately
            if self.strategy_initialize:
                self.strategy_initialize(self.context)
            logger.info("Strategy loaded successfully")
        except Exception as e:
            logger.error(f"Error loading strategy: {e}")
        finally:
            self.execution_times['load_strategy'] = time.time() - start_time

    def order_target_value(self, code: str, target_value: float):
        """
        调整持仓以达到目标价值
        """
        current_price = self._get_current_price(code)
        if current_price <= 0:
            logger.warning(f"Invalid price for {code}: {current_price}")
            return

        current_qty = self.context.portfolio["positions"].get(code, 0)
        current_val = current_qty * current_price
        
        diff_val = target_value - current_val
        
        if abs(diff_val) < current_price:
            return # 变化太小，忽略
            
        qty_to_trade = int(diff_val / current_price)
        
        if qty_to_trade == 0:
            return

        self._execute_trade(code, qty_to_trade, current_price)

    def order_target(self, code: str, target_qty: int):
        """
        调整持仓以达到目标数量
        """
        current_qty = self.context.portfolio["positions"].get(code, 0)
        qty_to_trade = int(target_qty - current_qty)
        
        if qty_to_trade == 0:
            return
            
        current_price = self._get_current_price(code)
        if current_price <= 0:
            logger.warning(f"Invalid price for {code}: {current_price}")
            return
            
        self._execute_trade(code, qty_to_trade, current_price)

    def _execute_trade(self, code: str, qty: int, price: float):
        """
        执行交易
        """
        try:
            cost = qty * price
            commission = abs(cost) * 0.0003 # 0.03% commission estimate
            
            # Check cash if buying
            if qty > 0:
                if self.context.portfolio["cash"] < cost + commission:
                    # Adjust qty
                    qty = int((self.context.portfolio["cash"] - commission) / price)
                    if qty <= 0:
                        return
                    cost = qty * price
                    commission = abs(cost) * 0.0003

            self.context.portfolio["cash"] -= (cost + commission)
            
            old_qty = self.context.portfolio["positions"].get(code, 0)
            new_qty = old_qty + qty
            
            if new_qty == 0:
                if code in self.context.portfolio["positions"]:
                    del self.context.portfolio["positions"][code]
            else:
                self.context.portfolio["positions"][code] = new_qty
                
            logger.debug(f"Trade: {code} Qty:{qty} Price:{price} Cash:{self.context.portfolio['cash']:.2f}")
        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    def _get_current_price(self, code: str) -> float:
        """
        获取当前价格
        """
        # 首先从当前价格字典中获取
        if code in self.context._current_prices:
            return self.context._current_prices[code]
        
        # 从缓存中获取
        cache_key = f"price:{code}:{self.context._current_date.strftime('%Y-%m-%d')}"
        cached_price = self.price_cache.get(cache_key)
        if cached_price is not None:
            return cached_price
        
        # 从数据提供者获取
        try:
            current_date_str = self.context._current_date.strftime("%Y-%m-%d")
            hist = self.provider.get_history(code, current_date_str, current_date_str)
            if hist and len(hist) > 0:
                price = hist[0]['close']
                self.price_cache.set(cache_key, price)
                return price
        except Exception as e:
            logger.error(f"Error getting price for {code}: {e}")
        
        return 0.0

    def run_backtest(self, start_date: str, end_date: str) -> List[Dict]:
        """
        运行回测
        """
        start_time = time.time()
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # 1. Generate trading days
        s_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        e_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        
        current_date = s_date
        total_days = (e_date - s_date).days + 1
        processed_days = 0
        
        # Pre-fetch history for universe to optimize
        universe_data = {}
        data_loading_start = time.time()
        for code in self.context.universe:
            try:
                hist = self.provider.get_history(code, start_date, end_date)
                # Convert to dict keyed by date string
                universe_data[code] = {d['date']: d for d in hist}
                logger.debug(f"Loaded data for {code}: {len(hist)} days")
            except Exception as e:
                logger.error(f"Error loading data for {code}: {e}")
                universe_data[code] = {}
        self.execution_times['data_loading'] = time.time() - data_loading_start
        
        backtest_start = time.time()
        while current_date <= e_date:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Prepare 'data' dict for handle_data
            bar_data = {}
            self.context._current_prices = {}
            
            has_data = False
            for code in self.context.universe:
                if date_str in universe_data[code]:
                    row = universe_data[code][date_str]
                    bar_data[code] = row
                    self.context._current_prices[code] = row['close']
                    has_data = True
            
            if has_data:
                self.context._current_date = current_date
                
                # Call Strategy
                if self.strategy_handle_data:
                    strategy_start = time.time()
                    try:
                        self.strategy_handle_data(self.context, bar_data)
                    except Exception as e:
                        logger.error(f"Error in strategy handle_data: {e}")
                    self.execution_times.setdefault('strategy_execution', []).append(time.time() - strategy_start)
                
                # Update Portfolio Value
                position_val = 0.0
                for code, qty in self.context.portfolio["positions"].items():
                    price = self.context._current_prices.get(code, 0.0)
                    if price == 0 and code in universe_data:
                        # 尝试获取前一天的价格
                        prev_date = current_date - datetime.timedelta(days=1)
                        prev_date_str = prev_date.strftime("%Y-%m-%d")
                        if prev_date_str in universe_data[code]:
                            price = universe_data[code][prev_date_str]['close']
                            self.context._current_prices[code] = price
                    position_val += qty * price
                
                self.context.portfolio["total_value"] = self.context.portfolio["cash"] + position_val
                
                # Record History
                self.history.append({
                    "date": date_str,
                    "value": self.context.portfolio["total_value"],
                    "cash": self.context.portfolio["cash"],
                    "positions": self.context.portfolio["positions"].copy()
                })
            
            processed_days += 1
            if processed_days % 10 == 0:
                progress = (processed_days / total_days) * 100
                logger.info(f"Backtest progress: {progress:.1f}%")
            
            current_date += datetime.timedelta(days=1)
        
        self.execution_times['backtest'] = time.time() - backtest_start
        self.execution_times['total'] = time.time() - start_time
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        logger.info(f"Backtest completed in {self.execution_times['total']:.2f} seconds")
        logger.info(f"Performance metrics: {self.performance_metrics}")
        
        return self.history
    
    def _calculate_performance_metrics(self):
        """
        计算性能指标
        """
        if not self.history:
            return
        
        # 计算总收益率
        initial_value = self.history[0]['value']
        final_value = self.history[-1]['value']
        total_return = (final_value - initial_value) / initial_value
        
        # 计算年化收益率
        start_date = datetime.datetime.strptime(self.history[0]['date'], "%Y-%m-%d")
        end_date = datetime.datetime.strptime(self.history[-1]['date'], "%Y-%m-%d")
        days = (end_date - start_date).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # 计算最大回撤
        max_drawdown = 0
        peak = initial_value
        for entry in self.history:
            current_value = entry['value']
            if current_value > peak:
                peak = current_value
            drawdown = (peak - current_value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 计算夏普比率（假设无风险利率为0）
        returns = []
        for i in range(1, len(self.history)):
            prev_value = self.history[i-1]['value']
            curr_value = self.history[i]['value']
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        self.performance_metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "initial_value": initial_value,
            "final_value": final_value,
            "trading_days": len(self.history)
        }

