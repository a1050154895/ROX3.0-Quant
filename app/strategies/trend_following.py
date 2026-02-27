#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势跟踪策略

该策略基于价格趋势的持续性，当价格突破一定阈值时，认为趋势形成，从而跟随趋势进行交易。

策略逻辑：
1. 计算短期和长期移动平均线
2. 当短期均线上穿长期均线时，认为形成上升趋势，买入
3. 当短期均线下穿长期均线时，认为形成下降趋势，卖出
4. 结合价格突破和成交量确认趋势强度

参数：
- short_ma_period: 短期移动平均线周期，默认10
- long_ma_period: 长期移动平均线周期，默认30
- breakout_threshold: 突破阈值，默认0.01
- stop_loss: 止损比例，默认0.06
- take_profit: 止盈比例，默认0.10
"""

import numpy as np
import pandas as pd

class TrendFollowingStrategy:
    """
    趋势跟踪策略类
    """
    
    def __init__(self):
        """
        初始化策略
        """
        self.name = "趋势跟踪策略"
        self.description = "基于移动平均线交叉和价格突破的趋势跟随策略"
        self.params = {
            "short_ma_period": 10,
            "long_ma_period": 30,
            "breakout_threshold": 0.01,
            "stop_loss": 0.06,
            "take_profit": 0.10
        }
        self.positions = {}
        self.trades = []
    
    def initialize(self, context):
        """
        初始化策略
        
        Args:
            context: 策略上下文对象
        """
        print(f"[{self.name}] 初始化策略...")
        context.universe = ["600519.SH", "000001.SZ", "300750.SZ", "601318.SH", "000858.SZ"]
        context.max_position_pct = 0.2
        
    def calculate_indicators(self, data):
        """
        计算技术指标
        
        Args:
            data: 价格数据
            
        Returns:
            dict: 包含指标的字典
        """
        indicators = {}
        
        short_period = self.params["short_ma_period"]
        long_period = self.params["long_ma_period"]
        
        for symbol, bars in data.items():
            if len(bars) < long_period:
                continue
            
            # 提取收盘价和成交量
            prices = [bar["close"] for bar in bars]
            volumes = [bar.get("volume", 0) for bar in bars]
            
            # 计算短期移动平均线
            short_ma = np.mean(prices[-short_period:])
            
            # 计算长期移动平均线
            long_ma = np.mean(prices[-long_period:])
            
            # 计算价格突破
            recent_high = max(prices[-short_period:])
            recent_low = min(prices[-short_period:])
            current_price = prices[-1]
            
            # 计算成交量变化
            if len(volumes) >= 2:
                volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] > 0 else 0
            else:
                volume_change = 0
            
            indicators[symbol] = {
                "short_ma": short_ma,
                "long_ma": long_ma,
                "ma_crossover": short_ma > long_ma,
                "price_breakout": (current_price - recent_high) / recent_high if recent_high > 0 else 0,
                "volume_change": volume_change,
                "current_price": current_price
            }
        
        return indicators
    
    def handle_data(self, context, data):
        """
        处理数据并生成交易信号
        
        Args:
            context: 策略上下文对象
            data: 价格数据
        """
        # 计算指标
        indicators = self.calculate_indicators(data)
        
        # 遍历股票
        for symbol in context.universe:
            if symbol not in data or symbol not in indicators:
                continue
            
            indicator = indicators[symbol]
            current_price = indicator["current_price"]
            ma_crossover = indicator["ma_crossover"]
            price_breakout = indicator["price_breakout"]
            volume_change = indicator["volume_change"]
            
            # 检查是否已持仓
            if symbol in self.positions:
                position = self.positions[symbol]
                entry_price = position["entry_price"]
                
                # 检查止盈止损
                if current_price >= entry_price * (1 + self.params["take_profit"]):
                    # 止盈
                    self.close_position(context, symbol, current_price, "止盈")
                elif current_price <= entry_price * (1 - self.params["stop_loss"]):
                    # 止损
                    self.close_position(context, symbol, current_price, "止损")
                elif not ma_crossover and position["side"] == "long":
                    # 趋势反转，平仓
                    self.close_position(context, symbol, current_price, "趋势反转")
                elif ma_crossover and position["side"] == "short":
                    # 趋势反转，平仓
                    self.close_position(context, symbol, current_price, "趋势反转")
            else:
                # 没有持仓，检查交易信号
                if ma_crossover and price_breakout > self.params["breakout_threshold"] and volume_change > 0:
                    # 上升趋势形成，买入
                    self.open_position(context, symbol, current_price, "long", "上升趋势")
                elif not ma_crossover and abs(price_breakout) > self.params["breakout_threshold"] and volume_change > 0:
                    # 下降趋势形成，卖出
                    self.open_position(context, symbol, current_price, "short", "下降趋势")
    
    def open_position(self, context, symbol, price, side, reason):
        """
        开仓
        
        Args:
            context: 策略上下文对象
            symbol: 股票代码
            price: 价格
            side: 方向，long或short
            reason: 开仓原因
        """
        # 计算仓位大小
        position_size = int(context.portfolio.cash * context.max_position_pct / price)
        
        if position_size <= 0:
            return
        
        # 记录持仓
        self.positions[symbol] = {
            "side": side,
            "entry_price": price,
            "size": position_size,
            "entry_time": pd.Timestamp.now()
        }
        
        # 记录交易
        trade = {
            "symbol": symbol,
            "side": side,
            "size": position_size,
            "price": price,
            "type": "open",
            "reason": reason,
            "time": pd.Timestamp.now()
        }
        self.trades.append(trade)
        
        print(f"[{self.name}] 开仓 {side} {symbol} {position_size}股 @ {price:.2f} - {reason}")
    
    def close_position(self, context, symbol, price, reason):
        """
        平仓
        
        Args:
            context: 策略上下文对象
            symbol: 股票代码
            price: 价格
            reason: 平仓原因
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        side = position["side"]
        entry_price = position["entry_price"]
        size = position["size"]
        
        # 计算盈亏
        if side == "long":
            pnl = (price - entry_price) * size
        else:  # short
            pnl = (entry_price - price) * size
        
        # 记录交易
        trade = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "type": "close",
            "reason": reason,
            "pnl": pnl,
            "time": pd.Timestamp.now()
        }
        self.trades.append(trade)
        
        # 更新账户现金
        context.portfolio.cash += pnl
        
        # 移除持仓
        del self.positions[symbol]
        
        print(f"[{self.name}] 平仓 {side} {symbol} {size}股 @ {price:.2f} - {reason} - 盈亏: {pnl:.2f}")
    
    def get_performance(self):
        """
        获取策略性能
        
        Returns:
            dict: 性能指标
        """
        # 计算总盈亏
        total_pnl = sum(trade.get("pnl", 0) for trade in self.trades if trade["type"] == "close")
        
        # 计算交易次数
        total_trades = len([trade for trade in self.trades if trade["type"] == "close"])
        
        # 计算胜率
        winning_trades = len([trade for trade in self.trades if trade["type"] == "close" and trade.get("pnl", 0) > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "trades": self.trades
        }

if __name__ == "__main__":
    # 简单的策略测试
    strategy = TrendFollowingStrategy()
    
    # 模拟上下文
    class MockContext:
        def __init__(self):
            self.universe = ["600519.SH"]
            self.max_position_pct = 0.2
            self.portfolio = type('obj', (object,), {'cash': 100000})
    
    context = MockContext()
    strategy.initialize(context)
    
    print(f"策略初始化完成: {strategy.name}")
    print(f"策略参数: {strategy.params}")
