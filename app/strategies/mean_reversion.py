#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
均值回归策略

该策略基于价格围绕均值波动的原理，当价格偏离均值超过一定阈值时，认为价格会回归均值，从而进行反向交易。

策略逻辑：
1. 计算股票价格的移动平均线作为均值
2. 计算价格与均值的偏离度（Z-score）
3. 当偏离度超过阈值时，进行反向交易
4. 当价格回归均值时，平仓

参数：
- ma_period: 移动平均线周期，默认20
- z_score_threshold: Z-score阈值，默认2.0
- stop_loss: 止损比例，默认0.05
- take_profit: 止盈比例，默认0.03
"""

import numpy as np
import pandas as pd

class MeanReversionStrategy:
    """
    均值回归策略类
    """
    
    def __init__(self):
        """
        初始化策略
        """
        self.name = "均值回归策略"
        self.description = "基于价格偏离均值的反向交易策略"
        self.params = {
            "ma_period": 20,
            "z_score_threshold": 2.0,
            "stop_loss": 0.05,
            "take_profit": 0.03
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
        
        for symbol, bars in data.items():
            if len(bars) < self.params["ma_period"]:
                continue
            
            # 提取收盘价
            prices = [bar["close"] for bar in bars]
            
            # 计算移动平均线
            ma = np.mean(prices[-self.params["ma_period"]:])
            
            # 计算标准差
            std = np.std(prices[-self.params["ma_period"]:])
            
            # 计算当前价格
            current_price = prices[-1]
            
            # 计算Z-score
            z_score = (current_price - ma) / std if std > 0 else 0
            
            indicators[symbol] = {
                "ma": ma,
                "std": std,
                "z_score": z_score,
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
            z_score = indicator["z_score"]
            
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
                elif abs(z_score) < 0.5:
                    # 回归均值，平仓
                    self.close_position(context, symbol, current_price, "回归均值")
            else:
                # 没有持仓，检查交易信号
                if z_score > self.params["z_score_threshold"]:
                    # 价格高估，卖出
                    self.open_position(context, symbol, current_price, "short", "价格高估")
                elif z_score < -self.params["z_score_threshold"]:
                    # 价格低估，买入
                    self.open_position(context, symbol, current_price, "long", "价格低估")
    
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
    strategy = MeanReversionStrategy()
    
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
