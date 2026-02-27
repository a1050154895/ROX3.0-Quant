#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI智能策略

该策略利用AI技术分析市场数据，生成交易信号。策略结合了技术分析、基本面分析和情感分析，
通过AI模型预测市场趋势并生成交易决策。

策略逻辑：
1. 收集多维度市场数据（价格、成交量、新闻、社交媒体情绪等）
2. 使用AI模型分析数据，预测市场趋势
3. 结合风险评估，生成交易信号
4. 动态调整策略参数以适应市场变化

参数：
- model_name: AI模型名称，默认"gpt-4"
- risk_level: 风险等级，默认"medium"
- lookback_period: 回测周期，默认30
- position_size: 仓位大小，默认0.1
- stop_loss: 止损比例，默认0.05
- take_profit: 止盈比例，默认0.15
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

class AIIntelligentStrategy:
    """
    AI智能策略类
    """
    
    def __init__(self):
        """
        初始化策略
        """
        self.name = "AI智能策略"
        self.description = "基于AI分析的多维度智能交易策略"
        self.params = {
            "model_name": "gpt-4",
            "risk_level": "medium",  # low, medium, high
            "lookback_period": 30,
            "position_size": 0.1,
            "stop_loss": 0.05,
            "take_profit": 0.15
        }
        self.positions = {}
        self.trades = []
        self.market_data = {}
        self.ai_predictions = {}
    
    def initialize(self, context):
        """
        初始化策略
        
        Args:
            context: 策略上下文对象
        """
        print(f"[{self.name}] 初始化策略...")
        context.universe = ["600519.SH", "000001.SZ", "300750.SZ", "601318.SH", "000858.SZ", "600036.SH", "002594.SZ", "601888.SH"]
        context.max_position_pct = self.params["position_size"]
        
    def collect_market_data(self, symbol, data):
        """
        收集市场数据
        
        Args:
            symbol: 股票代码
            data: 价格数据
            
        Returns:
            dict: 市场数据
        """
        if symbol not in data:
            return None
        
        bars = data[symbol]
        if len(bars) < self.params["lookback_period"]:
            return None
        
        # 提取价格和成交量数据
        prices = [bar["close"] for bar in bars]
        volumes = [bar.get("volume", 0) for bar in bars]
        
        # 计算技术指标
        short_ma = np.mean(prices[-10:])
        long_ma = np.mean(prices[-30:])
        rsi = self.calculate_rsi(prices, 14)
        macd, signal = self.calculate_macd(prices)
        
        # 模拟新闻情感分析（实际应用中应调用真实的情感分析API）
        sentiment = self.simulate_sentiment_analysis(symbol)
        
        market_data = {
            "prices": prices,
            "volumes": volumes,
            "short_ma": short_ma,
            "long_ma": long_ma,
            "rsi": rsi,
            "macd": macd,
            "signal": signal,
            "sentiment": sentiment,
            "current_price": prices[-1]
        }
        
        self.market_data[symbol] = market_data
        return market_data
    
    def calculate_rsi(self, prices, period=14):
        """
        计算RSI指标
        
        Args:
            prices: 价格序列
            period: 计算周期
            
        Returns:
            float: RSI值
        """
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        
        avg_gain = np.mean(gains[-period:]) if len(gains) > 0 else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        计算MACD指标
        
        Args:
            prices: 价格序列
            fast_period: 快速移动平均线周期
            slow_period: 慢速移动平均线周期
            signal_period: 信号线周期
            
        Returns:
            tuple: (macd, signal)
        """
        if len(prices) < slow_period + signal_period:
            return 0.0, 0.0
        
        fast_ema = self.calculate_ema(prices, fast_period)
        slow_ema = self.calculate_ema(prices, slow_period)
        macd = fast_ema - slow_ema
        signal = self.calculate_ema([macd], signal_period)
        
        return macd, signal
    
    def calculate_ema(self, prices, period):
        """
        计算指数移动平均线
        
        Args:
            prices: 价格序列
            period: 计算周期
            
        Returns:
            float: EMA值
        """
        if len(prices) < period:
            return np.mean(prices)
        
        k = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * k + ema * (1 - k)
        
        return ema
    
    def simulate_sentiment_analysis(self, symbol):
        """
        模拟新闻情感分析
        
        Args:
            symbol: 股票代码
            
        Returns:
            float: 情感得分 (-1 to 1)
        """
        # 模拟情感分析结果
        # 实际应用中应调用ClawFeed或其他情感分析API
        sentiment = np.random.uniform(-0.5, 0.5)
        
        # 根据股票代码添加一些随机性
        symbol_hash = hash(symbol) % 10
        if symbol_hash > 7:
            sentiment += 0.2
        elif symbol_hash < 3:
            sentiment -= 0.2
        
        return max(-1.0, min(1.0, sentiment))
    
    def ai_analysis(self, symbol, market_data):
        """
        AI分析市场数据
        
        Args:
            symbol: 股票代码
            market_data: 市场数据
            
        Returns:
            dict: AI分析结果
        """
        # 模拟AI分析过程
        # 实际应用中应调用OpenClaw或其他AI API
        time.sleep(0.1)  # 模拟API调用延迟
        
        # 基于市场数据生成预测
        price_change = (market_data["current_price"] - market_data["prices"][-2]) / market_data["prices"][-2]
        
        # 综合多维度因素
        factors = {
            "price_momentum": price_change,
            "ma_crossover": 1 if market_data["short_ma"] > market_data["long_ma"] else -1,
            "rsi": (market_data["rsi"] - 50) / 50,  # 归一化到 -1 到 1
            "macd": 1 if market_data["macd"] > market_data["signal"] else -1,
            "sentiment": market_data["sentiment"]
        }
        
        # 计算综合得分
        weights = {
            "price_momentum": 0.25,
            "ma_crossover": 0.2,
            "rsi": 0.15,
            "macd": 0.2,
            "sentiment": 0.2
        }
        
        score = sum(factors[key] * weights[key] for key in factors)
        
        # 根据得分生成预测
        if score > 0.3:
            prediction = "bullish"
            confidence = min(1.0, score)
        elif score < -0.3:
            prediction = "bearish"
            confidence = min(1.0, abs(score))
        else:
            prediction = "neutral"
            confidence = 0.5
        
        analysis = {
            "prediction": prediction,
            "confidence": confidence,
            "score": score,
            "factors": factors,
            "timestamp": datetime.now().isoformat()
        }
        
        self.ai_predictions[symbol] = analysis
        return analysis
    
    def handle_data(self, context, data):
        """
        处理数据并生成交易信号
        
        Args:
            context: 策略上下文对象
            data: 价格数据
        """
        # 遍历股票
        for symbol in context.universe:
            # 收集市场数据
            market_data = self.collect_market_data(symbol, data)
            if not market_data:
                continue
            
            # AI分析
            analysis = self.ai_analysis(symbol, market_data)
            current_price = market_data["current_price"]
            
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
                elif (analysis["prediction"] == "bearish" and position["side"] == "long") or \
                     (analysis["prediction"] == "bullish" and position["side"] == "short"):
                    # 趋势反转，平仓
                    self.close_position(context, symbol, current_price, "趋势反转")
            else:
                # 没有持仓，检查交易信号
                if analysis["prediction"] == "bullish" and analysis["confidence"] > 0.6:
                    # 看涨信号，买入
                    self.open_position(context, symbol, current_price, "long", f"AI看涨预测 (信心: {analysis['confidence']:.2f})")
                elif analysis["prediction"] == "bearish" and analysis["confidence"] > 0.6:
                    # 看跌信号，卖出
                    self.open_position(context, symbol, current_price, "short", f"AI看跌预测 (信心: {analysis['confidence']:.2f})")
    
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
        
        # 计算平均盈亏
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # 计算最大回撤（简化版）
        if not self.trades:
            max_drawdown = 0
        else:
            equity_curve = []
            current_equity = 0
            peak_equity = 0
            max_drawdown = 0
            
            for trade in self.trades:
                if trade["type"] == "close":
                    current_equity += trade.get("pnl", 0)
                    peak_equity = max(peak_equity, current_equity)
                    if peak_equity > 0:
                        drawdown = (peak_equity - current_equity) / peak_equity
                        max_drawdown = max(max_drawdown, drawdown)
            
        return {
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "max_drawdown": max_drawdown,
            "trades": self.trades,
            "ai_predictions": self.ai_predictions
        }

if __name__ == "__main__":
    # 简单的策略测试
    strategy = AIIntelligentStrategy()
    
    # 模拟上下文
    class MockContext:
        def __init__(self):
            self.universe = ["600519.SH"]
            self.max_position_pct = 0.1
            self.portfolio = type('obj', (object,), {'cash': 100000})
    
    context = MockContext()
    strategy.initialize(context)
    
    print(f"策略初始化完成: {strategy.name}")
    print(f"策略参数: {strategy.params}")
