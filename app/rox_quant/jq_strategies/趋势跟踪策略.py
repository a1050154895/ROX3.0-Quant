#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势跟踪策略
基于移动平均线和MACD指标的趋势跟踪策略
适用于各种市场环境，特别是趋势明显的市场
"""

import numpy as np
import pandas as pd

# 策略参数
strategy_params = {
    'short_ma': 20,      # 短期移动平均线
    'long_ma': 60,       # 长期移动平均线
    'macd_fast': 12,      # MACD快线
    'macd_slow': 26,      # MACD慢线
    'macd_signal': 9,     # MACD信号线
    'stop_loss': 0.05,    # 止损比例
    'take_profit': 0.15   # 止盈比例
}

# 全局变量
g = {
    'portfolio_value': 100000,  # 初始资金
    'holdings': {},             # 持仓
    'trades': [],               # 交易记录
    'params': strategy_params   # 策略参数
}

# 初始化函数
def initialize(context):
    """
    初始化策略
    """
    print("趋势跟踪策略初始化")
    print(f"策略参数: {g['params']}")
    print(f"初始资金: {g['portfolio_value']}")

# 交易函数
def handle_data(context, data):
    """
    处理数据并执行交易
    """
    # 获取股票列表
    stocks = get_stock_list()
    
    # 分析每个股票
    for stock in stocks:
        analyze_stock(stock, data)
    
    # 更新持仓价值
    update_portfolio_value()
    
    # 输出当前状态
    print(f"当前资金: {g['portfolio_value']:.2f}")
    print(f"当前持仓: {g['holdings']}")

# 获取股票列表
def get_stock_list():
    """
    获取股票列表
    """
    # 这里返回一些示例股票
    # 实际应用中可以根据行业、市值等筛选
    return ['600519', '000858', '000001', '601318', '600036']

# 分析单个股票
def analyze_stock(stock, data):
    """
    分析单个股票并执行交易
    """
    try:
        # 获取历史数据
        hist_data = get_historical_data(stock, 100)
        
        if len(hist_data) < g['params']['long_ma']:
            return
        
        # 计算技术指标
        indicators = calculate_indicators(hist_data)
        
        # 生成交易信号
        signal = generate_signal(indicators)
        
        # 执行交易
        execute_trade(stock, signal, hist_data['close'].iloc[-1])
        
    except Exception as e:
        print(f"分析股票 {stock} 时出错: {e}")

# 获取历史数据
def get_historical_data(stock, days):
    """
    获取历史数据
    """
    # 模拟获取历史数据
    # 实际应用中应该从数据源获取
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
    close_prices = np.random.rand(days) * 100 + 100
    
    data = pd.DataFrame({
        'date': dates,
        'open': close_prices * (1 + np.random.randn(days) * 0.01),
        'high': close_prices * (1 + np.random.randn(days) * 0.02),
        'low': close_prices * (1 - np.random.randn(days) * 0.02),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, days)
    })
    
    data.set_index('date', inplace=True)
    return data

# 计算技术指标
def calculate_indicators(data):
    """
    计算技术指标
    """
    indicators = {}
    
    # 计算移动平均线
    indicators['short_ma'] = data['close'].rolling(g['params']['short_ma']).mean().iloc[-1]
    indicators['long_ma'] = data['close'].rolling(g['params']['long_ma']).mean().iloc[-1]
    
    # 计算MACD
    exp1 = data['close'].ewm(span=g['params']['macd_fast'], adjust=False).mean()
    exp2 = data['close'].ewm(span=g['params']['macd_slow'], adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=g['params']['macd_signal'], adjust=False).mean()
    histogram = macd - signal
    
    indicators['macd'] = macd.iloc[-1]
    indicators['signal'] = signal.iloc[-1]
    indicators['histogram'] = histogram.iloc[-1]
    
    # 计算价格
    indicators['current_price'] = data['close'].iloc[-1]
    indicators['previous_price'] = data['close'].iloc[-2] if len(data) > 1 else data['close'].iloc[-1]
    
    return indicators

# 生成交易信号
def generate_signal(indicators):
    """
    生成交易信号
    """
    signal = 0  # 0: 无信号, 1: 买入, -1: 卖出
    
    # 移动平均线交叉
    if indicators['short_ma'] > indicators['long_ma']:
        # 金叉，可能买入
        if indicators['macd'] > indicators['signal'] and indicators['histogram'] > 0:
            signal = 1
    else:
        # 死叉，可能卖出
        if indicators['macd'] < indicators['signal'] and indicators['histogram'] < 0:
            signal = -1
    
    return signal

# 执行交易
def execute_trade(stock, signal, price):
    """
    执行交易
    """
    # 检查是否已经持仓
    if stock in g['holdings']:
        # 已经持仓，检查是否需要卖出
        if signal == -1 or check_stop_conditions(stock, price):
            sell_stock(stock, price)
    else:
        # 未持仓，检查是否需要买入
        if signal == 1:
            buy_stock(stock, price)

# 检查止盈止损条件
def check_stop_conditions(stock, current_price):
    """
    检查止盈止损条件
    """
    holding = g['holdings'][stock]
    buy_price = holding['price']
    
    # 计算盈亏比例
    profit_ratio = (current_price - buy_price) / buy_price
    
    # 检查止盈
    if profit_ratio >= g['params']['take_profit']:
        print(f"股票 {stock} 达到止盈条件: {profit_ratio:.2f}")
        return True
    
    # 检查止损
    if profit_ratio <= -g['params']['stop_loss']:
        print(f"股票 {stock} 达到止损条件: {profit_ratio:.2f}")
        return True
    
    return False

# 买入股票
def buy_stock(stock, price):
    """
    买入股票
    """
    # 计算买入数量
    buy_amount = g['portfolio_value'] * 0.1  # 每次买入10%的资金
    buy_shares = int(buy_amount / price)
    
    if buy_shares <= 0:
        return
    
    # 执行买入
    cost = buy_shares * price
    g['portfolio_value'] -= cost
    g['holdings'][stock] = {
        'shares': buy_shares,
        'price': price,
        'buy_date': pd.Timestamp.now()
    }
    
    # 记录交易
    g['trades'].append({
        'stock': stock,
        'type': 'buy',
        'price': price,
        'shares': buy_shares,
        'amount': cost,
        'date': pd.Timestamp.now()
    })
    
    print(f"买入 {stock}: {buy_shares}股, 价格: {price:.2f}, 花费: {cost:.2f}")

# 卖出股票
def sell_stock(stock, price):
    """
    卖出股票
    """
    holding = g['holdings'][stock]
    shares = holding['shares']
    buy_price = holding['price']
    
    # 计算卖出金额
    sell_amount = shares * price
    profit = sell_amount - (shares * buy_price)
    
    # 执行卖出
    g['portfolio_value'] += sell_amount
    del g['holdings'][stock]
    
    # 记录交易
    g['trades'].append({
        'stock': stock,
        'type': 'sell',
        'price': price,
        'shares': shares,
        'amount': sell_amount,
        'profit': profit,
        'date': pd.Timestamp.now()
    })
    
    print(f"卖出 {stock}: {shares}股, 价格: {price:.2f}, 收入: {sell_amount:.2f}, 利润: {profit:.2f}")

# 更新持仓价值
def update_portfolio_value():
    """
    更新持仓价值
    """
    # 这里简化处理，实际应用中应该根据最新价格计算
    pass

# 回测函数
def run_backtest():
    """
    运行回测
    """
    print("开始回测")
    
    # 初始化
    initialize({})
    
    # 模拟100天的交易
    for i in range(100):
        print(f"\n第 {i+1} 天")
        handle_data({}, {})
    
    # 输出回测结果
    print("\n回测完成")
    print(f"最终资金: {g['portfolio_value']:.2f}")
    print(f"总交易次数: {len(g['trades'])}")
    
    # 计算收益率
    initial_value = 100000
    final_value = g['portfolio_value']
    return_rate = (final_value - initial_value) / initial_value
    
    print(f"总收益率: {return_rate:.2%}")
    
    # 返回回测结果
    return {
        'initial_value': initial_value,
        'final_value': final_value,
        'return_rate': return_rate,
        'trades': g['trades'],
        'params': g['params']
    }

# 主函数
if __name__ == "__main__":
    # 运行回测
    result = run_backtest()
    
    # 输出结果
    print("\n回测结果:")
    print(result)

# 导出结果
def export_results():
    """
    导出回测结果
    """
    import json
    
    results = {
        'portfolio_value': g['portfolio_value'],
        'holdings': g['holdings'],
        'trades': g['trades'],
        'params': g['params']
    }
    
    with open('趋势跟踪策略回测结果.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("回测结果已导出到 趋势跟踪策略回测结果.json")

# 扩展功能：优化参数
def optimize_parameters():
    """
    优化策略参数
    """
    print("开始优化策略参数")
    
    # 这里可以实现参数优化逻辑
    # 例如使用网格搜索或遗传算法
    
    print("参数优化完成")
    return g['params']
