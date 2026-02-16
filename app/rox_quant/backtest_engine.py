"""
P3.5 Day 1: 回测引擎 - BacktestEngine
作用：加载历史K线数据，按时间顺序重放，根据交易信号模拟下单和平仓
关键概念：
  - K线：开盘价、最高价、最低价、收盘价、成交量
  - 信号：BUY(买入) / SELL(卖出) / HOLD(持仓)
  - 交易流程：收到BUY信号 → 以当前K线收盘价下单 → 收到SELL信号 → 平仓
  - 账户跟踪：记录每次进出场、盈亏、账户净值曲线
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置参数"""
    initial_capital: float = 100000.0        # 初始资金（元）
    commission_rate: float = 0.0003          # 手续费率 (0.03%)
    slippage: float = 0.0001                 # 滑点成本 (0.01%)
    stamp_duty: float = 0.0005               # 印花税 (0.05%, 仅卖出)
    min_commission: float = 5.0              # 最低佣金 (5元)
    position_size: float = 1.0               # 仓位大小 (0-1 为百分比, >1 为股数)
    min_price: float = 0.01                  # 最小价格精度
    risk_free_rate: float = 0.02             # 无风险利率（年化）
    max_position_size: float = 1.0          # 最大仓位限制 (0-1)
    stop_loss_pct: Optional[float] = None    # 止损百分比
    take_profit_pct: Optional[float] = None  # 止盈百分比


@dataclass
class TradeRecord:
    """单笔交易记录"""
    trade_id: int
    entry_time: datetime              # 进场时间
    entry_price: float                # 进场价格
    entry_qty: int                    # 进场数量（股）
    exit_time: Optional[datetime] = None     # 离场时间
    exit_price: Optional[float] = None       # 离场价格
    exit_qty: Optional[int] = None          # 离场数量（股）
    profit: Optional[float] = None           # 绝对利润（元）
    profit_pct: Optional[float] = None       # 收益率（%）
    commission: float = 0.0                  # 手续费总额（元）
    is_closed: bool = False                  # 是否平仓
    trade_type: str = "long"                 # 交易类型: long/short
    stop_loss: Optional[float] = None        # 止损价格
    take_profit: Optional[float] = None      # 止盈价格
    holding_period: Optional[int] = None     # 持有周期（天）


@dataclass
class BacktestResult:
    """回测结果"""
    portfolio_values: List[float]            # 账户净值曲线
    portfolio_dates: List[datetime]          # 净值对应日期
    trades: List[TradeRecord]                # 交易记录
    total_return: float                      # 总收益率
    sharpe_ratio: float                      # 夏普比率
    max_drawdown: float                      # 最大回撤
    win_rate: float                          # 胜率
    profit_factor: float                     # 盈利因子
    average_profit: float                    # 平均盈利
    average_loss: float                      # 平均亏损
    total_trades: int                        # 总交易次数
    average_holding_period: float            # 平均持有周期
    total_commission: float                  # 总手续费


class BacktestEngine:
    """
    回测引擎核心类
    流程：
      1. load_klines() - 加载K线数据
      2. run(signal_func) - 逐根K线重放，调用信号函数获取买卖信号
      3. get_trades() - 获取所有交易记录
      4. get_portfolio_values() - 获取账户净值曲线
      5. get_backtest_result() - 获取完整回测结果
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        
        # 账户状态
        self.cash = self.config.initial_capital              # 现金余额
        self.position_qty = 0                                # 持仓数量
        self.total_commission = 0.0                          # 总手续费
        
        # 数据存储
        self.klines: Optional[pd.DataFrame] = None          # K线数据
        self.trades: List[TradeRecord] = []                 # 交易记录
        self.portfolio_values: List[float] = []             # 账户净值曲线
        self.portfolio_dates: List[datetime] = []           # 净值对应日期
        
        # 当前进行中的交易
        self.current_trade: Optional[TradeRecord] = None    # 当前持仓交易
        self.trade_id_counter = 0                           # 交易ID计数器
    
    def load_klines(self, df: pd.DataFrame) -> None:
        """
        加载K线数据
        
        Args:
            df: DataFrame，必须包含列：
                - time/date: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - volume: 成交量
        """
        # 标准化列名（不区分大小写）
        df_copy = df.copy()
        df_copy.columns = [col.lower() for col in df_copy.columns]
        
        # 确保有必要的列
        required_cols = ['close', 'open', 'high', 'low', 'volume']
        for col in required_cols:
            if col not in df_copy.columns:
                raise ValueError(f"K线数据缺少必要列: {col}")
        
        # 排序确保时间顺序
        if 'time' in df_copy.columns:
            df_copy = df_copy.sort_values('time').reset_index(drop=True)
        elif 'date' in df_copy.columns:
            df_copy = df_copy.sort_values('date').reset_index(drop=True)
        
        self.klines = df_copy
        logger.info(f"✓ 加载K线数据: {len(self.klines)} 根")
    
    def run(self, signal_func) -> None:
        """
        逐根K线重放回测
        
        Args:
            signal_func: 信号函数，签名为 (klines_df, current_index) -> str
                         返回 'BUY' / 'SELL' / 'HOLD'
        """
        if self.klines is None:
            raise ValueError("请先通过 load_klines() 加载K线数据")
        
        # 重置状态
        self.trades = []
        self.portfolio_values = []
        self.portfolio_dates = []
        self.current_trade = None
        self.trade_id_counter = 0
        self.cash = self.config.initial_capital
        self.position_qty = 0
        self.total_commission = 0.0
        
        # 预计算所有信号（如果信号函数支持向量化）
        signals = []
        try:
            # 尝试向量化计算所有信号
            signals = signal_func(self.klines, None)  # 传入None表示计算所有信号
            if isinstance(signals, list) and len(signals) == len(self.klines):
                logger.info("使用向量化信号计算")
            else:
                signals = []
        except Exception:
            # 如果信号函数不支持向量化，使用逐根K线计算
            signals = []
        
        # 逐根K线处理
        for idx in range(len(self.klines)):
            kline = self.klines.iloc[idx]
            close_price = float(kline['close'])
            high_price = float(kline['high'])
            low_price = float(kline['low'])
            
            # 检查止损止盈
            if self.current_trade:
                if self.config.stop_loss_pct:
                    stop_loss_price = self.current_trade.entry_price * (1 - self.config.stop_loss_pct)
                    if low_price <= stop_loss_price:
                        self._execute_sell(idx, stop_loss_price)
                        continue
                if self.config.take_profit_pct:
                    take_profit_price = self.current_trade.entry_price * (1 + self.config.take_profit_pct)
                    if high_price >= take_profit_price:
                        self._execute_sell(idx, take_profit_price)
                        continue
            
            # 获取买卖信号
            if signals:
                signal = signals[idx]
            else:
                signal = signal_func(self.klines, idx)
            
            # 根据信号执行交易
            if signal == 'BUY' and self.position_qty == 0:
                self._execute_buy(idx, close_price)
            elif signal == 'SELL' and self.position_qty > 0:
                self._execute_sell(idx, close_price)
            
            # 记录账户净值（按收盘价更新）
            portfolio_value = self._calculate_portfolio_value(close_price)
            self.portfolio_values.append(portfolio_value)
            
            # 记录时间（优先使用time列，否则使用date列）
            if 'time' in self.klines.columns:
                self.portfolio_dates.append(kline['time'])
            elif 'date' in self.klines.columns:
                self.portfolio_dates.append(kline['date'])
            else:
                self.portfolio_dates.append(idx)
        
        # 回测结束时平仓
        if self.position_qty > 0 and len(self.klines) > 0:
            final_price = float(self.klines.iloc[-1]['close'])
            self._execute_sell(len(self.klines) - 1, final_price)
        
        logger.info(f"✓ 回测完成: 共 {len(self.trades)} 笔交易")
    
    def run_multiple_strategies(self, strategies: Dict[str, Callable]) -> Dict[str, BacktestResult]:
        """
        多策略回测
        
        Args:
            strategies: 策略字典，key为策略名称，value为信号函数
        
        Returns:
            各策略的回测结果
        """
        results = {}
        
        for strategy_name, signal_func in strategies.items():
            logger.info(f"开始回测策略: {strategy_name}")
            self.run(signal_func)
            results[strategy_name] = self.get_backtest_result()
        
        return results
    
    def optimize_parameters(self, signal_func_builder, param_grid: Dict[str, List[float]]) -> Dict:
        """
        参数优化
        
        Args:
            signal_func_builder: 参数化的信号函数构建器
            param_grid: 参数网格
        
        Returns:
            最优参数和对应的回测结果
        """
        import itertools
        
        # 生成参数组合
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        best_result = None
        best_params = None
        best_sharpe = -float('inf')
        
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            logger.info(f"测试参数: {param_dict}")
            
            # 构建信号函数
            signal_func = signal_func_builder(**param_dict)
            
            # 运行回测
            self.run(signal_func)
            result = self.get_backtest_result()
            
            # 评估结果
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_result = result
                best_params = param_dict
        
        logger.info(f"最优参数: {best_params}, 夏普比率: {best_sharpe:.4f}")
        return {
            'best_params': best_params,
            'best_result': best_result,
            'best_sharpe': best_sharpe
        }
    
    def _execute_buy(self, kline_idx: int, price: float) -> None:
        """执行买入操作"""
        # 计算手数
        if self.config.position_size <= 1:
            # 百分比仓位：用现金的一定比例买入
            qty = int(self.cash * self.config.position_size * self.config.max_position_size / price)
        else:
            # 固定手数：直接使用配置的手数
            qty = int(self.config.position_size)
        
        if qty <= 0 or self.cash < qty * price:
            return  # 资金不足，取消交易
        
        # 计算成本（包括手续费和滑点）
        cost_with_fee = price * (1 + self.config.commission_rate + self.config.slippage)
        total_cost = qty * cost_with_fee
        
        if total_cost > self.cash:
            # 资金不足，按可用资金调整手数
            qty = int(self.cash / cost_with_fee)
            if qty <= 0:
                return
        
        # 扣除现金
        commission = max(qty * price * self.config.commission_rate, self.config.min_commission)
        self.cash -= qty * price + commission
        self.position_qty = qty
        self.total_commission += commission
        
        # 计算止损止盈价格
        stop_loss = None
        take_profit = None
        if self.config.stop_loss_pct:
            stop_loss = price * (1 - self.config.stop_loss_pct)
        if self.config.take_profit_pct:
            take_profit = price * (1 + self.config.take_profit_pct)
        
        # 创建新交易记录
        self.trade_id_counter += 1
        kline = self.klines.iloc[kline_idx]
        entry_time = kline.get('time', kline.get('date', kline_idx))
        
        self.current_trade = TradeRecord(
            trade_id=self.trade_id_counter,
            entry_time=entry_time,
            entry_price=price,
            entry_qty=qty,
            commission=commission,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def _execute_sell(self, kline_idx: int, price: float) -> None:
        """
        执行卖出操作
        """
        if not self.current_trade or self.position_qty <= 0:
            return
        
        # 计算卖出收益（扣除手续费和滑点）
        sale_price = price * (1 - self.config.commission_rate - self.config.slippage - self.config.stamp_duty)
        revenue = self.position_qty * sale_price
        commission = max(self.position_qty * price * self.config.commission_rate, self.config.min_commission)
        
        # 更新现金和仓位
        self.cash += revenue
        self.total_commission += commission
        
        # 完成交易记录
        kline = self.klines.iloc[kline_idx]
        exit_time = kline.get('time', kline.get('date', kline_idx))
        
        self.current_trade.exit_time = exit_time
        self.current_trade.exit_price = price
        self.current_trade.exit_qty = self.position_qty
        self.current_trade.is_closed = True
        
        # 计算利润
        gross_profit = (price - self.current_trade.entry_price) * self.position_qty
        self.current_trade.profit = gross_profit - self.current_trade.commission - commission
        self.current_trade.profit_pct = (self.current_trade.profit / 
                                         (self.current_trade.entry_price * self.current_trade.entry_qty)) * 100
        
        # 计算持有周期
        if isinstance(self.current_trade.entry_time, datetime) and isinstance(exit_time, datetime):
            self.current_trade.holding_period = (exit_time - self.current_trade.entry_time).days
        
        # 添加到交易列表
        self.trades.append(self.current_trade)
        
        # 清空持仓
        self.position_qty = 0
        self.current_trade = None
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """
        计算当前账户总净值
        
        净值 = 现金 + 持仓市值
        """
        position_value = self.position_qty * current_price
        return self.cash + position_value
    
    def get_trades(self) -> List[TradeRecord]:
        """获取所有已平仓的交易记录"""
        return self.trades
    
    def get_portfolio_values(self) -> Tuple[List[float], List]:
        """
        获取账户净值曲线 (值, 日期)
        """
        return self.portfolio_values, self.portfolio_dates
    
    def get_current_status(self) -> Dict:
        """
        获取当前账户状态
        """
        return {
            'cash': self.cash,
            'position_qty': self.position_qty,
            'total_commission': self.total_commission,
            'closed_trades': len(self.trades),
            'open_trades': 1 if self.current_trade else 0
        }
    
    def get_backtest_result(self) -> BacktestResult:
        """
        获取完整回测结果
        """
        # 计算总收益率
        initial_value = self.config.initial_capital
        final_value = self.portfolio_values[-1] if self.portfolio_values else initial_value
        total_return = (final_value - initial_value) / initial_value
        
        # 计算夏普比率
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
            sharpe_ratio = (np.mean(returns) - self.config.risk_free_rate/252) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 计算最大回撤
        if self.portfolio_values:
            portfolio_array = np.array(self.portfolio_values)
            running_max = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - running_max) / running_max
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0.0
        
        # 计算交易统计
        if self.trades:
            win_trades = [trade for trade in self.trades if trade.profit > 0]
            win_rate = len(win_trades) / len(self.trades)
            
            total_profit = sum(trade.profit for trade in win_trades)
            total_loss = sum(abs(trade.profit) for trade in self.trades if trade.profit < 0)
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            average_profit = total_profit / len(win_trades) if win_trades else 0
            losing_trades = [trade for trade in self.trades if trade.profit < 0]
            average_loss = total_loss / len(losing_trades) if losing_trades else 0
            
            holding_periods = [trade.holding_period for trade in self.trades if trade.holding_period]
            average_holding_period = np.mean(holding_periods) if holding_periods else 0
        else:
            win_rate = 0.0
            profit_factor = 0.0
            average_profit = 0.0
            average_loss = 0.0
            average_holding_period = 0.0
        
        return BacktestResult(
            portfolio_values=self.portfolio_values,
            portfolio_dates=self.portfolio_dates,
            trades=self.trades,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_profit=average_profit,
            average_loss=average_loss,
            total_trades=len(self.trades),
            average_holding_period=average_holding_period,
            total_commission=self.total_commission
        )
    
    def generate_backtest_report(self) -> str:
        """
        生成回测报告
        """
        result = self.get_backtest_result()
        
        report = f"""
╔════════════════════════════════════════╗
║        回测报告 (Rox Quant)             ║
╚════════════════════════════════════════╝

【回测概览】
- 初始资金: ¥{self.config.initial_capital:.2f}
- 最终资金: ¥{result.portfolio_values[-1]:.2f}
- 总收益率: {result.total_return:.2%}
- 夏普比率: {result.sharpe_ratio:.4f}
- 最大回撤: {result.max_drawdown:.2%}
- 总交易次数: {result.total_trades}
- 总手续费: ¥{result.total_commission:.2f}

【交易统计】
- 胜率: {result.win_rate:.2%}
- 盈利因子: {result.profit_factor:.4f}
- 平均盈利: ¥{result.average_profit:.2f}
- 平均亏损: ¥{result.average_loss:.2f}
- 平均持有周期: {result.average_holding_period:.1f} 天

【最优交易】
"""
        
        # 添加最优交易
        if self.trades:
            best_trade = max(self.trades, key=lambda x: x.profit)
            report += f"- 交易ID: {best_trade.trade_id}"
            report += f"- 进场时间: {best_trade.entry_time}"
            report += f"- 进场价格: ¥{best_trade.entry_price:.2f}"
            report += f"- 离场时间: {best_trade.exit_time}"
            report += f"- 离场价格: ¥{best_trade.exit_price:.2f}"
            report += f"- 盈利: ¥{best_trade.profit:.2f} ({best_trade.profit_pct:.2}%)"
            report += f"- 持有周期: {best_trade.holding_period} 天\n"
        
        return report


def create_sample_klines(rows: int = 100) -> pd.DataFrame:
    """
    创建示例K线数据（用于测试）
    
    模拟一个月的日K线数据，价格随机游走
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(rows)]
    
    # 生成随机价格（几何布朗运动）
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, rows)  # 日收益率均值0.1%，波动2%
    prices = 100 * np.exp(np.cumsum(returns))
    
    klines = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, rows)),
        'high': prices * (1 + np.random.uniform(0.01, 0.03, rows)),
        'low': prices * (1 + np.random.uniform(-0.03, -0.01, rows)),
        'close': prices,
        'volume': np.random.randint(10000, 100000, rows)
    })
    
    return klines


def create_multi_asset_klines(assets: List[str] = ['AAPL', 'MSFT', 'GOOGL'], rows: int = 100) -> Dict[str, pd.DataFrame]:
    """
    创建多资产示例K线数据
    
    Args:
        assets: 资产列表
        rows: 数据行数
    
    Returns:
        各资产的K线数据
    """
    result = {}
    for asset in assets:
        df = create_sample_klines(rows)
        df['asset'] = asset
        result[asset] = df
    return result
