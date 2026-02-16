import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    技术指标计算类
    提供多种技术指标的计算功能
    """
    
    @staticmethod
    def ma(close: pd.Series, n: int, min_periods: int = 1) -> pd.Series:
        """
        计算移动平均线（Moving Average）
        
        Args:
            close: 收盘价序列
            n: 周期
            min_periods: 最小周期数
            
        Returns:
            移动平均线序列
        """
        return close.rolling(window=int(n), min_periods=min_periods).mean()
    
    @staticmethod
    def ema(close: pd.Series, n: int) -> pd.Series:
        """
        计算指数移动平均线（Exponential Moving Average）
        
        Args:
            close: 收盘价序列
            n: 周期
            
        Returns:
            指数移动平均线序列
        """
        return close.ewm(span=int(n), adjust=False).mean()
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        计算MACD指标（Moving Average Convergence Divergence）
        
        Args:
            close: 收盘价序列
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            包含 dif, dea, macd_hist 的 DataFrame
        """
        dif = TechnicalIndicators.ema(close, fast) - TechnicalIndicators.ema(close, slow)
        dea = dif.ewm(span=int(signal), adjust=False).mean()
        hist = dif - dea
        return pd.DataFrame({"dif": dif, "dea": dea, "macd_hist": hist})
    
    @staticmethod
    def rsi(close: pd.Series, n: int = 14, method: str = 'ema') -> pd.Series:
        """
        计算RSI指标（Relative Strength Index）
        
        Args:
            close: 收盘价序列
            n: 周期
            method: 计算方法 ('ema', 'sma', 'china')
            
        Returns:
            RSI序列（0-100）
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        if method == 'ema':
            # 国际标准：Wilder's指数移动平均
            avg_gain = gain.ewm(alpha=1 / float(n), adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / float(n), adjust=False).mean()
        elif method == 'sma':
            # 简单移动平均
            avg_gain = gain.rolling(window=int(n), min_periods=1).mean()
            avg_loss = loss.rolling(window=int(n), min_periods=1).mean()
        elif method == 'china':
            # 中国式SMA：同花顺/通达信风格
            avg_gain = gain.ewm(com=int(n) - 1, adjust=True).mean()
            avg_loss = loss.ewm(com=int(n) - 1, adjust=True).mean()
        else:
            raise ValueError(f"不支持的RSI计算方法: {method}")
        
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    @staticmethod
    def boll(close: pd.Series, n: int = 20, k: float = 2.0, min_periods: int = 1) -> pd.DataFrame:
        """
        计算布林带指标（Bollinger Bands）
        
        Args:
            close: 收盘价序列
            n: 周期
            k: 标准差倍数
            min_periods: 最小周期数
            
        Returns:
            包含 boll_mid, boll_upper, boll_lower 的 DataFrame
        """
        mid = close.rolling(window=int(n), min_periods=min_periods).mean()
        std = close.rolling(window=int(n), min_periods=min_periods).std()
        upper = mid + k * std
        lower = mid - k * std
        return pd.DataFrame({"boll_mid": mid, "boll_upper": upper, "boll_lower": lower})
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
        """
        计算ATR指标（Average True Range）
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            n: 周期
            
        Returns:
            ATR序列
        """
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(window=int(n), min_periods=int(n)).mean()
    
    @staticmethod
    def kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """
        计算KDJ指标
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            n: 周期
            m1: K线平滑周期
            m2: D线平滑周期
            
        Returns:
            包含 kdj_k, kdj_d, kdj_j 的 DataFrame
        """
        lowest_low = low.rolling(window=int(n), min_periods=int(n)).min()
        highest_high = high.rolling(window=int(n), min_periods=int(n)).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        # 处理除零与起始NaN
        rsv = rsv.replace([np.inf, -np.inf], np.nan)
        
        # 按经典公式递推（初始化 50）
        k = pd.Series(np.nan, index=close.index)
        d = pd.Series(np.nan, index=close.index)
        alpha_k = 1 / float(m1)
        alpha_d = 1 / float(m2)
        last_k = 50.0
        last_d = 50.0
        for i in range(len(close)):
            rv = rsv.iloc[i]
            if np.isnan(rv):
                k.iloc[i] = np.nan
                d.iloc[i] = np.nan
                continue
            curr_k = (1 - alpha_k) * last_k + alpha_k * rv
            curr_d = (1 - alpha_d) * last_d + alpha_d * curr_k
            k.iloc[i] = curr_k
            d.iloc[i] = curr_d
            last_k, last_d = curr_k, curr_d
        j = 3 * k - 2 * d
        return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
        """
        计算CCI指标（Commodity Channel Index）
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            n: 周期
            
        Returns:
            CCI序列
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=int(n), min_periods=int(n)).mean()
        mean_deviation = typical_price.rolling(window=int(n), min_periods=int(n)).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    @staticmethod
    def roc(close: pd.Series, n: int = 12) -> pd.Series:
        """
        计算ROC指标（Rate of Change）
        
        Args:
            close: 收盘价序列
            n: 周期
            
        Returns:
            ROC序列
        """
        return ((close / close.shift(int(n))) - 1) * 100
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
        """
        计算Williams %R指标
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            n: 周期
            
        Returns:
            Williams %R序列（-100到0）
        """
        highest_high = high.rolling(window=int(n), min_periods=int(n)).max()
        lowest_low = low.rolling(window=int(n), min_periods=int(n)).min()
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        return williams_r
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        计算OBV指标（On-Balance Volume）
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
            
        Returns:
            OBV序列
        """
        delta = close.diff()
        obv = pd.Series(0, index=close.index)
        for i in range(1, len(close)):
            if delta.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif delta.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        为DataFrame添加所有常用技术指标
        
        Args:
            df: 包含价格数据的DataFrame，需要包含 close, high, low, volume 列
            
        Returns:
            添加了技术指标列的DataFrame
        """
        # 检查必要的列
        required_cols = ['close', 'high', 'low', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame缺少必要列: {col}")
        
        # 计算移动平均线
        df['ma5'] = TechnicalIndicators.ma(df['close'], 5)
        df['ma10'] = TechnicalIndicators.ma(df['close'], 10)
        df['ma20'] = TechnicalIndicators.ma(df['close'], 20)
        df['ma60'] = TechnicalIndicators.ma(df['close'], 60)
        
        # 计算指数移动平均线
        df['ema12'] = TechnicalIndicators.ema(df['close'], 12)
        df['ema26'] = TechnicalIndicators.ema(df['close'], 26)
        
        # 计算MACD
        macd_df = TechnicalIndicators.macd(df['close'])
        df['macd_dif'] = macd_df['dif']
        df['macd_dea'] = macd_df['dea']
        df['macd_hist'] = macd_df['macd_hist']
        
        # 计算RSI
        df['rsi14'] = TechnicalIndicators.rsi(df['close'], 14)
        df['rsi6'] = TechnicalIndicators.rsi(df['close'], 6, method='china')
        df['rsi12'] = TechnicalIndicators.rsi(df['close'], 12, method='china')
        
        # 计算布林带
        boll_df = TechnicalIndicators.boll(df['close'])
        df['boll_mid'] = boll_df['boll_mid']
        df['boll_upper'] = boll_df['boll_upper']
        df['boll_lower'] = boll_df['boll_lower']
        
        # 计算ATR
        df['atr14'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14)
        
        # 计算KDJ
        kdj_df = TechnicalIndicators.kdj(df['high'], df['low'], df['close'])
        df['kdj_k'] = kdj_df['kdj_k']
        df['kdj_d'] = kdj_df['kdj_d']
        df['kdj_j'] = kdj_df['kdj_j']
        
        # 计算CCI
        df['cci14'] = TechnicalIndicators.cci(df['high'], df['low'], df['close'], 14)
        
        # 计算ROC
        df['roc12'] = TechnicalIndicators.roc(df['close'], 12)
        
        # 计算Williams %R
        df['williams_r14'] = TechnicalIndicators.williams_r(df['high'], df['low'], df['close'], 14)
        
        # 计算OBV
        df['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
        
        return df


class StrategyTemplates:
    """
    策略模板类
    提供多种交易策略的模板
    """
    
    @staticmethod
    def double_ma_strategy(data: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.Series:
        """
        双均线策略
        
        Args:
            data: 包含价格数据的DataFrame
            short_window: 短期均线周期
            long_window: 长期均线周期
            
        Returns:
            信号序列（1=买入，-1=卖出，0=持有）
        """
        # 计算均线
        data['ma_short'] = TechnicalIndicators.ma(data['close'], short_window)
        data['ma_long'] = TechnicalIndicators.ma(data['close'], long_window)
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        
        # 金叉：短期均线上穿长期均线
        signals[(data['ma_short'] > data['ma_long']) & (data['ma_short'].shift(1) <= data['ma_long'].shift(1))] = 1
        
        # 死叉：短期均线下穿长期均线
        signals[(data['ma_short'] < data['ma_long']) & (data['ma_short'].shift(1) >= data['ma_long'].shift(1))] = -1
        
        return signals
    
    @staticmethod
    def macd_strategy(data: pd.DataFrame) -> pd.Series:
        """
        MACD策略
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            信号序列（1=买入，-1=卖出，0=持有）
        """
        # 计算MACD
        macd_df = TechnicalIndicators.macd(data['close'])
        data['macd_dif'] = macd_df['dif']
        data['macd_dea'] = macd_df['dea']
        data['macd_hist'] = macd_df['macd_hist']
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        
        # 金叉：DIF上穿DEA
        signals[(data['macd_dif'] > data['macd_dea']) & (data['macd_dif'].shift(1) <= data['macd_dea'].shift(1))] = 1
        
        # 死叉：DIF下穿DEA
        signals[(data['macd_dif'] < data['macd_dea']) & (data['macd_dif'].shift(1) >= data['macd_dea'].shift(1))] = -1
        
        return signals
    
    @staticmethod
    def rsi_strategy(data: pd.DataFrame, overbought: int = 70, oversold: int = 30) -> pd.Series:
        """
        RSI策略
        
        Args:
            data: 包含价格数据的DataFrame
            overbought: 超买阈值
            oversold: 超卖阈值
            
        Returns:
            信号序列（1=买入，-1=卖出，0=持有）
        """
        # 计算RSI
        data['rsi14'] = TechnicalIndicators.rsi(data['close'], 14)
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        
        # 超卖：RSI低于超卖阈值
        signals[data['rsi14'] < oversold] = 1
        
        # 超买：RSI高于超买阈值
        signals[data['rsi14'] > overbought] = -1
        
        return signals
    
    @staticmethod
    def bollinger_bands_strategy(data: pd.DataFrame) -> pd.Series:
        """
        布林带策略
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            信号序列（1=买入，-1=卖出，0=持有）
        """
        # 计算布林带
        boll_df = TechnicalIndicators.boll(data['close'])
        data['boll_mid'] = boll_df['boll_mid']
        data['boll_upper'] = boll_df['boll_upper']
        data['boll_lower'] = boll_df['boll_lower']
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        
        # 下轨附近买入
        signals[data['close'] < data['boll_lower']] = 1
        
        # 上轨附近卖出
        signals[data['close'] > data['boll_upper']] = -1
        
        return signals
    
    @staticmethod
    def kdj_strategy(data: pd.DataFrame) -> pd.Series:
        """
        KDJ策略
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            信号序列（1=买入，-1=卖出，0=持有）
        """
        # 计算KDJ
        kdj_df = TechnicalIndicators.kdj(data['high'], data['low'], data['close'])
        data['kdj_k'] = kdj_df['kdj_k']
        data['kdj_d'] = kdj_df['kdj_d']
        data['kdj_j'] = kdj_df['kdj_j']
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        
        # K线上穿D线（金叉）
        signals[(data['kdj_k'] > data['kdj_d']) & (data['kdj_k'].shift(1) <= data['kdj_d'].shift(1))] = 1
        
        # K线下穿D线（死叉）
        signals[(data['kdj_k'] < data['kdj_d']) & (data['kdj_k'].shift(1) >= data['kdj_d'].shift(1))] = -1
        
        return signals
    
    @staticmethod
    def momentum_strategy(data: pd.DataFrame, window: int = 10) -> pd.Series:
        """
        动量策略
        
        Args:
            data: 包含价格数据的DataFrame
            window: 动量计算周期
            
        Returns:
            信号序列（1=买入，-1=卖出，0=持有）
        """
        # 计算动量
        data['momentum'] = data['close'] - data['close'].shift(window)
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        
        # 动量为正买入
        signals[data['momentum'] > 0] = 1
        
        # 动量为负卖出
        signals[data['momentum'] < 0] = -1
        
        return signals
    
    @staticmethod
    def mean_reversion_strategy(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        均值回归策略
        
        Args:
            data: 包含价格数据的DataFrame
            window: 均值计算周期
            
        Returns:
            信号序列（1=买入，-1=卖出，0=持有）
        """
        # 计算移动平均和标准差
        data['ma'] = TechnicalIndicators.ma(data['close'], window)
        data['std'] = data['close'].rolling(window=window).std()
        
        # 计算Z-score
        data['z_score'] = (data['close'] - data['ma']) / data['std']
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        
        # Z-score低于-1买入（超跌）
        signals[data['z_score'] < -1] = 1
        
        # Z-score高于1卖出（超涨）
        signals[data['z_score'] > 1] = -1
        
        return signals
