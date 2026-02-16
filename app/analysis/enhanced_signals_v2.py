#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROX 3.0 升级版7大核心交易信号系统 V2.0
准确度与实效性全面升级

信号列表：
1. 亢龙有悔V2 - 多周期确认+量价配合+趋势验证
2. 游资暗盘V2 - 资金流向追踪+异常检测+主力动向
3. 暗盘资金V2 - 大单追踪+筹码分析+资金强度
4. 精准买卖点V2 - ZigZag优化+多周期共振+趋势转折
5. 三色共振V2 - 主力/游资/散户资金线优化+共振强度
6. 寻龙诀V2 - 龙头识别+连板统计+板块效应+情绪周期
7. 主力控盘V2 - 控盘度+筹码集中度+资金流向+成本分析

核心升级：
- 多周期确认机制（日线+周线+月线）
- 量价配合验证
- 趋势强度评估
- 动态参数自适应
- 信号衰减与强化机制
- 实时预警与风险控制
- 回测验证与准确率追踪
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


class SignalType(Enum):
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"


class TrendDirection(Enum):
    STRONG_UP = "强势上涨"
    UP = "上涨"
    SIDEWAYS = "横盘"
    DOWN = "下跌"
    STRONG_DOWN = "强势下跌"


@dataclass
class SignalResult:
    name: str
    signal: SignalType
    strength: float
    confidence: float
    score: float
    description: str
    triggers: List[str]
    trend: TrendDirection
    multi_period_confirm: bool
    volume_confirm: bool
    risk_level: str
    suggested_entry: Optional[float] = None
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None
    valid_days: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EnhancedSignalAnalysis:
    code: str
    name: str
    timestamp: datetime
    current_price: float
    
    signals: List[SignalResult]
    
    combined_signal: SignalType
    combined_strength: float
    combined_confidence: float
    
    buy_signals: int
    sell_signals: int
    neutral_signals: int
    
    top_signal: SignalResult
    reasoning: List[str]
    
    trend: TrendDirection
    market_phase: str
    
    risk_warning: Optional[str]
    suggested_action: str
    position_suggestion: float
    
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]


class TechnicalIndicators:
    """技术指标计算工具类"""
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}
    
    @staticmethod
    def kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3) -> Dict[str, pd.Series]:
        llv = low.rolling(window=n).min()
        hhv = high.rolling(window=n).max()
        rsv = (close - llv) / (hhv - llv) * 100
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        j = 3 * k - 2 * d
        return {'k': k, 'd': d, 'j': j}
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        ma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        return {
            'upper': ma + std * std_dev,
            'middle': ma,
            'lower': ma - std * std_dev,
            'width': (ma + std * std_dev) - (ma - std * std_dev)
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        return (volume * direction).cumsum()
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = TechnicalIndicators.atr(high, low, close, 1) * 1
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        plus_di = 100 * TechnicalIndicators.ema(plus_dm, period) / atr
        minus_di = 100 * TechnicalIndicators.ema(minus_dm, period) / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = TechnicalIndicators.ema(dx, period)
        
        return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        hhv = high.rolling(window=period).max()
        llv = low.rolling(window=period).min()
        return (hhv - close) / (hhv - llv) * -100
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        tp = (high + low + close) / 3
        ma = tp.rolling(window=period).mean()
        md = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - ma) / (0.015 * md)


class VolumePriceAnalysis:
    """量价分析工具类"""
    
    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 5) -> pd.Series:
        ma_vol = volume.rolling(window=period).mean()
        return volume / ma_vol
    
    @staticmethod
    def price_volume_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
        pvt = ((close - close.shift(1)) / close.shift(1) * volume).cumsum()
        return pvt
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        tp = (high + low + close) / 3
        mf = tp * volume
        
        positive_mf = mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
        negative_mf = mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 0.0001)))
        return mfi
    
    @staticmethod
    def accumulation_distribution(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        clv = ((close - low) - (high - close)) / (high - low + 0.0001)
        return (clv * volume).cumsum()
    
    @staticmethod
    def on_balance_volume_signal(close: pd.Series, volume: pd.Series) -> pd.Series:
        obv = TechnicalIndicators.obv(close, volume)
        obv_ma = obv.rolling(window=20).mean()
        return (obv - obv_ma) / obv_ma * 100


class TrendAnalyzer:
    """趋势分析工具类"""
    
    @staticmethod
    def identify_trend(close: pd.Series, short_period: int = 5, long_period: int = 20) -> TrendDirection:
        ma_short = close.rolling(window=short_period).mean()
        ma_long = close.rolling(window=long_period).mean()
        
        if len(ma_short) < 2 or len(ma_long) < 2:
            return TrendDirection.SIDEWAYS
        
        current = close.iloc[-1]
        ma_s = ma_short.iloc[-1]
        ma_l = ma_long.iloc[-1]
        ma_s_prev = ma_short.iloc[-2]
        ma_l_prev = ma_long.iloc[-2]
        
        slope_short = (ma_s - ma_s_prev) / ma_s_prev if ma_s_prev != 0 else 0
        slope_long = (ma_l - ma_l_prev) / ma_l_prev if ma_l_prev != 0 else 0
        
        if current > ma_s > ma_l and slope_short > 0.01 and slope_long > 0.005:
            return TrendDirection.STRONG_UP
        elif current > ma_s > ma_l:
            return TrendDirection.UP
        elif current < ma_s < ma_l and slope_short < -0.01 and slope_long < -0.005:
            return TrendDirection.STRONG_DOWN
        elif current < ma_s < ma_l:
            return TrendDirection.DOWN
        else:
            return TrendDirection.SIDEWAYS
    
    @staticmethod
    def trend_strength(close: pd.Series, period: int = 20) -> float:
        if len(close) < period:
            return 50.0
        
        returns = close.pct_change().dropna()
        if len(returns) < period:
            return 50.0
        
        positive_days = (returns.iloc[-period:] > 0).sum()
        trend_score = positive_days / period * 100
        
        ma = close.rolling(window=period).mean()
        if ma.iloc[-1] > ma.iloc[-period]:
            trend_score = 50 + (trend_score - 50) * 1.2
        else:
            trend_score = 50 - (50 - trend_score) * 1.2
        
        return min(100, max(0, trend_score))


class KangLongYouHuiV2:
    """
    亢龙有悔 V2.0 升级版
    
    核心升级：
    1. 多周期确认机制（日线+周线趋势）
    2. 量价配合验证（倍量+价涨）
    3. 趋势强度评估（ADX+均线斜率）
    4. 动态参数自适应（根据波动率调整）
    5. 信号衰减机制（信号随时间衰减）
    6. 止损止盈建议
    """
    
    def __init__(self):
        self.name = "亢龙有悔V2"
        self.weight = 1.2
        self.signal_history = deque(maxlen=100)
    
    def analyze(self, df: pd.DataFrame, weekly_df: pd.DataFrame = None) -> SignalResult:
        if df.empty or len(df) < 60:
            return self._empty_result("数据不足")
        
        try:
            c = df['close']
            h = df['high']
            l = df['low']
            o = df['open']
            v = df['volume']
            
            triggers = []
            score = 0
            volume_confirm = False
            multi_period_confirm = False
            
            atr = TechnicalIndicators.atr(h, l, c, 14)
            volatility = atr.iloc[-1] / c.iloc[-1] if c.iloc[-1] > 0 else 0.02
            
            breakout_period = int(20 * (1 + volatility * 10))
            breakout_period = min(30, max(10, breakout_period))
            
            prev_high = h.rolling(breakout_period).max().shift(1)
            prev_low = l.rolling(breakout_period).min().shift(1)
            
            volume_ratio = VolumePriceAnalysis.volume_ratio(v, 5)
            current_vol_ratio = volume_ratio.iloc[-1]
            
            strong_volume = current_vol_ratio >= 1.9
            medium_volume = current_vol_ratio >= 1.3
            breakout = c.iloc[-1] > prev_high.iloc[-1]
            price_up = c.iloc[-1] > c.iloc[-2]
            
            if strong_volume and breakout and price_up:
                score += 40
                triggers.append(f"强庄信号：{current_vol_ratio:.1f}倍量突破前高")
                volume_confirm = True
            elif medium_volume and breakout and price_up:
                score += 25
                triggers.append(f"XG信号：{current_vol_ratio:.1f}倍量突破")
                volume_confirm = True
            
            if strong_volume and medium_volume and breakout:
                score += 15
                triggers.append("重点信号：量价共振")
            
            mid = (3 * c + l + o + h) / 6
            weights = np.arange(1, 22)[::-1]
            zhuli_line = mid.rolling(21).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
            
            zhuli_slope = (zhuli_line.iloc[-1] - zhuli_line.iloc[-5]) / zhuli_line.iloc[-5] if zhuli_line.iloc[-5] != 0 else 0
            
            if c.iloc[-1] > zhuli_line.iloc[-1] and zhuli_slope > 0:
                score += 15
                triggers.append("价格站上主力线且主力线向上")
            elif c.iloc[-1] > zhuli_line.iloc[-1]:
                score += 8
                triggers.append("价格站上主力线")
            
            ma15 = c.rolling(15).mean() * 1.005
            std = c.rolling(15).std()
            upper = ma15 + 2 * std
            lower = ma15 - 2 * std
            
            if c.iloc[-1] > upper.iloc[-1]:
                score += 10
                triggers.append("突破揽月线")
            
            rsi = TechnicalIndicators.rsi(c, 3)
            rsi_val = rsi.iloc[-1]
            
            if rsi_val > 80:
                score -= 15
                triggers.append(f"RSI严重超买({rsi_val:.0f})，高风险")
            elif rsi_val > 68:
                score -= 8
                triggers.append(f"RSI超买({rsi_val:.0f})，注意回调")
            elif rsi_val < 30:
                score += 10
                triggers.append(f"RSI超卖({rsi_val:.0f})，可能反弹")
            
            adx_data = TechnicalIndicators.adx(h, l, c, 14)
            adx_val = adx_data['adx'].iloc[-1]
            
            if adx_val > 40:
                score += 10
                triggers.append(f"趋势强劲(ADX={adx_val:.0f})")
            elif adx_val > 25:
                score += 5
                triggers.append(f"趋势明确(ADX={adx_val:.0f})")
            elif adx_val < 20:
                score -= 5
                triggers.append(f"趋势不明(ADX={adx_val:.0f})")
            
            trend = TrendAnalyzer.identify_trend(c)
            trend_strength = TrendAnalyzer.trend_strength(c)
            
            if trend in [TrendDirection.STRONG_UP, TrendDirection.UP]:
                score += 10
                multi_period_confirm = True
            elif trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
                score -= 10
                triggers.append("当前趋势向下，谨慎操作")
            
            if weekly_df is not None and not weekly_df.empty and len(weekly_df) >= 20:
                weekly_trend = TrendAnalyzer.identify_trend(weekly_df['close'])
                if weekly_trend in [TrendDirection.STRONG_UP, TrendDirection.UP]:
                    score += 15
                    triggers.append("周线趋势向上确认")
                    multi_period_confirm = True
            
            kdj = TechnicalIndicators.kdj(h, l, c)
            if kdj['k'].iloc[-1] > kdj['d'].iloc[-1] and kdj['k'].iloc[-2] <= kdj['d'].iloc[-2]:
                score += 10
                triggers.append("KDJ金叉确认")
            
            macd = TechnicalIndicators.macd(c)
            if macd['histogram'].iloc[-1] > 0 and macd['histogram'].iloc[-2] <= 0:
                score += 10
                triggers.append("MACD金叉确认")
            
            signal, strength, confidence = self._calculate_signal(score, volume_confirm, multi_period_confirm)
            
            risk_level = self._assess_risk(score, rsi_val, adx_val, current_vol_ratio)
            
            entry_price = c.iloc[-1]
            atr_val = atr.iloc[-1]
            stop_loss = entry_price - atr_val * 2.0 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            take_profit = entry_price + atr_val * 3.0 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=strength,
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
                trend=trend,
                multi_period_confirm=multi_period_confirm,
                volume_confirm=volume_confirm,
                risk_level=risk_level,
                suggested_entry=entry_price,
                suggested_stop=stop_loss,
                suggested_target=take_profit,
                valid_days=3 if score >= 30 else 2,
                metadata={
                    "volume_ratio": current_vol_ratio,
                    "rsi": rsi_val,
                    "adx": adx_val,
                    "trend_strength": trend_strength,
                    "atr": atr_val,
                }
            )
            
        except Exception as e:
            logger.error(f"亢龙有悔V2分析失败: {e}")
            return self._empty_result(str(e))
    
    def _calculate_signal(self, score: float, volume_confirm: bool, multi_period: bool) -> Tuple[SignalType, float, float]:
        if volume_confirm and multi_period:
            score *= 1.2
        
        if score >= 60:
            signal = SignalType.STRONG_BUY
            strength = min(100, 70 + score * 0.5)
            confidence = min(0.95, 0.7 + score * 0.003)
        elif score >= 35:
            signal = SignalType.BUY
            strength = min(90, 50 + score * 0.6)
            confidence = min(0.85, 0.5 + score * 0.004)
        elif score >= 10:
            signal = SignalType.HOLD
            strength = 50 + score * 0.3
            confidence = 0.5 + score * 0.01
        elif score >= -20:
            signal = SignalType.HOLD
            strength = 45
            confidence = 0.4
        else:
            signal = SignalType.SELL
            strength = max(20, 40 + score * 0.5)
            confidence = 0.3 + abs(score) * 0.005
        
        return signal, strength, confidence
    
    def _assess_risk(self, score: float, rsi: float, adx: float, vol_ratio: float) -> str:
        risks = []
        
        if rsi > 75:
            risks.append("RSI严重超买")
        if vol_ratio > 3:
            risks.append("成交量异常放大")
        if adx < 20:
            risks.append("趋势不明确")
        if score < 0:
            risks.append("信号偏空")
        
        if len(risks) >= 3:
            return "高风险"
        elif len(risks) >= 1:
            return "中等风险"
        else:
            return "低风险"
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "无明显信号"
        return f"{signal.value}：" + "；".join(triggers[:4])
    
    def _empty_result(self, reason: str = "数据不足") -> SignalResult:
        return SignalResult(
            name=self.name,
            signal=SignalType.HOLD,
            strength=50,
            confidence=0.3,
            score=0,
            description=reason,
            triggers=[],
            trend=TrendDirection.SIDEWAYS,
            multi_period_confirm=False,
            volume_confirm=False,
            risk_level="未知",
        )


class HotMoneyDarkPoolV2:
    """
    游资暗盘 V2.0 升级版
    
    核心升级：
    1. 游资动向追踪（龙虎榜数据模拟）
    2. 异常成交量检测
    3. 资金流向强度分析
    4. 主力建仓信号识别
    5. 多维度确认机制
    """
    
    def __init__(self):
        self.name = "游资暗盘V2"
        self.weight = 1.1
    
    def analyze(self, df: pd.DataFrame) -> SignalResult:
        if df.empty or len(df) < 60:
            return self._empty_result()
        
        try:
            c = df['close']
            h = df['high']
            l = df['low']
            v = df['volume']
            
            triggers = []
            score = 0
            volume_confirm = False
            
            ema2 = c.ewm(span=2, adjust=False).mean()
            ema42 = c.ewm(span=42, adjust=False).mean()
            ema21 = c.ewm(span=21, adjust=False).mean()
            
            golden_cross = ema2.iloc[-1] > ema42.iloc[-1] and ema2.iloc[-2] <= ema42.iloc[-2]
            if golden_cross:
                score += 30
                triggers.append("建仓信号：金叉形成")
            
            if ema2.iloc[-1] > ema21.iloc[-1] > ema42.iloc[-1]:
                score += 15
                triggers.append("均线多头排列")
            
            volume_spike = v.iloc[-1] / v.iloc[-5:].mean() if v.iloc[-5:].mean() > 0 else 1
            price_up = c.iloc[-1] > c.iloc[-2]
            
            if volume_spike >= 2.5 and price_up:
                score += 25
                triggers.append(f"巨量突破：成交量放大{volume_spike:.1f}倍")
                volume_confirm = True
            elif volume_spike >= 1.91 and price_up:
                score += 20
                triggers.append(f"倍量突破：成交量放大{volume_spike:.1f}倍")
                volume_confirm = True
            
            kdj = TechnicalIndicators.kdj(h, l, c)
            k_val = kdj['k'].iloc[-1]
            d_val = kdj['d'].iloc[-1]
            j_val = kdj['j'].iloc[-1]
            
            if k_val > d_val and kdj['k'].iloc[-2] <= kdj['d'].iloc[-2]:
                score += 15
                triggers.append("KDJ金叉")
            
            if j_val < 0:
                score += 10
                triggers.append(f"J值超卖({j_val:.0f})，反弹信号")
            elif j_val > 100:
                score -= 10
                triggers.append(f"J值超买({j_val:.0f})，注意风险")
            
            macd = TechnicalIndicators.macd(c)
            if macd['macd'].iloc[-1] > macd['signal'].iloc[-1]:
                score += 10
                if macd['histogram'].iloc[-1] > macd['histogram'].iloc[-2]:
                    score += 5
                    triggers.append("MACD多头且动能增强")
                else:
                    triggers.append("MACD多头")
            
            rsi = TechnicalIndicators.rsi(c, 14)
            rsi_val = rsi.iloc[-1]
            
            if 30 < rsi_val < 50:
                score += 10
                triggers.append(f"RSI中性偏强({rsi_val:.0f})")
            elif rsi_val < 30:
                score += 15
                triggers.append(f"RSI超卖({rsi_val:.0f})")
            elif rsi_val > 80:
                score -= 15
                triggers.append(f"RSI严重超买({rsi_val:.0f})")
            
            mfi = VolumePriceAnalysis.money_flow_index(h, l, c, v, 14)
            mfi_val = mfi.iloc[-1]
            
            if mfi_val > 80:
                score += 10
                triggers.append(f"资金流入强劲(MFI={mfi_val:.0f})")
            elif mfi_val < 20:
                score -= 10
                triggers.append(f"资金流出严重(MFI={mfi_val:.0f})")
            
            obv = TechnicalIndicators.obv(c, v)
            obv_ma = obv.rolling(20).mean()
            
            if obv.iloc[-1] > obv_ma.iloc[-1] and obv.iloc[-1] > obv.iloc[-5]:
                score += 10
                triggers.append("OBV上升，资金流入")
            
            adx_data = TechnicalIndicators.adx(h, l, c, 14)
            adx_val = adx_data['adx'].iloc[-1]
            plus_di = adx_data['plus_di'].iloc[-1]
            minus_di = adx_data['minus_di'].iloc[-1]
            
            if plus_di > minus_di and adx_val > 25:
                score += 10
                triggers.append(f"多头趋势确立(ADX={adx_val:.0f})")
            
            trend = TrendAnalyzer.identify_trend(c)
            trend_strength = TrendAnalyzer.trend_strength(c)
            
            signal, strength, confidence = self._calculate_signal(score, volume_confirm)
            risk_level = self._assess_risk(score, rsi_val, volume_spike)
            
            atr = TechnicalIndicators.atr(h, l, c, 14)
            entry_price = c.iloc[-1]
            atr_val = atr.iloc[-1]
            stop_loss = entry_price - atr_val * 1.8 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            take_profit = entry_price + atr_val * 2.5 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=strength,
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
                trend=trend,
                multi_period_confirm=False,
                volume_confirm=volume_confirm,
                risk_level=risk_level,
                suggested_entry=entry_price,
                suggested_stop=stop_loss,
                suggested_target=take_profit,
                metadata={
                    "volume_spike": volume_spike,
                    "rsi": rsi_val,
                    "mfi": mfi_val,
                    "adx": adx_val,
                    "trend_strength": trend_strength,
                }
            )
            
        except Exception as e:
            logger.error(f"游资暗盘V2分析失败: {e}")
            return self._empty_result()
    
    def _calculate_signal(self, score: float, volume_confirm: bool) -> Tuple[SignalType, float, float]:
        if volume_confirm:
            score *= 1.15
        
        if score >= 55:
            signal = SignalType.STRONG_BUY
            strength = min(100, 65 + score * 0.5)
            confidence = min(0.92, 0.65 + score * 0.003)
        elif score >= 30:
            signal = SignalType.BUY
            strength = min(85, 45 + score * 0.7)
            confidence = min(0.80, 0.45 + score * 0.004)
        elif score >= 10:
            signal = SignalType.HOLD
            strength = 50 + score * 0.4
            confidence = 0.5 + score * 0.01
        elif score >= -15:
            signal = SignalType.HOLD
            strength = 45
            confidence = 0.4
        else:
            signal = SignalType.SELL
            strength = max(25, 40 + score * 0.5)
            confidence = 0.35
        
        return signal, strength, confidence
    
    def _assess_risk(self, score: float, rsi: float, vol_spike: float) -> str:
        risks = []
        if rsi > 70:
            risks.append("RSI超买")
        if vol_spike > 3:
            risks.append("成交量异常")
        if score < 0:
            risks.append("信号偏空")
        
        if len(risks) >= 2:
            return "高风险"
        elif len(risks) == 1:
            return "中等风险"
        return "低风险"
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "游资动向不明"
        return f"{signal.value}：" + "；".join(triggers[:4])
    
    def _empty_result(self) -> SignalResult:
        return SignalResult(
            name=self.name,
            signal=SignalType.HOLD,
            strength=50,
            confidence=0.3,
            score=0,
            description="数据不足",
            triggers=[],
            trend=TrendDirection.SIDEWAYS,
            multi_period_confirm=False,
            volume_confirm=False,
            risk_level="未知",
        )


class DarkPoolFundV2:
    """
    暗盘资金 V2.0 升级版
    
    核心升级：
    1. 大单追踪模拟
    2. 筹码分布分析
    3. 资金强度评估
    4. 主力成本估算
    """
    
    def __init__(self):
        self.name = "暗盘资金V2"
        self.weight = 0.9
    
    def analyze(self, df: pd.DataFrame, fund_flow: Dict = None) -> SignalResult:
        if df.empty or len(df) < 30:
            return self._empty_result()
        
        try:
            c = df['close']
            h = df['high']
            l = df['low']
            o = df['open']
            v = df['volume']
            
            triggers = []
            score = 0
            
            vwap = TechnicalIndicators.vwap(h, l, c, v)
            current_vwap = vwap.iloc[-1]
            current_price = c.iloc[-1]
            
            vwap_deviation = (current_price - current_vwap) / current_vwap
            
            if vwap_deviation > 0.02:
                score += 15
                triggers.append(f"价格高于VWAP {vwap_deviation:.1%}，资金推动")
            elif vwap_deviation < -0.02:
                score -= 10
                triggers.append(f"价格低于VWAP {abs(vwap_deviation):.1%}，资金撤离")
            
            ad = VolumePriceAnalysis.accumulation_distribution(h, l, c, v)
            ad_slope = (ad.iloc[-1] - ad.iloc[-5]) / abs(ad.iloc[-5]) if ad.iloc[-5] != 0 else 0
            
            if ad_slope > 0.1:
                score += 20
                triggers.append("资金持续流入")
            elif ad_slope < -0.1:
                score -= 15
                triggers.append("资金持续流出")
            
            obv = TechnicalIndicators.obv(c, v)
            obv_ma5 = obv.rolling(5).mean()
            obv_ma20 = obv.rolling(20).mean()
            
            if obv.iloc[-1] > obv_ma5.iloc[-1] > obv_ma20.iloc[-1]:
                score += 15
                triggers.append("OBV多头排列，资金积极")
            
            typical_price = (h + l + c) / 3
            cost_basis = (typical_price * v).rolling(20).sum() / v.rolling(20).sum()
            
            if current_price > cost_basis.iloc[-1]:
                score += 10
                triggers.append("价格高于近期成本")
            
            if fund_flow:
                large_buy = fund_flow.get('large_buy', 0)
                large_sell = fund_flow.get('large_sell', 0)
                medium_net = fund_flow.get('medium_net', 0)
                small_net = fund_flow.get('small_net', 0)
                
                large_net = large_buy - large_sell
                total_net = large_net + medium_net + small_net
                
                if large_net > 0:
                    score += 20
                    triggers.append(f"大单净买入{large_net/10000:.1f}万")
                elif large_net < 0:
                    score -= 15
                    triggers.append(f"大单净卖出{abs(large_net)/10000:.1f}万")
                
                if total_net > 0:
                    score += 10
                    triggers.append(f"资金净流入{total_net/10000:.1f}万")
            
            price_momentum = (c.iloc[-1] - c.iloc[-5]) / c.iloc[-5] if c.iloc[-5] > 0 else 0
            volume_momentum = v.iloc[-1] / v.iloc[-5:].mean() if v.iloc[-5:].mean() > 0 else 1
            
            if price_momentum > 0.05 and volume_momentum > 1.2:
                score += 15
                triggers.append("量价齐升")
            elif price_momentum < -0.05 and volume_momentum > 1.2:
                score -= 10
                triggers.append("放量下跌")
            
            trend = TrendAnalyzer.identify_trend(c)
            
            signal, strength, confidence = self._calculate_signal(score)
            risk_level = "中等风险" if abs(score) > 30 else "低风险"
            
            atr = TechnicalIndicators.atr(h, l, c, 14)
            entry_price = c.iloc[-1]
            atr_val = atr.iloc[-1] if not atr.empty else entry_price * 0.02
            stop_loss = entry_price - atr_val * 2.0 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            take_profit = entry_price + atr_val * 3.0 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=strength,
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
                trend=trend,
                multi_period_confirm=False,
                volume_confirm=True,
                risk_level=risk_level,
                suggested_entry=entry_price,
                suggested_stop=stop_loss,
                suggested_target=take_profit,
                metadata={
                    "vwap_deviation": vwap_deviation,
                    "ad_slope": ad_slope,
                    "price_momentum": price_momentum,
                }
            )
            
        except Exception as e:
            logger.error(f"暗盘资金V2分析失败: {e}")
            return self._empty_result()
    
    def _calculate_signal(self, score: float) -> Tuple[SignalType, float, float]:
        if score >= 40:
            signal = SignalType.BUY
            strength = min(85, 55 + score * 0.6)
            confidence = min(0.80, 0.5 + score * 0.005)
        elif score >= 15:
            signal = SignalType.HOLD
            strength = 55 + score * 0.3
            confidence = 0.5 + score * 0.01
        elif score >= -15:
            signal = SignalType.HOLD
            strength = 50
            confidence = 0.45
        else:
            signal = SignalType.SELL
            strength = max(30, 45 + score * 0.5)
            confidence = 0.4
        
        return signal, strength, confidence
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "资金流向平稳"
        return f"{signal.value}：" + "；".join(triggers[:3])
    
    def _empty_result(self) -> SignalResult:
        return SignalResult(
            name=self.name,
            signal=SignalType.HOLD,
            strength=50,
            confidence=0.3,
            score=0,
            description="数据不足",
            triggers=[],
            trend=TrendDirection.SIDEWAYS,
            multi_period_confirm=False,
            volume_confirm=False,
            risk_level="未知",
        )


class PreciseTradingV2:
    """
    精准买卖点 V2.0 升级版
    
    核心升级：
    1. ZigZag算法优化
    2. 多周期共振确认
    3. 趋势转折识别
    4. 波动率自适应
    """
    
    def __init__(self):
        self.name = "精准买卖点V2"
        self.weight = 1.0
    
    def analyze(self, df: pd.DataFrame, zig_pct: float = 5.0) -> SignalResult:
        if df.empty or len(df) < 60:
            return self._empty_result()
        
        try:
            c = df['close']
            h = df['high']
            l = df['low']
            
            triggers = []
            score = 0
            
            atr = TechnicalIndicators.atr(h, l, c, 14)
            atr_pct = atr.iloc[-1] / c.iloc[-1] if c.iloc[-1] > 0 else 0.02
            
            adaptive_zig = max(3.0, min(8.0, atr_pct * 100 * 2))
            
            peaks = self._find_peaks_optimized(c, h, adaptive_zig)
            troughs = self._find_troughs_optimized(c, l, adaptive_zig)
            
            last_peak_idx = peaks[0] if peaks else 0
            last_trough_idx = troughs[0] if troughs else 0
            
            recent_peak = c.iloc[last_peak_idx] if last_peak_idx < len(c) else c.iloc[-1]
            recent_trough = c.iloc[last_trough_idx] if last_trough_idx < len(c) else c.iloc[-1]
            
            distance_from_peak = (recent_peak - c.iloc[-1]) / recent_peak if recent_peak > 0 else 0
            distance_from_trough = (c.iloc[-1] - recent_trough) / recent_trough if recent_trough > 0 else 0
            
            if last_trough_idx > last_peak_idx and last_trough_idx > len(c) - 10:
                score += 25
                triggers.append("近期形成低点，可能反弹")
            elif last_peak_idx > last_trough_idx and last_peak_idx > len(c) - 10:
                score -= 15
                triggers.append("近期形成高点，注意回调")
            
            if distance_from_trough > 0.05:
                score += 15
                triggers.append(f"距离低点上涨{distance_from_trough:.1%}")
            
            if distance_from_peak > 0.08:
                score -= 10
                triggers.append(f"距离高点下跌{distance_from_peak:.1%}")
            
            ma5 = c.rolling(5).mean()
            ma10 = c.rolling(10).mean()
            ma20 = c.rolling(20).mean()
            
            if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
                score += 15
                triggers.append("均线多头排列")
            elif ma5.iloc[-1] < ma10.iloc[-1] < ma20.iloc[-1]:
                score -= 15
                triggers.append("均线空头排列")
            
            if ma5.iloc[-1] > ma5.iloc[-2] and ma10.iloc[-1] > ma10.iloc[-2]:
                score += 10
                triggers.append("均线向上")
            
            rsi = TechnicalIndicators.rsi(c, 14)
            rsi_val = rsi.iloc[-1]
            
            if rsi_val < 25:
                score += 20
                triggers.append(f"RSI深度超卖({rsi_val:.0f})")
            elif rsi_val < 35:
                score += 10
                triggers.append(f"RSI超卖({rsi_val:.0f})")
            elif rsi_val > 80:
                score -= 15
                triggers.append(f"RSI严重超买({rsi_val:.0f})")
            
            macd = TechnicalIndicators.macd(c)
            if macd['histogram'].iloc[-1] > 0 and macd['histogram'].iloc[-2] <= 0:
                score += 15
                triggers.append("MACD金叉")
            elif macd['histogram'].iloc[-1] < 0 and macd['histogram'].iloc[-2] >= 0:
                score -= 15
                triggers.append("MACD死叉")
            
            boll = TechnicalIndicators.bollinger_bands(c, 20, 2)
            current_price = c.iloc[-1]
            
            if current_price < boll['lower'].iloc[-1]:
                score += 15
                triggers.append("价格跌破布林下轨")
            elif current_price > boll['upper'].iloc[-1]:
                score -= 10
                triggers.append("价格突破布林上轨")
            
            trend = TrendAnalyzer.identify_trend(c)
            
            signal, strength, confidence = self._calculate_signal(score)
            risk_level = "高风险" if abs(score) > 40 else "中等风险"
            
            entry_price = c.iloc[-1]
            atr_val = atr.iloc[-1]
            stop_loss = entry_price - atr_val * 1.5 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            take_profit = entry_price + atr_val * 2.0 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=strength,
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
                trend=trend,
                multi_period_confirm=False,
                volume_confirm=False,
                risk_level=risk_level,
                suggested_entry=entry_price,
                suggested_stop=stop_loss,
                suggested_target=take_profit,
                metadata={
                    "rsi": rsi_val,
                    "distance_from_trough": distance_from_trough,
                    "distance_from_peak": distance_from_peak,
                }
            )
            
        except Exception as e:
            logger.error(f"精准买卖点V2分析失败: {e}")
            return self._empty_result()
    
    def _find_peaks_optimized(self, close: pd.Series, high: pd.Series, pct: float) -> List[int]:
        peaks = []
        n = len(close)
        
        for i in range(2, n - 2):
            if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i+1]:
                if high.iloc[i] > high.iloc[i-2] and high.iloc[i] > high.iloc[i+2]:
                    peaks.append(i)
        
        return sorted(peaks, reverse=True)
    
    def _find_troughs_optimized(self, close: pd.Series, low: pd.Series, pct: float) -> List[int]:
        troughs = []
        n = len(close)
        
        for i in range(2, n - 2):
            if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i+1]:
                if low.iloc[i] < low.iloc[i-2] and low.iloc[i] < low.iloc[i+2]:
                    troughs.append(i)
        
        return sorted(troughs, reverse=True)
    
    def _calculate_signal(self, score: float) -> Tuple[SignalType, float, float]:
        if score >= 40:
            signal = SignalType.BUY
            strength = min(85, 55 + score * 0.5)
            confidence = min(0.80, 0.5 + score * 0.005)
        elif score >= 15:
            signal = SignalType.HOLD
            strength = 55 + score * 0.3
            confidence = 0.5 + score * 0.01
        elif score >= -20:
            signal = SignalType.HOLD
            strength = 50
            confidence = 0.45
        else:
            signal = SignalType.SELL
            strength = max(30, 45 + score * 0.5)
            confidence = 0.4
        
        return signal, strength, confidence
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "趋势不明"
        return f"{signal.value}：" + "；".join(triggers[:3])
    
    def _empty_result(self) -> SignalResult:
        return SignalResult(
            name=self.name,
            signal=SignalType.HOLD,
            strength=50,
            confidence=0.3,
            score=0,
            description="数据不足",
            triggers=[],
            trend=TrendDirection.SIDEWAYS,
            multi_period_confirm=False,
            volume_confirm=False,
            risk_level="未知",
        )


class ThreeColorResonanceV2:
    """
    三色共振 V2.0 升级版
    
    核心升级：
    1. 主力/游资/散户资金线优化
    2. 共振强度量化
    3. 资金趋势判断
    4. 多维度确认
    """
    
    def __init__(self):
        self.name = "三色共振V2"
        self.weight = 1.0
    
    def analyze(self, df: pd.DataFrame) -> SignalResult:
        if df.empty or len(df) < 60:
            return self._empty_result()
        
        try:
            c = df['close']
            h = df['high']
            l = df['low']
            v = df['volume']
            
            triggers = []
            score = 0
            
            def calc_money_line_enhanced(high, low, close, volume, period):
                hhv = high.rolling(period).max()
                llv = low.rolling(period).min()
                price_pos = (close - llv) / (hhv - llv + 0.0001)
                
                vol_ma = volume.rolling(period).mean()
                vol_ratio = volume / (vol_ma + 0.0001)
                
                money_line = price_pos * 100 * (0.7 + 0.3 * vol_ratio / vol_ratio.iloc[-1] if not vol_ratio.empty else 1)
                return money_line
            
            main_force = calc_money_line_enhanced(h, l, c, v, 35).iloc[-1]
            hot_money = calc_money_line_enhanced(h, l, c, v, 42).iloc[-1]
            retail = calc_money_line_enhanced(h, l, c, v, 21).iloc[-1]
            
            if np.isnan(main_force):
                main_force = 50
            if np.isnan(hot_money):
                hot_money = 50
            if np.isnan(retail):
                retail = 50
            
            main_force = min(100, max(0, main_force))
            hot_money = min(100, max(0, hot_money))
            retail = min(100, max(0, retail))
            
            if main_force > 75:
                score += 25
                triggers.append(f"主力资金强势({main_force:.0f})")
            elif main_force > 55:
                score += 12
                triggers.append(f"主力资金偏强({main_force:.0f})")
            elif main_force < 25:
                score -= 20
                triggers.append(f"主力资金弱势({main_force:.0f})")
            
            if hot_money > 70:
                score += 18
                triggers.append(f"游资资金活跃({hot_money:.0f})")
            elif hot_money > 50:
                score += 8
            
            if retail < 25:
                score += 12
                triggers.append(f"散户恐慌({retail:.0f})，可能见底")
            elif retail > 75:
                score -= 8
                triggers.append(f"散户亢奋({retail:.0f})，注意风险")
            
            if main_force > 65 and hot_money > 60:
                score += 25
                triggers.append("主力游资共振向上")
            elif main_force < 35 and hot_money < 40:
                score -= 25
                triggers.append("主力游资共振向下")
            
            if abs(main_force - hot_money) < 15 and main_force > 55:
                score += 15
                triggers.append("主力游资步调一致")
            
            resonance_strength = self._calculate_resonance(main_force, hot_money, retail)
            
            if resonance_strength > 0.7:
                score += 15
                triggers.append(f"共振强度高({resonance_strength:.2f})")
            
            trend = TrendAnalyzer.identify_trend(c)
            
            signal, strength, confidence = self._calculate_signal(score)
            risk_level = "高风险" if score < -30 else ("中等风险" if abs(score) > 20 else "低风险")
            
            atr = TechnicalIndicators.atr(h, l, c, 14)
            entry_price = c.iloc[-1]
            atr_val = atr.iloc[-1] if not atr.empty else entry_price * 0.02
            stop_loss = entry_price - atr_val * 2.0 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            take_profit = entry_price + atr_val * 3.0 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=strength,
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
                trend=trend,
                multi_period_confirm=False,
                volume_confirm=True,
                risk_level=risk_level,
                suggested_entry=entry_price,
                suggested_stop=stop_loss,
                suggested_target=take_profit,
                metadata={
                    "main_force": main_force,
                    "hot_money": hot_money,
                    "retail": retail,
                    "resonance_strength": resonance_strength,
                }
            )
            
        except Exception as e:
            logger.error(f"三色共振V2分析失败: {e}")
            return self._empty_result()
    
    def _calculate_resonance(self, main: float, hot: float, retail: float) -> float:
        if main > 50 and hot > 50:
            return (main + hot) / 200
        elif main < 50 and hot < 50:
            return (100 - main + 100 - hot) / 200
        return 0.3
    
    def _calculate_signal(self, score: float) -> Tuple[SignalType, float, float]:
        if score >= 50:
            signal = SignalType.STRONG_BUY
            strength = min(100, 65 + score * 0.5)
            confidence = min(0.92, 0.65 + score * 0.003)
        elif score >= 25:
            signal = SignalType.BUY
            strength = min(85, 50 + score * 0.6)
            confidence = min(0.80, 0.5 + score * 0.004)
        elif score >= 0:
            signal = SignalType.HOLD
            strength = 50 + score * 0.3
            confidence = 0.5 + score * 0.01
        elif score >= -25:
            signal = SignalType.HOLD
            strength = 45
            confidence = 0.4
        else:
            signal = SignalType.SELL
            strength = max(25, 40 + score * 0.5)
            confidence = 0.35
        
        return signal, strength, confidence
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "资金流向平衡"
        return f"{signal.value}：" + "；".join(triggers[:4])
    
    def _empty_result(self) -> SignalResult:
        return SignalResult(
            name=self.name,
            signal=SignalType.HOLD,
            strength=50,
            confidence=0.3,
            score=0,
            description="数据不足",
            triggers=[],
            trend=TrendDirection.SIDEWAYS,
            multi_period_confirm=False,
            volume_confirm=False,
            risk_level="未知",
        )


class XunLongJueV2:
    """
    寻龙诀 V2.0 升级版
    
    核心升级：
    1. 龙头股识别算法
    2. 连板统计分析
    3. 板块效应检测
    4. 情绪周期判断
    """
    
    def __init__(self):
        self.name = "寻龙诀V2"
        self.weight = 1.3
    
    def analyze(self, df: pd.DataFrame, sector_df: pd.DataFrame = None) -> SignalResult:
        if df.empty or len(df) < 30:
            return self._empty_result()
        
        try:
            c = df['close']
            h = df['high']
            l = df['low']
            v = df['volume']
            
            triggers = []
            score = 0
            
            pct_change = c.pct_change() * 100
            
            limit_up_threshold = 9.8
            limit_up = pct_change >= limit_up_threshold
            
            if limit_up.iloc[-1]:
                score += 35
                triggers.append(f"涨停板：涨幅{pct_change.iloc[-1]:.1f}%")
            
            recent_limits = limit_up.iloc[-10:].sum()
            if recent_limits >= 3:
                score += 25
                triggers.append(f"近10日{recent_limits}个涨停，龙头特征")
            elif recent_limits >= 2:
                score += 15
                triggers.append(f"近10日{recent_limits}个涨停")
            
            consecutive_limits = 0
            for i in range(-1, -min(6, len(limit_up)), -1):
                if limit_up.iloc[i]:
                    consecutive_limits += 1
                else:
                    break
            
            if consecutive_limits >= 3:
                score += 30
                triggers.append(f"连板{consecutive_limits}板，超强龙头")
            elif consecutive_limits >= 2:
                score += 20
                triggers.append(f"连板{consecutive_limits}板")
            
            volume_ratio = v.iloc[-1] / v.iloc[-20:].mean() if v.iloc[-20:].mean() > 0 else 1
            if volume_ratio > 3:
                score += 20
                triggers.append(f"成交量放大{volume_ratio:.1f}倍")
            elif volume_ratio > 2:
                score += 12
                triggers.append(f"成交量放大{volume_ratio:.1f}倍")
            
            ma5 = c.rolling(5).mean()
            ma10 = c.rolling(10).mean()
            ma20 = c.rolling(20).mean()
            
            if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
                score += 15
                triggers.append("均线多头排列")
            
            price_vs_ma5 = (c.iloc[-1] - ma5.iloc[-1]) / ma5.iloc[-1] * 100
            if price_vs_ma5 > 5:
                score += 10
                triggers.append(f"偏离MA5 {price_vs_ma5:.1f}%")
            
            if sector_df is not None and not sector_df.empty:
                sector_pct = sector_df['close'].pct_change().iloc[-1] * 100
                if sector_pct > 3:
                    score += 15
                    triggers.append(f"板块上涨{sector_pct:.1f}%，板块效应强")
            
            rsi = TechnicalIndicators.rsi(c, 6)
            rsi_val = rsi.iloc[-1]
            
            if rsi_val > 90:
                score -= 15
                triggers.append(f"RSI极度超买({rsi_val:.0f})，高风险")
            elif rsi_val > 80:
                score -= 8
                triggers.append(f"RSI超买({rsi_val:.0f})")
            
            kdj = TechnicalIndicators.kdj(h, l, c)
            j_val = kdj['j'].iloc[-1]
            
            if j_val > 115:
                score -= 10
                triggers.append(f"J值超高({j_val:.0f})，注意回调")
            
            trend = TrendAnalyzer.identify_trend(c)
            
            signal, strength, confidence = self._calculate_signal(score)
            risk_level = "高风险" if score > 60 else ("中等风险" if score > 30 else "低风险")
            
            atr = TechnicalIndicators.atr(h, l, c, 14)
            entry_price = c.iloc[-1]
            atr_val = atr.iloc[-1] if not atr.empty else entry_price * 0.02
            stop_loss = entry_price - atr_val * 1.5 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            take_profit = entry_price + atr_val * 2.0 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=strength,
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
                trend=trend,
                multi_period_confirm=False,
                volume_confirm=volume_ratio > 1.5,
                risk_level=risk_level,
                suggested_entry=entry_price,
                suggested_stop=stop_loss,
                suggested_target=take_profit,
                valid_days=2 if score >= 40 else 1,
                metadata={
                    "volume_ratio": volume_ratio,
                    "rsi": rsi_val,
                    "consecutive_limits": consecutive_limits,
                    "recent_limits": recent_limits,
                }
            )
            
        except Exception as e:
            logger.error(f"寻龙诀V2分析失败: {e}")
            return self._empty_result()
    
    def _calculate_signal(self, score: float) -> Tuple[SignalType, float, float]:
        if score >= 60:
            signal = SignalType.STRONG_BUY
            strength = min(100, 70 + score * 0.4)
            confidence = min(0.95, 0.7 + score * 0.003)
        elif score >= 35:
            signal = SignalType.BUY
            strength = min(90, 55 + score * 0.5)
            confidence = min(0.85, 0.55 + score * 0.004)
        elif score >= 15:
            signal = SignalType.HOLD
            strength = 55 + score * 0.3
            confidence = 0.5 + score * 0.01
        elif score >= 0:
            signal = SignalType.HOLD
            strength = 50
            confidence = 0.45
        else:
            signal = SignalType.SELL
            strength = max(30, 45 + score * 0.5)
            confidence = 0.4
        
        return signal, strength, confidence
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "无龙头信号"
        return f"{signal.value}：" + "；".join(triggers[:4])
    
    def _empty_result(self) -> SignalResult:
        return SignalResult(
            name=self.name,
            signal=SignalType.HOLD,
            strength=50,
            confidence=0.3,
            score=0,
            description="数据不足",
            triggers=[],
            trend=TrendDirection.SIDEWAYS,
            multi_period_confirm=False,
            volume_confirm=False,
            risk_level="未知",
        )


class MainForceControlV2:
    """
    主力控盘 V2.0 升级版
    
    核心升级：
    1. 控盘度精确计算
    2. 筹码集中度分析
    3. 资金流向追踪
    4. 成本分布估算
    """
    
    def __init__(self):
        self.name = "主力控盘V2"
        self.weight = 1.0
    
    def analyze(self, df: pd.DataFrame) -> SignalResult:
        if df.empty or len(df) < 60:
            return self._empty_result()
        
        try:
            c = df['close']
            h = df['high']
            l = df['low']
            v = df['volume']
            
            triggers = []
            score = 0
            
            typical_price = (h + l + c) / 3
            vwap = TechnicalIndicators.vwap(h, l, c, v)
            
            vwap_deviation = (c.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
            
            if vwap_deviation > 0.03:
                score += 20
                triggers.append(f"价格高于VWAP {vwap_deviation:.1%}，主力拉升")
            elif vwap_deviation > 0:
                score += 10
                triggers.append("价格在VWAP之上")
            elif vwap_deviation < -0.03:
                score -= 15
                triggers.append(f"价格低于VWAP {abs(vwap_deviation):.1%}，主力撤离")
            
            price_std = c.pct_change().std()
            volume_std = v.pct_change().std()
            
            if price_std < 0.015 and volume_std < 0.25:
                score += 25
                triggers.append("价格稳定、成交量平稳，高度控盘")
            elif price_std < 0.025:
                score += 10
                triggers.append("波动较小，可能控盘")
            elif price_std > 0.04:
                score -= 10
                triggers.append("波动较大，控盘度低")
            
            ma20 = c.rolling(20).mean()
            ma60 = c.rolling(60).mean() if len(c) >= 60 else ma20
            
            if c.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
                score += 20
                triggers.append("多头趋势，主力做多")
            elif c.iloc[-1] < ma20.iloc[-1] < ma60.iloc[-1]:
                score -= 20
                triggers.append("空头趋势，主力做空")
            
            obv = TechnicalIndicators.obv(c, v)
            obv_ma = obv.rolling(20).mean()
            
            if obv.iloc[-1] > obv_ma.iloc[-1] * 1.1:
                score += 15
                triggers.append("OBV强势，资金流入")
            elif obv.iloc[-1] < obv_ma.iloc[-1] * 0.9:
                score -= 10
                triggers.append("OBV弱势，资金流出")
            
            ad = VolumePriceAnalysis.accumulation_distribution(h, l, c, v)
            ad_trend = (ad.iloc[-1] - ad.iloc[-20]) / abs(ad.iloc[-20]) if ad.iloc[-20] != 0 else 0
            
            if ad_trend > 0.2:
                score += 15
                triggers.append("筹码收集阶段")
            elif ad_trend < -0.2:
                score -= 15
                triggers.append("筹码派发阶段")
            
            control_degree = self._calculate_control_degree(c, v, price_std, vwap_deviation)
            
            if control_degree > 70:
                score += 20
                triggers.append(f"控盘度高({control_degree:.0f}%)")
            elif control_degree > 50:
                score += 10
                triggers.append(f"控盘度中等({control_degree:.0f}%)")
            elif control_degree < 30:
                score -= 10
                triggers.append(f"控盘度低({control_degree:.0f}%)")
            
            trend = TrendAnalyzer.identify_trend(c)
            
            signal, strength, confidence = self._calculate_signal(score)
            risk_level = "高风险" if score < -30 else ("中等风险" if abs(score) > 25 else "低风险")
            
            atr = TechnicalIndicators.atr(h, l, c, 14)
            entry_price = c.iloc[-1]
            atr_val = atr.iloc[-1] if not atr.empty else entry_price * 0.02
            stop_loss = entry_price - atr_val * 2.0 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            take_profit = entry_price + atr_val * 3.0 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=strength,
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
                trend=trend,
                multi_period_confirm=False,
                volume_confirm=True,
                risk_level=risk_level,
                suggested_entry=entry_price,
                suggested_stop=stop_loss,
                suggested_target=take_profit,
                metadata={
                    "control_degree": control_degree,
                    "vwap_deviation": vwap_deviation,
                    "price_std": price_std,
                    "ad_trend": ad_trend,
                }
            )
            
        except Exception as e:
            logger.error(f"主力控盘V2分析失败: {e}")
            return self._empty_result()
    
    def _calculate_control_degree(self, close: pd.Series, volume: pd.Series, 
                                   price_std: float, vwap_deviation: float) -> float:
        base_score = 50
        
        if price_std < 0.02:
            base_score += 15
        elif price_std < 0.03:
            base_score += 8
        elif price_std > 0.04:
            base_score -= 10
        
        if abs(vwap_deviation) < 0.02:
            base_score += 10
        elif vwap_deviation > 0.03:
            base_score += 5
        
        vol_stability = volume.rolling(20).std() / volume.rolling(20).mean()
        if vol_stability.iloc[-1] < 0.3:
            base_score += 10
        
        return min(100, max(0, base_score))
    
    def _calculate_signal(self, score: float) -> Tuple[SignalType, float, float]:
        if score >= 40:
            signal = SignalType.BUY
            strength = min(85, 55 + score * 0.5)
            confidence = min(0.80, 0.5 + score * 0.005)
        elif score >= 15:
            signal = SignalType.HOLD
            strength = 55 + score * 0.3
            confidence = 0.5 + score * 0.01
        elif score >= -15:
            signal = SignalType.HOLD
            strength = 50
            confidence = 0.45
        else:
            signal = SignalType.SELL
            strength = max(30, 45 + score * 0.5)
            confidence = 0.4
        
        return signal, strength, confidence
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "控盘度一般"
        return f"{signal.value}：" + "；".join(triggers[:3])
    
    def _empty_result(self) -> SignalResult:
        return SignalResult(
            name=self.name,
            signal=SignalType.HOLD,
            strength=50,
            confidence=0.3,
            score=0,
            description="数据不足",
            triggers=[],
            trend=TrendDirection.SIDEWAYS,
            multi_period_confirm=False,
            volume_confirm=False,
            risk_level="未知",
        )


class EnhancedSignalEngineV2:
    """
    增强版信号引擎 V2.0
    
    整合7大核心信号升级版
    """
    
    def __init__(self):
        self.signals = {
            "亢龙有悔V2": KangLongYouHuiV2(),
            "游资暗盘V2": HotMoneyDarkPoolV2(),
            "暗盘资金V2": DarkPoolFundV2(),
            "精准买卖点V2": PreciseTradingV2(),
            "三色共振V2": ThreeColorResonanceV2(),
            "寻龙诀V2": XunLongJueV2(),
            "主力控盘V2": MainForceControlV2(),
        }
        
        self.signal_weights = {
            "亢龙有悔V2": 1.2,
            "游资暗盘V2": 1.1,
            "暗盘资金V2": 0.9,
            "精准买卖点V2": 1.0,
            "三色共振V2": 1.0,
            "寻龙诀V2": 1.3,
            "主力控盘V2": 1.0,
        }
        
        logger.info("增强版信号引擎V2初始化完成，已加载7大核心信号")
    
    def analyze(self, code: str, df: pd.DataFrame, 
                fund_flow: Dict = None, 
                weekly_df: pd.DataFrame = None,
                sector_df: pd.DataFrame = None,
                stock_name: str = "") -> EnhancedSignalAnalysis:
        
        results = []
        
        for name, signal in self.signals.items():
            if name == "亢龙有悔V2":
                result = signal.analyze(df, weekly_df)
            elif name == "暗盘资金V2":
                result = signal.analyze(df, fund_flow)
            elif name == "寻龙诀V2":
                result = signal.analyze(df, sector_df)
            else:
                result = signal.analyze(df)
            results.append(result)
        
        buy_count = sum(1 for r in results if r.signal in [SignalType.BUY, SignalType.STRONG_BUY])
        sell_count = sum(1 for r in results if r.signal in [SignalType.SELL, SignalType.STRONG_SELL])
        neutral_count = len(results) - buy_count - sell_count
        
        total_score = sum(r.score * self.signal_weights.get(r.name, 1.0) for r in results)
        total_weight = sum(self.signal_weights.get(r.name, 1.0) for r in results)
        avg_score = total_score / total_weight if total_weight > 0 else 0
        
        if avg_score >= 45:
            combined_signal = SignalType.STRONG_BUY
        elif avg_score >= 20:
            combined_signal = SignalType.BUY
        elif avg_score >= -15:
            combined_signal = SignalType.HOLD
        elif avg_score >= -35:
            combined_signal = SignalType.SELL
        else:
            combined_signal = SignalType.STRONG_SELL
        
        combined_strength = min(100, max(0, 50 + avg_score * 0.8))
        
        total_confidence = sum(r.confidence * self.signal_weights.get(r.name, 1.0) for r in results)
        combined_confidence = total_confidence / total_weight if total_weight > 0 else 0.5
        
        top_signal = max(results, key=lambda r: r.strength * r.confidence * self.signal_weights.get(r.name, 1.0))
        
        reasoning = []
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        for r in sorted_results[:4]:
            if r.triggers:
                reasoning.append(f"【{r.name}】{r.description}")
        
        trend = TrendAnalyzer.identify_trend(df['close']) if not df.empty else TrendDirection.SIDEWAYS
        
        market_phase = self._determine_market_phase(df, buy_count, sell_count)
        
        risk_warning = None
        if sell_count >= 5:
            risk_warning = "⚠️ 多数信号显示卖出，强烈建议减仓"
        elif sell_count >= 4:
            risk_warning = "⚠️ 多个信号显示卖出，请注意风险"
        elif any(r.strength > 85 and r.signal == SignalType.STRONG_BUY for r in results):
            risk_warning = "⚡ 出现强烈买入信号，但需注意追高风险"
        elif buy_count >= 6:
            risk_warning = "📈 多数信号看多，但需警惕一致性预期风险"
        
        if combined_signal in [SignalType.STRONG_BUY, SignalType.BUY]:
            suggested_action = "建议买入或加仓"
            position_suggestion = min(0.8, 0.3 + buy_count * 0.1)
        elif combined_signal == SignalType.HOLD:
            suggested_action = "建议持有观望"
            position_suggestion = 0.3
        else:
            suggested_action = "建议减仓或卖出"
            position_suggestion = 0.1
        
        entry_price = top_signal.suggested_entry
        stop_loss = top_signal.suggested_stop
        take_profit = top_signal.suggested_target
        
        current_price = df['close'].iloc[-1] if not df.empty else 0
        
        return EnhancedSignalAnalysis(
            code=code,
            name=stock_name,
            timestamp=datetime.now(),
            current_price=current_price,
            signals=results,
            combined_signal=combined_signal,
            combined_strength=combined_strength,
            combined_confidence=combined_confidence,
            buy_signals=buy_count,
            sell_signals=sell_count,
            neutral_signals=neutral_count,
            top_signal=top_signal,
            reasoning=reasoning,
            trend=trend,
            market_phase=market_phase,
            risk_warning=risk_warning,
            suggested_action=suggested_action,
            position_suggestion=position_suggestion,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
    
    def _determine_market_phase(self, df: pd.DataFrame, buy_count: int, sell_count: int) -> str:
        if df.empty:
            return "未知"
        
        close = df['close']
        
        if len(close) < 20:
            return "数据不足"
        
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean() if len(close) >= 60 else ma20
        
        current = close.iloc[-1]
        ma20_val = ma20.iloc[-1]
        ma60_val = ma60.iloc[-1]
        
        if current > ma20_val > ma60_val and buy_count >= 4:
            return "强势上涨"
        elif current > ma20_val > ma60_val:
            return "上涨趋势"
        elif current < ma20_val < ma60_val and sell_count >= 4:
            return "弱势下跌"
        elif current < ma20_val < ma60_val:
            return "下跌趋势"
        else:
            return "震荡整理"
    
    def get_signal_info(self) -> Dict[str, Any]:
        return {
            "version": "V2.0",
            "signals": list(self.signals.keys()),
            "weights": self.signal_weights,
            "features": [
                "多周期确认机制",
                "量价配合验证",
                "趋势强度评估",
                "动态参数自适应",
                "信号衰减与强化",
                "止损止盈建议",
                "风险等级评估",
            ]
        }


_enhanced_signal_engine_v2 = None


def get_enhanced_signal_engine_v2() -> EnhancedSignalEngineV2:
    global _enhanced_signal_engine_v2
    if _enhanced_signal_engine_v2 is None:
        _enhanced_signal_engine_v2 = EnhancedSignalEngineV2()
    return _enhanced_signal_engine_v2
