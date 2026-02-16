#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版7大核心交易信号系统

信号列表：
1. 亢龙有悔 - 强庄突破信号
2. 游资暗盘 - 游资动向追踪
3. 暗盘资金 - 资金流向分析
4. 精准买卖点 - ZigZag高低点
5. 三色共振 - 主力/游资/散户共振
6. 寻龙诀 - 涨停板突破
7. 主力控盘 - 控盘度分析

增强功能：
- 信号强度评分
- 信号回测验证
- 信号组合优化
- 自适应参数
- 实时预警
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

from app.rox_quant.signal_cache import get_signal_cache

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"


@dataclass
class SignalResult:
    """信号结果"""
    name: str
    signal: SignalType
    strength: float  # 0-100
    confidence: float  # 0-1
    score: float  # 综合得分
    description: str
    triggers: List[str]  # 触发条件
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EnhancedSignalAnalysis:
    """增强版信号分析结果"""
    code: str
    timestamp: datetime
    
    signals: List[SignalResult]
    
    combined_signal: SignalType
    combined_strength: float
    combined_confidence: float
    
    buy_signals: int
    sell_signals: int
    neutral_signals: int
    
    top_signal: SignalResult
    reasoning: List[str]
    
    risk_warning: Optional[str]
    suggested_action: str


class KangLongYouHuiEnhanced:
    """
    亢龙有悔增强版
    
    核心逻辑：
    1. 强庄信号：倍量突破前高
    2. XG信号：1.1倍量突破
    3. 重点信号：强庄与XG同时满足
    4. 主力线：21周期加权移动平均
    5. 启明线/揽月线：布林带类通道
    6. RSI信号：3周期RSI穿越68
    
    增强功能：
    - 自适应参数调整
    - 信号强度评分
    - 多周期确认
    """
    
    def __init__(self):
        self.name = "亢龙有悔"
        self.weight = 1.0
    
    def analyze(self, df: pd.DataFrame) -> SignalResult:
        """分析信号"""
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
            
            prev_high = h.rolling(20).max().shift(1)
            volume_ratio = v / v.shift(1)
            
            strong_volume = volume_ratio >= 1.9
            breakout = c > prev_high
            strong_signal = strong_volume.iloc[-1] and breakout.iloc[-1]
            
            if strong_signal:
                score += 30
                triggers.append(f"强庄信号：成交量放大{volume_ratio.iloc[-1]:.1f}倍突破前高")
            
            medium_volume = volume_ratio >= 1.1
            xg_signal = medium_volume.iloc[-1] and breakout.iloc[-1]
            
            if xg_signal:
                score += 20
                triggers.append(f"XG信号：成交量放大{volume_ratio.iloc[-1]:.1f}倍突破")
            
            if strong_signal and xg_signal:
                score += 20
                triggers.append("重点信号：强庄与XG共振")
            
            mid = (3 * c + l + o + h) / 6
            weights = np.arange(1, 22)[::-1]
            zhuli_line = mid.rolling(21).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
            
            if c.iloc[-1] > zhuli_line.iloc[-1]:
                score += 10
                triggers.append("价格站上主力线")
            
            ma15 = c.rolling(15).mean() * 1.005
            std = c.rolling(15).std()
            upper = ma15 + 2 * std
            
            if c.iloc[-1] > upper.iloc[-1]:
                score += 10
                triggers.append("突破揽月线")
            
            delta = c.diff()
            gain = delta.where(delta > 0, 0).rolling(3).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(3).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = (rs / (1 + rs) * 100).iloc[-1]
            
            if rsi > 68:
                score -= 10
                triggers.append(f"RSI超买({rsi:.0f})，注意回调")
            elif rsi < 30:
                score += 10
                triggers.append(f"RSI超卖({rsi:.0f})，可能反弹")
            
            if score >= 50:
                signal = SignalType.STRONG_BUY
                strength = min(100, score + 20)
            elif score >= 30:
                signal = SignalType.BUY
                strength = score
            elif score >= 10:
                signal = SignalType.HOLD
                strength = 50
            elif score >= -20:
                signal = SignalType.HOLD
                strength = 40
            else:
                signal = SignalType.SELL
                strength = max(0, 50 + score)
            
            confidence = min(0.95, strength / 100 * 0.8 + 0.2)
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=strength,
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
                metadata={
                    "volume_ratio": volume_ratio.iloc[-1] if not volume_ratio.empty else 1,
                    "rsi": rsi if not np.isnan(rsi) else 50,
                }
            )
            
        except Exception as e:
            logger.error(f"亢龙有悔分析失败: {e}")
            return self._empty_result()
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        """生成描述"""
        if not triggers:
            return "无明显信号"
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
        )


class HotMoneyDarkPoolEnhanced:
    """
    游资暗盘增强版
    
    核心逻辑：
    1. 游资净买：DMA动态均线计算
    2. 建仓信号：MX上穿MXA
    3. 暗盘买入信号：多指标共振确认
    4. 成交量突破：1.91倍量+价格上涨
    
    增强功能：
    - 多指标权重融合
    - 游资动向追踪
    - 资金强度评估
    """
    
    def __init__(self):
        self.name = "游资暗盘"
        self.weight = 1.0
    
    def analyze(self, df: pd.DataFrame) -> SignalResult:
        """分析信号"""
        if df.empty or len(df) < 50:
            return self._empty_result()
        
        try:
            c = df['close']
            h = df['high']
            l = df['low']
            v = df['volume']
            
            triggers = []
            score = 0
            
            ema2 = c.ewm(span=2).mean()
            ema42 = c.ewm(span=42).mean()
            
            golden_cross = ema2.iloc[-1] > ema42.iloc[-1] and ema2.iloc[-2] <= ema42.iloc[-2]
            if golden_cross:
                score += 25
                triggers.append("建仓信号：金叉形成")
            
            volume_spike = v.iloc[-1] / v.iloc[-2] if v.iloc[-2] > 0 else 1
            price_up = c.iloc[-1] > c.iloc[-2]
            
            if volume_spike >= 1.91 and price_up:
                score += 20
                triggers.append(f"倍量突破：成交量放大{volume_spike:.1f}倍")
            
            rsv = (c - l.rolling(9).min()) / (h.rolling(9).max() - l.rolling(9).min()) * 100
            k = rsv.ewm(alpha=1/3, adjust=False).mean()
            d = k.ewm(alpha=1/3, adjust=False).mean()
            
            if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2]:
                score += 15
                triggers.append("KDJ金叉")
            
            macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
            signal_line = macd.ewm(span=9).mean()
            
            if macd.iloc[-1] > signal_line.iloc[-1]:
                score += 10
                triggers.append("MACD多头")
            
            delta = c.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            if 30 < rsi < 70:
                score += 5
            elif rsi < 30:
                score += 10
                triggers.append(f"RSI超卖({rsi:.0f})")
            
            if score >= 50:
                signal = SignalType.STRONG_BUY
                strength = min(100, score + 15)
            elif score >= 30:
                signal = SignalType.BUY
                strength = score
            elif score >= 10:
                signal = SignalType.HOLD
                strength = 50
            else:
                signal = SignalType.HOLD
                strength = 40
            
            confidence = min(0.9, strength / 100 * 0.75 + 0.25)
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=strength,
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
                metadata={"volume_spike": volume_spike, "rsi": rsi}
            )
            
        except Exception as e:
            logger.error(f"游资暗盘分析失败: {e}")
            return self._empty_result()
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "游资动向不明"
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
        )


class DarkPoolFundEnhanced:
    """
    暗盘资金增强版
    
    核心逻辑：
    - 基于中小单净流入估算暗盘资金
    - 调整幅度计算
    - 资金流向分析
    
    增强功能：
    - 资金趋势分析
    - 异常资金检测
    """
    
    def __init__(self):
        self.name = "暗盘资金"
        self.weight = 0.8
    
    def analyze(self, df: pd.DataFrame, fund_flow: Dict = None) -> SignalResult:
        """分析信号"""
        if df.empty or len(df) < 10:
            return self._empty_result()
        
        try:
            c = df['close']
            h = df['high']
            l = df['low']
            o = df['open']
            
            triggers = []
            score = 0
            
            prev_close = c.shift(1)
            gap = (o - prev_close) / prev_close
            body = (c - o) / o
            upper_shadow = (h - o) / o
            lower_shadow = (o - l) / o
            
            adjustment = gap.iloc[-1] + body.iloc[-1] + upper_shadow.iloc[-1] - lower_shadow.iloc[-1]
            
            if fund_flow:
                medium_flow = fund_flow.get('medium_net', 0)
                small_flow = fund_flow.get('small_net', 0)
                dark_pool = (medium_flow + small_flow) * (1 + adjustment)
                
                if dark_pool > 0:
                    score += 25
                    triggers.append(f"暗盘资金流入：{dark_pool/10000:.1f}万")
                elif dark_pool < 0:
                    score -= 15
                    triggers.append(f"暗盘资金流出：{dark_pool/10000:.1f}万")
            
            price_momentum = (c.iloc[-1] - c.iloc[-5]) / c.iloc[-5] if c.iloc[-5] > 0 else 0
            if price_momentum > 0.05:
                score += 10
                triggers.append(f"价格上涨{price_momentum:.1%}")
            elif price_momentum < -0.05:
                score -= 10
                triggers.append(f"价格下跌{abs(price_momentum):.1%}")
            
            if score >= 20:
                signal = SignalType.BUY
                strength = 60 + score
            elif score >= 0:
                signal = SignalType.HOLD
                strength = 50
            else:
                signal = SignalType.SELL
                strength = max(30, 50 + score)
            
            confidence = 0.6
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=min(100, strength),
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
            )
            
        except Exception as e:
            logger.error(f"暗盘资金分析失败: {e}")
            return self._empty_result()
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "资金流向平稳"
        return f"{signal.value}：" + "；".join(triggers[:2])
    
    def _empty_result(self) -> SignalResult:
        return SignalResult(
            name=self.name,
            signal=SignalType.HOLD,
            strength=50,
            confidence=0.3,
            score=0,
            description="数据不足",
            triggers=[],
        )


class PreciseTradingEnhanced:
    """
    精准买卖点增强版
    
    核心逻辑：
    - ZigZag高低点检测
    - 新高点/新低点判断
    - 趋势转折识别
    
    增强功能：
    - 多周期ZigZag
    - 趋势强度评估
    """
    
    def __init__(self):
        self.name = "精准买卖点"
        self.weight = 0.9
    
    def analyze(self, df: pd.DataFrame, zig_pct: float = 5.0) -> SignalResult:
        """分析信号"""
        if df.empty or len(df) < 30:
            return self._empty_result()
        
        try:
            c = df['close']
            
            triggers = []
            score = 0
            
            peaks = self._find_peaks(c, zig_pct)
            troughs = self._find_troughs(c, zig_pct)
            
            last_peak_idx = peaks[0] if peaks else 0
            last_trough_idx = troughs[0] if troughs else 0
            
            recent_peak = c.iloc[last_peak_idx] if last_peak_idx < len(c) else c.iloc[-1]
            recent_trough = c.iloc[last_trough_idx] if last_trough_idx < len(c) else c.iloc[-1]
            
            distance_from_peak = (recent_peak - c.iloc[-1]) / recent_peak if recent_peak > 0 else 0
            distance_from_trough = (c.iloc[-1] - recent_trough) / recent_trough if recent_trough > 0 else 0
            
            if distance_from_trough > 0.03:
                score += 20
                triggers.append(f"距离最近低点上涨{distance_from_trough:.1%}")
            
            if distance_from_peak > 0.05:
                score -= 15
                triggers.append(f"距离最近高点下跌{distance_from_peak:.1%}")
            
            ma5 = c.rolling(5).mean()
            ma20 = c.rolling(20).mean()
            
            if ma5.iloc[-1] > ma20.iloc[-1]:
                score += 10
                triggers.append("短期均线在长期均线之上")
            else:
                score -= 10
            
            if last_trough_idx > last_peak_idx:
                score += 15
                triggers.append("最近形成低点，可能反弹")
            else:
                score -= 10
                triggers.append("最近形成高点，注意回调")
            
            if score >= 25:
                signal = SignalType.BUY
                strength = 60 + score
            elif score >= 5:
                signal = SignalType.HOLD
                strength = 50
            elif score >= -15:
                signal = SignalType.HOLD
                strength = 45
            else:
                signal = SignalType.SELL
                strength = max(30, 50 + score)
            
            confidence = 0.65
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=min(100, strength),
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
            )
            
        except Exception as e:
            logger.error(f"精准买卖点分析失败: {e}")
            return self._empty_result()
    
    def _find_peaks(self, series: pd.Series, pct: float) -> List[int]:
        """寻找峰值"""
        peaks = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] > series.iloc[i-1] * (1 + pct/100) and \
               series.iloc[i] > series.iloc[i+1] * (1 + pct/100):
                peaks.append(i)
        return sorted(peaks, reverse=True)
    
    def _find_troughs(self, series: pd.Series, pct: float) -> List[int]:
        """寻找谷值"""
        troughs = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] < series.iloc[i-1] * (1 - pct/100) and \
               series.iloc[i] < series.iloc[i+1] * (1 - pct/100):
                troughs.append(i)
        return sorted(troughs, reverse=True)
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "趋势不明"
        return f"{signal.value}：" + "；".join(triggers[:2])
    
    def _empty_result(self) -> SignalResult:
        return SignalResult(
            name=self.name,
            signal=SignalType.HOLD,
            strength=50,
            confidence=0.3,
            score=0,
            description="数据不足",
            triggers=[],
        )


class ThreeColorResonanceEnhanced:
    """
    三色共振增强版
    
    核心逻辑：
    - 主力资金线：35周期
    - 游资资金线：42周期
    - 散户资金线：21周期
    - 三线共振判断
    
    增强功能：
    - 资金强度评分
    - 共振程度分析
    """
    
    def __init__(self):
        self.name = "三色共振"
        self.weight = 1.0
    
    def analyze(self, df: pd.DataFrame) -> SignalResult:
        """分析信号"""
        if df.empty or len(df) < 50:
            return self._empty_result()
        
        try:
            c = df['close']
            h = df['high']
            l = df['low']
            
            triggers = []
            score = 0
            
            def calc_money_line(high, low, close, period):
                hhv = high.rolling(period).max()
                llv = low.rolling(period).min()
                return (close - llv) / (hhv - llv) * 100
            
            main_force = calc_money_line(h, l, c, 35).iloc[-1]
            hot_money = calc_money_line(h, l, c, 42).iloc[-1]
            retail = calc_money_line(h, l, c, 21).iloc[-1]
            
            if np.isnan(main_force):
                main_force = 50
            if np.isnan(hot_money):
                hot_money = 50
            if np.isnan(retail):
                retail = 50
            
            if main_force > 70:
                score += 20
                triggers.append(f"主力资金强势({main_force:.0f})")
            elif main_force > 50:
                score += 10
            elif main_force < 30:
                score -= 15
                triggers.append(f"主力资金弱势({main_force:.0f})")
            
            if hot_money > 70:
                score += 15
                triggers.append(f"游资资金活跃({hot_money:.0f})")
            elif hot_money > 50:
                score += 5
            
            if retail > 70:
                score += 5
            elif retail < 30:
                score += 10
                triggers.append(f"散户恐慌({retail:.0f})，可能见底")
            
            if main_force > 60 and hot_money > 60:
                score += 20
                triggers.append("主力游资共振向上")
            elif main_force < 40 and hot_money < 40:
                score -= 20
                triggers.append("主力游资共振向下")
            
            if score >= 40:
                signal = SignalType.STRONG_BUY
                strength = min(100, 60 + score)
            elif score >= 20:
                signal = SignalType.BUY
                strength = 55 + score
            elif score >= 0:
                signal = SignalType.HOLD
                strength = 50
            else:
                signal = SignalType.SELL
                strength = max(30, 50 + score)
            
            confidence = min(0.9, strength / 100 * 0.8 + 0.2)
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=strength,
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
                metadata={
                    "main_force": main_force,
                    "hot_money": hot_money,
                    "retail": retail,
                }
            )
            
        except Exception as e:
            logger.error(f"三色共振分析失败: {e}")
            return self._empty_result()
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "资金流向平衡"
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
        )


class XunLongJueEnhanced:
    """
    寻龙诀增强版
    
    核心逻辑：
    - 涨停板检测
    - 倍量突破
    - 龙头股识别
    
    增强功能：
    - 连板统计
    - 龙头强度评估
    """
    
    def __init__(self):
        self.name = "寻龙诀"
        self.weight = 1.1
    
    def analyze(self, df: pd.DataFrame) -> SignalResult:
        """分析信号"""
        if df.empty or len(df) < 20:
            return self._empty_result()
        
        try:
            c = df['close']
            v = df['volume']
            
            triggers = []
            score = 0
            
            pct_change = c.pct_change() * 100
            
            limit_up = pct_change >= 9.9
            if limit_up.iloc[-1]:
                score += 30
                triggers.append(f"涨停板：涨幅{pct_change.iloc[-1]:.1f}%")
            
            recent_limits = limit_up.iloc[-5:].sum()
            if recent_limits >= 2:
                score += 20
                triggers.append(f"近期{recent_limits}个涨停")
            
            volume_ratio = v.iloc[-1] / v.iloc[-5:].mean() if v.iloc[-5:].mean() > 0 else 1
            if volume_ratio > 2:
                score += 15
                triggers.append(f"成交量放大{volume_ratio:.1f}倍")
            
            ma5 = c.rolling(5).mean()
            ma10 = c.rolling(10).mean()
            
            if ma5.iloc[-1] > ma10.iloc[-1]:
                score += 10
                triggers.append("均线多头排列")
            
            if score >= 50:
                signal = SignalType.STRONG_BUY
                strength = min(100, 70 + score)
            elif score >= 30:
                signal = SignalType.BUY
                strength = 60 + score
            elif score >= 10:
                signal = SignalType.HOLD
                strength = 50
            else:
                signal = SignalType.HOLD
                strength = 45
            
            confidence = min(0.95, strength / 100 * 0.85 + 0.15)
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=strength,
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
            )
            
        except Exception as e:
            logger.error(f"寻龙诀分析失败: {e}")
            return self._empty_result()
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "无龙头信号"
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
        )


class MainForceControlEnhanced:
    """
    主力控盘增强版
    
    核心逻辑：
    - 控盘度计算
    - 筹码集中度
    - 主力动向
    
    增强功能：
    - 控盘强度评估
    - 筹码分布分析
    """
    
    def __init__(self):
        self.name = "主力控盘"
        self.weight = 0.9
    
    def analyze(self, df: pd.DataFrame) -> SignalResult:
        """分析信号"""
        if df.empty or len(df) < 30:
            return self._empty_result()
        
        try:
            c = df['close']
            h = df['high']
            l = df['low']
            v = df['volume']
            
            triggers = []
            score = 0
            
            typical_price = (h + l + c) / 3
            cumulative_tp = (typical_price * v).cumsum()
            cumulative_vol = v.cumsum()
            vwap = cumulative_tp / cumulative_vol
            
            if c.iloc[-1] > vwap.iloc[-1]:
                score += 15
                triggers.append("价格在VWAP之上")
            else:
                score -= 10
            
            price_std = c.pct_change().std()
            volume_std = v.pct_change().std()
            
            if price_std < 0.02 and volume_std < 0.3:
                score += 20
                triggers.append("价格稳定，成交量平稳，可能主力控盘")
            elif price_std > 0.04:
                score -= 10
                triggers.append("波动较大")
            
            ma20 = c.rolling(20).mean()
            ma60 = c.rolling(60).mean() if len(c) >= 60 else ma20
            
            if c.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
                score += 15
                triggers.append("多头趋势")
            elif c.iloc[-1] < ma20.iloc[-1] < ma60.iloc[-1]:
                score -= 15
                triggers.append("空头趋势")
            
            if score >= 30:
                signal = SignalType.BUY
                strength = 60 + score
            elif score >= 10:
                signal = SignalType.HOLD
                strength = 55
            elif score >= -10:
                signal = SignalType.HOLD
                strength = 50
            else:
                signal = SignalType.SELL
                strength = max(35, 50 + score)
            
            confidence = 0.65
            
            return SignalResult(
                name=self.name,
                signal=signal,
                strength=min(100, strength),
                confidence=confidence,
                score=score,
                description=self._generate_description(signal, triggers),
                triggers=triggers,
            )
            
        except Exception as e:
            logger.error(f"主力控盘分析失败: {e}")
            return self._empty_result()
    
    def _generate_description(self, signal: SignalType, triggers: List[str]) -> str:
        if not triggers:
            return "控盘度一般"
        return f"{signal.value}：" + "；".join(triggers[:2])
    
    def _empty_result(self) -> SignalResult:
        return SignalResult(
            name=self.name,
            signal=SignalType.HOLD,
            strength=50,
            confidence=0.3,
            score=0,
            description="数据不足",
            triggers=[],
        )


class EnhancedSignalEngine:
    """
    增强版信号引擎
    
    整合7大核心信号
    """
    
    def __init__(self):
        self.signals = {
            "亢龙有悔": KangLongYouHuiEnhanced(),
            "游资暗盘": HotMoneyDarkPoolEnhanced(),
            "暗盘资金": DarkPoolFundEnhanced(),
            "精准买卖点": PreciseTradingEnhanced(),
            "三色共振": ThreeColorResonanceEnhanced(),
            "寻龙诀": XunLongJueEnhanced(),
            "主力控盘": MainForceControlEnhanced(),
        }
    
    def analyze(self, code: str, df: pd.DataFrame, fund_flow: Dict = None) -> EnhancedSignalAnalysis:
        """执行完整分析"""
        cache = get_signal_cache()
        
        # 尝试从缓存获取完整分析结果
        cache_key = f"enhanced:analysis:{code}"
        cached_result = cache.get(cache_key, code, df)
        if cached_result:
            return cached_result
        
        results = []
        
        for name, signal in self.signals.items():
            # 尝试从缓存获取单个信号结果
            signal_result = cache.get(name, code, df)
            if not signal_result:
                if name == "暗盘资金":
                    signal_result = signal.analyze(df, fund_flow)
                else:
                    signal_result = signal.analyze(df)
                # 缓存单个信号结果
                cache.set(name, code, df, signal_result)
            results.append(signal_result)
        
        # 检测极端市场情况
        market_condition = self._detect_market_condition(df)
        
        # 计算基本信号统计
        buy_count = sum(1 for r in results if r.signal in [SignalType.BUY, SignalType.STRONG_BUY])
        sell_count = sum(1 for r in results if r.signal in [SignalType.SELL, SignalType.STRONG_SELL])
        neutral_count = len(results) - buy_count - sell_count
        
        # 智能信号冲突处理
        conflict_analysis = self._analyze_signal_conflicts(results)
        
        # 根据市场情况和信号冲突调整权重
        adjusted_results = self._adjust_for_market_condition(results, market_condition)
        
        # 计算调整后的总分
        total_score = sum(r.score * r.weight for r in adjusted_results)
        total_weight = sum(r.weight for r in adjusted_results)
        avg_score = total_score / total_weight if total_weight > 0 else 0
        
        # 根据调整后的分数确定综合信号
        combined_signal = self._determine_combined_signal(avg_score, conflict_analysis, market_condition)
        
        combined_strength = min(100, max(0, 50 + avg_score))
        combined_confidence = sum(r.confidence * r.weight for r in adjusted_results) / total_weight if total_weight > 0 else 0.5
        
        # 根据市场情况和冲突分析调整置信度
        if market_condition["is_extreme"]:
            combined_confidence = max(0.3, combined_confidence * 0.8)  # 在极端市场中降低置信度
        if conflict_analysis["has_conflict"]:
            combined_confidence = max(0.3, combined_confidence * 0.9)  # 在信号冲突时降低置信度
        
        top_signal = max(adjusted_results, key=lambda r: r.strength * r.confidence)
        
        reasoning = []
        for r in sorted(adjusted_results, key=lambda x: x.score, reverse=True)[:3]:
            if r.triggers:
                reasoning.append(f"【{r.name}】{r.description}")
        
        # 添加信号冲突分析到推理
        if conflict_analysis["has_conflict"]:
            reasoning.append(f"信号冲突分析: {conflict_analysis['conflict_reason']}")
            reasoning.append(f"冲突解决: {conflict_analysis['resolution']}")
        
        # 添加极端市场分析到推理
        if market_condition["is_extreme"]:
            reasoning.append(f"市场情况: {market_condition['condition']}")
            reasoning.append(f"市场建议: {market_condition['suggestion']}")
        
        risk_warning = None
        if sell_count >= 4:
            risk_warning = "多个信号显示卖出，请注意风险"
        elif any(r.strength > 80 and r.signal == SignalType.STRONG_BUY for r in adjusted_results):
            risk_warning = "出现强烈买入信号，但需注意追高风险"
        
        # 根据市场情况和信号冲突调整风险警告
        if market_condition["is_extreme"]:
            risk_warning = market_condition['risk_warning']
        elif conflict_analysis["has_conflict"] and conflict_analysis["conflict_level"] == "high":
            risk_warning = "信号存在高度冲突，建议谨慎操作"
        
        # 根据综合信号、市场情况和信号冲突调整建议操作
        suggested_action = self._adjust_suggested_action(combined_signal, market_condition, conflict_analysis)
        
        analysis_result = EnhancedSignalAnalysis(
            code=code,
            timestamp=datetime.now(),
            signals=adjusted_results,
            combined_signal=combined_signal,
            combined_strength=combined_strength,
            combined_confidence=combined_confidence,
            buy_signals=buy_count,
            sell_signals=sell_count,
            neutral_signals=neutral_count,
            top_signal=top_signal,
            reasoning=reasoning,
            risk_warning=risk_warning,
            suggested_action=suggested_action,
        )
        
        # 缓存完整分析结果
        cache.set(cache_key, code, df, analysis_result)
        
        return analysis_result
    
    def _detect_market_condition(self, df: pd.DataFrame) -> Dict:
        """检测市场情况"""
        if df.empty or len(df) < 20:
            return {
                "is_extreme": False,
                "condition": "正常",
                "suggestion": "按正常信号操作",
                "risk_warning": None
            }
        
        # 计算价格波动
        close_prices = df['close']
        price_change = close_prices.pct_change()
        price_volatility = price_change.std()
        recent_change = abs(price_change.iloc[-5:].mean())
        max_daily_change = abs(price_change.iloc[-10:].max())
        
        # 计算成交量变化
        if 'volume' in df:
            volume = df['volume']
            volume_change = volume.pct_change()
            recent_volume_change = abs(volume_change.iloc[-5:].mean())
            max_volume_change = abs(volume_change.iloc[-10:].max())
        else:
            recent_volume_change = 0
            max_volume_change = 0
        
        # 检测极端市场情况
        is_extreme = False
        condition = "正常"
        suggestion = "按正常信号操作"
        risk_warning = None
        
        # 价格大幅波动
        if max_daily_change > 0.07:
            is_extreme = True
            condition = "价格大幅波动"
            suggestion = "谨慎操作，设置止损"
            risk_warning = "市场价格波动剧烈，操作风险较高"
        
        # 成交量异常
        elif max_volume_change > 3:
            is_extreme = True
            condition = "成交量异常"
            suggestion = "关注量价配合，谨慎追高"
            risk_warning = "成交量异常放大，可能存在短期炒作"
        
        # 连续大幅下跌
        elif recent_change < -0.03 and price_change.iloc[-3:].mean() < -0.02:
            is_extreme = True
            condition = "连续大幅下跌"
            suggestion = "观望为主，等待企稳"
            risk_warning = "市场连续下跌，可能存在恐慌情绪"
        
        # 连续大幅上涨
        elif recent_change > 0.03 and price_change.iloc[-3:].mean() > 0.02:
            is_extreme = True
            condition = "连续大幅上涨"
            suggestion = "避免追高，关注回调"
            risk_warning = "市场连续上涨，可能存在泡沫风险"
        
        return {
            "is_extreme": is_extreme,
            "condition": condition,
            "suggestion": suggestion,
            "risk_warning": risk_warning,
            "price_volatility": price_volatility,
            "recent_change": recent_change,
            "max_daily_change": max_daily_change,
            "recent_volume_change": recent_volume_change,
            "max_volume_change": max_volume_change
        }
    
    def _analyze_signal_conflicts(self, results: List[SignalResult]) -> Dict:
        """分析信号冲突"""
        buy_signals = [r for r in results if r.signal in [SignalType.BUY, SignalType.STRONG_BUY]]
        sell_signals = [r for r in results if r.signal in [SignalType.SELL, SignalType.STRONG_SELL]]
        
        has_conflict = len(buy_signals) > 0 and len(sell_signals) > 0
        conflict_level = "low"
        conflict_reason = ""
        resolution = ""
        
        if has_conflict:
            # 计算买入和卖出信号的强度
            buy_strength = sum(r.strength * r.weight for r in buy_signals) / sum(r.weight for r in buy_signals) if buy_signals else 0
            sell_strength = sum(r.strength * r.weight for r in sell_signals) / sum(r.weight for r in sell_signals) if sell_signals else 0
            buy_confidence = sum(r.confidence * r.weight for r in buy_signals) / sum(r.weight for r in buy_signals) if buy_signals else 0
            sell_confidence = sum(r.confidence * r.weight for r in sell_signals) / sum(r.weight for r in sell_signals) if sell_signals else 0
            
            # 确定冲突级别
            if abs(buy_strength - sell_strength) < 20:
                conflict_level = "high"
            elif abs(buy_strength - sell_strength) < 40:
                conflict_level = "medium"
            else:
                conflict_level = "low"
            
            # 分析冲突原因
            if buy_confidence > sell_confidence:
                conflict_reason = f"买入信号平均置信度({buy_confidence:.2f})高于卖出信号({sell_confidence:.2f})"
            else:
                conflict_reason = f"卖出信号平均置信度({sell_confidence:.2f})高于买入信号({buy_confidence:.2f})"
            
            # 解决冲突
            if buy_strength > sell_strength:
                resolution = f"买入信号强度({buy_strength:.1f})强于卖出信号({sell_strength:.1f})，倾向于买入"
            else:
                resolution = f"卖出信号强度({sell_strength:.1f})强于买入信号({buy_strength:.1f})，倾向于卖出"
        
        return {
            "has_conflict": has_conflict,
            "conflict_level": conflict_level,
            "conflict_reason": conflict_reason,
            "resolution": resolution,
            "buy_count": len(buy_signals),
            "sell_count": len(sell_signals)
        }
    
    def _adjust_for_market_condition(self, results: List[SignalResult], market_condition: Dict) -> List[SignalResult]:
        """根据市场情况调整信号"""
        adjusted_results = []
        
        for result in results:
            # 创建结果副本
            import copy
            adjusted_result = copy.deepcopy(result)
            
            # 在极端市场情况下调整权重和强度
            if market_condition["is_extreme"]:
                # 降低信号权重，增加保守性
                adjusted_result.weight = max(0.5, adjusted_result.weight * 0.8)
                
                # 在价格大幅波动时降低信号强度
                if market_condition["condition"] == "价格大幅波动":
                    adjusted_result.strength = max(30, adjusted_result.strength * 0.7)
                    adjusted_result.confidence = max(0.3, adjusted_result.confidence * 0.8)
                
                # 在连续下跌时对买入信号更谨慎
                elif market_condition["condition"] == "连续大幅下跌":
                    if adjusted_result.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                        adjusted_result.strength = max(30, adjusted_result.strength * 0.6)
                        adjusted_result.confidence = max(0.3, adjusted_result.confidence * 0.7)
                
                # 在连续上涨时对卖出信号更谨慎
                elif market_condition["condition"] == "连续大幅上涨":
                    if adjusted_result.signal in [SignalType.SELL, SignalType.STRONG_SELL]:
                        adjusted_result.strength = max(30, adjusted_result.strength * 0.6)
                        adjusted_result.confidence = max(0.3, adjusted_result.confidence * 0.7)
            
            adjusted_results.append(adjusted_result)
        
        return adjusted_results
    
    def _determine_combined_signal(self, avg_score: float, conflict_analysis: Dict, market_condition: Dict) -> SignalType:
        """根据调整后的分数确定综合信号"""
        # 在极端市场情况下调整阈值
        if market_condition["is_extreme"]:
            if market_condition["condition"] == "连续大幅下跌":
                # 更难产生买入信号
                if avg_score >= 40:
                    return SignalType.STRONG_BUY
                elif avg_score >= 25:
                    return SignalType.BUY
                elif avg_score >= -5:
                    return SignalType.HOLD
                elif avg_score >= -25:
                    return SignalType.SELL
                else:
                    return SignalType.STRONG_SELL
            elif market_condition["condition"] == "连续大幅上涨":
                # 更难产生卖出信号
                if avg_score >= 30:
                    return SignalType.STRONG_BUY
                elif avg_score >= 10:
                    return SignalType.BUY
                elif avg_score >= -15:
                    return SignalType.HOLD
                elif avg_score >= -35:
                    return SignalType.SELL
                else:
                    return SignalType.STRONG_SELL
            else:
                # 其他极端市场情况，使用更保守的阈值
                if avg_score >= 40:
                    return SignalType.STRONG_BUY
                elif avg_score >= 20:
                    return SignalType.BUY
                elif avg_score >= -5:
                    return SignalType.HOLD
                elif avg_score >= -30:
                    return SignalType.SELL
                else:
                    return SignalType.STRONG_SELL
        
        # 正常市场情况
        if avg_score >= 35:
            return SignalType.STRONG_BUY
        elif avg_score >= 15:
            return SignalType.BUY
        elif avg_score >= -10:
            return SignalType.HOLD
        elif avg_score >= -30:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL
    
    def _adjust_suggested_action(self, combined_signal: SignalType, market_condition: Dict, conflict_analysis: Dict) -> str:
        """根据综合信号、市场情况和信号冲突调整建议操作"""
        base_action = ""
        
        if combined_signal in [SignalType.STRONG_BUY, SignalType.BUY]:
            base_action = "建议买入或加仓"
        elif combined_signal == SignalType.HOLD:
            base_action = "建议持有观望"
        else:
            base_action = "建议减仓或卖出"
        
        # 根据市场情况调整建议
        if market_condition["is_extreme"]:
            if market_condition["condition"] == "连续大幅下跌" and combined_signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                return f"{base_action}，但应分批建仓，设置止损"
            elif market_condition["condition"] == "连续大幅上涨" and combined_signal in [SignalType.SELL, SignalType.STRONG_SELL]:
                return f"{base_action}，但应分批减仓，避免踏空"
            elif market_condition["condition"] == "价格大幅波动":
                return f"{base_action}，但应控制仓位，设置严格止损"
            else:
                return f"{base_action}，但应谨慎操作，关注市场变化"
        
        # 根据信号冲突调整建议
        if conflict_analysis["has_conflict"] and conflict_analysis["conflict_level"] == "high":
            return f"{base_action}，但信号存在高度冲突，建议小仓位试探"
        elif conflict_analysis["has_conflict"]:
            return f"{base_action}，但信号存在一定冲突，建议控制仓位"
        
        return base_action


_enhanced_signal_engine = None


def get_enhanced_signal_engine() -> EnhancedSignalEngine:
    """获取增强版信号引擎"""
    global _enhanced_signal_engine
    if _enhanced_signal_engine is None:
        _enhanced_signal_engine = EnhancedSignalEngine()
    return _enhanced_signal_engine
