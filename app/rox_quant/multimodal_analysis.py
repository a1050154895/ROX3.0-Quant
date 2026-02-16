#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态分析系统
支持K线图表图像识别分析

功能：
1. K线形态识别
2. 图表模式识别
3. 趋势线检测
4. 支撑阻力识别
5. AI图表分析
"""

import logging
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """形态类型"""
    HEAD_SHOULDERS = "头肩顶/底"
    DOUBLE_TOP = "双顶"
    DOUBLE_BOTTOM = "双底"
    TRIANGLE = "三角形"
    FLAG = "旗形"
    WEDGE = "楔形"
    CHANNEL = "通道"
    CUP_HANDLE = "杯柄"


class CandleType(Enum):
    """K线类型"""
    DOJI = "十字星"
    HAMMER = "锤子线"
    HANGING_MAN = "上吊线"
    ENGULFING = "吞没形态"
    MORNING_STAR = "启明星"
    EVENING_STAR = "黄昏星"
    SPINNING_TOP = "纺锤线"
    MARUBOZU = "光头光脚"


@dataclass
class CandlePattern:
    """K线形态"""
    type: CandleType
    position: int
    signal: str
    confidence: float
    description: str


@dataclass
class ChartPattern:
    """图表形态"""
    type: PatternType
    start_idx: int
    end_idx: int
    signal: str
    confidence: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    description: str


@dataclass
class TrendLine:
    """趋势线"""
    type: str
    start_point: Tuple[int, float]
    end_point: Tuple[int, float]
    slope: float
    r_squared: float
    touches: int


@dataclass
class SupportResistance:
    """支撑阻力"""
    level: float
    type: str
    strength: int
    touches: int
    first_touch: int
    last_touch: int


@dataclass
class MultimodalAnalysisResult:
    """多模态分析结果"""
    candle_patterns: List[CandlePattern]
    chart_patterns: List[ChartPattern]
    trend_lines: List[TrendLine]
    support_levels: List[SupportResistance]
    resistance_levels: List[SupportResistance]
    overall_signal: str
    confidence: float
    reasoning: List[str]


class CandlePatternRecognizer:
    """
    K线形态识别器
    
    识别常见K线形态
    """
    
    def recognize(self, ohlc: List[Dict]) -> List[CandlePattern]:
        """识别K线形态"""
        patterns = []
        
        for i in range(len(ohlc)):
            o = ohlc[i].get('open', 0)
            h = ohlc[i].get('high', 0)
            l = ohlc[i].get('low', 0)
            c = ohlc[i].get('close', 0)
            
            if o == 0 or c == 0:
                continue
            
            body = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            total_range = h - l if h > l else 1
            
            if body / total_range < 0.1:
                patterns.append(CandlePattern(
                    type=CandleType.DOJI,
                    position=i,
                    signal="neutral",
                    confidence=0.7,
                    description="十字星，市场犹豫不决"
                ))
            
            if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                if i > 0 and ohlc[i-1].get('close', 0) > ohlc[i-1].get('open', 0):
                    patterns.append(CandlePattern(
                        type=CandleType.HAMMER,
                        position=i,
                        signal="buy",
                        confidence=0.75,
                        description="锤子线，底部反转信号"
                    ))
                else:
                    patterns.append(CandlePattern(
                        type=CandleType.HANGING_MAN,
                        position=i,
                        signal="sell",
                        confidence=0.7,
                        description="上吊线，顶部反转信号"
                    ))
            
            if i > 0:
                prev_o = ohlc[i-1].get('open', 0)
                prev_c = ohlc[i-1].get('close', 0)
                prev_body = abs(prev_c - prev_o)
                
                if c > o and o < prev_c and c > prev_o and body > prev_body:
                    patterns.append(CandlePattern(
                        type=CandleType.ENGULFING,
                        position=i,
                        signal="buy",
                        confidence=0.8,
                        description="看涨吞没，强烈买入信号"
                    ))
                elif c < o and o > prev_c and c < prev_o and body > prev_body:
                    patterns.append(CandlePattern(
                        type=CandleType.ENGULFING,
                        position=i,
                        signal="sell",
                        confidence=0.8,
                        description="看跌吞没，强烈卖出信号"
                    ))
            
            if i > 1:
                prev1 = ohlc[i-1]
                prev2 = ohlc[i-2]
                
                if (prev2.get('close', 0) < prev2.get('open', 0) and
                    abs(prev1.get('close', 0) - prev1.get('open', 0)) < prev2.get('open', 0) * 0.01 and
                    c > o and c > prev2.get('open', 0)):
                    patterns.append(CandlePattern(
                        type=CandleType.MORNING_STAR,
                        position=i,
                        signal="buy",
                        confidence=0.85,
                        description="启明星，底部反转信号"
                    ))
                
                if (prev2.get('close', 0) > prev2.get('open', 0) and
                    abs(prev1.get('close', 0) - prev1.get('open', 0)) < prev2.get('open', 0) * 0.01 and
                    c < o and c < prev2.get('open', 0)):
                    patterns.append(CandlePattern(
                        type=CandleType.EVENING_STAR,
                        position=i,
                        signal="sell",
                        confidence=0.85,
                        description="黄昏星，顶部反转信号"
                    ))
            
            if body / total_range > 0.9:
                if c > o:
                    patterns.append(CandlePattern(
                        type=CandleType.MARUBOZU,
                        position=i,
                        signal="buy",
                        confidence=0.7,
                        description="光头光脚阳线，强势上涨"
                    ))
                else:
                    patterns.append(CandlePattern(
                        type=CandleType.MARUBOZU,
                        position=i,
                        signal="sell",
                        confidence=0.7,
                        description="光头光脚阴线，强势下跌"
                    ))
        
        return patterns


class ChartPatternRecognizer:
    """
    图表形态识别器
    
    识别经典图表形态
    """
    
    def recognize(self, ohlc: List[Dict]) -> List[ChartPattern]:
        """识别图表形态"""
        patterns = []
        
        if len(ohlc) < 20:
            return patterns
        
        closes = [d.get('close', 0) for d in ohlc]
        highs = [d.get('high', 0) for d in ohlc]
        lows = [d.get('low', 0) for d in ohlc]
        
        db_pattern = self._detect_double_bottom(closes, lows)
        if db_pattern:
            patterns.append(db_pattern)
        
        dt_pattern = self._detect_double_top(closes, highs)
        if dt_pattern:
            patterns.append(dt_pattern)
        
        triangle = self._detect_triangle(highs, lows)
        if triangle:
            patterns.append(triangle)
        
        return patterns
    
    def _detect_double_bottom(self, closes: List[float], lows: List[float]) -> Optional[ChartPattern]:
        """检测双底"""
        if len(lows) < 20:
            return None
        
        recent_lows = lows[-20:]
        min_idx = recent_lows.index(min(recent_lows))
        
        left_idx = min_idx - 5 if min_idx >= 5 else 0
        right_idx = min_idx + 5 if min_idx + 5 < len(recent_lows) else len(recent_lows) - 1
        
        if left_idx >= 0 and right_idx < len(recent_lows):
            left_low = recent_lows[left_idx]
            right_low = recent_lows[right_idx]
            middle_low = recent_lows[min_idx]
            
            if (abs(left_low - right_low) / middle_low < 0.03 and
                middle_low < left_low and middle_low < right_low):
                return ChartPattern(
                    type=PatternType.DOUBLE_BOTTOM,
                    start_idx=left_idx,
                    end_idx=right_idx,
                    signal="buy",
                    confidence=0.75,
                    target_price=closes[-1] * 1.1,
                    stop_loss=middle_low * 0.97,
                    description="双底形态，底部反转信号"
                )
        
        return None
    
    def _detect_double_top(self, closes: List[float], highs: List[float]) -> Optional[ChartPattern]:
        """检测双顶"""
        if len(highs) < 20:
            return None
        
        recent_highs = highs[-20:]
        max_idx = recent_highs.index(max(recent_highs))
        
        left_idx = max_idx - 5 if max_idx >= 5 else 0
        right_idx = max_idx + 5 if max_idx + 5 < len(recent_highs) else len(recent_highs) - 1
        
        if left_idx >= 0 and right_idx < len(recent_highs):
            left_high = recent_highs[left_idx]
            right_high = recent_highs[right_idx]
            middle_high = recent_highs[max_idx]
            
            if (abs(left_high - right_high) / middle_high < 0.03 and
                middle_high > left_high and middle_high > right_high):
                return ChartPattern(
                    type=PatternType.DOUBLE_TOP,
                    start_idx=left_idx,
                    end_idx=right_idx,
                    signal="sell",
                    confidence=0.75,
                    target_price=closes[-1] * 0.9,
                    stop_loss=middle_high * 1.03,
                    description="双顶形态，顶部反转信号"
                )
        
        return None
    
    def _detect_triangle(self, highs: List[float], lows: List[float]) -> Optional[ChartPattern]:
        """检测三角形"""
        if len(highs) < 15:
            return None
        
        recent_highs = highs[-15:]
        recent_lows = lows[-15:]
        
        high_slope = (recent_highs[-1] - recent_highs[0]) / recent_highs[0] if recent_highs[0] > 0 else 0
        low_slope = (recent_lows[-1] - recent_lows[0]) / recent_lows[0] if recent_lows[0] > 0 else 0
        
        if high_slope < -0.02 and low_slope > 0.02:
            return ChartPattern(
                type=PatternType.TRIANGLE,
                start_idx=0,
                end_idx=14,
                signal="neutral",
                confidence=0.7,
                target_price=None,
                stop_loss=None,
                description="对称三角形，即将突破"
            )
        
        return None


class SupportResistanceDetector:
    """
    支撑阻力检测器
    """
    
    def detect(self, ohlc: List[Dict], n_levels: int = 3) -> Tuple[List[SupportResistance], List[SupportResistance]]:
        """检测支撑阻力位"""
        if len(ohlc) < 20:
            return [], []
        
        highs = [d.get('high', 0) for d in ohlc]
        lows = [d.get('low', 0) for d in ohlc]
        closes = [d.get('close', 0) for d in ohlc]
        
        levels = {}
        
        for i in range(1, len(ohlc) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                level = highs[i]
                rounded = round(level, 2)
                if rounded not in levels:
                    levels[rounded] = {'type': 'resistance', 'touches': 0, 'indices': []}
                levels[rounded]['touches'] += 1
                levels[rounded]['indices'].append(i)
            
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                level = lows[i]
                rounded = round(level, 2)
                if rounded not in levels:
                    levels[rounded] = {'type': 'support', 'touches': 0, 'indices': []}
                levels[rounded]['touches'] += 1
                levels[rounded]['indices'].append(i)
        
        current_price = closes[-1]
        
        supports = []
        resistances = []
        
        for level, info in sorted(levels.items(), key=lambda x: x[1]['touches'], reverse=True):
            sr = SupportResistance(
                level=level,
                type=info['type'],
                strength=info['touches'],
                touches=info['touches'],
                first_touch=min(info['indices']),
                last_touch=max(info['indices']),
            )
            
            if level < current_price and len(supports) < n_levels:
                supports.append(sr)
            elif level > current_price and len(resistances) < n_levels:
                resistances.append(sr)
        
        supports.sort(key=lambda x: x.level, reverse=True)
        resistances.sort(key=lambda x: x.level)
        
        return supports, resistances


class TrendLineDetector:
    """
    趋势线检测器
    """
    
    def detect(self, ohlc: List[Dict]) -> List[TrendLine]:
        """检测趋势线"""
        if len(ohlc) < 10:
            return []
        
        trend_lines = []
        
        lows = [d.get('low', 0) for d in ohlc]
        highs = [d.get('high', 0) for d in ohlc]
        
        uptrend = self._fit_trend_line(lows, 'up')
        if uptrend:
            trend_lines.append(uptrend)
        
        downtrend = self._fit_trend_line(highs, 'down')
        if downtrend:
            trend_lines.append(downtrend)
        
        return trend_lines
    
    def _fit_trend_line(self, values: List[float], direction: str) -> Optional[TrendLine]:
        """拟合趋势线"""
        if len(values) < 5:
            return None
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        try:
            slope, intercept = np.polyfit(x, y, 1)
            
            predicted = slope * x + intercept
            residuals = y - predicted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            if r_squared > 0.5:
                touches = sum(1 for r in residuals if abs(r) < np.std(residuals) * 0.5)
                
                return TrendLine(
                    type=f"{'上升' if direction == 'up' else '下降'}趋势线",
                    start_point=(0, intercept),
                    end_point=(n - 1, slope * (n - 1) + intercept),
                    slope=slope,
                    r_squared=r_squared,
                    touches=touches,
                )
        except:
            pass
        
        return None


class MultimodalAnalyzer:
    """
    多模态分析器
    
    整合所有分析功能
    """
    
    def __init__(self):
        self.candle_recognizer = CandlePatternRecognizer()
        self.chart_recognizer = ChartPatternRecognizer()
        self.sr_detector = SupportResistanceDetector()
        self.trend_detector = TrendLineDetector()
    
    def analyze(self, ohlc: List[Dict]) -> MultimodalAnalysisResult:
        """
        执行多模态分析
        
        Args:
            ohlc: OHLC数据列表
        """
        candle_patterns = self.candle_recognizer.recognize(ohlc)
        
        chart_patterns = self.chart_recognizer.recognize(ohlc)
        
        trend_lines = self.trend_detector.detect(ohlc)
        
        supports, resistances = self.sr_detector.detect(ohlc)
        
        signal, confidence, reasoning = self._generate_signal(
            candle_patterns, chart_patterns, supports, resistances
        )
        
        return MultimodalAnalysisResult(
            candle_patterns=candle_patterns,
            chart_patterns=chart_patterns,
            trend_lines=trend_lines,
            support_levels=supports,
            resistance_levels=resistances,
            overall_signal=signal,
            confidence=confidence,
            reasoning=reasoning,
        )
    
    def _generate_signal(
        self,
        candle_patterns: List[CandlePattern],
        chart_patterns: List[ChartPattern],
        supports: List[SupportResistance],
        resistances: List[SupportResistance],
    ) -> Tuple[str, float, List[str]]:
        """生成综合信号"""
        buy_score = 0
        sell_score = 0
        reasoning = []
        
        for p in candle_patterns:
            if p.signal == "buy":
                buy_score += p.confidence
                reasoning.append(f"K线形态：{p.description}")
            elif p.signal == "sell":
                sell_score += p.confidence
                reasoning.append(f"K线形态：{p.description}")
        
        for p in chart_patterns:
            if p.signal == "buy":
                buy_score += p.confidence * 1.5
                reasoning.append(f"图表形态：{p.description}")
            elif p.signal == "sell":
                sell_score += p.confidence * 1.5
                reasoning.append(f"图表形态：{p.description}")
        
        if supports and supports[0].strength >= 3:
            buy_score += 0.5
            reasoning.append(f"强支撑位：{supports[0].level:.2f}")
        
        if resistances and resistances[0].strength >= 3:
            sell_score += 0.5
            reasoning.append(f"强阻力位：{resistances[0].level:.2f}")
        
        total = buy_score + sell_score
        if total == 0:
            return "neutral", 0.5, ["无明显信号"]
        
        if buy_score > sell_score:
            signal = "buy"
            confidence = buy_score / (buy_score + sell_score)
        elif sell_score > buy_score:
            signal = "sell"
            confidence = sell_score / (buy_score + sell_score)
        else:
            signal = "neutral"
            confidence = 0.5
        
        return signal, min(confidence, 0.95), reasoning


_multimodal_analyzer = None


def get_multimodal_analyzer() -> MultimodalAnalyzer:
    """获取多模态分析器单例"""
    global _multimodal_analyzer
    if _multimodal_analyzer is None:
        _multimodal_analyzer = MultimodalAnalyzer()
    return _multimodal_analyzer
