#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卢麒元方法论增强预测系统
整合矛盾分析、价值规律、334法则、律动操作、宏观周期

核心目标：提高预测准确度至95%以上，帮助用户提高收益率

方法论核心：
1. 矛盾分析法 - 识别市场主矛盾，把握主要矛盾的主要方面
2. 价值规律 - 价格围绕价值波动，识别背离机会
3. 334法则 - 30%底仓 + 30%律动 + 40%预备队
4. 律动操作 - 通过波段降低持仓成本
5. 宏观周期 - 资本周转效率决定大方向
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """信号强度等级"""
    VERY_STRONG_BUY = 5
    STRONG_BUY = 4
    BUY = 3
    WEAK_BUY = 2
    NEUTRAL = 1
    WEAK_SELL = -2
    SELL = -3
    STRONG_SELL = -4
    VERY_STRONG_SELL = -5


class MarketPhase(Enum):
    """市场阶段"""
    VALUE_ACCUMULATION = "价值积累期"
    PRICE_DEVIATION_UP = "价格向上背离期"
    PRICE_DEVIATION_DOWN = "价格向下背离期"
    LAW_RHYTHM = "律动操作期"
    CRISIS_WARNING = "危机预警期"
    UNCERTAIN = "不确定期"


@dataclass
class PredictionResult:
    """预测结果"""
    signal: str
    confidence: float
    direction: float
    strength: int
    target_price: Optional[float]
    stop_loss: Optional[float]
    position_ratio: float
    holding_period: int
    reasoning: List[str]
    risk_level: str
    expected_return: float
    win_probability: float


@dataclass
class ContradictionSignal:
    """矛盾分析信号"""
    primary_contradiction: str
    contradiction_strength: float
    direction: float
    confidence: float
    description: str


@dataclass
class ValueSignal:
    """价值规律信号"""
    intrinsic_value: float
    current_price: float
    deviation_ratio: float
    value_grade: str
    signal: str
    confidence: float


@dataclass
class MacroSignal:
    """宏观周期信号"""
    capital_turnover: float
    tax_health: float
    crisis_probability: float
    phase: str
    asset_allocation: Dict[str, float]


@dataclass
class TechnicalSignal:
    """技术分析信号"""
    macd_signal: str
    rsi_value: float
    volume_trend: str
    price_momentum: float
    support_level: float
    resistance_level: float


class LuQiyuanPredictionEngine:
    """
    卢麒元方法论增强预测引擎
    
    整合五大核心方法论：
    1. 矛盾分析法 - 识别主要矛盾
    2. 价值规律 - 价格与价值关系
    3. 334法则 - 仓位管理
    4. 律动操作 - 波段降成本
    5. 宏观周期 - 大方向判断
    """
    
    def __init__(self):
        self.weights = {
            'contradiction': 0.25,
            'value': 0.30,
            'macro': 0.20,
            'technical': 0.15,
            'sentiment': 0.10,
        }
        self._signal_history: List[Dict] = []
        self._accuracy_records: List[Dict] = []
    
    def predict(
        self,
        code: str,
        price_data: pd.DataFrame,
        fundamental_data: Dict,
        market_data: Dict,
        macro_data: Dict,
    ) -> PredictionResult:
        """
        综合预测
        
        Args:
            code: 股票代码
            price_data: 价格数据(OHLCV)
            fundamental_data: 基本面数据(ROE, PE, PB等)
            market_data: 市场数据(成交量, 涨跌比等)
            macro_data: 宏观数据(GDP, M2, 利率等)
        
        Returns:
            PredictionResult: 预测结果
        """
        signals = {}
        confidences = {}
        
        contradiction_signal = self._analyze_contradiction(market_data)
        signals['contradiction'] = contradiction_signal.direction
        confidences['contradiction'] = contradiction_signal.confidence
        
        value_signal = self._analyze_value(fundamental_data, price_data)
        signals['value'] = self._value_to_score(value_signal)
        confidences['value'] = value_signal.confidence
        
        macro_signal = self._analyze_macro(macro_data)
        signals['macro'] = self._macro_to_score(macro_signal)
        confidences['macro'] = 0.7 if macro_signal.crisis_probability < 0.3 else 0.9
        
        technical_signal = self._analyze_technical(price_data)
        signals['technical'] = self._technical_to_score(technical_signal)
        confidences['technical'] = 0.6
        
        sentiment_score = self._analyze_sentiment(market_data)
        signals['sentiment'] = sentiment_score
        confidences['sentiment'] = 0.5
        
        final_score = 0
        total_weight = 0
        for key, weight in self.weights.items():
            if key in signals and key in confidences:
                adjusted_weight = weight * confidences[key]
                final_score += signals[key] * adjusted_weight
                total_weight += adjusted_weight
        
        if total_weight > 0:
            final_score = final_score / total_weight
        
        signal, strength = self._score_to_signal(final_score)
        confidence = self._calculate_confidence(confidences)
        
        current_price = price_data['close'].iloc[-1] if not price_data.empty else 0
        target_price, stop_loss = self._calculate_targets(
            current_price, final_score, technical_signal
        )
        
        position_ratio = self._calculate_position_334(
            final_score, confidence, macro_signal.crisis_probability
        )
        
        holding_period = self._calculate_holding_period(
            value_signal, contradiction_signal, macro_signal
        )
        
        reasoning = self._generate_reasoning(
            contradiction_signal, value_signal, macro_signal, technical_signal
        )
        
        risk_level = self._assess_risk(
            macro_signal, contradiction_signal, confidence
        )
        
        expected_return = self._calculate_expected_return(
            final_score, confidence, holding_period
        )
        
        win_probability = self._calculate_win_probability(
            confidence, abs(final_score), risk_level
        )
        
        result = PredictionResult(
            signal=signal,
            confidence=confidence,
            direction=final_score,
            strength=strength,
            target_price=target_price,
            stop_loss=stop_loss,
            position_ratio=position_ratio,
            holding_period=holding_period,
            reasoning=reasoning,
            risk_level=risk_level,
            expected_return=expected_return,
            win_probability=win_probability,
        )
        
        self._record_prediction(code, result)
        
        return result
    
    def _analyze_contradiction(self, market_data: Dict) -> ContradictionSignal:
        """
        矛盾分析法
        
        识别市场主要矛盾：
        1. 量能 vs 赚钱效应的矛盾
        2. 外资 vs 内资的矛盾
        3. 行业分化 vs 指数共振的矛盾
        4. 政策预期 vs 经济现实的矛盾
        """
        contradictions = []
        
        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume', 1)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        up_count = market_data.get('up_count', 0)
        down_count = market_data.get('down_count', 0)
        total = up_count + down_count
        up_ratio = up_count / total if total > 0 else 0.5
        
        if volume_ratio > 1.5 and up_ratio < 0.4:
            contradictions.append({
                'name': '量价背离',
                'strength': 80,
                'direction': -1,
                'desc': '放量下跌，主力出货迹象明显'
            })
        elif volume_ratio > 1.5 and up_ratio > 0.6:
            contradictions.append({
                'name': '量价齐升',
                'strength': 70,
                'direction': 1,
                'desc': '放量上涨，资金积极入场'
            })
        
        north_flow = market_data.get('north_flow', 0)
        main_flow = market_data.get('main_flow', 0)
        
        if abs(north_flow - main_flow) > 50:
            contradictions.append({
                'name': '资金分歧',
                'strength': 60,
                'direction': 1 if north_flow > 0 else -1,
                'desc': f'北向资金{"流入" if north_flow > 0 else "流出"}，与主力资金存在分歧'
            })
        
        if contradictions:
            primary = max(contradictions, key=lambda x: x['strength'])
            return ContradictionSignal(
                primary_contradiction=primary['name'],
                contradiction_strength=primary['strength'],
                direction=primary['direction'],
                confidence=primary['strength'] / 100,
                description=primary['desc']
            )
        
        return ContradictionSignal(
            primary_contradiction="无明显矛盾",
            contradiction_strength=30,
            direction=0,
            confidence=0.3,
            description="市场处于平衡状态"
        )
    
    def _analyze_value(
        self,
        fundamental_data: Dict,
        price_data: pd.DataFrame
    ) -> ValueSignal:
        """
        价值规律分析
        
        核心原理：价格围绕价值波动
        
        方法：
        1. 计算内在价值（基于ROE、增长率、风险溢价）
        2. 计算价格偏离度
        3. 判断价值等级
        """
        roe = fundamental_data.get('roe', 0.10)
        pe = fundamental_data.get('pe', 15)
        pb = fundamental_data.get('pb', 1.5)
        growth_rate = fundamental_data.get('growth_rate', 0.05)
        dividend_yield = fundamental_data.get('dividend_yield', 0.02)
        
        risk_free_rate = 0.03
        market_risk_premium = 0.06
        beta = fundamental_data.get('beta', 1.0)
        required_return = risk_free_rate + beta * market_risk_premium
        
        if roe > required_return:
            growth_adjusted = min(growth_rate, 0.20)
            intrinsic_value = pb * (1 + growth_adjusted) * (roe / required_return)
        else:
            intrinsic_value = pb * (roe / required_return)
        
        current_price = price_data['close'].iloc[-1] if not price_data.empty else 1
        book_value = current_price / pb if pb > 0 else current_price
        intrinsic_price = intrinsic_value * book_value
        
        deviation_ratio = (current_price - intrinsic_price) / intrinsic_price if intrinsic_price > 0 else 0
        
        if deviation_ratio < -0.30:
            value_grade = "深度低估"
            signal = "strong_buy"
            confidence = 0.85
        elif deviation_ratio < -0.15:
            value_grade = "低估"
            signal = "buy"
            confidence = 0.75
        elif deviation_ratio < 0.15:
            value_grade = "合理"
            signal = "hold"
            confidence = 0.50
        elif deviation_ratio < 0.30:
            value_grade = "高估"
            signal = "sell"
            confidence = 0.70
        else:
            value_grade = "严重高估"
            signal = "strong_sell"
            confidence = 0.80
        
        return ValueSignal(
            intrinsic_value=intrinsic_price,
            current_price=current_price,
            deviation_ratio=deviation_ratio,
            value_grade=value_grade,
            signal=signal,
            confidence=confidence
        )
    
    def _analyze_macro(self, macro_data: Dict) -> MacroSignal:
        """
        宏观周期分析
        
        基于卢麒元宏观理论：
        1. 资本周转效率 = GDP / M2
        2. 税政健康度 = 直接税占比
        3. 危机信号 = 资本周转受阻 + 食利结构固化
        """
        gdp = macro_data.get('gdp', 100)
        m2 = macro_data.get('m2', 200)
        direct_tax_ratio = macro_data.get('direct_tax_ratio', 0.3)
        gini = macro_data.get('gini', 0.4)
        
        capital_turnover = gdp / m2 if m2 > 0 else 0.5
        
        if capital_turnover > 0.6:
            turnover_grade = "高效"
        elif capital_turnover > 0.45:
            turnover_grade = "正常"
        else:
            turnover_grade = "低效"
        
        tax_health = direct_tax_ratio * (1 - gini)
        
        crisis_probability = 0
        if capital_turnover < 0.45:
            crisis_probability += 0.3
        if gini > 0.45:
            crisis_probability += 0.3
        if direct_tax_ratio < 0.25:
            crisis_probability += 0.2
        
        crisis_probability = min(crisis_probability, 0.9)
        
        if crisis_probability > 0.5:
            phase = "危机预警期"
            asset_allocation = {"现金": 0.4, "黄金": 0.3, "债券": 0.2, "股票": 0.1}
        elif crisis_probability > 0.3:
            phase = "风险期"
            asset_allocation = {"现金": 0.2, "黄金": 0.2, "债券": 0.3, "股票": 0.3}
        else:
            phase = "安全期"
            asset_allocation = {"现金": 0.1, "黄金": 0.1, "债券": 0.2, "股票": 0.6}
        
        return MacroSignal(
            capital_turnover=capital_turnover,
            tax_health=tax_health,
            crisis_probability=crisis_probability,
            phase=phase,
            asset_allocation=asset_allocation
        )
    
    def _analyze_technical(self, price_data: pd.DataFrame) -> TechnicalSignal:
        """技术分析"""
        if price_data.empty or len(price_data) < 26:
            return TechnicalSignal(
                macd_signal="neutral",
                rsi_value=50,
                volume_trend="neutral",
                price_momentum=0,
                support_level=0,
                resistance_level=0
            )
        
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        volume = price_data['volume']
        
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        macd = (dif - dea) * 2
        
        if macd.iloc[-1] > 0 and dif.iloc[-1] > dea.iloc[-1]:
            macd_signal = "golden_cross"
        elif macd.iloc[-1] < 0 and dif.iloc[-1] < dea.iloc[-1]:
            macd_signal = "death_cross"
        else:
            macd_signal = "neutral"
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1] if not rsi.empty else 50
        
        avg_volume = volume.rolling(window=20).mean()
        recent_volume = volume.iloc[-5:].mean()
        volume_trend = "increasing" if recent_volume > avg_volume.iloc[-1] * 1.2 else "decreasing" if recent_volume < avg_volume.iloc[-1] * 0.8 else "neutral"
        
        momentum = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if close.iloc[-20] != 0 else 0
        
        support_level = low.rolling(window=20).min().iloc[-1]
        resistance_level = high.rolling(window=20).max().iloc[-1]
        
        return TechnicalSignal(
            macd_signal=macd_signal,
            rsi_value=rsi_value,
            volume_trend=volume_trend,
            price_momentum=momentum,
            support_level=support_level,
            resistance_level=resistance_level
        )
    
    def _analyze_sentiment(self, market_data: Dict) -> float:
        """市场情绪分析"""
        up_ratio = market_data.get('up_ratio', 0.5)
        limit_up = market_data.get('limit_up', 0)
        limit_down = market_data.get('limit_down', 0)
        turnover_rate = market_data.get('turnover_rate', 0.02)
        
        sentiment = 0
        
        if up_ratio > 0.7:
            sentiment += 2
        elif up_ratio > 0.55:
            sentiment += 1
        elif up_ratio < 0.3:
            sentiment -= 2
        elif up_ratio < 0.45:
            sentiment -= 1
        
        if limit_up > limit_down * 2:
            sentiment += 1
        elif limit_down > limit_up * 2:
            sentiment -= 1
        
        if turnover_rate > 0.05:
            sentiment += 0.5
        elif turnover_rate < 0.01:
            sentiment -= 0.5
        
        return max(-5, min(5, sentiment))
    
    def _value_to_score(self, value_signal: ValueSignal) -> float:
        """价值信号转分数"""
        score_map = {
            "strong_buy": 4,
            "buy": 2,
            "hold": 0,
            "sell": -2,
            "strong_sell": -4,
        }
        return score_map.get(value_signal.signal, 0)
    
    def _macro_to_score(self, macro_signal: MacroSignal) -> float:
        """宏观信号转分数"""
        if macro_signal.crisis_probability > 0.6:
            return -4
        elif macro_signal.crisis_probability > 0.4:
            return -2
        elif macro_signal.crisis_probability < 0.2:
            return 2
        return 0
    
    def _technical_to_score(self, technical: TechnicalSignal) -> float:
        """技术信号转分数"""
        score = 0
        
        if technical.macd_signal == "golden_cross":
            score += 1.5
        elif technical.macd_signal == "death_cross":
            score -= 1.5
        
        if technical.rsi_value < 30:
            score += 1
        elif technical.rsi_value > 70:
            score -= 1
        
        if technical.volume_trend == "increasing":
            score += 0.5
        elif technical.volume_trend == "decreasing":
            score -= 0.5
        
        if technical.price_momentum > 0.05:
            score += 1
        elif technical.price_momentum < -0.05:
            score -= 1
        
        return score
    
    def _score_to_signal(self, score: float) -> Tuple[str, int]:
        """分数转信号"""
        if score >= 3.5:
            return "强烈买入", 5
        elif score >= 2.5:
            return "买入", 4
        elif score >= 1.5:
            return "偏多", 3
        elif score >= 0.5:
            return "观望偏多", 2
        elif score >= -0.5:
            return "中性", 1
        elif score >= -1.5:
            return "观望偏空", -2
        elif score >= -2.5:
            return "偏空", -3
        elif score >= -3.5:
            return "卖出", -4
        else:
            return "强烈卖出", -5
    
    def _calculate_confidence(self, confidences: Dict[str, float]) -> float:
        """计算综合置信度"""
        values = list(confidences.values())
        if not values:
            return 0.5
        return sum(values) / len(values)
    
    def _calculate_targets(
        self,
        current_price: float,
        score: float,
        technical: TechnicalSignal
    ) -> Tuple[Optional[float], Optional[float]]:
        """计算目标价和止损价"""
        if current_price <= 0:
            return None, None
        
        if score > 0:
            target_return = min(score * 0.05, 0.30)
            target_price = current_price * (1 + target_return)
            stop_loss = max(technical.support_level, current_price * 0.92)
        else:
            target_price = None
            stop_loss = current_price * 1.05
        
        return target_price, stop_loss
    
    def _calculate_position_334(
        self,
        score: float,
        confidence: float,
        crisis_prob: float
    ) -> float:
        """
        334法则仓位计算
        
        - 30% 底仓：长期持有
        - 30% 律动仓：波段操作
        - 40% 预备队：应对风险
        """
        base_position = 0.30
        
        if score > 2 and confidence > 0.7:
            rhythm_position = 0.30
        elif score > 1:
            rhythm_position = 0.15
        else:
            rhythm_position = 0
        
        reserve_reduction = crisis_prob * 0.40
        
        total_position = base_position + rhythm_position - reserve_reduction
        
        return max(0.1, min(0.8, total_position))
    
    def _calculate_holding_period(
        self,
        value: ValueSignal,
        contradiction: ContradictionSignal,
        macro: MacroSignal
    ) -> int:
        """计算建议持仓周期（天）"""
        if macro.crisis_probability > 0.5:
            return 5
        
        if value.deviation_ratio < -0.2:
            return 60
        elif value.deviation_ratio < -0.1:
            return 30
        
        if contradiction.contradiction_strength > 60:
            return 10
        
        return 20
    
    def _generate_reasoning(
        self,
        contradiction: ContradictionSignal,
        value: ValueSignal,
        macro: MacroSignal,
        technical: TechnicalSignal
    ) -> List[str]:
        """生成决策理由"""
        reasons = []
        
        reasons.append(f"【矛盾分析】{contradiction.primary_contradiction}：{contradiction.description}")
        
        reasons.append(f"【价值规律】当前{value.value_grade}，偏离度{value.deviation_ratio:.1%}")
        
        reasons.append(f"【宏观周期】{macro.phase}，危机概率{macro.crisis_probability:.0%}")
        
        reasons.append(f"【技术分析】MACD {technical.macd_signal}，RSI {technical.rsi_value:.0f}")
        
        return reasons
    
    def _assess_risk(
        self,
        macro: MacroSignal,
        contradiction: ContradictionSignal,
        confidence: float
    ) -> str:
        """评估风险等级"""
        if macro.crisis_probability > 0.5:
            return "高风险"
        elif macro.crisis_probability > 0.3 or contradiction.contradiction_strength > 70:
            return "中高风险"
        elif confidence < 0.5:
            return "中风险"
        elif confidence > 0.8:
            return "低风险"
        else:
            return "中低风险"
    
    def _calculate_expected_return(
        self,
        score: float,
        confidence: float,
        holding_period: int
    ) -> float:
        """计算预期收益率"""
        base_return = score * 0.02
        confidence_adj = confidence * 0.5 + 0.5
        period_adj = min(holding_period / 30, 2)
        
        return base_return * confidence_adj * period_adj
    
    def _calculate_win_probability(
        self,
        confidence: float,
        signal_strength: float,
        risk_level: str
    ) -> float:
        """计算胜率"""
        base_prob = 0.5
        
        base_prob += confidence * 0.2
        
        base_prob += signal_strength * 0.02
        
        risk_adj = {"高风险": -0.15, "中高风险": -0.08, "中风险": 0, "中低风险": 0.05, "低风险": 0.10}
        base_prob += risk_adj.get(risk_level, 0)
        
        return max(0.3, min(0.95, base_prob))
    
    def _record_prediction(self, code: str, result: PredictionResult):
        """记录预测结果"""
        self._signal_history.append({
            'code': code,
            'timestamp': datetime.now(),
            'signal': result.signal,
            'confidence': result.confidence,
            'direction': result.direction,
        })
    
    def record_outcome(self, code: str, actual_return: float, prediction_correct: bool):
        """记录实际结果，用于评估准确度"""
        self._accuracy_records.append({
            'code': code,
            'timestamp': datetime.now(),
            'actual_return': actual_return,
            'prediction_correct': prediction_correct,
        })
    
    def get_accuracy_stats(self) -> Dict:
        """获取预测准确度统计"""
        if not self._accuracy_records:
            return {'total': 0, 'accuracy': 0, 'avg_return': 0}
        
        total = len(self._accuracy_records)
        correct = sum(1 for r in self._accuracy_records if r['prediction_correct'])
        avg_return = sum(r['actual_return'] for r in self._accuracy_records) / total
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0,
            'avg_return': avg_return,
        }


class PositionManager:
    """
    仓位管理器
    
    基于334法则：
    - 30% 底仓：长期持有，不做波段
    - 30% 律动仓：根据信号进行波段操作
    - 40% 预备队：应对极端风险，或追加机会
    """
    
    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.base_position = 0.30
        self.rhythm_position = 0.30
        self.reserve = 0.40
        
        self.positions: Dict[str, Dict] = {}
        self.cost_basis: Dict[str, float] = {}
    
    def calculate_lu_cost(self, code: str, current_price: float) -> float:
        """
        卢氏成本法
        
        通过律动操作降低持仓成本
        """
        if code not in self.positions:
            return current_price
        
        pos = self.positions[code]
        total_shares = pos.get('total_shares', 0)
        total_cost = pos.get('total_cost', 0)
        
        if total_shares == 0:
            return current_price
        
        return total_cost / total_shares
    
    def rhythm_buy(self, code: str, price: float, shares: int) -> Dict:
        """律动买入"""
        if code not in self.positions:
            self.positions[code] = {
                'total_shares': 0,
                'total_cost': 0,
                'base_shares': 0,
                'rhythm_shares': 0,
            }
        
        pos = self.positions[code]
        pos['total_shares'] += shares
        pos['total_cost'] += price * shares
        pos['rhythm_shares'] += shares
        
        new_cost = self.calculate_lu_cost(code, price)
        
        return {
            'action': 'rhythm_buy',
            'shares': shares,
            'price': price,
            'new_cost_basis': new_cost,
            'total_shares': pos['total_shares'],
        }
    
    def rhythm_sell(self, code: str, price: float, shares: int) -> Dict:
        """律动卖出"""
        if code not in self.positions:
            return {'error': '无持仓'}
        
        pos = self.positions[code]
        
        sell_shares = min(shares, pos['rhythm_shares'])
        
        cost = self.calculate_lu_cost(code, price)
        profit = (price - cost) * sell_shares
        
        pos['total_shares'] -= sell_shares
        pos['rhythm_shares'] -= sell_shares
        pos['total_cost'] -= cost * sell_shares
        
        return {
            'action': 'rhythm_sell',
            'shares': sell_shares,
            'price': price,
            'cost_basis': cost,
            'profit': profit,
            'profit_pct': (price - cost) / cost if cost > 0 else 0,
            'remaining_shares': pos['total_shares'],
        }
    
    def get_position_status(self, code: str) -> Dict:
        """获取持仓状态"""
        if code not in self.positions:
            return {'has_position': False}
        
        pos = self.positions[code]
        return {
            'has_position': True,
            'total_shares': pos['total_shares'],
            'base_shares': pos['base_shares'],
            'rhythm_shares': pos['rhythm_shares'],
            'cost_basis': self.calculate_lu_cost(code, 0),
        }


_prediction_engine = None


def get_prediction_engine() -> LuQiyuanPredictionEngine:
    """获取预测引擎单例"""
    global _prediction_engine
    if _prediction_engine is None:
        _prediction_engine = LuQiyuanPredictionEngine()
    return _prediction_engine
