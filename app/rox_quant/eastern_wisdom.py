#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
东方智慧量化系统
整合道家、儒家、孙子兵法、王阳明心学、周易、缠论、索罗斯反身性、复杂系统理论

核心思想体系：
1. 道家哲学 - 道法自然、无为而治、顺势而为
2. 儒家思想 - 中庸之道、过犹不及、恰到好处
3. 孙子兵法 - 知彼知己、避实击虚、兵不厌诈
4. 王阳明心学 - 知行合一、致良知、心即理
5. 周易 - 阴阳变化、否极泰来、周期规律
6. 缠论 - 缠中说禅、走势终完美、级别递归
7. 索罗斯反身性 - 认知影响现实、偏见强化趋势
8. 复杂系统理论 - 涌现、自组织、熵增定律
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


class YinYangState(Enum):
    """阴阳状态"""
    PURE_YANG = "纯阳"
    YANG_DOMINANT = "阳盛"
    YIN_YANG_BALANCE = "阴阳平衡"
    YIN_DOMINANT = "阴盛"
    PURE_YIN = "纯阴"


class TrendState(Enum):
    """趋势状态"""
    STRONG_UP = "强势上涨"
    WEAK_UP = "弱势上涨"
    SIDEWAYS = "横盘震荡"
    WEAK_DOWN = "弱势下跌"
    STRONG_DOWN = "强势下跌"


class ChanLunLevel(Enum):
    """缠论级别"""
    LEVEL_1 = "1分钟"
    LEVEL_5 = "5分钟"
    LEVEL_30 = "30分钟"
    LEVEL_DAY = "日线"
    LEVEL_WEEK = "周线"


@dataclass
class DaoistAnalysis:
    """道家哲学分析结果"""
    tao_alignment: float
    wu_wei_score: float
    natural_rhythm: str
    flow_direction: str
    counter_action_signal: bool
    advice: str


@dataclass
class ConfucianAnalysis:
    """儒家思想分析结果"""
    zhongyong_score: float
    excess_warning: bool
    deficiency_warning: bool
    golden_mean_position: float
    timing_score: float
    advice: str


@dataclass
class SunziAnalysis:
    """孙子兵法分析结果"""
    know_enemy_score: float
    know_self_score: float
    victory_probability: float
    avoid_strong_attack_weak: str
    terrain_advantage: float
    deception_signal: bool
    advice: str


@dataclass
class YangmingAnalysis:
    """王阳明心学分析结果"""
    unity_of_knowledge_action: float
    conscience_alignment: float
    mind_discipline_score: float
    action_consistency: float
    inner_wisdom: str
    advice: str


@dataclass
class IChingAnalysis:
    """周易分析结果"""
    yin_yang_state: YinYangState
    hexagram: str
    changing_line: int
    cycle_position: float
    tai_pai_signal: bool
    transformation_hint: str
    advice: str


@dataclass
class ChanLunAnalysis:
    """缠论分析结果"""
    current_level: ChanLunLevel
    bi_direction: str
    zhong_shu_status: str
    mai_dian_signal: str
    bei_chi_type: Optional[str]
    trend_perfection: bool
    three_buy_sell: Optional[str]
    advice: str


@dataclass
class SorosReflexivity:
    """索罗斯反身性分析结果"""
    bias_strength: float
    trend_reinforcement: float
    bubble_probability: float
    disconnection_degree: float
    turning_point_signal: bool
    feedback_loop: str
    advice: str


@dataclass
class ComplexityAnalysis:
    """复杂系统分析结果"""
    entropy_level: float
    emergence_signal: bool
    self_organization: float
    black_swan_probability: float
    system_stability: float
    critical_point_distance: float
    advice: str


@dataclass
class EasternWisdomPrediction:
    """东方智慧综合预测结果"""
    signal: str
    confidence: float
    direction: float
    
    daoist_score: float
    confucian_score: float
    sunzi_score: float
    yangming_score: float
    iching_score: float
    chanlun_score: float
    soros_score: float
    complexity_score: float
    
    reasoning: List[str]
    risk_level: str
    position_advice: str
    holding_period: int
    stop_loss_hint: str


class DaoistPhilosophy:
    """
    道家哲学分析
    
    核心思想：
    1. 道法自然 - 顺应市场规律
    2. 无为而治 - 不强行干预，顺势而为
    3. 反者道之动 - 物极必反，逆向思维
    4. 上善若水 - 灵活应变，不争而善胜
    """
    
    def analyze(self, price_data: pd.DataFrame, market_data: Dict) -> DaoistAnalysis:
        """道家哲学分析"""
        if price_data.empty:
            return self._default_result()
        
        close = price_data['close']
        volume = price_data.get('volume', pd.Series([1] * len(close)))
        
        tao_alignment = self._calculate_tao_alignment(close)
        
        wu_wei_score = self._calculate_wu_wei(close, volume)
        
        rhythm = self._analyze_natural_rhythm(close)
        
        flow = self._determine_flow_direction(close)
        
        counter_signal = self._detect_counter_action(close, tao_alignment)
        
        advice = self._generate_daoist_advice(tao_alignment, wu_wei_score, counter_signal)
        
        return DaoistAnalysis(
            tao_alignment=round(tao_alignment, 2),
            wu_wei_score=round(wu_wei_score, 2),
            natural_rhythm=rhythm,
            flow_direction=flow,
            counter_action_signal=counter_signal,
            advice=advice,
        )
    
    def _calculate_tao_alignment(self, close: pd.Series) -> float:
        """计算与道的契合度 - 趋势与自然的契合"""
        if len(close) < 20:
            return 50
        
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean() if len(close) >= 60 else ma20
        
        trend_alignment = 0
        for i in range(-5, 0):
            if ma20.iloc[i] > ma60.iloc[i]:
                trend_alignment += 1
            else:
                trend_alignment -= 1
        
        return 50 + trend_alignment * 10
    
    def _calculate_wu_wei(self, close: pd.Series, volume: pd.Series) -> float:
        """计算无为而治分数 - 减少不必要的操作"""
        if len(close) < 10:
            return 50
        
        volatility = close.pct_change().std()
        
        if volatility < 0.01:
            return 90
        elif volatility < 0.02:
            return 70
        elif volatility < 0.03:
            return 50
        else:
            return 30
    
    def _analyze_natural_rhythm(self, close: pd.Series) -> str:
        """分析自然韵律"""
        if len(close) < 20:
            return "韵律不明"
        
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        
        if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-5] < ma20.iloc[-5]:
            return "春生-趋势初起"
        elif ma5.iloc[-1] > ma20.iloc[-1]:
            return "夏长-趋势发展"
        elif ma5.iloc[-1] < ma20.iloc[-1] and ma5.iloc[-5] > ma20.iloc[-5]:
            return "秋收-趋势转折"
        else:
            return "冬藏-趋势衰退"
    
    def _determine_flow_direction(self, close: pd.Series) -> str:
        """确定流向 - 上善若水"""
        if len(close) < 5:
            return "静止"
        
        change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
        
        if change > 0.05:
            return "奔流向上"
        elif change > 0:
            return "缓缓流动"
        elif change > -0.05:
            return "回流"
        else:
            return "倾泻而下"
    
    def _detect_counter_action(self, close: pd.Series, tao_alignment: float) -> bool:
        """检测反向信号 - 反者道之动"""
        if len(close) < 20:
            return False
        
        rsi = self._calculate_rsi(close, 14)
        
        if tao_alignment > 80 and rsi > 80:
            return True
        if tao_alignment < 20 and rsi < 20:
            return True
        
        return False
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        if len(close) < period:
            return 50
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _generate_daoist_advice(self, tao: float, wu_wei: float, counter: bool) -> str:
        """生成道家建议"""
        if counter:
            return "反者道之动，物极必反，考虑逆向操作"
        elif tao > 70 and wu_wei > 60:
            return "道法自然，顺势而为，无需多虑"
        elif tao < 30:
            return "趋势不明，无为而治，静待时机"
        else:
            return "上善若水，灵活应变，不争而善胜"
    
    def _default_result(self) -> DaoistAnalysis:
        return DaoistAnalysis(
            tao_alignment=50,
            wu_wei_score=50,
            natural_rhythm="韵律不明",
            flow_direction="静止",
            counter_action_signal=False,
            advice="数据不足，静观其变",
        )


class ConfucianPhilosophy:
    """
    儒家思想分析
    
    核心思想：
    1. 中庸之道 - 不偏不倚，恰到好处
    2. 过犹不及 - 过度和不足都不好
    3. 时中 - 因时制宜，恰到好处
    4. 慎独 - 独处时也要谨慎
    """
    
    def analyze(self, price_data: pd.DataFrame, position_data: Dict) -> ConfucianAnalysis:
        """儒家思想分析"""
        if price_data.empty:
            return self._default_result()
        
        close = price_data['close']
        
        zhongyong = self._calculate_zhongyong(close)
        
        excess, deficiency = self._check_excess_deficiency(close)
        
        golden_mean = self._calculate_golden_mean_position(close)
        
        timing = self._calculate_timing_score(close)
        
        advice = self._generate_confucian_advice(zhongyong, excess, deficiency)
        
        return ConfucianAnalysis(
            zhongyong_score=round(zhongyong, 2),
            excess_warning=excess,
            deficiency_warning=deficiency,
            golden_mean_position=round(golden_mean, 2),
            timing_score=round(timing, 2),
            advice=advice,
        )
    
    def _calculate_zhongyong(self, close: pd.Series) -> float:
        """计算中庸分数"""
        if len(close) < 20:
            return 50
        
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        
        deviation = abs(close.iloc[-1] - ma20.iloc[-1]) / std20.iloc[-1] if std20.iloc[-1] > 0 else 0
        
        if deviation < 0.5:
            return 90
        elif deviation < 1:
            return 70
        elif deviation < 2:
            return 50
        else:
            return 30
    
    def _check_excess_deficiency(self, close: pd.Series) -> Tuple[bool, bool]:
        """检查过犹不及"""
        if len(close) < 20:
            return False, False
        
        rsi = self._calculate_rsi(close)
        
        excess = rsi > 75
        deficiency = rsi < 25
        
        return excess, deficiency
    
    def _calculate_golden_mean_position(self, close: pd.Series) -> float:
        """计算中位位置"""
        if len(close) < 60:
            return 50
        
        high = close.rolling(60).max().iloc[-1]
        low = close.rolling(60).min().iloc[-1]
        current = close.iloc[-1]
        
        if high == low:
            return 50
        
        position = (current - low) / (high - low) * 100
        
        return position
    
    def _calculate_timing_score(self, close: pd.Series) -> float:
        """计算时中分数"""
        if len(close) < 20:
            return 50
        
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        
        cross_distance = abs(ma5.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1]
        
        if cross_distance < 0.01:
            return 90
        elif cross_distance < 0.02:
            return 70
        else:
            return 50
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _generate_confucian_advice(self, zhongyong: float, excess: bool, deficiency: bool) -> str:
        """生成儒家建议"""
        if excess:
            return "过犹不及，过度上涨需警惕回调"
        if deficiency:
            return "过犹不及，过度下跌或存反弹机会"
        if zhongyong > 70:
            return "中庸之道，位置适中，可稳健操作"
        else:
            return "时中为贵，等待更好的时机"
    
    def _default_result(self) -> ConfucianAnalysis:
        return ConfucianAnalysis(
            zhongyong_score=50,
            excess_warning=False,
            deficiency_warning=False,
            golden_mean_position=50,
            timing_score=50,
            advice="数据不足，谨慎为上",
        )


class SunziArtOfWar:
    """
    孙子兵法分析
    
    核心思想：
    1. 知彼知己，百战不殆
    2. 避实击虚
    3. 兵不厌诈
    4. 善战者，求之于势
    5. 不战而屈人之兵
    """
    
    def analyze(
        self,
        price_data: pd.DataFrame,
        market_data: Dict,
        position_data: Dict,
    ) -> SunziAnalysis:
        """孙子兵法分析"""
        if price_data.empty:
            return self._default_result()
        
        know_enemy = self._analyze_enemy(market_data)
        
        know_self = self._analyze_self(position_data)
        
        victory_prob = self._calculate_victory_probability(know_enemy, know_self)
        
        avoid_attack = self._analyze_avoid_strong_attack_weak(price_data)
        
        terrain = self._analyze_terrain(price_data)
        
        deception = self._detect_deception(price_data, market_data)
        
        advice = self._generate_sunzi_advice(victory_prob, avoid_attack, deception)
        
        return SunziAnalysis(
            know_enemy_score=round(know_enemy, 2),
            know_self_score=round(know_self, 2),
            victory_probability=round(victory_prob, 2),
            avoid_strong_attack_weak=avoid_attack,
            terrain_advantage=round(terrain, 2),
            deception_signal=deception,
            advice=advice,
        )
    
    def _analyze_enemy(self, market_data: Dict) -> float:
        """分析敌人（市场）"""
        score = 50
        
        trend = market_data.get('trend', 0)
        if abs(trend) > 0.02:
            score += 20
        elif abs(trend) > 0.01:
            score += 10
        
        volume_ratio = market_data.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            score += 15
        
        return min(100, score)
    
    def _analyze_self(self, position_data: Dict) -> float:
        """分析自己（持仓）"""
        score = 50
        
        position_ratio = position_data.get('position_ratio', 0)
        if 0.3 <= position_ratio <= 0.6:
            score += 20
        
        profit_ratio = position_data.get('profit_ratio', 0)
        if profit_ratio > 0:
            score += 15
        elif profit_ratio < -0.05:
            score -= 10
        
        return min(100, max(0, score))
    
    def _calculate_victory_probability(self, enemy: float, self_score: float) -> float:
        """计算胜率"""
        return (enemy + self_score) / 2 / 100
    
    def _analyze_avoid_strong_attack_weak(self, price_data: pd.DataFrame) -> str:
        """避实击虚分析"""
        if len(price_data) < 20:
            return "虚实难辨"
        
        close = price_data['close']
        ma20 = close.rolling(20).mean()
        
        deviation = (close.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1]
        
        if deviation > 0.1:
            return "敌方强势，避其锋芒"
        elif deviation < -0.1:
            return "敌方虚弱，可击其软肋"
        else:
            return "虚实相当，静观其变"
    
    def _analyze_terrain(self, price_data: pd.DataFrame) -> float:
        """分析地形（支撑阻力）"""
        if len(price_data) < 60:
            return 50
        
        close = price_data['close']
        high = close.rolling(60).max().iloc[-1]
        low = close.rolling(60).min().iloc[-1]
        current = close.iloc[-1]
        
        position = (current - low) / (high - low) if high != low else 0.5
        
        if position < 0.3:
            return 80
        elif position > 0.7:
            return 20
        else:
            return 50
    
    def _detect_deception(self, price_data: pd.DataFrame, market_data: Dict) -> bool:
        """检测兵不厌诈"""
        if len(price_data) < 10:
            return False
        
        close = price_data['close']
        volume = price_data.get('volume', pd.Series([1] * len(close)))
        
        price_change = close.pct_change().iloc[-1]
        volume_change = volume.pct_change().iloc[-1] if len(volume) > 1 else 0
        
        if price_change > 0.03 and volume_change < -0.2:
            return True
        if price_change < -0.03 and volume_change < -0.2:
            return True
        
        return False
    
    def _generate_sunzi_advice(self, victory: float, avoid_attack: str, deception: bool) -> str:
        """生成兵法建议"""
        if deception:
            return "兵不厌诈，警惕假突破，不可轻信"
        if victory > 0.7:
            return "知彼知己，胜券在握，可果断出击"
        elif victory > 0.5:
            return f"{avoid_attack}，谨慎为上"
        else:
            return "敌情不明，不可轻进，以守为攻"
    
    def _default_result(self) -> SunziAnalysis:
        return SunziAnalysis(
            know_enemy_score=50,
            know_self_score=50,
            victory_probability=0.5,
            avoid_strong_attack_weak="虚实难辨",
            terrain_advantage=50,
            deception_signal=False,
            advice="情报不足，不可轻战",
        )


class YangmingMindPhilosophy:
    """
    王阳明心学分析
    
    核心思想：
    1. 知行合一 - 认知与行动统一
    2. 致良知 - 发挥内心良知
    3. 心即理 - 心外无理
    4. 事上磨练 - 在实践中修行
    """
    
    def analyze(
        self,
        trade_history: List[Dict],
        current_plan: Dict,
        market_view: Dict,
    ) -> YangmingAnalysis:
        """王阳明心学分析"""
        unity = self._calculate_unity_of_knowledge_action(trade_history)
        
        conscience = self._calculate_conscience_alignment(trade_history)
        
        discipline = self._calculate_mind_discipline(trade_history)
        
        consistency = self._calculate_action_consistency(current_plan, market_view)
        
        wisdom = self._derive_inner_wisdom(unity, conscience, discipline)
        
        advice = self._generate_yangming_advice(unity, discipline, consistency)
        
        return YangmingAnalysis(
            unity_of_knowledge_action=round(unity, 2),
            conscience_alignment=round(conscience, 2),
            mind_discipline_score=round(discipline, 2),
            action_consistency=round(consistency, 2),
            inner_wisdom=wisdom,
            advice=advice,
        )
    
    def _calculate_unity_of_knowledge_action(self, history: List[Dict]) -> float:
        """计算知行合一程度"""
        if not history:
            return 50
        
        consistent_count = 0
        for trade in history[-20:]:
            planned = trade.get('planned_action', '')
            actual = trade.get('actual_action', '')
            if planned == actual:
                consistent_count += 1
        
        return consistent_count / min(len(history), 20) * 100
    
    def _calculate_conscience_alignment(self, history: List[Dict]) -> float:
        """计算良知契合度"""
        if not history:
            return 50
        
        good_trades = 0
        for trade in history[-20:]:
            profit = trade.get('profit', 0)
            risk_taken = trade.get('risk_taken', 0)
            
            if profit > 0 and risk_taken < 0.1:
                good_trades += 1
            elif profit < 0 and risk_taken > 0.1:
                good_trades += 0.5
        
        return good_trades / min(len(history), 20) * 100
    
    def _calculate_mind_discipline(self, history: List[Dict]) -> float:
        """计算心性修养"""
        if not history:
            return 50
        
        disciplined_count = 0
        for trade in history[-20:]:
            stop_loss = trade.get('stop_loss_triggered', False)
            take_profit = trade.get('take_profit_triggered', False)
            emotional = trade.get('emotional_trade', False)
            
            if not emotional:
                disciplined_count += 1
            if stop_loss or take_profit:
                disciplined_count += 0.5
        
        return min(100, disciplined_count / min(len(history), 20) * 100)
    
    def _calculate_action_consistency(self, plan: Dict, view: Dict) -> float:
        """计算行动一致性"""
        planned_direction = plan.get('direction', 0)
        view_direction = view.get('direction', 0)
        
        if planned_direction * view_direction > 0:
            return 90
        elif planned_direction * view_direction == 0:
            return 50
        else:
            return 20
    
    def _derive_inner_wisdom(self, unity: float, conscience: float, discipline: float) -> str:
        """推导内心智慧"""
        avg = (unity + conscience + discipline) / 3
        
        if avg > 80:
            return "良知清明，心性纯正，可信任直觉"
        elif avg > 60:
            return "心性尚可，需继续事上磨练"
        else:
            return "心性未定，需加强修养，不可妄动"
    
    def _generate_yangming_advice(self, unity: float, discipline: float, consistency: float) -> str:
        """生成心学建议"""
        if unity < 50:
            return "知行不一，当反思认知与行动的差距"
        elif discipline < 50:
            return "心性不坚，需事上磨练，增强定力"
        elif consistency < 50:
            return "计划与判断相悖，当审视内心真实想法"
        else:
            return "知行合一，致良知，可依心而行"


class IChingAnalyzer:
    """
    周易分析
    
    核心思想：
    1. 阴阳变化 - 万物负阴而抱阳
    2. 否极泰来 - 物极必反
    3. 周期规律 - 循环往复
    4. 变易不易 - 变中有常
    """
    
    TRIGRAMS = {
        '乾': [1, 1, 1],
        '坤': [0, 0, 0],
        '震': [0, 0, 1],
        '艮': [1, 0, 0],
        '离': [1, 0, 1],
        '坎': [0, 1, 0],
        '兑': [0, 1, 1],
        '巽': [1, 1, 0],
    }
    
    HEXAGRAMS = {
        '乾乾': '元亨利贞',
        '坤坤': '厚德载物',
        '泰': '小往大来',
        '否': '大往小来',
        '既济': '事已成',
        '未济': '事未成',
    }
    
    def analyze(self, price_data: pd.DataFrame) -> IChingAnalysis:
        """周易分析"""
        if price_data.empty or len(price_data) < 60:
            return self._default_result()
        
        close = price_data['close']
        
        yin_yang = self._determine_yin_yang_state(close)
        
        hexagram = self._calculate_hexagram(close)
        
        changing_line = self._determine_changing_line(close)
        
        cycle_pos = self._calculate_cycle_position(close)
        
        tai_pai = self._detect_tai_pai_signal(close, yin_yang)
        
        transformation = self._get_transformation_hint(yin_yang, changing_line)
        
        advice = self._generate_iching_advice(yin_yang, tai_pai, hexagram)
        
        return IChingAnalysis(
            yin_yang_state=yin_yang,
            hexagram=hexagram,
            changing_line=changing_line,
            cycle_position=round(cycle_pos, 2),
            tai_pai_signal=tai_pai,
            transformation_hint=transformation,
            advice=advice,
        )
    
    def _determine_yin_yang_state(self, close: pd.Series) -> YinYangState:
        """确定阴阳状态"""
        if len(close) < 20:
            return YinYangState.YIN_YANG_BALANCE
        
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean() if len(close) >= 60 else ma20
        
        current = close.iloc[-1]
        
        if current > ma5.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
            return YinYangState.PURE_YANG
        elif current > ma20.iloc[-1] and ma5.iloc[-1] > ma20.iloc[-1]:
            return YinYangState.YANG_DOMINANT
        elif current < ma5.iloc[-1] < ma20.iloc[-1] < ma60.iloc[-1]:
            return YinYangState.PURE_YIN
        elif current < ma20.iloc[-1] and ma5.iloc[-1] < ma20.iloc[-1]:
            return YinYangState.YIN_DOMINANT
        else:
            return YinYangState.YIN_YANG_BALANCE
    
    def _calculate_hexagram(self, close: pd.Series) -> str:
        """计算卦象"""
        if len(close) < 60:
            return "未济"
        
        changes = close.pct_change().tail(6)
        
        lines = []
        for change in changes:
            if change > 0:
                lines.append(1)
            else:
                lines.append(0)
        
        lower = lines[:3]
        upper = lines[3:]
        
        lower_trigram = self._get_trigram_name(lower)
        upper_trigram = self._get_trigram_name(upper)
        
        return f"{upper_trigram}{lower_trigram}"
    
    def _get_trigram_name(self, lines: List[int]) -> str:
        """获取卦名"""
        for name, pattern in self.TRIGRAMS.items():
            if lines == pattern:
                return name
        return "坎"
    
    def _determine_changing_line(self, close: pd.Series) -> int:
        """确定变爻"""
        if len(close) < 14:
            return 0
        
        rsi = self._calculate_rsi(close)
        
        if rsi > 80:
            return 6
        elif rsi > 70:
            return 5
        elif rsi > 60:
            return 4
        elif rsi < 20:
            return 1
        elif rsi < 30:
            return 2
        elif rsi < 40:
            return 3
        else:
            return 0
    
    def _calculate_cycle_position(self, close: pd.Series) -> float:
        """计算周期位置"""
        if len(close) < 60:
            return 50
        
        high = close.rolling(60).max().iloc[-1]
        low = close.rolling(60).min().iloc[-1]
        current = close.iloc[-1]
        
        if high == low:
            return 50
        
        return (current - low) / (high - low) * 100
    
    def _detect_tai_pai_signal(self, close: pd.Series, state: YinYangState) -> bool:
        """检测否极泰来信号"""
        if state == YinYangState.PURE_YIN:
            return True
        if state == YinYangState.PURE_YANG:
            return True
        return False
    
    def _get_transformation_hint(self, state: YinYangState, changing: int) -> str:
        """获取变化提示"""
        if state == YinYangState.PURE_YIN:
            return "阴极阳生，否极泰来"
        elif state == YinYangState.PURE_YANG:
            return "阳极阴生，泰极否来"
        elif changing > 0:
            return f"第{changing}爻动，变数将生"
        else:
            return "阴阳调和，变化未显"
    
    def _generate_iching_advice(self, state: YinYangState, tai_pai: bool, hexagram: str) -> str:
        """生成周易建议"""
        if tai_pai:
            if state == YinYangState.PURE_YIN:
                return "否极泰来，阴极阳生，可考虑建仓"
            else:
                return "泰极否来，阳极阴生，需警惕风险"
        
        if "乾" in hexagram:
            return "乾卦刚健，自强不息，可积极进取"
        elif "坤" in hexagram:
            return "坤卦厚德，顺势而为，宜稳健保守"
        elif "泰" in hexagram:
            return "泰卦通达，小往大来，形势向好"
        elif "否" in hexagram:
            return "否卦闭塞，大往小来，宜谨慎观望"
        else:
            return "观象玩辞，审时度势"
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _default_result(self) -> 'IChingAnalysis':
        return IChingAnalysis(
            yin_yang_state=YinYangState.YIN_YANG_BALANCE,
            hexagram="未济",
            changing_line=0,
            cycle_position=50,
            tai_pai_signal=False,
            transformation_hint="数据不足",
            advice="观象待时",
        )


class ChanLunTheory:
    """
    缠论分析
    
    核心思想：
    1. 走势终完美 - 任何走势都会完成
    2. 级别递归 - 从小级别到大级别
    3. 中枢 - 走势的核心结构
    4. 买卖点 - 三类买卖点
    5. 背驰 - 趋势转折的信号
    """
    
    def __init__(self):
        self.min_bi_length = 4
        self.min_zhongshu_length = 6
    
    def analyze(self, price_data: pd.DataFrame, level: ChanLunLevel = ChanLunLevel.LEVEL_DAY) -> ChanLunAnalysis:
        """缠论分析"""
        if price_data.empty or len(price_data) < 30:
            return self._default_result()
        
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        bi_direction = self._determine_bi_direction(high, low)
        
        zhongshu_status = self._analyze_zhongshu(high, low)
        
        maidian = self._detect_mai_dian(high, low, close)
        
        beichi = self._detect_beichi(close, high, low)
        
        trend_perfect = self._check_trend_perfection(high, low, close)
        
        three_signal = self._detect_three_buy_sell(high, low, close)
        
        advice = self._generate_chanlun_advice(maidian, beichi, three_signal)
        
        return ChanLunAnalysis(
            current_level=level,
            bi_direction=bi_direction,
            zhong_shu_status=zhongshu_status,
            mai_dian_signal=maidian,
            bei_chi_type=beichi,
            trend_perfection=trend_perfect,
            three_buy_sell=three_signal,
            advice=advice,
        )
    
    def _determine_bi_direction(self, high: pd.Series, low: pd.Series) -> str:
        """确定笔方向"""
        if len(high) < 5:
            return "方向不明"
        
        recent_high = high.iloc[-5:].max()
        recent_low = low.iloc[-5:].min()
        
        current = high.iloc[-1]
        
        if current >= recent_high * 0.99:
            return "向上笔"
        elif current <= recent_low * 1.01:
            return "向下笔"
        else:
            return "笔震荡"
    
    def _analyze_zhongshu(self, high: pd.Series, low: pd.Series) -> str:
        """分析中枢"""
        if len(high) < 20:
            return "中枢未形成"
        
        recent_high = high.iloc[-20:]
        recent_low = low.iloc[-20:]
        
        zg = recent_low.min()
        zd = recent_high.max()
        
        if zg < zd:
            return "中枢震荡中"
        else:
            return "中枢移动中"
    
    def _detect_mai_dian(self, high: pd.Series, low: pd.Series, close: pd.Series) -> str:
        """检测买卖点"""
        if len(close) < 20:
            return "无明确买卖点"
        
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        
        if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-2] <= ma20.iloc[-2]:
            return "一买信号"
        elif ma5.iloc[-1] < ma20.iloc[-1] and ma5.iloc[-2] >= ma20.iloc[-2]:
            return "一卖信号"
        elif close.iloc[-1] > high.iloc[-20:-1].max():
            return "二买信号"
        elif close.iloc[-1] < low.iloc[-20:-1].min():
            return "二卖信号"
        else:
            return "无明确买卖点"
    
    def _detect_beichi(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Optional[str]:
        """检测背驰"""
        if len(close) < 30:
            return None
        
        macd = self._calculate_macd(close)
        
        price_trend = close.iloc[-1] - close.iloc[-10]
        macd_trend = macd[-1] - macd[-10] if len(macd) >= 10 else 0
        
        if price_trend > 0 and macd_trend < 0:
            return "顶背驰"
        elif price_trend < 0 and macd_trend > 0:
            return "底背驰"
        else:
            return None
    
    def _calculate_macd(self, close: pd.Series) -> List[float]:
        """计算MACD"""
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        macd = (dif - dea) * 2
        
        return macd.tolist()
    
    def _check_trend_perfection(self, high: pd.Series, low: pd.Series, close: pd.Series) -> bool:
        """检查走势是否完美"""
        if len(close) < 20:
            return False
        
        ma20 = close.rolling(20).mean()
        
        deviation = abs(close.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1]
        
        return deviation < 0.05
    
    def _detect_three_buy_sell(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Optional[str]:
        """检测三类买卖点"""
        if len(close) < 60:
            return None
        
        high_60 = high.iloc[-60:].max()
        low_60 = low.iloc[-60:].min()
        current = close.iloc[-1]
        
        if current > high_60 * 0.98:
            return "三买"
        elif current < low_60 * 1.02:
            return "三卖"
        
        return None
    
    def _generate_chanlun_advice(self, maidian: str, beichi: Optional[str], three: Optional[str]) -> str:
        """生成缠论建议"""
        if beichi == "底背驰":
            return "底背驰出现，走势终完美，可考虑买入"
        elif beichi == "顶背驰":
            return "顶背驰出现，趋势或将转折，注意风险"
        elif "买" in maidian:
            return f"{maidian}，可考虑建仓"
        elif "卖" in maidian:
            return f"{maidian}，可考虑减仓"
        elif three:
            return f"{three}信号，突破确认"
        else:
            return "走势未明，等待买卖点"
    
    def _default_result(self) -> ChanLunAnalysis:
        return ChanLunAnalysis(
            current_level=ChanLunLevel.LEVEL_DAY,
            bi_direction="方向不明",
            zhong_shu_status="中枢未形成",
            mai_dian_signal="无明确买卖点",
            bei_chi_type=None,
            trend_perfection=False,
            three_buy_sell=None,
            advice="数据不足，走势未明",
        )


class SorosReflexivityAnalyzer:
    """
    索罗斯反身性分析
    
    核心思想：
    1. 认知影响现实 - 市场参与者的偏见影响价格
    2. 反身性循环 - 偏见与现实的相互作用
    3. 泡沫形成 - 正反馈循环
    4. 趋势转折 - 负反馈或认知修正
    """
    
    def analyze(
        self,
        price_data: pd.DataFrame,
        sentiment_data: Dict,
        fundamental_data: Dict,
    ) -> SorosReflexivity:
        """索罗斯反身性分析"""
        if price_data.empty:
            return self._default_result()
        
        close = price_data['close']
        
        bias = self._calculate_bias_strength(close, sentiment_data)
        
        reinforcement = self._calculate_trend_reinforcement(close)
        
        bubble_prob = self._calculate_bubble_probability(close, fundamental_data)
        
        disconnection = self._calculate_disconnection(close, fundamental_data)
        
        turning_point = self._detect_turning_point(close, bias, reinforcement)
        
        feedback = self._determine_feedback_loop(bias, reinforcement)
        
        advice = self._generate_soros_advice(bubble_prob, turning_point, feedback)
        
        return SorosReflexivity(
            bias_strength=round(bias, 2),
            trend_reinforcement=round(reinforcement, 2),
            bubble_probability=round(bubble_prob, 2),
            disconnection_degree=round(disconnection, 2),
            turning_point_signal=turning_point,
            feedback_loop=feedback,
            advice=advice,
        )
    
    def _calculate_bias_strength(self, close: pd.Series, sentiment: Dict) -> float:
        """计算偏见强度"""
        if len(close) < 20:
            return 0
        
        price_momentum = close.pct_change(20).iloc[-1]
        sentiment_score = sentiment.get('score', 0.5)
        
        bias = abs(price_momentum * 10) + abs(sentiment_score - 0.5) * 50
        
        return min(100, bias)
    
    def _calculate_trend_reinforcement(self, close: pd.Series) -> float:
        """计算趋势强化程度"""
        if len(close) < 60:
            return 0
        
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        
        ma_deviation = (ma20.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1]
        
        return min(100, abs(ma_deviation) * 500)
    
    def _calculate_bubble_probability(self, close: pd.Series, fundamental: Dict) -> float:
        """计算泡沫概率"""
        if len(close) < 60:
            return 0
        
        pe = fundamental.get('pe', 15)
        pb = fundamental.get('pb', 1.5)
        
        price_increase = (close.iloc[-1] - close.iloc[-60]) / close.iloc[-60]
        
        bubble_score = 0
        
        if pe > 50:
            bubble_score += 30
        elif pe > 30:
            bubble_score += 15
        
        if pb > 5:
            bubble_score += 30
        elif pb > 3:
            bubble_score += 15
        
        if price_increase > 0.5:
            bubble_score += 40
        elif price_increase > 0.3:
            bubble_score += 20
        
        return min(100, bubble_score)
    
    def _calculate_disconnection(self, close: pd.Series, fundamental: Dict) -> float:
        """计算价格与基本面的脱节程度"""
        intrinsic_value = fundamental.get('intrinsic_value', close.iloc[-1])
        current_price = close.iloc[-1]
        
        disconnection = abs(current_price - intrinsic_value) / intrinsic_value * 100
        
        return min(100, disconnection)
    
    def _detect_turning_point(self, close: pd.Series, bias: float, reinforcement: float) -> bool:
        """检测转折点"""
        if bias > 80 and reinforcement > 70:
            return True
        if bias > 70 and reinforcement > 80:
            return True
        return False
    
    def _determine_feedback_loop(self, bias: float, reinforcement: float) -> str:
        """确定反馈循环类型"""
        if bias > 60 and reinforcement > 60:
            return "正反馈循环-趋势自我强化"
        elif bias < 30 and reinforcement < 30:
            return "负反馈循环-趋势自我修正"
        else:
            return "反馈平衡"
    
    def _generate_soros_advice(self, bubble: float, turning: bool, feedback: str) -> str:
        """生成索罗斯建议"""
        if turning:
            return "反身性转折点可能到来，警惕趋势反转"
        if bubble > 70:
            return "泡沫风险高，认知与现实严重脱节"
        if "正反馈" in feedback:
            return "正反馈循环中，趋势自我强化，可顺势而为"
        elif "负反馈" in feedback:
            return "负反馈修正中，趋势可能逆转"
        else:
            return "市场相对理性，偏见影响有限"
    
    def _default_result(self) -> SorosReflexivity:
        return SorosReflexivity(
            bias_strength=0,
            trend_reinforcement=0,
            bubble_probability=0,
            disconnection_degree=0,
            turning_point_signal=False,
            feedback_loop="数据不足",
            advice="无法判断",
        )


class ComplexitySystem:
    """
    复杂系统分析
    
    核心思想：
    1. 熵增定律 - 系统趋向无序
    2. 涌现现象 - 整体大于部分之和
    3. 自组织 - 系统自我调节
    4. 黑天鹅 - 极端事件不可预测
    5. 临界点 - 系统相变
    """
    
    def analyze(
        self,
        price_data: pd.DataFrame,
        market_data: Dict,
        system_data: Dict,
    ) -> ComplexityAnalysis:
        """复杂系统分析"""
        if price_data.empty:
            return self._default_result()
        
        close = price_data['close']
        
        entropy = self._calculate_entropy(close)
        
        emergence = self._detect_emergence(close, market_data)
        
        self_org = self._calculate_self_organization(close)
        
        black_swan = self._calculate_black_swan_probability(close, market_data)
        
        stability = self._calculate_system_stability(close)
        
        critical_dist = self._calculate_critical_point_distance(close)
        
        advice = self._generate_complexity_advice(entropy, black_swan, stability)
        
        return ComplexityAnalysis(
            entropy_level=round(entropy, 2),
            emergence_signal=emergence,
            self_organization=round(self_org, 2),
            black_swan_probability=round(black_swan, 2),
            system_stability=round(stability, 2),
            critical_point_distance=round(critical_dist, 2),
            advice=advice,
        )
    
    def _calculate_entropy(self, close: pd.Series) -> float:
        """计算系统熵"""
        if len(close) < 20:
            return 50
        
        returns = close.pct_change().dropna()
        
        if returns.empty or len(returns) < 10:
            return 50
        
        try:
            hist, _ = np.histogram(returns, bins=20, density=True)
            hist = hist[hist > 0]
            
            if len(hist) == 0:
                return 50
            
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            normalized = min(100, entropy * 20)
            
            return normalized
        except:
            return 50
    
    def _detect_emergence(self, close: pd.Series, market: Dict) -> bool:
        """检测涌现现象"""
        if len(close) < 30:
            return False
        
        volume = market.get('volume_ratio', 1)
        volatility = close.pct_change().std()
        
        if volume > 2 and volatility > 0.03:
            return True
        
        return False
    
    def _calculate_self_organization(self, close: pd.Series) -> float:
        """计算自组织程度"""
        if len(close) < 60:
            return 50
        
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        
        alignment = 0
        if (ma5.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]) or \
           (ma5.iloc[-1] < ma20.iloc[-1] < ma60.iloc[-1]):
            alignment = 80
        else:
            alignment = 40
        
        return alignment
    
    def _calculate_black_swan_probability(self, close: pd.Series, market: Dict) -> float:
        """计算黑天鹅概率"""
        if len(close) < 60:
            return 10
        
        returns = close.pct_change()
        
        std = returns.std()
        recent_std = returns.iloc[-20:].std()
        
        vol_increase = recent_std / std if std > 0 else 1
        
        correlation = market.get('market_correlation', 0.5)
        
        black_swan_prob = 10
        
        if vol_increase > 2:
            black_swan_prob += 30
        elif vol_increase > 1.5:
            black_swan_prob += 15
        
        if correlation > 0.9:
            black_swan_prob += 20
        
        return min(100, black_swan_prob)
    
    def _calculate_system_stability(self, close: pd.Series) -> float:
        """计算系统稳定性"""
        if len(close) < 30:
            return 50
        
        returns = close.pct_change()
        
        volatility = returns.std()
        
        if volatility < 0.01:
            return 90
        elif volatility < 0.02:
            return 70
        elif volatility < 0.03:
            return 50
        else:
            return 30
    
    def _calculate_critical_point_distance(self, close: pd.Series) -> float:
        """计算距离临界点的距离"""
        if len(close) < 60:
            return 50
        
        high = close.rolling(60).max().iloc[-1]
        low = close.rolling(60).min().iloc[-1]
        current = close.iloc[-1]
        
        range_val = high - low
        
        if range_val == 0:
            return 50
        
        dist_to_high = (high - current) / range_val * 100
        dist_to_low = (current - low) / range_val * 100
        
        return min(dist_to_high, dist_to_low)
    
    def _generate_complexity_advice(self, entropy: float, black_swan: float, stability: float) -> str:
        """生成复杂系统建议"""
        if entropy > 80:
            return "系统熵增严重，混乱度高，需谨慎"
        if black_swan > 50:
            return "黑天鹅风险上升，系统可能发生相变"
        if stability < 40:
            return "系统稳定性差，波动加剧"
        if stability > 80:
            return "系统稳定，但需警惕临界点突破"
        return "系统相对稳定，可正常操作"
    
    def _default_result(self) -> ComplexityAnalysis:
        return ComplexityAnalysis(
            entropy_level=50,
            emergence_signal=False,
            self_organization=50,
            black_swan_probability=10,
            system_stability=50,
            critical_point_distance=50,
            advice="数据不足",
        )


class EasternWisdomEngine:
    """
    东方智慧综合预测引擎
    
    整合八大思想体系：
    1. 道家哲学 - 顺势而为
    2. 儒家思想 - 中庸之道
    3. 孙子兵法 - 知彼知己
    4. 王阳明心学 - 知行合一
    5. 周易 - 阴阳变化
    6. 缠论 - 走势终完美
    7. 索罗斯反身性 - 认知影响现实
    8. 复杂系统理论 - 熵增与涌现
    """
    
    def __init__(self):
        self.daoist = DaoistPhilosophy()
        self.confucian = ConfucianPhilosophy()
        self.sunzi = SunziArtOfWar()
        self.yangming = YangmingMindPhilosophy()
        self.iching = IChingAnalyzer()
        self.chanlun = ChanLunTheory()
        self.soros = SorosReflexivityAnalyzer()
        self.complexity = ComplexitySystem()
        
        self.weights = {
            'daoist': 0.15,
            'confucian': 0.10,
            'sunzi': 0.15,
            'yangming': 0.10,
            'iching': 0.15,
            'chanlun': 0.15,
            'soros': 0.10,
            'complexity': 0.10,
        }
    
    def predict(
        self,
        code: str,
        price_data: pd.DataFrame,
        market_data: Dict,
        position_data: Dict,
        fundamental_data: Dict,
        trade_history: List[Dict],
    ) -> EasternWisdomPrediction:
        """综合预测"""
        daoist_result = self.daoist.analyze(price_data, market_data)
        confucian_result = self.confucian.analyze(price_data, position_data)
        sunzi_result = self.sunzi.analyze(price_data, market_data, position_data)
        yangming_result = self.yangming.analyze(trade_history, position_data, market_data)
        iching_result = self.iching.analyze(price_data)
        chanlun_result = self.chanlun.analyze(price_data)
        soros_result = self.soros.analyze(price_data, market_data, fundamental_data)
        complexity_result = self.complexity.analyze(price_data, market_data, {})
        
        daoist_score = self._daoist_to_score(daoist_result)
        confucian_score = self._confucian_to_score(confucian_result)
        sunzi_score = self._sunzi_to_score(sunzi_result)
        yangming_score = self._yangming_to_score(yangming_result)
        iching_score = self._iching_to_score(iching_result)
        chanlun_score = self._chanlun_to_score(chanlun_result)
        soros_score = self._soros_to_score(soros_result)
        complexity_score = self._complexity_to_score(complexity_result)
        
        final_score = (
            daoist_score * self.weights['daoist'] +
            confucian_score * self.weights['confucian'] +
            sunzi_score * self.weights['sunzi'] +
            yangming_score * self.weights['yangming'] +
            iching_score * self.weights['iching'] +
            chanlun_score * self.weights['chanlun'] +
            soros_score * self.weights['soros'] +
            complexity_score * self.weights['complexity']
        )
        
        signal = self._score_to_signal(final_score)
        confidence = self._calculate_confidence(
            daoist_result, confucian_result, sunzi_result,
            yangming_result, iching_result, chanlun_result,
            soros_result, complexity_result
        )
        
        reasoning = [
            f"【道家】道法自然{daoist_result.tao_alignment}分，{daoist_result.advice}",
            f"【儒家】中庸之道{confucian_result.zhongyong_score}分，{confucian_result.advice}",
            f"【孙子】知彼知己{sunzi_result.victory_probability:.0%}，{sunzi_result.advice}",
            f"【心学】知行合一{yangming_result.unity_of_knowledge_action}分，{yangming_result.advice}",
            f"【周易】{iching_result.yin_yang_state.value}，{iching_result.advice}",
            f"【缠论】{chanlun_result.bi_direction}，{chanlun_result.advice}",
            f"【反身性】泡沫概率{soros_result.bubble_probability:.0%}，{soros_result.advice}",
            f"【复杂系统】熵{complexity_result.entropy_level}，{complexity_result.advice}",
        ]
        
        risk = self._assess_risk(complexity_result, soros_result, iching_result)
        position = self._generate_position_advice(final_score, risk)
        holding = self._calculate_holding_period(iching_result, chanlun_result)
        stop_loss = self._generate_stop_loss_hint(complexity_result, soros_result)
        
        return EasternWisdomPrediction(
            signal=signal,
            confidence=round(confidence, 2),
            direction=round(final_score, 2),
            daoist_score=round(daoist_score, 2),
            confucian_score=round(confucian_score, 2),
            sunzi_score=round(sunzi_score, 2),
            yangming_score=round(yangming_score, 2),
            iching_score=round(iching_score, 2),
            chanlun_score=round(chanlun_score, 2),
            soros_score=round(soros_score, 2),
            complexity_score=round(complexity_score, 2),
            reasoning=reasoning,
            risk_level=risk,
            position_advice=position,
            holding_period=holding,
            stop_loss_hint=stop_loss,
        )
    
    def _daoist_to_score(self, result: DaoistAnalysis) -> float:
        if result.counter_action_signal:
            return -3 if result.tao_alignment > 70 else 3
        return (result.tao_alignment - 50) / 10
    
    def _confucian_to_score(self, result: ConfucianAnalysis) -> float:
        if result.excess_warning:
            return -2
        if result.deficiency_warning:
            return 2
        return (result.zhongyong_score - 50) / 25
    
    def _sunzi_to_score(self, result: SunziAnalysis) -> float:
        return (result.victory_probability - 0.5) * 6
    
    def _yangming_to_score(self, result: YangmingAnalysis) -> float:
        return (result.unity_of_knowledge_action - 50) / 25
    
    def _iching_to_score(self, result: IChingAnalysis) -> float:
        if result.tai_pai_signal:
            if result.yin_yang_state == YinYangState.PURE_YIN:
                return 4
            else:
                return -4
        return (result.cycle_position - 50) / 25
    
    def _chanlun_to_score(self, result: ChanLunAnalysis) -> float:
        if "买" in result.mai_dian_signal:
            return 3
        if "卖" in result.mai_dian_signal:
            return -3
        if result.bei_chi_type == "底背驰":
            return 2
        if result.bei_chi_type == "顶背驰":
            return -2
        return 0
    
    def _soros_to_score(self, result: SorosReflexivity) -> float:
        if result.turning_point_signal:
            return -3 if result.bias_strength > 70 else 3
        return (50 - result.bubble_probability) / 25
    
    def _complexity_to_score(self, result: ComplexityAnalysis) -> float:
        if result.black_swan_probability > 50:
            return -3
        return (result.system_stability - 50) / 25
    
    def _score_to_signal(self, score: float) -> str:
        if score >= 3:
            return "强烈买入"
        elif score >= 2:
            return "买入"
        elif score >= 1:
            return "偏多"
        elif score >= -1:
            return "中性"
        elif score >= -2:
            return "偏空"
        elif score >= -3:
            return "卖出"
        else:
            return "强烈卖出"
    
    def _calculate_confidence(self, *results) -> float:
        return 0.65
    
    def _assess_risk(self, complexity: ComplexityAnalysis, soros: SorosReflexivity, iching: IChingAnalysis) -> str:
        if complexity.black_swan_probability > 50:
            return "极高风险"
        if soros.bubble_probability > 70:
            return "高风险"
        if complexity.entropy_level > 70:
            return "中高风险"
        if iching.yin_yang_state in [YinYangState.PURE_YIN, YinYangState.PURE_YANG]:
            return "中风险"
        return "中低风险"
    
    def _generate_position_advice(self, score: float, risk: str) -> str:
        if "极高" in risk:
            return "空仓观望，等待系统稳定"
        if score >= 2:
            return "按334法则建仓，30%底仓+30%律动"
        elif score >= 0:
            return "维持底仓，静待时机"
        else:
            return "降低仓位，防范风险"
    
    def _calculate_holding_period(self, iching: IChingAnalysis, chanlun: ChanLunAnalysis) -> int:
        if iching.tai_pai_signal:
            return 60
        if chanlun.bei_chi_type:
            return 30
        return 20
    
    def _generate_stop_loss_hint(self, complexity: ComplexityAnalysis, soros: SorosReflexivity) -> str:
        if complexity.critical_point_distance < 10:
            return "接近临界点，严格止损"
        if soros.turning_point_signal:
            return "转折点附近，动态止损"
        return "常规止损，控制风险"


_eastern_engine = None


def get_eastern_wisdom_engine() -> EasternWisdomEngine:
    """获取东方智慧引擎单例"""
    global _eastern_engine
    if _eastern_engine is None:
        _eastern_engine = EasternWisdomEngine()
    return _eastern_engine
