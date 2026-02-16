#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
哲学思想量化系统
整合马克思、毛泽东、卢麒元、凯恩斯、货币哲学等思想

核心理论体系：
1. 马克思主义政治经济学
   - 剩余价值理论
   - 资本有机构成
   - 利润率趋势分析

2. 毛泽东哲学思想
   - 矛盾分析法（主要矛盾与次要矛盾）
   - 实践论（实践-认识-再实践）
   - 量变质变规律

3. 卢麒元方法论
   - 货币哲学（货币本质与信用）
   - 财政分析（税政健康度）
   - 资本流转（跨境流动）

4. 凯恩斯经济学
   - 有效需求理论
   - 流动性偏好
   - 乘数效应

5. 货币哲学
   - 货币时间价值
   - 信用周期
   - 通胀预期
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


class EconomicPhase(Enum):
    """经济周期阶段"""
    RECOVERY = "复苏期"
    EXPANSION = "扩张期"
    OVERHEAT = "过热期"
    RECESSION = "衰退期"
    DEPRESSION = "萧条期"
    STAGFLATION = "滞胀期"


class ContradictionType(Enum):
    """矛盾类型"""
    PRIMARY = "主要矛盾"
    SECONDARY = "次要矛盾"
    ANTAGONISTIC = "对抗性矛盾"
    NON_ANTAGONISTIC = "非对抗性矛盾"


@dataclass
class MarxistAnalysis:
    """马克思主义分析结果"""
    surplus_value_rate: float
    organic_composition: float
    profit_rate_trend: str
    exploitation_degree: float
    capital_accumulation: str
    crisis_tendency: float


@dataclass
class MaoistAnalysis:
    """毛泽东思想分析结果"""
    primary_contradiction: str
    secondary_contradictions: List[str]
    contradiction_development: str
    practice_cycle: int
    qualitative_change_signal: bool
    strategic_direction: str


@dataclass
class LuQiyuanAnalysis:
    """卢麒元方法论分析结果"""
    monetary_health: float
    fiscal_health: float
    capital_flow_direction: str
    credit_cycle_phase: str
    tax_structure_quality: float
    crisis_warning_level: int


@dataclass
class KeynesianAnalysis:
    """凯恩斯主义分析结果"""
    effective_demand_gap: float
    liquidity_preference: str
    multiplier_effect: float
    investment_incentive: float
    government_intervention_needed: bool
    interest_rate_outlook: str


@dataclass
class MonetaryPhilosophy:
    """货币哲学分析结果"""
    money_time_value: float
    credit_cycle_phase: str
    inflation_expectation: float
    currency_strength: float
    interest_rate_trend: str
    liquidity_status: str


@dataclass
class IntegratedPrediction:
    """综合预测结果"""
    signal: str
    confidence: float
    direction: float
    marxist_score: float
    maoist_score: float
    lu_qiyuan_score: float
    keynesian_score: float
    monetary_score: float
    reasoning: List[str]
    risk_level: str
    position_advice: str
    holding_period: int


class MarxistEconomics:
    """
    马克思主义政治经济学分析
    
    核心理论：
    1. 剩余价值率 = 剩余价值 / 可变资本
    2. 资本有机构成 = 不变资本 / 可变资本
    3. 利润率 = 剩余价值 / (不变资本 + 可变资本)
    4. 利润率下降趋势规律
    """
    
    def analyze(
        self,
        revenue: float,
        cost: float,
        labor_cost: float,
        fixed_assets: float,
        inventory: float,
    ) -> MarxistAnalysis:
        """
        马克思主义经济分析
        
        Args:
            revenue: 营业收入
            cost: 总成本
            labor_cost: 人工成本（可变资本）
            fixed_assets: 固定资产（不变资本-固定）
            inventory: 存货（不变资本-流动）
        """
        surplus_value = revenue - cost
        variable_capital = labor_cost
        constant_capital = fixed_assets + inventory
        
        surplus_value_rate = surplus_value / variable_capital if variable_capital > 0 else 0
        
        organic_composition = constant_capital / variable_capital if variable_capital > 0 else 0
        
        total_capital = constant_capital + variable_capital
        profit_rate = surplus_value / total_capital if total_capital > 0 else 0
        
        exploitation_degree = surplus_value_rate * 100
        
        if organic_composition > 5:
            capital_accumulation = "资本密集型，技术替代劳动明显"
        elif organic_composition > 2:
            capital_accumulation = "中等资本构成，处于转型期"
        else:
            capital_accumulation = "劳动密集型，剩余价值空间大"
        
        crisis_tendency = 0
        if organic_composition > 4 and profit_rate < 0.05:
            crisis_tendency += 0.3
        if surplus_value_rate < 0.5:
            crisis_tendency += 0.2
        if profit_rate < 0.03:
            crisis_tendency += 0.3
        
        crisis_tendency = min(crisis_tendency, 1.0)
        
        if profit_rate > 0.15:
            trend = "利润率健康，资本积累活跃"
        elif profit_rate > 0.08:
            trend = "利润率正常，存在竞争压力"
        elif profit_rate > 0.03:
            trend = "利润率偏低，利润率下降趋势显现"
        else:
            trend = "利润率危机水平，资本积累受阻"
        
        return MarxistAnalysis(
            surplus_value_rate=round(surplus_value_rate, 3),
            organic_composition=round(organic_composition, 2),
            profit_rate_trend=trend,
            exploitation_degree=round(exploitation_degree, 1),
            capital_accumulation=capital_accumulation,
            crisis_tendency=round(crisis_tendency, 2),
        )
    
    def calculate_profit_rate_tendency(
        self,
        historical_profit_rates: List[float]
    ) -> Tuple[str, float]:
        """
        分析利润率下降趋势
        
        马克思：一般利润率有下降趋势
        """
        if len(historical_profit_rates) < 3:
            return "数据不足", 0
        
        rates = np.array(historical_profit_rates)
        x = np.arange(len(rates))
        
        slope = np.polyfit(x, rates, 1)[0]
        
        avg_rate = np.mean(rates)
        tendency = slope / avg_rate if avg_rate != 0 else 0
        
        if tendency < -0.1:
            return "利润率显著下降趋势", abs(tendency)
        elif tendency < -0.03:
            return "利润率温和下降", abs(tendency)
        elif tendency > 0.1:
            return "利润率上升趋势（可能存在超额利润）", 0
        else:
            return "利润率相对稳定", 0


class MaoistPhilosophy:
    """
    毛泽东哲学思想分析
    
    核心理论：
    1. 矛盾论 - 主要矛盾与次要矛盾
    2. 实践论 - 实践-认识-再实践-再认识
    3. 量变质变规律
    4. 否定之否定
    """
    
    def analyze_contradictions(
        self,
        market_data: Dict,
        economic_data: Dict,
        policy_data: Dict,
    ) -> MaoistAnalysis:
        """
        矛盾分析法
        
        识别市场中的主要矛盾和次要矛盾
        """
        contradictions = []
        
        liquidity = market_data.get('liquidity', 0)
        valuation = market_data.get('valuation', 0)
        
        if abs(liquidity - valuation) > 30:
            contradictions.append({
                'name': '流动性vs估值',
                'strength': abs(liquidity - valuation),
                'type': ContradictionType.PRIMARY,
                'direction': 1 if liquidity > valuation else -1,
            })
        
        foreign_capital = market_data.get('foreign_capital', 0)
        domestic_capital = market_data.get('domestic_capital', 0)
        
        if abs(foreign_capital - domestic_capital) > 20:
            contradictions.append({
                'name': '内外资分歧',
                'strength': abs(foreign_capital - domestic_capital),
                'type': ContradictionType.SECONDARY,
                'direction': 1 if foreign_capital > 0 else -1,
            })
        
        policy_intent = policy_data.get('intent', 0)
        economic_reality = economic_data.get('growth', 0)
        
        if abs(policy_intent - economic_reality) > 20:
            contradictions.append({
                'name': '政策预期vs经济现实',
                'strength': abs(policy_intent - economic_reality),
                'type': ContradictionType.SECONDARY,
                'direction': 1 if policy_intent > economic_reality else -1,
            })
        
        if not contradictions:
            return MaoistAnalysis(
                primary_contradiction="无明显矛盾",
                secondary_contradictions=[],
                contradiction_development="市场处于相对平衡状态",
                practice_cycle=0,
                qualitative_change_signal=False,
                strategic_direction="观望",
            )
        
        contradictions.sort(key=lambda x: x['strength'], reverse=True)
        
        primary = contradictions[0]
        secondary = [c['name'] for c in contradictions[1:]]
        
        if primary['strength'] > 70:
            development = f"主要矛盾{primary['name']}激化，即将发生质变"
        elif primary['strength'] > 40:
            development = f"主要矛盾{primary['name']}发展中，量变积累阶段"
        else:
            development = f"主要矛盾{primary['name']}初步显现"
        
        practice_cycle = self._calculate_practice_cycle(market_data)
        
        qualitative_signal = self._detect_qualitative_change(market_data)
        
        if primary['direction'] > 0 and primary['strength'] > 50:
            strategic_direction = "积极进攻，抓住主要矛盾的主要方面"
        elif primary['direction'] < 0 and primary['strength'] > 50:
            strategic_direction = "战略防御，等待矛盾转化"
        else:
            strategic_direction = "战略相持，观察矛盾发展"
        
        return MaoistAnalysis(
            primary_contradiction=primary['name'],
            secondary_contradictions=secondary,
            contradiction_development=development,
            practice_cycle=practice_cycle,
            qualitative_change_signal=qualitative_signal,
            strategic_direction=strategic_direction,
        )
    
    def _calculate_practice_cycle(self, market_data: Dict) -> int:
        """
        计算实践周期
        
        实践论：实践-认识-再实践-再认识
        """
        trend_duration = market_data.get('trend_duration', 0)
        
        if trend_duration < 5:
            return 1
        elif trend_duration < 20:
            return 2
        elif trend_duration < 60:
            return 3
        else:
            return 4
    
    def _detect_qualitative_change(self, market_data: Dict) -> bool:
        """
        检测量变质变信号
        
        量变积累到一定程度会引起质变
        """
        volume_change = market_data.get('volume_change', 0)
        price_breakthrough = market_data.get('price_breakthrough', False)
        sentiment_extreme = market_data.get('sentiment_extreme', False)
        
        signals = 0
        
        if abs(volume_change) > 50:
            signals += 1
        if price_breakthrough:
            signals += 1
        if sentiment_extreme:
            signals += 1
        
        return signals >= 2
    
    def practice_knowledge_cycle(
        self,
        historical_predictions: List[Dict],
        actual_results: List[Dict],
    ) -> Dict:
        """
        实践论：实践-认识-再实践
        
        通过历史预测和实际结果，优化认知
        """
        if not historical_predictions or not actual_results:
            return {'insight': '数据不足', 'adjustment': 0}
        
        correct_count = 0
        total = min(len(historical_predictions), len(actual_results))
        
        for i in range(total):
            pred = historical_predictions[i]
            actual = actual_results[i]
            
            if pred.get('direction', 0) * actual.get('return', 0) > 0:
                correct_count += 1
        
        accuracy = correct_count / total if total > 0 else 0
        
        if accuracy > 0.7:
            insight = "认知基本正确，可加大实践力度"
            adjustment = 0.1
        elif accuracy > 0.5:
            insight = "认知部分正确，需要修正方法论"
            adjustment = 0
        else:
            insight = "认知存在偏差，需要重新认识市场"
            adjustment = -0.2
        
        return {
            'insight': insight,
            'accuracy': accuracy,
            'adjustment': adjustment,
            'practice_count': total,
        }


class LuQiyuanMethodology:
    """
    卢麒元方法论
    
    核心理论：
    1. 货币哲学 - 货币的本质是信用
    2. 财政分析 - 直接税vs间接税结构
    3. 资本流转 - 跨境资本流动分析
    4. 危机预警 - 资本周转效率下降
    """
    
    def analyze(
        self,
        monetary_data: Dict,
        fiscal_data: Dict,
        capital_flow_data: Dict,
    ) -> LuQiyuanAnalysis:
        """
        卢麒元方法论综合分析
        """
        monetary_health = self._analyze_monetary_health(monetary_data)
        
        fiscal_health = self._analyze_fiscal_health(fiscal_data)
        
        capital_flow_direction = self._analyze_capital_flow(capital_flow_data)
        
        credit_cycle = self._determine_credit_cycle(monetary_data, fiscal_data)
        
        tax_quality = self._analyze_tax_structure(fiscal_data)
        
        crisis_level = self._calculate_crisis_warning(
            monetary_health, fiscal_health, capital_flow_data
        )
        
        return LuQiyuanAnalysis(
            monetary_health=round(monetary_health, 2),
            fiscal_health=round(fiscal_health, 2),
            capital_flow_direction=capital_flow_direction,
            credit_cycle_phase=credit_cycle,
            tax_structure_quality=round(tax_quality, 2),
            crisis_warning_level=crisis_level,
        )
    
    def _analyze_monetary_health(self, data: Dict) -> float:
        """
        货币健康度分析
        
        货币哲学：货币的本质是信用
        """
        m2_growth = data.get('m2_growth', 0.08)
        gdp_growth = data.get('gdp_growth', 0.05)
        credit_spread = data.get('credit_spread', 0.02)
        
        efficiency = gdp_growth / m2_growth if m2_growth != 0 else 0
        
        health = 50
        
        if efficiency > 0.8:
            health += 20
        elif efficiency > 0.5:
            health += 10
        elif efficiency < 0.3:
            health -= 20
        
        if credit_spread < 0.02:
            health += 10
        elif credit_spread > 0.05:
            health -= 15
        
        return max(0, min(100, health))
    
    def _analyze_fiscal_health(self, data: Dict) -> float:
        """
        财政健康度分析
        
        直接税占比反映税政质量
        """
        direct_tax_ratio = data.get('direct_tax_ratio', 0.3)
        deficit_ratio = data.get('deficit_ratio', 0.03)
        debt_ratio = data.get('debt_ratio', 0.6)
        
        health = 50
        
        if direct_tax_ratio > 0.4:
            health += 20
        elif direct_tax_ratio > 0.3:
            health += 10
        elif direct_tax_ratio < 0.2:
            health -= 15
        
        if deficit_ratio < 0.03:
            health += 10
        elif deficit_ratio > 0.05:
            health -= 15
        
        if debt_ratio < 0.5:
            health += 10
        elif debt_ratio > 0.8:
            health -= 20
        
        return max(0, min(100, health))
    
    def _analyze_capital_flow(self, data: Dict) -> str:
        """
        资本流向分析
        
        资本总是流向效率最高的地方
        """
        net_flow = data.get('net_flow', 0)
        trend = data.get('trend', 'stable')
        
        if net_flow > 100 and trend == 'increasing':
            return "资本大幅流入"
        elif net_flow > 0:
            return "资本温和流入"
        elif net_flow < -100 and trend == 'decreasing':
            return "资本大幅流出"
        elif net_flow < 0:
            return "资本温和流出"
        else:
            return "资本流动平衡"
    
    def _determine_credit_cycle(self, monetary: Dict, fiscal: Dict) -> str:
        """
        信用周期判断
        """
        m2_growth = monetary.get('m2_growth', 0.08)
        credit_growth = monetary.get('credit_growth', 0.1)
        
        if credit_growth > m2_growth * 1.5 and credit_growth > 0.12:
            return "信用扩张期"
        elif credit_growth > m2_growth:
            return "信用温和扩张"
        elif credit_growth < m2_growth * 0.5:
            return "信用收缩期"
        else:
            return "信用平稳期"
    
    def _analyze_tax_structure(self, data: Dict) -> float:
        """
        税收结构质量
        
        直接税占比高 = 税政质量高
        """
        direct_tax = data.get('direct_tax_ratio', 0.3)
        indirect_tax = 1 - direct_tax
        
        quality = direct_tax * 100
        
        if direct_tax > 0.5:
            quality += 20
        elif direct_tax < 0.25:
            quality -= 20
        
        return max(0, min(100, quality))
    
    def _calculate_crisis_warning(
        self,
        monetary_health: float,
        fiscal_health: float,
        capital_flow: Dict,
    ) -> int:
        """
        危机预警等级
        
        0=安全, 1=关注, 2=警告, 3=严重警告, 4=危机
        """
        level = 0
        
        if monetary_health < 40:
            level += 1
        if monetary_health < 20:
            level += 1
        
        if fiscal_health < 40:
            level += 1
        if fiscal_health < 20:
            level += 1
        
        net_flow = capital_flow.get('net_flow', 0)
        if net_flow < -200:
            level += 1
        
        return min(level, 4)


class KeynesianEconomics:
    """
    凯恩斯经济学分析
    
    核心理论：
    1. 有效需求理论
    2. 流动性偏好
    3. 乘数效应
    4. 政府干预
    """
    
    def analyze(
        self,
        demand_data: Dict,
        liquidity_data: Dict,
        investment_data: Dict,
        government_data: Dict,
    ) -> KeynesianAnalysis:
        """
        凯恩斯主义经济分析
        """
        demand_gap = self._calculate_effective_demand_gap(demand_data)
        
        liquidity_pref = self._analyze_liquidity_preference(liquidity_data)
        
        multiplier = self._calculate_multiplier(investment_data)
        
        incentive = self._calculate_investment_incentive(investment_data)
        
        intervention_needed = self._assess_intervention_need(
            demand_gap, incentive, liquidity_data
        )
        
        rate_outlook = self._forecast_interest_rate(
            liquidity_data, demand_gap, government_data
        )
        
        return KeynesianAnalysis(
            effective_demand_gap=round(demand_gap, 3),
            liquidity_preference=liquidity_pref,
            multiplier_effect=round(multiplier, 2),
            investment_incentive=round(incentive, 2),
            government_intervention_needed=intervention_needed,
            interest_rate_outlook=rate_outlook,
        )
    
    def _calculate_effective_demand_gap(self, data: Dict) -> float:
        """
        有效需求缺口
        
        有效需求 = 消费 + 投资 + 政府支出 + 净出口
        潜在产出 - 有效需求 = 需求缺口
        """
        potential_output = data.get('potential_output', 100)
        actual_demand = data.get('actual_demand', 95)
        
        gap = (potential_output - actual_demand) / potential_output
        
        return gap
    
    def _analyze_liquidity_preference(self, data: Dict) -> str:
        """
        流动性偏好分析
        
        三大动机：交易动机、预防动机、投机动机
        """
        money_demand = data.get('money_demand', 0.5)
        interest_rate = data.get('interest_rate', 0.03)
        
        if money_demand > 0.7 and interest_rate < 0.02:
            return "流动性陷阱"
        elif money_demand > 0.6:
            return "高流动性偏好"
        elif money_demand < 0.3 and interest_rate > 0.05:
            return "流动性偏好低"
        else:
            return "流动性偏好正常"
    
    def _calculate_multiplier(self, data: Dict) -> float:
        """
        乘数效应计算
        
        乘数 = 1 / (1 - 边际消费倾向)
        """
        mpc = data.get('marginal_propensity_consume', 0.6)
        
        multiplier = 1 / (1 - mpc) if mpc < 1 else 1
        
        return multiplier
    
    def _calculate_investment_incentive(self, data: Dict) -> float:
        """
        投资激励计算
        
        MEC（资本边际效率）vs 利率
        """
        mec = data.get('marginal_efficiency_capital', 0.08)
        interest_rate = data.get('interest_rate', 0.03)
        
        incentive = mec - interest_rate
        
        return incentive
    
    def _assess_intervention_need(
        self,
        demand_gap: float,
        incentive: float,
        liquidity: Dict,
    ) -> bool:
        """
        评估是否需要政府干预
        """
        if demand_gap > 0.05:
            return True
        if incentive < 0.02:
            return True
        if liquidity.get('liquidity_trap', False):
            return True
        return False
    
    def _forecast_interest_rate(
        self,
        liquidity: Dict,
        demand_gap: float,
        gov: Dict,
    ) -> str:
        """
        利率走势预测
        """
        if demand_gap > 0.05:
            return "下行（刺激需求）"
        elif demand_gap < -0.02:
            return "上行（抑制过热）"
        else:
            return "稳定"


class MonetaryPhilosophyAnalyzer:
    """
    货币哲学分析
    
    核心理论：
    1. 货币时间价值
    2. 信用周期
    3. 通胀预期
    4. 货币强弱
    """
    
    def analyze(
        self,
        inflation_data: Dict,
        interest_data: Dict,
        currency_data: Dict,
        credit_data: Dict,
    ) -> MonetaryPhilosophy:
        """
        货币哲学综合分析
        """
        time_value = self._calculate_time_value(interest_data, inflation_data)
        
        credit_phase = self._determine_credit_cycle(credit_data)
        
        inflation_exp = self._estimate_inflation_expectation(inflation_data)
        
        currency_str = self._assess_currency_strength(currency_data)
        
        rate_trend = self._forecast_rate_trend(
            inflation_data, interest_data, credit_data
        )
        
        liquidity = self._assess_liquidity(interest_data, credit_data)
        
        return MonetaryPhilosophy(
            money_time_value=round(time_value, 4),
            credit_cycle_phase=credit_phase,
            inflation_expectation=round(inflation_exp, 3),
            currency_strength=round(currency_str, 2),
            interest_rate_trend=rate_trend,
            liquidity_status=liquidity,
        )
    
    def _calculate_time_value(
        self,
        interest: Dict,
        inflation: Dict,
    ) -> float:
        """
        货币时间价值 = 名义利率 - 通胀率
        """
        nominal_rate = interest.get('nominal_rate', 0.03)
        inflation_rate = inflation.get('current', 0.02)
        
        real_rate = nominal_rate - inflation_rate
        
        return real_rate
    
    def _determine_credit_cycle(self, data: Dict) -> str:
        """
        信用周期判断
        """
        credit_growth = data.get('growth', 0.05)
        default_rate = data.get('default_rate', 0.01)
        
        if credit_growth > 0.1 and default_rate < 0.01:
            return "信用扩张早期"
        elif credit_growth > 0.05 and default_rate < 0.02:
            return "信用扩张中期"
        elif credit_growth < 0.02 or default_rate > 0.03:
            return "信用收缩期"
        else:
            return "信用平稳期"
    
    def _estimate_inflation_expectation(self, data: Dict) -> float:
        """
        通胀预期估计
        """
        current = data.get('current', 0.02)
        trend = data.get('trend', 0)
        expectation = current + trend * 0.5
        
        return expectation
    
    def _assess_currency_strength(self, data: Dict) -> float:
        """
        货币强弱评估
        """
        exchange_rate_change = data.get('exchange_rate_change', 0)
        reserve_change = data.get('reserve_change', 0)
        
        strength = 50
        
        if exchange_rate_change > 0:
            strength += exchange_rate_change * 100
        else:
            strength += exchange_rate_change * 100
        
        if reserve_change > 0:
            strength += reserve_change * 50
        
        return max(0, min(100, strength))
    
    def _forecast_rate_trend(
        self,
        inflation: Dict,
        interest: Dict,
        credit: Dict,
    ) -> str:
        """
        利率趋势预测
        """
        inflation_trend = inflation.get('trend', 0)
        credit_phase = credit.get('phase', 'stable')
        
        if inflation_trend > 0.01:
            return "上行（抗通胀）"
        elif inflation_trend < -0.01:
            return "下行（防通缩）"
        else:
            return "稳定"
    
    def _assess_liquidity(
        self,
        interest: Dict,
        credit: Dict,
    ) -> str:
        """
        流动性状态评估
        """
        money_supply_growth = credit.get('money_supply_growth', 0.08)
        velocity = credit.get('velocity', 0.5)
        
        if money_supply_growth > 0.1 and velocity < 0.4:
            return "流动性陷阱风险"
        elif money_supply_growth > 0.08:
            return "流动性充裕"
        elif money_supply_growth < 0.05:
            return "流动性偏紧"
        else:
            return "流动性适中"


class IntegratedPhilosophyEngine:
    """
    综合哲学思想预测引擎
    
    整合五大思想体系：
    1. 马克思主义政治经济学
    2. 毛泽东哲学思想
    3. 卢麒元方法论
    4. 凯恩斯经济学
    5. 货币哲学
    """
    
    def __init__(self):
        self.marxist = MarxistEconomics()
        self.maoist = MaoistPhilosophy()
        self.lu_qiyuan = LuQiyuanMethodology()
        self.keynesian = KeynesianEconomics()
        self.monetary = MonetaryPhilosophyAnalyzer()
        
        self.weights = {
            'marxist': 0.20,
            'maoist': 0.25,
            'lu_qiyuan': 0.25,
            'keynesian': 0.15,
            'monetary': 0.15,
        }
    
    def predict(
        self,
        code: str,
        fundamental_data: Dict,
        market_data: Dict,
        macro_data: Dict,
        policy_data: Dict,
    ) -> IntegratedPrediction:
        """
        综合预测
        """
        marxist_result = self.marxist.analyze(
            revenue=fundamental_data.get('revenue', 100),
            cost=fundamental_data.get('cost', 80),
            labor_cost=fundamental_data.get('labor_cost', 10),
            fixed_assets=fundamental_data.get('fixed_assets', 50),
            inventory=fundamental_data.get('inventory', 20),
        )
        
        maoist_result = self.maoist.analyze_contradictions(
            market_data=market_data,
            economic_data=macro_data,
            policy_data=policy_data,
        )
        
        lu_result = self.lu_qiyuan.analyze(
            monetary_data=macro_data.get('monetary', {}),
            fiscal_data=macro_data.get('fiscal', {}),
            capital_flow_data=macro_data.get('capital_flow', {}),
        )
        
        keynesian_result = self.keynesian.analyze(
            demand_data=macro_data.get('demand', {}),
            liquidity_data=macro_data.get('liquidity', {}),
            investment_data=macro_data.get('investment', {}),
            government_data=policy_data,
        )
        
        monetary_result = self.monetary.analyze(
            inflation_data=macro_data.get('inflation', {}),
            interest_data=macro_data.get('interest', {}),
            currency_data=macro_data.get('currency', {}),
            credit_data=macro_data.get('credit', {}),
        )
        
        marxist_score = self._marxist_to_score(marxist_result)
        maoist_score = self._maoist_to_score(maoist_result)
        lu_score = self._lu_to_score(lu_result)
        keynesian_score = self._keynesian_to_score(keynesian_result)
        monetary_score = self._monetary_to_score(monetary_result)
        
        final_score = (
            marxist_score * self.weights['marxist'] +
            maoist_score * self.weights['maoist'] +
            lu_score * self.weights['lu_qiyuan'] +
            keynesian_score * self.weights['keynesian'] +
            monetary_score * self.weights['monetary']
        )
        
        signal = self._score_to_signal(final_score)
        confidence = self._calculate_confidence(
            marxist_result, maoist_result, lu_result,
            keynesian_result, monetary_result
        )
        
        reasoning = self._generate_reasoning(
            marxist_result, maoist_result, lu_result,
            keynesian_result, monetary_result
        )
        
        risk_level = self._assess_risk(lu_result, maoist_result)
        position_advice = self._generate_position_advice(
            final_score, lu_result, maoist_result
        )
        holding_period = self._calculate_holding_period(
            marxist_result, maoist_result, monetary_result
        )
        
        return IntegratedPrediction(
            signal=signal,
            confidence=round(confidence, 2),
            direction=round(final_score, 2),
            marxist_score=round(marxist_score, 2),
            maoist_score=round(maoist_score, 2),
            lu_qiyuan_score=round(lu_score, 2),
            keynesian_score=round(keynesian_score, 2),
            monetary_score=round(monetary_score, 2),
            reasoning=reasoning,
            risk_level=risk_level,
            position_advice=position_advice,
            holding_period=holding_period,
        )
    
    def _marxist_to_score(self, result: MarxistAnalysis) -> float:
        """马克思主义分析转分数"""
        score = 0
        
        if result.surplus_value_rate > 1:
            score += 1
        elif result.surplus_value_rate < 0.3:
            score -= 1
        
        if result.crisis_tendency < 0.2:
            score += 1
        elif result.crisis_tendency > 0.5:
            score -= 2
        
        return score
    
    def _maoist_to_score(self, result: MaoistAnalysis) -> float:
        """毛泽东思想分析转分数"""
        score = 0
        
        if "进攻" in result.strategic_direction:
            score += 2
        elif "防御" in result.strategic_direction:
            score -= 2
        
        if result.qualitative_change_signal:
            score += 1
        
        return score
    
    def _lu_to_score(self, result: LuQiyuanAnalysis) -> float:
        """卢麒元方法论转分数"""
        score = 0
        
        if result.crisis_warning_level >= 3:
            score -= 3
        elif result.crisis_warning_level >= 2:
            score -= 1
        
        if result.monetary_health > 60:
            score += 1
        elif result.monetary_health < 30:
            score -= 1
        
        if "流入" in result.capital_flow_direction:
            score += 1
        elif "流出" in result.capital_flow_direction:
            score -= 1
        
        return score
    
    def _keynesian_to_score(self, result: KeynesianAnalysis) -> float:
        """凯恩斯分析转分数"""
        score = 0
        
        if result.effective_demand_gap > 0.05:
            score -= 1
        
        if result.multiplier_effect > 2.5:
            score += 1
        
        if result.investment_incentive > 0.05:
            score += 1
        elif result.investment_incentive < 0:
            score -= 1
        
        return score
    
    def _monetary_to_score(self, result: MonetaryPhilosophy) -> float:
        """货币哲学分析转分数"""
        score = 0
        
        if result.money_time_value > 0.02:
            score += 1
        elif result.money_time_value < 0:
            score -= 1
        
        if result.inflation_expectation > 0.05:
            score -= 1
        elif result.inflation_expectation < 0:
            score -= 0.5
        
        if "充裕" in result.liquidity_status:
            score += 1
        elif "紧" in result.liquidity_status:
            score -= 1
        
        return score
    
    def _score_to_signal(self, score: float) -> str:
        """分数转信号"""
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
        """计算置信度"""
        return 0.65
    
    def _generate_reasoning(self, *results) -> List[str]:
        """生成决策理由"""
        marxist, maoist, lu, keynesian, monetary = results
        
        return [
            f"【马克思主义】剩余价值率{marxist.surplus_value_rate:.1%}，{marxist.profit_rate_trend}",
            f"【毛泽东思想】{maoist.primary_contradiction}，{maoist.strategic_direction}",
            f"【卢麒元】货币健康度{lu.monetary_health}，危机预警等级{lu.crisis_warning_level}",
            f"【凯恩斯】有效需求缺口{keynesian.effective_demand_gap:.1%}，投资激励{keynesian.investment_incentive:.2f}",
            f"【货币哲学】信用周期{monetary.credit_cycle_phase}，流动性{monetary.liquidity_status}",
        ]
    
    def _assess_risk(self, lu_result, maoist_result) -> str:
        """评估风险"""
        if lu_result.crisis_warning_level >= 3:
            return "高风险"
        elif lu_result.crisis_warning_level >= 2:
            return "中高风险"
        elif maoist_result.qualitative_change_signal:
            return "中风险"
        else:
            return "中低风险"
    
    def _generate_position_advice(self, score, lu_result, maoist_result) -> str:
        """生成仓位建议"""
        if lu_result.crisis_warning_level >= 3:
            return "建议空仓或极低仓位，等待危机过去"
        elif score >= 2:
            return "建议按334法则建仓，30%底仓+30%律动"
        elif score >= 0:
            return "建议维持底仓，等待更明确信号"
        else:
            return "建议降低仓位，防范风险"
    
    def _calculate_holding_period(self, marxist, maoist, monetary) -> int:
        """计算持仓周期"""
        if marxist.crisis_tendency > 0.5:
            return 5
        elif maoist.practice_cycle >= 3:
            return 60
        elif "收缩" in monetary.credit_cycle_phase:
            return 10
        else:
            return 30


_philosophy_engine = None


def get_philosophy_engine() -> IntegratedPhilosophyEngine:
    """获取哲学引擎单例"""
    global _philosophy_engine
    if _philosophy_engine is None:
        _philosophy_engine = IntegratedPhilosophyEngine()
    return _philosophy_engine
