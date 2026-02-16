#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
哲学思想量化预测API
整合马克思、毛泽东、卢麒元、凯恩斯、货币哲学
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rox_quant.philosophy_engine import (
    get_philosophy_engine,
    MarxistAnalysis,
    MaoistAnalysis,
    LuQiyuanAnalysis,
    KeynesianAnalysis,
    MonetaryPhilosophy,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/philosophy-prediction", tags=["哲学思想量化系统"])


class PredictionRequest(BaseModel):
    """预测请求"""
    code: str
    revenue: float = 100
    cost: float = 80
    labor_cost: float = 10
    fixed_assets: float = 50
    inventory: float = 20


@router.get("/predict/{code}")
async def philosophy_predict(code: str) -> Dict[str, Any]:
    """
    哲学思想综合预测
    
    整合五大思想体系：
    - 马克思主义政治经济学
    - 毛泽东哲学思想
    - 卢麒元方法论
    - 凯恩斯经济学
    - 货币哲学
    """
    try:
        engine = get_philosophy_engine()
        
        fundamental_data = {
            'revenue': 100,
            'cost': 80,
            'labor_cost': 10,
            'fixed_assets': 50,
            'inventory': 20,
        }
        
        market_data = {
            'liquidity': 60,
            'valuation': 50,
            'foreign_capital': 10,
            'domestic_capital': 5,
            'volume_change': 20,
            'price_breakthrough': False,
            'sentiment_extreme': False,
            'trend_duration': 15,
        }
        
        macro_data = {
            'growth': 5,
            'monetary': {
                'm2_growth': 0.08,
                'gdp_growth': 0.05,
                'credit_spread': 0.02,
                'credit_growth': 0.1,
            },
            'fiscal': {
                'direct_tax_ratio': 0.35,
                'deficit_ratio': 0.03,
                'debt_ratio': 0.6,
            },
            'capital_flow': {
                'net_flow': 50,
                'trend': 'stable',
            },
            'demand': {
                'potential_output': 100,
                'actual_demand': 95,
            },
            'liquidity': {
                'money_demand': 0.5,
                'interest_rate': 0.03,
                'liquidity_trap': False,
            },
            'investment': {
                'marginal_propensity_consume': 0.6,
                'marginal_efficiency_capital': 0.08,
                'interest_rate': 0.03,
            },
            'inflation': {
                'current': 0.02,
                'trend': 0.005,
            },
            'interest': {
                'nominal_rate': 0.03,
            },
            'currency': {
                'exchange_rate_change': 0,
                'reserve_change': 0,
            },
            'credit': {
                'growth': 0.05,
                'default_rate': 0.01,
                'money_supply_growth': 0.08,
                'velocity': 0.5,
                'phase': 'stable',
            },
        }
        
        policy_data = {
            'intent': 60,
        }
        
        result = engine.predict(
            code=code,
            fundamental_data=fundamental_data,
            market_data=market_data,
            macro_data=macro_data,
            policy_data=policy_data,
        )
        
        return {
            "code": code,
            "signal": result.signal,
            "confidence": result.confidence,
            "direction": result.direction,
            "scores": {
                "marxist": result.marxist_score,
                "maoist": result.maoist_score,
                "lu_qiyuan": result.lu_qiyuan_score,
                "keynesian": result.keynesian_score,
                "monetary": result.monetary_score,
            },
            "reasoning": result.reasoning,
            "risk_level": result.risk_level,
            "position_advice": result.position_advice,
            "holding_period": result.holding_period,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"哲学预测失败 {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/marxist/{code}")
async def marxist_analysis(code: str) -> Dict[str, Any]:
    """
    马克思主义政治经济学分析
    
    核心理论：
    - 剩余价值理论
    - 资本有机构成
    - 利润率下降趋势
    """
    try:
        from app.rox_quant.philosophy_engine import MarxistEconomics
        
        engine = MarxistEconomics()
        result = engine.analyze(
            revenue=100,
            cost=80,
            labor_cost=10,
            fixed_assets=50,
            inventory=20,
        )
        
        return {
            "code": code,
            "surplus_value_rate": result.surplus_value_rate,
            "organic_composition": result.organic_composition,
            "profit_rate_trend": result.profit_rate_trend,
            "exploitation_degree": result.exploitation_degree,
            "capital_accumulation": result.capital_accumulation,
            "crisis_tendency": result.crisis_tendency,
            "theory": "《资本论》剩余价值理论",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/maoist/{code}")
async def maoist_analysis(code: str) -> Dict[str, Any]:
    """
    毛泽东哲学思想分析
    
    核心理论：
    - 矛盾论：主要矛盾与次要矛盾
    - 实践论：实践-认识-再实践
    - 量变质变规律
    """
    try:
        from app.rox_quant.philosophy_engine import MaoistPhilosophy
        
        engine = MaoistPhilosophy()
        result = engine.analyze_contradictions(
            market_data={
                'liquidity': 60,
                'valuation': 50,
                'foreign_capital': 10,
                'domestic_capital': 5,
                'volume_change': 20,
                'price_breakthrough': False,
                'sentiment_extreme': False,
                'trend_duration': 15,
            },
            economic_data={'growth': 5},
            policy_data={'intent': 60},
        )
        
        return {
            "code": code,
            "primary_contradiction": result.primary_contradiction,
            "secondary_contradictions": result.secondary_contradictions,
            "contradiction_development": result.contradiction_development,
            "practice_cycle": result.practice_cycle,
            "qualitative_change_signal": result.qualitative_change_signal,
            "strategic_direction": result.strategic_direction,
            "theory": "《矛盾论》《实践论》",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lu-qiyuan/{code}")
async def lu_qiyuan_analysis(code: str) -> Dict[str, Any]:
    """
    卢麒元方法论分析
    
    核心理论：
    - 货币哲学
    - 财政健康度
    - 资本流转
    - 危机预警
    """
    try:
        from app.rox_quant.philosophy_engine import LuQiyuanMethodology
        
        engine = LuQiyuanMethodology()
        result = engine.analyze(
            monetary_data={
                'm2_growth': 0.08,
                'gdp_growth': 0.05,
                'credit_spread': 0.02,
                'credit_growth': 0.1,
            },
            fiscal_data={
                'direct_tax_ratio': 0.35,
                'deficit_ratio': 0.03,
                'debt_ratio': 0.6,
            },
            capital_flow_data={
                'net_flow': 50,
                'trend': 'stable',
            },
        )
        
        return {
            "code": code,
            "monetary_health": result.monetary_health,
            "fiscal_health": result.fiscal_health,
            "capital_flow_direction": result.capital_flow_direction,
            "credit_cycle_phase": result.credit_cycle_phase,
            "tax_structure_quality": result.tax_structure_quality,
            "crisis_warning_level": result.crisis_warning_level,
            "theory": "卢麒元货币哲学与财政分析",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/keynesian/{code}")
async def keynesian_analysis(code: str) -> Dict[str, Any]:
    """
    凯恩斯经济学分析
    
    核心理论：
    - 有效需求理论
    - 流动性偏好
    - 乘数效应
    - 政府干预
    """
    try:
        from app.rox_quant.philosophy_engine import KeynesianEconomics
        
        engine = KeynesianEconomics()
        result = engine.analyze(
            demand_data={
                'potential_output': 100,
                'actual_demand': 95,
            },
            liquidity_data={
                'money_demand': 0.5,
                'interest_rate': 0.03,
                'liquidity_trap': False,
            },
            investment_data={
                'marginal_propensity_consume': 0.6,
                'marginal_efficiency_capital': 0.08,
                'interest_rate': 0.03,
            },
            government_data={'spending': 20},
        )
        
        return {
            "code": code,
            "effective_demand_gap": result.effective_demand_gap,
            "liquidity_preference": result.liquidity_preference,
            "multiplier_effect": result.multiplier_effect,
            "investment_incentive": result.investment_incentive,
            "government_intervention_needed": result.government_intervention_needed,
            "interest_rate_outlook": result.interest_rate_outlook,
            "theory": "《就业、利息和货币通论》",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monetary/{code}")
async def monetary_analysis(code: str) -> Dict[str, Any]:
    """
    货币哲学分析
    
    核心理论：
    - 货币时间价值
    - 信用周期
    - 通胀预期
    """
    try:
        from app.rox_quant.philosophy_engine import MonetaryPhilosophyAnalyzer
        
        engine = MonetaryPhilosophyAnalyzer()
        result = engine.analyze(
            inflation_data={
                'current': 0.02,
                'trend': 0.005,
            },
            interest_data={
                'nominal_rate': 0.03,
            },
            currency_data={
                'exchange_rate_change': 0,
                'reserve_change': 0,
            },
            credit_data={
                'growth': 0.05,
                'default_rate': 0.01,
                'money_supply_growth': 0.08,
                'velocity': 0.5,
                'phase': 'stable',
            },
        )
        
        return {
            "code": code,
            "money_time_value": result.money_time_value,
            "credit_cycle_phase": result.credit_cycle_phase,
            "inflation_expectation": result.inflation_expectation,
            "currency_strength": result.currency_strength,
            "interest_rate_trend": result.interest_rate_trend,
            "liquidity_status": result.liquidity_status,
            "theory": "货币哲学与信用周期",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/theory-guide")
async def get_theory_guide() -> Dict[str, Any]:
    """获取理论指导说明"""
    return {
        "title": "哲学思想量化系统理论指导",
        "theories": [
            {
                "name": "马克思主义政治经济学",
                "source": "《资本论》",
                "core_concepts": [
                    "剩余价值率 = 剩余价值/可变资本",
                    "资本有机构成 = 不变资本/可变资本",
                    "利润率下降趋势规律",
                    "危机理论",
                ],
                "application": "分析企业盈利质量、资本结构、危机风险",
                "weight": 0.20,
            },
            {
                "name": "毛泽东哲学思想",
                "source": "《矛盾论》《实践论》",
                "core_concepts": [
                    "主要矛盾与次要矛盾",
                    "矛盾的主要方面",
                    "实践-认识-再实践",
                    "量变质变规律",
                ],
                "application": "识别市场主矛盾，把握投资方向",
                "weight": 0.25,
            },
            {
                "name": "卢麒元方法论",
                "source": "卢麒元文集",
                "core_concepts": [
                    "货币哲学（货币=信用）",
                    "财政健康度（直接税占比）",
                    "资本流转效率",
                    "危机预警系统",
                ],
                "application": "宏观判断，大方向把握",
                "weight": 0.25,
            },
            {
                "name": "凯恩斯经济学",
                "source": "《就业、利息和货币通论》",
                "core_concepts": [
                    "有效需求理论",
                    "流动性偏好",
                    "乘数效应",
                    "政府干预",
                ],
                "application": "判断经济周期，把握政策方向",
                "weight": 0.15,
            },
            {
                "name": "货币哲学",
                "source": "货币理论",
                "core_concepts": [
                    "货币时间价值",
                    "信用周期",
                    "通胀预期",
                    "流动性分析",
                ],
                "application": "利率判断，流动性分析",
                "weight": 0.15,
            },
        ],
        "integration_method": "加权融合，动态调整权重",
        "target_accuracy": "95%+",
    }
