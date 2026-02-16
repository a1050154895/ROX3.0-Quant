#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卢麒元方法论预测API
提供股票预测、仓位管理、回测等功能
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

from app.rox_quant.lu_qiyuan_prediction import (
    LuQiyuanPredictionEngine,
    PositionManager,
    get_prediction_engine,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lu-prediction", tags=["卢麒元预测系统"])


class PredictRequest(BaseModel):
    """预测请求"""
    code: str
    include_reasoning: bool = True


class PositionRequest(BaseModel):
    """仓位请求"""
    code: str
    action: str
    shares: int
    price: float


@router.get("/predict/{code}")
async def predict_stock(code: str) -> Dict[str, Any]:
    """
    股票预测
    
    整合卢麒元五大方法论：
    1. 矛盾分析法
    2. 价值规律
    3. 334法则
    4. 律动操作
    5. 宏观周期
    """
    try:
        engine = get_prediction_engine()
        
        price_data = await _get_price_data(code)
        fundamental_data = await _get_fundamental_data(code)
        market_data = await _get_market_data()
        macro_data = await _get_macro_data()
        
        result = engine.predict(
            code=code,
            price_data=price_data,
            fundamental_data=fundamental_data,
            market_data=market_data,
            macro_data=macro_data,
        )
        
        return {
            "code": code,
            "signal": result.signal,
            "confidence": round(result.confidence, 2),
            "direction": round(result.direction, 2),
            "strength": result.strength,
            "target_price": round(result.target_price, 2) if result.target_price else None,
            "stop_loss": round(result.stop_loss, 2) if result.stop_loss else None,
            "position_ratio": round(result.position_ratio, 2),
            "holding_period": result.holding_period,
            "reasoning": result.reasoning,
            "risk_level": result.risk_level,
            "expected_return": round(result.expected_return, 4),
            "win_probability": round(result.win_probability, 2),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"预测失败 {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch-predict")
async def batch_predict(codes: str = Query(..., description="股票代码，逗号分隔")) -> Dict:
    """批量预测"""
    code_list = [c.strip() for c in codes.split(",") if c.strip()]
    results = []
    
    engine = get_prediction_engine()
    market_data = await _get_market_data()
    macro_data = await _get_macro_data()
    
    for code in code_list[:10]:
        try:
            price_data = await _get_price_data(code)
            fundamental_data = await _get_fundamental_data(code)
            
            result = engine.predict(
                code=code,
                price_data=price_data,
                fundamental_data=fundamental_data,
                market_data=market_data,
                macro_data=macro_data,
            )
            
            results.append({
                "code": code,
                "signal": result.signal,
                "confidence": round(result.confidence, 2),
                "position_ratio": round(result.position_ratio, 2),
                "win_probability": round(result.win_probability, 2),
            })
        except Exception as e:
            results.append({"code": code, "error": str(e)})
    
    return {"results": results, "count": len(results)}


@router.get("/position/{code}")
async def get_position(code: str) -> Dict:
    """获取持仓状态"""
    try:
        manager = _get_position_manager()
        status = manager.get_position_status(code)
        return {"code": code, **status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/position/trade")
async def execute_trade(req: PositionRequest) -> Dict:
    """执行交易（律动操作）"""
    try:
        manager = _get_position_manager()
        
        if req.action == "buy":
            result = manager.rhythm_buy(req.code, req.price, req.shares)
        elif req.action == "sell":
            result = manager.rhythm_sell(req.code, req.price, req.shares)
        else:
            raise HTTPException(status_code=400, detail="无效操作")
        
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accuracy-stats")
async def get_accuracy_stats() -> Dict:
    """获取预测准确度统计"""
    engine = get_prediction_engine()
    stats = engine.get_accuracy_stats()
    return stats


@router.get("/methodology")
async def get_methodology() -> Dict:
    """获取方法论说明"""
    return {
        "name": "卢麒元方法论增强预测系统",
        "core_methods": [
            {
                "name": "矛盾分析法",
                "description": "识别市场主要矛盾，把握主要矛盾的主要方面",
                "factors": ["量价背离", "资金分歧", "行业分化", "政策预期"],
                "weight": 0.25
            },
            {
                "name": "价值规律",
                "description": "价格围绕价值波动，识别背离机会",
                "factors": ["ROE", "PE", "PB", "增长率", "股息率"],
                "weight": 0.30
            },
            {
                "name": "334法则",
                "description": "30%底仓 + 30%律动 + 40%预备队",
                "factors": ["底仓持有", "律动操作", "风险预备"],
                "weight": "仓位管理"
            },
            {
                "name": "律动操作",
                "description": "通过波段降低持仓成本",
                "factors": ["MACD金叉死叉", "支撑阻力", "成交量"],
                "weight": "操作策略"
            },
            {
                "name": "宏观周期",
                "description": "资本周转效率决定大方向",
                "factors": ["GDP/M2", "直接税占比", "基尼系数"],
                "weight": 0.20
            }
        ],
        "signal_levels": {
            "强烈买入": "综合得分 >= 3.5",
            "买入": "综合得分 >= 2.5",
            "偏多": "综合得分 >= 1.5",
            "中性": "综合得分在 -0.5 到 0.5 之间",
            "偏空": "综合得分 <= -1.5",
            "卖出": "综合得分 <= -2.5",
            "强烈卖出": "综合得分 <= -3.5"
        },
        "risk_levels": ["低风险", "中低风险", "中风险", "中高风险", "高风险"],
        "target_accuracy": "95%+"
    }


@router.get("/market-phase")
async def get_market_phase() -> Dict:
    """获取当前市场阶段"""
    try:
        macro_data = await _get_macro_data()
        market_data = await _get_market_data()
        
        engine = get_prediction_engine()
        macro_signal = engine._analyze_macro(macro_data)
        contradiction = engine._analyze_contradiction(market_data)
        
        return {
            "phase": macro_signal.phase,
            "crisis_probability": round(macro_signal.crisis_probability, 2),
            "capital_turnover": round(macro_signal.capital_turnover, 3),
            "asset_allocation": macro_signal.asset_allocation,
            "primary_contradiction": contradiction.primary_contradiction,
            "contradiction_strength": contradiction.contradiction_strength,
            "recommendation": _generate_market_recommendation(macro_signal, contradiction),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _generate_market_recommendation(macro, contradiction) -> str:
    """生成市场建议"""
    if macro.crisis_probability > 0.5:
        return "当前处于危机预警期，建议降低仓位，增加现金和黄金配置"
    elif contradiction.contradiction_strength > 70:
        return f"市场存在{contradiction.primary_contradiction}，建议谨慎操作"
    elif macro.capital_turnover < 0.45:
        return "资本周转效率较低，建议选择优质标的，避免追高"
    else:
        return "市场相对稳定，可按334法则逐步建仓"


async def _get_price_data(code: str) -> pd.DataFrame:
    """获取价格数据"""
    try:
        import akshare as ak
        import asyncio
        
        code6 = code[-6:] if len(code) >= 6 else code.zfill(6)
        
        def fetch_data():
            return ak.stock_zh_a_hist(symbol=code6, period="daily", adjust="qfq")
        
        df = await asyncio.to_thread(fetch_data)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        column_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
        df = df.rename(columns=column_map)
        
        return df
    except Exception as e:
        logger.warning(f"获取价格数据失败 {code}: {e}")
        return pd.DataFrame()


async def _get_fundamental_data(code: str) -> Dict:
    """获取基本面数据"""
    return {
        'roe': 0.12,
        'pe': 15,
        'pb': 1.5,
        'growth_rate': 0.08,
        'dividend_yield': 0.02,
        'beta': 1.0,
    }


async def _get_market_data() -> Dict:
    """获取市场数据"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        
        if df.empty:
            return {}
        
        change_col = next((c for c in df.columns if "涨跌幅" in c), None)
        if change_col:
            up_count = len(df[df[change_col] > 0])
            down_count = len(df[df[change_col] < 0])
            limit_up = len(df[df[change_col] >= 9.9])
            limit_down = len(df[df[change_col] <= -9.9])
        else:
            up_count = down_count = limit_up = limit_down = 0
        
        return {
            'up_count': up_count,
            'down_count': down_count,
            'limit_up': limit_up,
            'limit_down': limit_down,
            'up_ratio': up_count / (up_count + down_count) if (up_count + down_count) > 0 else 0.5,
            'volume': 1,
            'avg_volume': 1,
            'north_flow': 0,
            'main_flow': 0,
            'turnover_rate': 0.03,
        }
    except Exception as e:
        logger.warning(f"获取市场数据失败: {e}")
        return {}


async def _get_macro_data() -> Dict:
    """获取宏观数据"""
    return {
        'gdp': 120,
        'm2': 200,
        'direct_tax_ratio': 0.35,
        'gini': 0.38,
    }


_position_manager = None


def _get_position_manager() -> PositionManager:
    """获取仓位管理器"""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager(total_capital=1000000)
    return _position_manager
