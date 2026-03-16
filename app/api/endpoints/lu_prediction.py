#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卢麒元方法论预测 API — Protocol v5（精简路由层）

路由层只负责 HTTP 接口，所有业务逻辑委托到：
  - app.services.lu_protocol  : 输入模型 + 常量配置
  - app.services.lu_regime    : 市场状态识别 + 六维评分
  - app.services.lu_portfolio : 组合优化（v2 规则 / v3 协方差）
  - app.rox_quant.lu_qiyuan_prediction : 核心预测引擎
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

# ── 服务子模块 ──────────────────────────────────────────────────
from app.services.lu_protocol import (
    PROTOCOL_VERSION, AnalyzerInput, ScanInput, PortfolioInput,
    RISK_CAPS, ADAPTIVE_WEIGHTS,
    normalize_weights, normalize_code, normalize_code_list,
)
from app.services.lu_regime import (
    identify_market_regime,
    build_component_scores,
    calc_composite_score,
    build_action_plan,
    get_top_drivers,
)
from app.services.lu_portfolio import allocate_v2, allocate_v3_cov

from app.rox_quant.lu_qiyuan_prediction import (
    LuQiyuanPredictionEngine,
    PositionManager,
    get_prediction_engine,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lu-prediction", tags=["卢麒元预测系统"])


# ── 旧版兼容模型 ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    code: str
    include_reasoning: bool = True


class PositionRequest(BaseModel):
    code: str
    action: str
    shares: int
    price: float


# ── 旧版兼容路由 ────────────────────────────────────────────────

@router.get("/predict/{code}")
async def predict_stock(code: str) -> Dict[str, Any]:
    """股票预测（旧版兼容，建议改用 POST /predict-v2）"""
    code = normalize_code(code)
    try:
        engine = get_prediction_engine()
        price_data = await _get_price_data(code)
        fundamental_data = await _get_fundamental_data(code)
        market_data = await _get_market_data()
        macro_data = _mock_macro()
        result = engine.predict(
            code=code, price_data=price_data,
            fundamental_data=fundamental_data,
            market_data=market_data, macro_data=macro_data,
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
            "compat": {"protocol": PROTOCOL_VERSION, "use_v2": "POST /api/lu-prediction/predict-v2"},
        }
    except Exception as e:
        logger.error(f"预测失败 {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── v5 核心路由 ─────────────────────────────────────────────────

@router.post("/predict-v2")
async def predict_v2(inp: AnalyzerInput) -> Dict[str, Any]:
    """v5 协议标准化预测：六维评分 + market_regime + action_plan"""
    if inp.market != "CN_A":
        raise HTTPException(status_code=400, detail="当前版本仅支持 CN_A 市场")

    code = normalize_code(inp.code)
    price_data, fundamental_data, market_data = await asyncio.gather(
        _get_price_data(code, lookback=inp.lookback),
        _get_fundamental_data(code),
        _get_market_data(),
        return_exceptions=True,
    )
    if isinstance(price_data, Exception): price_data = pd.DataFrame()
    if isinstance(fundamental_data, Exception): fundamental_data = _mock_fundamental()
    if isinstance(market_data, Exception): market_data = {}

    price_ok = isinstance(price_data, pd.DataFrame) and not price_data.empty
    data_quality = {
        "price_available": price_ok,
        "price_rows": len(price_data) if price_ok else 0,
        "fundamental_available": bool(fundamental_data),
        "warnings": [] if price_ok else ["price_data_unavailable"],
    }

    engine = get_prediction_engine()
    try:
        result = engine.predict(
            code=code,
            price_data=price_data if price_ok else pd.DataFrame(),
            fundamental_data=fundamental_data,
            market_data=market_data,
            macro_data=_mock_macro(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测引擎异常: {e}")

    market_regime, regime_confidence = identify_market_regime(market_data, price_data if price_ok else pd.DataFrame())
    adaptive_weights = normalize_weights(ADAPTIVE_WEIGHTS.get(market_regime, ADAPTIVE_WEIGHTS["range"]))
    component_scores = build_component_scores(result, market_data, fundamental_data)
    composite_score = calc_composite_score(component_scores, market_regime)

    caps = RISK_CAPS.get(inp.risk_preference, RISK_CAPS["balanced"])
    regime_cap = caps.get(f"regime_{market_regime}", caps["portfolio_max"])
    position_ratio = min(result.position_ratio, regime_cap)

    action_plan = build_action_plan(composite_score, market_regime, inp.risk_preference, result, position_ratio)
    top_drivers = get_top_drivers(component_scores)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "input": {"code": code, "market": inp.market, "period": inp.period, "risk_preference": inp.risk_preference},
        "data_quality": data_quality,
        "analysis": {
            "composite_score": composite_score,
            "component_scores": component_scores,
            "top_drivers": top_drivers,
            "market_regime": market_regime,
            "regime_confidence": round(regime_confidence, 2),
            "adaptive_weights": adaptive_weights,
            "action_plan": action_plan,
            "signal": result.signal,
            "confidence": round(result.confidence, 2),
            "position_ratio": round(position_ratio, 2),
            "risk_level": result.risk_level,
        },
        "compat": {
            "signal": result.signal,
            "direction": round(result.direction, 2),
            "target_price": round(result.target_price, 2) if result.target_price else None,
            "stop_loss": round(result.stop_loss, 2) if result.stop_loss else None,
            "reasoning": result.reasoning,
        },
        "as_of": datetime.now().isoformat(),
    }


@router.post("/scan-v2")
async def scan_v2(inp: ScanInput) -> Dict[str, Any]:
    """批量扫描候选池，按综合评分排序，返回 TopN leaders"""
    codes = normalize_code_list(inp.codes)
    if not codes:
        raise HTTPException(status_code=400, detail="codes 不能为空")

    market_data = await _get_market_data()
    engine = get_prediction_engine()
    results = []

    for code in codes:
        try:
            price_data = await _get_price_data(code, lookback=inp.lookback)
            fundamental_data = await _get_fundamental_data(code)
            result = engine.predict(
                code=code, price_data=price_data,
                fundamental_data=fundamental_data,
                market_data=market_data, macro_data=_mock_macro(),
            )
            composite_score = round(min(100, max(0, 50 + result.direction * 10)), 2)
            results.append({
                "code": code, "composite_score": composite_score,
                "signal": result.signal,
                "confidence": round(result.confidence, 2),
                "position_ratio": round(result.position_ratio, 2),
                "win_probability": round(result.win_probability, 2),
            })
        except Exception as e:
            results.append({"code": code, "error": str(e), "composite_score": 0})

    results.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    leaders = [r for r in results if "error" not in r][:inp.top_n]

    return {
        "protocol_version": PROTOCOL_VERSION,
        "input_count": len(inp.codes),
        "normalized_count": len(codes),
        "results": results,
        "leaders": leaders,
        "as_of": datetime.now().isoformat(),
    }


@router.post("/portfolio-v2")
async def portfolio_v2(inp: PortfolioInput) -> Dict[str, Any]:
    """组合权重分配 v2：规则分配 + single_max/portfolio_max 约束"""
    codes = normalize_code_list(inp.codes)
    if not codes:
        raise HTTPException(status_code=400, detail="codes 不能为空")

    scan_result = await scan_v2(ScanInput(
        codes=codes, market=inp.market,
        risk_preference=inp.risk_preference, lookback=inp.lookback,
    ))
    leaders = scan_result["leaders"]
    weights, risk_summary = allocate_v2(leaders, inp.risk_preference)
    total_weight = round(sum(weights.values()), 4)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "input_count": len(inp.codes),
        "normalized_count": len(codes),
        "weights": weights,
        "total_invested": total_weight,
        "cash_weight": risk_summary["cash_weight"],
        "risk_summary": risk_summary,
        "leaders_used": len(leaders),
        "as_of": datetime.now().isoformat(),
    }


@router.post("/portfolio-v3")
async def portfolio_v3(inp: PortfolioInput) -> Dict[str, Any]:
    """组合权重分配 v3：协方差驱动 + 桶约束"""
    codes = normalize_code_list(inp.codes)
    if not codes:
        raise HTTPException(status_code=400, detail="codes 不能为空")

    price_map: Dict[str, pd.Series] = {}
    for code in codes:
        try:
            df = await _get_price_data(code, lookback=inp.lookback)
            if isinstance(df, pd.DataFrame) and not df.empty and "close" in df.columns:
                price_map[code] = df["close"].astype(float)
        except Exception:
            pass

    weights, risk_summary = allocate_v3_cov(codes, price_map, inp.risk_preference)
    total_weight = round(sum(weights.values()), 4)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "input_count": len(inp.codes),
        "normalized_count": len(codes),
        "weights": weights,
        "total_invested": total_weight,
        "cash_weight": risk_summary["cash_weight"],
        "risk_summary": risk_summary,
        "valid_codes_used": len(price_map),
        "as_of": datetime.now().isoformat(),
    }


# ── 其他路由 ────────────────────────────────────────────────────

@router.get("/batch-predict")
async def batch_predict(codes: str = Query(..., description="股票代码，逗号分隔")):
    """批量预测（旧版兼容）"""
    code_list = normalize_code_list([c.strip() for c in codes.split(",") if c.strip()])
    results = []
    engine = get_prediction_engine()
    market_data = await _get_market_data()
    for code in code_list[:10]:
        try:
            price_data = await _get_price_data(code)
            result = engine.predict(
                code=code, price_data=price_data,
                fundamental_data=_mock_fundamental(),
                market_data=market_data, macro_data=_mock_macro(),
            )
            results.append({
                "code": code, "signal": result.signal,
                "confidence": round(result.confidence, 2),
                "position_ratio": round(result.position_ratio, 2),
            })
        except Exception as e:
            results.append({"code": code, "error": str(e)})
    return {"results": results, "count": len(results)}


@router.get("/position/{code}")
async def get_position(code: str):
    try:
        return {"code": code, **_get_position_manager().get_position_status(code)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/position/trade")
async def execute_trade(req: PositionRequest):
    try:
        manager = _get_position_manager()
        if req.action == "buy":
            result = manager.rhythm_buy(req.code, req.price, req.shares)
        elif req.action == "sell":
            result = manager.rhythm_sell(req.code, req.price, req.shares)
        else:
            raise HTTPException(status_code=400, detail="无效操作")
        return {"success": True, **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accuracy-stats")
async def get_accuracy_stats():
    return get_prediction_engine().get_accuracy_stats()


@router.get("/methodology")
async def get_methodology():
    return {
        "name": "卢麒元方法论增强预测系统",
        "protocol_version": PROTOCOL_VERSION,
        "phase": "phase5",
        "services": {
            "lu_protocol": "输入模型 + 风险约束配置",
            "lu_regime": "市场状态识别 + 六维评分",
            "lu_portfolio": "组合优化（v2规则/v3协方差）",
        },
        "examples": {
            "predict_v2": {"method": "POST", "url": "/api/lu-prediction/predict-v2",
                           "body": {"code": "600519", "market": "CN_A", "lookback": 120}},
            "scan_v2": {"method": "POST", "url": "/api/lu-prediction/scan-v2",
                        "body": {"codes": ["600519", "000858", "601318"], "top_n": 3}},
            "portfolio_v3": {"method": "POST", "url": "/api/lu-prediction/portfolio-v3",
                             "body": {"codes": ["600519", "000858"], "risk_preference": "balanced"}},
        },
    }


@router.get("/market-phase")
async def get_market_phase():
    try:
        market_data = await _get_market_data()
        engine = get_prediction_engine()
        macro_data = _mock_macro()
        macro_signal = engine._analyze_macro(macro_data)
        contradiction = engine._analyze_contradiction(market_data)
        market_regime, regime_confidence = identify_market_regime(market_data, pd.DataFrame())
        return {
            "phase": macro_signal.phase,
            "market_regime": market_regime,
            "regime_confidence": round(regime_confidence, 2),
            "crisis_probability": round(macro_signal.crisis_probability, 2),
            "capital_turnover": round(macro_signal.capital_turnover, 3),
            "asset_allocation": macro_signal.asset_allocation,
            "primary_contradiction": contradiction.primary_contradiction,
            "contradiction_strength": contradiction.contradiction_strength,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 内部数据获取 ────────────────────────────────────────────────

async def _get_price_data(code: str, lookback: int = 120) -> pd.DataFrame:
    try:
        code6 = normalize_code(code)
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=lookback * 2)).strftime("%Y%m%d")
        import akshare as ak

        def fetch():
            return ak.stock_zh_a_hist(symbol=code6, period="daily", start_date=start, end_date=end, adjust="qfq")

        df = await asyncio.to_thread(fetch)
        if df is None or df.empty:
            return pd.DataFrame()
        col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"}
        df = df.rename(columns=col_map)
        for col in ["open", "close", "high", "low", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["close"]).tail(lookback)
    except Exception as e:
        logger.warning(f"获取价格数据失败 {code}: {e}")
        return pd.DataFrame()


async def _get_fundamental_data(code: str) -> Dict:
    try:
        from app.api.endpoints.stock import calculate_fundamentals
        result = await asyncio.to_thread(calculate_fundamentals, normalize_code(code))
        if result and isinstance(result, dict):
            m = result.get("metrics", {})
            return {
                "roe": float(m.get("roe", 0.12)) if m.get("roe") else 0.12,
                "pe": float(m.get("pe", 15)) if m.get("pe") else 15,
                "pb": float(m.get("pb", 1.5)) if m.get("pb") else 1.5,
                "growth_rate": 0.08, "dividend_yield": 0.02, "beta": 1.0,
                "industry": m.get("industry", ""),
            }
    except Exception:
        pass
    return _mock_fundamental()


async def _get_market_data() -> Dict:
    try:
        import akshare as ak
        df = await asyncio.to_thread(ak.stock_zh_a_spot_em)
        if df is None or df.empty:
            return {}
        chg_col = next((c for c in df.columns if "涨跌幅" in c), None)
        if chg_col:
            up = int((df[chg_col] > 0).sum())
            down = int((df[chg_col] < 0).sum())
            return {
                "up_count": up, "down_count": down,
                "limit_up": int((df[chg_col] >= 9.9).sum()),
                "limit_down": int((df[chg_col] <= -9.9).sum()),
                "up_ratio": up / max(up + down, 1),
                "volume": 1, "avg_volume": 1, "north_flow": 0, "main_flow": 0,
            }
    except Exception as e:
        logger.warning(f"获取市场数据失败: {e}")
    return {}


def _mock_fundamental() -> Dict:
    return {"roe": 0.12, "pe": 15, "pb": 1.5, "growth_rate": 0.08, "dividend_yield": 0.02, "beta": 1.0}


def _mock_macro() -> Dict:
    return {"gdp": 120, "m2": 200, "direct_tax_ratio": 0.35, "gini": 0.38}


# ── 仓位管理 ────────────────────────────────────────────────────
_position_manager = None


def _get_position_manager() -> PositionManager:
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager(total_capital=1_000_000)
    return _position_manager
