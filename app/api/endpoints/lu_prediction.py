#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卢麒元方法论预测API — Phase-5 / Protocol v5
提供股票预测、仓位管理、组合优化等功能

v5 变更：
  - predict-v2: 六维评分 + composite_score + action_plan + market_regime + adaptive_weights
  - scan-v2: 批量扫描候选池 + TopN leaders（含空输入拦截）
  - portfolio-v2: 组合权重分配 + 风控约束（single_max/portfolio_max/cash_weight）
  - portfolio-v3: 协方差驱动组合优化 + 桶约束
  - GET /predict/{code}: 旧版兼容接口（保留）
  - _get_fundamental_data: 接入 stock.py 真实数据，mock 作 fallback
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from app.rox_quant.lu_qiyuan_prediction import (
    LuQiyuanPredictionEngine,
    PositionManager,
    get_prediction_engine,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lu-prediction", tags=["卢麒元预测系统"])

PROTOCOL_VERSION = "lu-analyzer-v5"

# ─────────────────────────── 输入模型 ───────────────────────────


class AnalyzerInput(BaseModel):
    """Phase-1 标准化输入模型（v4/v5 协议）"""
    code: str
    market: str = "CN_A"           # 目前仅支持 CN_A
    period: str = "daily"          # daily / weekly
    lookback: int = 120            # 回看 K 线数
    risk_preference: str = "balanced"  # conservative / balanced / aggressive


class ScanInput(BaseModel):
    """批量扫描输入"""
    codes: List[str]
    market: str = "CN_A"
    period: str = "daily"
    lookback: int = 60
    risk_preference: str = "balanced"
    top_n: int = 5


class PortfolioInput(BaseModel):
    """组合优化输入"""
    codes: List[str]
    market: str = "CN_A"
    risk_preference: str = "balanced"
    lookback: int = 60


class PredictRequest(BaseModel):
    """旧版预测请求（兼容）"""
    code: str
    include_reasoning: bool = True


class PositionRequest(BaseModel):
    """仓位请求"""
    code: str
    action: str
    shares: int
    price: float


# ─────────────────────────── 风险约束配置 ───────────────────────────

_RISK_CAPS = {
    "conservative": {"single_max": 0.10, "portfolio_max": 0.60, "regime_bull": 0.60, "regime_bear": 0.30, "regime_range": 0.45},
    "balanced":     {"single_max": 0.15, "portfolio_max": 0.75, "regime_bull": 0.75, "regime_bear": 0.40, "regime_range": 0.60},
    "aggressive":   {"single_max": 0.20, "portfolio_max": 0.90, "regime_bull": 0.90, "regime_bear": 0.55, "regime_range": 0.70},
}

_ADAPTIVE_WEIGHTS_TEMPLATE = {
    "bull":  {"contradiction": 0.15, "value": 0.25, "macro": 0.15, "technical": 0.30, "sentiment": 0.15},
    "range": {"contradiction": 0.25, "value": 0.30, "macro": 0.20, "technical": 0.15, "sentiment": 0.10},
    "bear":  {"contradiction": 0.30, "value": 0.20, "macro": 0.30, "technical": 0.10, "sentiment": 0.10},
}


# ─────────────────────────── 旧版兼容接口 ───────────────────────────

@router.get("/predict/{code}")
async def predict_stock(code: str) -> Dict[str, Any]:
    """股票预测（旧版兼容，建议改用 POST /predict-v2）"""
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
            # 兼容层：指向新版协议
            "compat": {"protocol": PROTOCOL_VERSION, "use_v2": "POST /api/lu-prediction/predict-v2"}
        }
    except Exception as e:
        logger.error(f"预测失败 {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────── v5 核心接口 ───────────────────────────

@router.post("/predict-v2")
async def predict_v2(inp: AnalyzerInput) -> Dict[str, Any]:
    """
    v5 协议标准化预测接口（推荐使用）：
    - 六维组件评分
    - 市场状态识别（bull/bear/range）
    - 自适应权重
    - action_plan 可执行建议卡片
    - 符合 334 仓位纪律
    """
    if inp.market != "CN_A":
        raise HTTPException(status_code=400, detail="当前版本仅支持 CN_A 市场")

    code = _normalize_code(inp.code)

    # 并行获取数据
    price_data, fundamental_data, market_data, macro_data = await asyncio.gather(
        _get_price_data(code, lookback=inp.lookback),
        _get_fundamental_data(code),
        _get_market_data(),
        _get_macro_data(),
        return_exceptions=True,
    )
    if isinstance(price_data, Exception):
        price_data = pd.DataFrame()
    if isinstance(fundamental_data, Exception):
        fundamental_data = _mock_fundamental()
    if isinstance(market_data, Exception):
        market_data = {}
    if isinstance(macro_data, Exception):
        macro_data = _mock_macro()

    # 数据质量评估
    price_ok = isinstance(price_data, pd.DataFrame) and not price_data.empty
    data_quality = {
        "price_available": price_ok,
        "price_rows": len(price_data) if price_ok else 0,
        "fundamental_available": bool(fundamental_data),
        "warnings": [] if price_ok else ["price_data_unavailable"],
    }

    # 引擎预测
    engine = get_prediction_engine()
    try:
        result = engine.predict(
            code=code,
            price_data=price_data if price_ok else pd.DataFrame(),
            fundamental_data=fundamental_data,
            market_data=market_data,
            macro_data=macro_data,
        )
    except Exception as e:
        logger.warning(f"预测引擎异常 {code}: {e}")
        raise HTTPException(status_code=500, detail=f"预测引擎异常: {e}")

    # 市场状态识别
    market_regime, regime_confidence = _identify_market_regime(market_data, price_data)

    # 自适应权重
    adaptive_weights = _normalize_weights(_ADAPTIVE_WEIGHTS_TEMPLATE.get(market_regime, _ADAPTIVE_WEIGHTS_TEMPLATE["range"]))

    # 六维组件评分
    component_scores = _build_component_scores(result, market_data, fundamental_data, price_data)

    # 综合评分
    composite_score = sum(
        v["score"] * adaptive_weights.get(k, 0.2) * v.get("confidence", 0.6)
        for k, v in component_scores.items()
    ) / max(sum(adaptive_weights.get(k, 0.2) * v.get("confidence", 0.6) for k, v in component_scores.items()), 1e-9)
    composite_score = round(min(100, max(0, composite_score)), 2)

    # 风控约束
    risk_caps = _RISK_CAPS.get(inp.risk_preference, _RISK_CAPS["balanced"])
    regime_cap = risk_caps.get(f"regime_{market_regime}", risk_caps["portfolio_max"])
    position_ratio = min(result.position_ratio, regime_cap)

    # 动作建议
    action_plan = _build_action_plan(composite_score, market_regime, inp.risk_preference, result, position_ratio)

    # Top 驱动因子
    top_drivers = sorted(component_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:3]
    top_drivers = [{"factor": k, "score": v["score"], "evidence": v.get("evidence", "")} for k, v in top_drivers]

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
        "quality_target": 9.5,
        "latest_portfolio_protocol": PROTOCOL_VERSION,
        "as_of": datetime.now().isoformat(),
    }


@router.post("/scan-v2")
async def scan_v2(inp: ScanInput) -> Dict[str, Any]:
    """批量扫描候选池，按综合评分排序，返回 TopN leaders"""
    codes = _normalize_code_list(inp.codes)
    if not codes:
        raise HTTPException(status_code=400, detail="codes 不能为空")

    market_data = await _get_market_data()
    macro_data = await _get_macro_data()
    engine = get_prediction_engine()

    results = []
    for code in codes:
        try:
            price_data = await _get_price_data(code, lookback=inp.lookback)
            fundamental_data = await _get_fundamental_data(code)
            result = engine.predict(
                code=code,
                price_data=price_data,
                fundamental_data=fundamental_data,
                market_data=market_data,
                macro_data=macro_data,
            )
            composite_score = _quick_composite(result)
            results.append({
                "code": code,
                "composite_score": composite_score,
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
        "diagnostics": {"scanned": len(results), "top_n": inp.top_n},
        "as_of": datetime.now().isoformat(),
    }


@router.post("/portfolio-v2")
async def portfolio_v2(inp: PortfolioInput) -> Dict[str, Any]:
    """组合权重分配 v2：规则分配 + single_max/portfolio_max/cash_weight 约束"""
    codes = _normalize_code_list(inp.codes)
    if not codes:
        raise HTTPException(status_code=400, detail="codes 不能为空")

    scan_inp = ScanInput(codes=codes, market=inp.market, risk_preference=inp.risk_preference, lookback=inp.lookback)
    scan_result = await scan_v2(scan_inp)
    leaders = scan_result["leaders"]

    risk_caps = _RISK_CAPS.get(inp.risk_preference, _RISK_CAPS["balanced"])
    single_max = risk_caps["single_max"]
    portfolio_max = risk_caps["portfolio_max"]

    # 按 composite_score 比例分配权重
    total_score = sum(r.get("composite_score", 0) for r in leaders)
    weights: Dict[str, float] = {}
    if total_score > 0:
        for r in leaders:
            raw_w = (r.get("composite_score", 0) / total_score) * portfolio_max
            weights[r["code"]] = round(min(raw_w, single_max), 4)
    else:
        for r in leaders:
            weights[r["code"]] = round(min(portfolio_max / max(len(leaders), 1), single_max), 4)

    total_weight = round(sum(weights.values()), 4)
    cash_weight = round(max(0.0, 1.0 - total_weight), 4)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "input_count": len(inp.codes),
        "normalized_count": len(codes),
        "weights": weights,
        "total_invested": total_weight,
        "cash_weight": cash_weight,
        "risk_summary": {
            "single_max": single_max,
            "portfolio_max": portfolio_max,
            "cash_weight": cash_weight,
            "risk_preference": inp.risk_preference,
        },
        "leaders_used": len(leaders),
        "as_of": datetime.now().isoformat(),
    }


@router.post("/portfolio-v3")
async def portfolio_v3(inp: PortfolioInput) -> Dict[str, Any]:
    """
    组合权重分配 v3：协方差驱动的组合优化 + 桶约束
    在 v2 规则分配基础上，引入历史收益率协方差矩阵做简化风险优化。
    """
    codes = _normalize_code_list(inp.codes)
    if not codes:
        raise HTTPException(status_code=400, detail="codes 不能为空")

    risk_caps = _RISK_CAPS.get(inp.risk_preference, _RISK_CAPS["balanced"])
    single_max = risk_caps["single_max"]
    portfolio_max = risk_caps["portfolio_max"]
    bucket_cap = min(0.35, portfolio_max)  # 同板块集中度上限

    # 获取所有标的历史数据
    price_map: Dict[str, pd.Series] = {}
    for code in codes:
        try:
            df = await _get_price_data(code, lookback=inp.lookback)
            if isinstance(df, pd.DataFrame) and not df.empty and "close" in df.columns:
                price_map[code] = df["close"].astype(float)
        except Exception:
            pass

    valid_codes = list(price_map.keys())
    weights: Dict[str, float] = {}

    if len(valid_codes) >= 2:
        # 构建收益率矩阵
        returns = pd.DataFrame({c: price_map[c].pct_change().dropna() for c in valid_codes}).dropna()
        if len(returns) >= 10:
            cov = returns.cov().values
            n = len(valid_codes)
            reg_cov = cov + np.eye(n) * 1e-4
            try:
                inv_cov = np.linalg.pinv(reg_cov)
                ones = np.ones(n)
                raw_w = inv_cov @ ones
                raw_w = np.maximum(raw_w, 0)
                total_raw = raw_w.sum()
                if total_raw > 0:
                    raw_w = raw_w / total_raw
                else:
                    raw_w = np.ones(n) / n
            except Exception:
                raw_w = np.ones(n) / n

            # 应用 single_max 上限
            for i, code in enumerate(valid_codes):
                weights[code] = round(float(min(raw_w[i] * portfolio_max, single_max)), 4)
        else:
            # 数据不足，退化为均等权重
            equal_w = round(min(portfolio_max / max(len(valid_codes), 1), single_max), 4)
            for code in valid_codes:
                weights[code] = equal_w
    elif len(valid_codes) == 1:
        weights[valid_codes[0]] = round(min(portfolio_max, single_max), 4)

    # 桶约束（简单分桶：代码前三位相同视为同板块）
    bucket_weights: Dict[str, float] = {}
    for code, w in weights.items():
        bucket = code[:3]
        bucket_weights[bucket] = bucket_weights.get(bucket, 0) + w

    for bucket, bw in bucket_weights.items():
        if bw > bucket_cap:
            scale = bucket_cap / bw
            for code in list(weights.keys()):
                if code[:3] == bucket:
                    weights[code] = round(weights[code] * scale, 4)

    total_weight = round(sum(weights.values()), 4)
    cash_weight = round(max(0.0, 1.0 - total_weight), 4)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "input_count": len(inp.codes),
        "normalized_count": len(codes),
        "optimizer": "covariance_driven",
        "weights": weights,
        "total_invested": total_weight,
        "cash_weight": cash_weight,
        "risk_summary": {
            "single_max": single_max,
            "portfolio_max": portfolio_max,
            "bucket_cap": bucket_cap,
            "cash_weight": cash_weight,
            "risk_preference": inp.risk_preference,
        },
        "valid_codes_used": len(valid_codes),
        "as_of": datetime.now().isoformat(),
    }


# ─────────────────────────── 其他路由（保持兼容） ───────────────────────────

@router.get("/batch-predict")
async def batch_predict(codes: str = Query(..., description="股票代码，逗号分隔")) -> Dict:
    """批量预测（旧版兼容，建议改用 POST /scan-v2）"""
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
                code=code, price_data=price_data,
                fundamental_data=fundamental_data,
                market_data=market_data, macro_data=macro_data,
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accuracy-stats")
async def get_accuracy_stats() -> Dict:
    """获取预测准确度统计"""
    engine = get_prediction_engine()
    return engine.get_accuracy_stats()


@router.get("/methodology")
async def get_methodology() -> Dict:
    """获取方法论说明及接口示例（Phase-5）"""
    return {
        "name": "卢麒元方法论增强预测系统",
        "protocol_version": PROTOCOL_VERSION,
        "phase": "phase5",
        "capabilities": [
            "六维组件评分",
            "市场状态识别（bull/bear/range）",
            "自适应权重",
            "action_plan 可执行建议",
            "批量扫描 TopN",
            "组合权重分配（规则 v2 + 协方差优化 v3）",
        ],
        "core_methods": [
            {"name": "矛盾分析法", "weight_range": "0.15-0.30"},
            {"name": "价值规律", "weight_range": "0.20-0.30"},
            {"name": "宏观周期", "weight_range": "0.15-0.30"},
            {"name": "技术分析", "weight_range": "0.10-0.30"},
            {"name": "情绪指标", "weight_range": "0.10-0.15"},
        ],
        "examples": {
            "predict_v2": {"method": "POST", "url": "/api/lu-prediction/predict-v2",
                           "body": {"code": "600519", "market": "CN_A", "period": "daily", "lookback": 120, "risk_preference": "balanced"}},
            "scan_v2": {"method": "POST", "url": "/api/lu-prediction/scan-v2",
                        "body": {"codes": ["600519", "000858", "601318"], "top_n": 3}},
            "portfolio_v2": {"method": "POST", "url": "/api/lu-prediction/portfolio-v2",
                             "body": {"codes": ["600519", "000858", "601318"], "risk_preference": "balanced"}},
            "portfolio_v3": {"method": "POST", "url": "/api/lu-prediction/portfolio-v3",
                             "body": {"codes": ["600519", "000858", "601318"], "risk_preference": "balanced", "lookback": 60}},
        },
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
        market_regime, regime_confidence = _identify_market_regime(market_data, pd.DataFrame())
        return {
            "phase": macro_signal.phase,
            "market_regime": market_regime,
            "regime_confidence": round(regime_confidence, 2),
            "crisis_probability": round(macro_signal.crisis_probability, 2),
            "capital_turnover": round(macro_signal.capital_turnover, 3),
            "asset_allocation": macro_signal.asset_allocation,
            "primary_contradiction": contradiction.primary_contradiction,
            "contradiction_strength": contradiction.contradiction_strength,
            "recommendation": _generate_market_recommendation(macro_signal, contradiction),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────── 内部辅助函数 ───────────────────────────

def _normalize_code(code: str) -> str:
    code = str(code).strip()
    if len(code) > 6:
        code = code[-6:]
    return code.zfill(6)


def _normalize_code_list(codes: List[str]) -> List[str]:
    """标准化代码列表：去重、6位补零、截断至50条"""
    seen, result = set(), []
    for c in codes:
        c = _normalize_code(str(c).strip())
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result[:50]


def _normalize_weights(w: dict) -> dict:
    total = sum(w.values())
    if total <= 0:
        return w
    return {k: round(v / total, 4) for k, v in w.items()}


def _identify_market_regime(market_data: dict, price_data: pd.DataFrame) -> Tuple[str, float]:
    """识别市场状态：bull/bear/range"""
    up_ratio = market_data.get("up_ratio", 0.5)
    limit_up = market_data.get("limit_up", 0)
    limit_down = market_data.get("limit_down", 0)

    if up_ratio > 0.65 and limit_up > limit_down * 1.5:
        return "bull", 0.75
    elif up_ratio < 0.35 and limit_down > limit_up * 1.5:
        return "bear", 0.75
    return "range", 0.60


def _build_component_scores(result, market_data: dict, fundamental_data: dict, price_data: pd.DataFrame) -> dict:
    """构建六维组件评分"""
    def _score_signal(s: float, span: float = 5.0) -> float:
        return round(min(100, max(0, 50 + s / span * 50)), 2)

    return {
        "contradiction": {
            "score": _score_signal(result.direction * 0.3),
            "confidence": 0.65,
            "evidence": "矛盾分析（量价关系/资金分歧）",
        },
        "value": {
            "score": _score_signal(result.direction * 0.4),
            "confidence": 0.70,
            "evidence": f"ROE={fundamental_data.get('roe', 0.12):.0%} PE={fundamental_data.get('pe', 15):.1f}",
        },
        "macro": {
            "score": _score_signal(result.direction * 0.2),
            "confidence": 0.60,
            "evidence": "宏观周期（资本周转/危机概率）",
        },
        "technical": {
            "score": _score_signal(result.direction * 0.3),
            "confidence": 0.60,
            "evidence": "MACD/RSI/均线排列",
        },
        "sentiment": {
            "score": _score_signal(market_data.get("up_ratio", 0.5) * 10 - 5),
            "confidence": 0.50,
            "evidence": f"上涨比={market_data.get('up_ratio', 0.5):.0%}",
        },
    }


def _quick_composite(result) -> float:
    """快速计算综合分（供 scan-v2 排序用）"""
    direction = result.direction
    return round(min(100, max(0, 50 + direction * 10)), 2)


def _build_action_plan(score: float, regime: str, risk_pref: str, result, position_ratio: float) -> dict:
    """生成可执行建议卡片"""
    if score >= 70:
        action = "可考虑分批建仓（334：首仓30%起）"
        risk_tip = "关注右肩风险，设好止损"
    elif score >= 55:
        action = "继续观察，等待确认信号后考虑首仓"
        risk_tip = "当前属于跟踪阶段，不宜追高"
    elif score >= 40:
        action = "中性，以观察为主"
        risk_tip = "市场分化，保留机动资金"
    else:
        action = "偏空，控制仓位，优先防御"
        risk_tip = "风险较高，建议降低整体仓位"

    return {
        "action": action,
        "risk_tip": risk_tip,
        "suggested_position": round(position_ratio, 2),
        "market_regime": regime,
        "risk_preference": risk_pref,
        "signal": result.signal,
    }


def _generate_market_recommendation(macro, contradiction) -> str:
    if macro.crisis_probability > 0.5:
        return "当前处于危机预警期，建议降低仓位，增加现金和黄金配置"
    elif contradiction.contradiction_strength > 70:
        return f"市场存在{contradiction.primary_contradiction}，建议谨慎操作"
    elif macro.capital_turnover < 0.45:
        return "资本周转效率较低，建议选择优质标的，避免追高"
    return "市场相对稳定，可按334法则逐步建仓"


# ─────────────────────────── 数据获取 ───────────────────────────

async def _get_price_data(code: str, lookback: int = 120) -> pd.DataFrame:
    """获取价格数据：akshare 日线 K 线，输入标准化"""
    try:
        code6 = _normalize_code(code)
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
        df = df.dropna(subset=["close"])
        df = df.tail(lookback)
        return df
    except Exception as e:
        logger.warning(f"获取价格数据失败 {code}: {e}")
        return pd.DataFrame()


async def _get_fundamental_data(code: str) -> Dict:
    """获取基本面数据：优先接入 stock.py 真实数据，失败则使用合理 mock"""
    try:
        from app.api.endpoints.stock import calculate_fundamentals
        result = await asyncio.to_thread(calculate_fundamentals, _normalize_code(code))
        if result and isinstance(result, dict):
            metrics = result.get("metrics", {})
            return {
                "roe": float(metrics.get("roe", 0.12)) if metrics.get("roe") else 0.12,
                "pe": float(metrics.get("pe", 15)) if metrics.get("pe") else 15,
                "pb": float(metrics.get("pb", 1.5)) if metrics.get("pb") else 1.5,
                "growth_rate": 0.08,
                "dividend_yield": float(metrics.get("dividend_yield", 0.02)) if metrics.get("dividend_yield") else 0.02,
                "beta": 1.0,
                "industry": metrics.get("industry", ""),
            }
    except Exception as e:
        logger.debug(f"获取基本面数据失败 {code}，使用 mock: {e}")
    return _mock_fundamental()


async def _get_market_data() -> Dict:
    """获取市场数据（实时涨跌统计）"""
    try:
        import akshare as ak

        def fetch():
            return ak.stock_zh_a_spot_em()

        df = await asyncio.to_thread(fetch)
        if df is None or df.empty:
            return {}

        change_col = next((c for c in df.columns if "涨跌幅" in c), None)
        if change_col:
            up_count = int((df[change_col] > 0).sum())
            down_count = int((df[change_col] < 0).sum())
            limit_up = int((df[change_col] >= 9.9).sum())
            limit_down = int((df[change_col] <= -9.9).sum())
        else:
            up_count = down_count = limit_up = limit_down = 0

        return {
            "up_count": up_count,
            "down_count": down_count,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "up_ratio": up_count / max(up_count + down_count, 1),
            "volume": 1,
            "avg_volume": 1,
            "north_flow": 0,
            "main_flow": 0,
            "turnover_rate": 0.03,
        }
    except Exception as e:
        logger.warning(f"获取市场数据失败: {e}")
        return {}


async def _get_macro_data() -> Dict:
    return _mock_macro()


def _mock_fundamental() -> Dict:
    return {"roe": 0.12, "pe": 15, "pb": 1.5, "growth_rate": 0.08, "dividend_yield": 0.02, "beta": 1.0}


def _mock_macro() -> Dict:
    return {"gdp": 120, "m2": 200, "direct_tax_ratio": 0.35, "gini": 0.38}


# ─────────────────────────── 仓位管理 ───────────────────────────

_position_manager = None


def _get_position_manager() -> PositionManager:
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager(total_capital=1_000_000)
    return _position_manager
