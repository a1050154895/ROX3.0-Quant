#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卢麒元方法论预测API
提供股票预测、仓位管理、回测等功能
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from app.rox_quant.lu_qiyuan_prediction import (
    PositionManager,
    get_prediction_engine,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lu-prediction", tags=["卢麒元预测系统"])


class PredictRequest(BaseModel):
    """预测请求（旧版）"""
    code: str
    include_reasoning: bool = True


class PositionRequest(BaseModel):
    """仓位请求"""
    code: str
    action: str
    shares: int
    price: float


class AnalyzerInput(BaseModel):
    """卢式分析器标准输入协议（Phase-4）"""
    code: str = Field(..., description="股票代码")
    market: Literal["CN_A", "US", "CRYPTO"] = Field(default="CN_A", description="市场类型")
    timeframe: Literal["1d", "1w", "1m"] = Field(default="1d", description="分析周期")
    lookback_days: int = Field(default=240, ge=60, le=1500, description="回看窗口天数")
    risk_profile: Literal["conservative", "balanced", "aggressive"] = Field(
        default="balanced", description="风险偏好"
    )
    include_reasoning: bool = True


class DataQuality(BaseModel):
    has_price_data: bool
    has_market_data: bool
    has_macro_data: bool
    has_fundamental_data: bool
    degraded: bool
    warnings: List[str]


def _build_action_plan(signal: str, risk_level: str, confidence: float) -> Dict[str, str]:
    """Phase-2: 决策动作建议卡片"""
    if "卖" in signal:
        action = "减仓/止盈止损"
    elif "买" in signal:
        action = "分批建仓"
    else:
        action = "观望"

    if risk_level in {"高风险", "中高风险"}:
        risk_advice = "单票仓位控制在10%以内，务必设置止损"
    else:
        risk_advice = "可按计划执行，保持纪律化交易"

    if confidence >= 0.75:
        execution = "可执行主计划，盘中仅做小幅动态调整"
    elif confidence >= 0.55:
        execution = "建议半仓试错，等待二次确认"
    else:
        execution = "仅跟踪，不建议立即重仓"

    return {
        "action": action,
        "risk_advice": risk_advice,
        "execution": execution,
    }


def _build_phase2_components(engine, price_data: pd.DataFrame, fundamental_data: Dict, market_data: Dict, macro_data: Dict) -> Dict[str, Any]:
    """Phase-2: 输出多维分项评分与证据。"""
    contradiction = engine._analyze_contradiction(market_data)
    value_signal = engine._analyze_value(fundamental_data, price_data)
    macro_signal = engine._analyze_macro(macro_data)
    technical_signal = engine._analyze_technical(price_data)
    sentiment_score = engine._analyze_sentiment(market_data)

    components = {
        "contradiction": {
            "score": round(max(-5.0, min(5.0, contradiction.direction)), 4),
            "confidence": round(_safe_float(contradiction.confidence), 4),
            "weight": engine.weights.get("contradiction", 0.0),
            "evidence": contradiction.description,
        },
        "value": {
            "score": round(max(-5.0, min(5.0, engine._value_to_score(value_signal))), 4),
            "confidence": round(_safe_float(value_signal.confidence), 4),
            "weight": engine.weights.get("value", 0.0),
            "evidence": f"deviation={round(_safe_float(value_signal.deviation_ratio), 4)}, grade={value_signal.value_grade}",
        },
        "macro": {
            "score": round(max(-5.0, min(5.0, engine._macro_to_score(macro_signal))), 4),
            "confidence": round(0.7 if macro_signal.crisis_probability < 0.3 else 0.9, 4),
            "weight": engine.weights.get("macro", 0.0),
            "evidence": f"phase={macro_signal.phase}, crisis_probability={round(_safe_float(macro_signal.crisis_probability), 4)}",
        },
        "technical": {
            "score": round(max(-5.0, min(5.0, engine._technical_to_score(technical_signal))), 4),
            "confidence": 0.6,
            "weight": engine.weights.get("technical", 0.0),
            "evidence": f"macd={technical_signal.macd_signal}, rsi={round(_safe_float(technical_signal.rsi_value), 2)}",
        },
        "sentiment": {
            "score": round(max(-5.0, min(5.0, sentiment_score)), 4),
            "confidence": 0.5,
            "weight": engine.weights.get("sentiment", 0.0),
            "evidence": f"up_ratio={round(_safe_float(market_data.get('up_ratio', 0.5)), 4)}",
        },
    }

    weighted_score = 0.0
    total_weight = 0.0
    for item in components.values():
        eff = _safe_float(item["weight"]) * _safe_float(item["confidence"])
        weighted_score += _safe_float(item["score"]) * eff
        total_weight += eff
    final_score = weighted_score / total_weight if total_weight > 0 else 0.0

    return {
        "components": components,
        "composite_score": round(final_score, 4),
        "top_drivers": sorted(
            [{"name": k, "impact": round(abs(_safe_float(v["score"]) * _safe_float(v["weight"])), 4)} for k, v in components.items()],
            key=lambda x: x["impact"],
            reverse=True,
        )[:3],
    }


def _detect_market_regime(market_data: Dict, macro_data: Dict) -> Dict[str, Any]:
    """Phase-3: 市场状态识别（牛/熊/震荡）"""
    up_ratio = _safe_float(market_data.get("up_ratio", 0.5), 0.5)
    limit_up = _safe_float(market_data.get("limit_up", 0.0), 0.0)
    limit_down = _safe_float(market_data.get("limit_down", 0.0), 0.0)
    crisis = _safe_float(macro_data.get("crisis_probability", 0.2), 0.2)

    if crisis > 0.45 or (up_ratio < 0.42 and limit_down > limit_up):
        regime = "bear"
    elif up_ratio > 0.58 and limit_up >= limit_down:
        regime = "bull"
    else:
        regime = "range"

    confidence = min(0.95, max(0.5, abs(up_ratio - 0.5) * 2 + abs(limit_up - limit_down) / 200))
    return {
        "regime": regime,
        "confidence": round(confidence, 4),
        "metrics": {
            "up_ratio": round(up_ratio, 4),
            "limit_up": int(limit_up),
            "limit_down": int(limit_down),
            "macro_crisis_probability": round(crisis, 4),
        },
    }


def _adaptive_weights(base_weights: Dict[str, float], regime: str, risk_profile: str) -> Dict[str, float]:
    """Phase-3: 自适应权重"""
    w = dict(base_weights)
    if regime == "bull":
        w["technical"] = w.get("technical", 0.0) + 0.07
        w["sentiment"] = w.get("sentiment", 0.0) + 0.03
        w["macro"] = max(0.05, w.get("macro", 0.0) - 0.05)
    elif regime == "bear":
        w["macro"] = w.get("macro", 0.0) + 0.08
        w["value"] = w.get("value", 0.0) + 0.05
        w["sentiment"] = max(0.03, w.get("sentiment", 0.0) - 0.04)
    else:
        w["value"] = w.get("value", 0.0) + 0.04
        w["contradiction"] = w.get("contradiction", 0.0) + 0.03

    if risk_profile == "aggressive":
        w["technical"] = w.get("technical", 0.0) + 0.04
        w["macro"] = max(0.05, w.get("macro", 0.0) - 0.03)
    elif risk_profile == "conservative":
        w["macro"] = w.get("macro", 0.0) + 0.04
        w["sentiment"] = max(0.02, w.get("sentiment", 0.0) - 0.03)

    s = sum(max(0.0, x) for x in w.values()) or 1.0
    return {k: round(max(0.0, v) / s, 6) for k, v in w.items()}


def _risk_caps(risk_profile: str, regime: str) -> Dict[str, float]:
    """Phase-4: 组合级风控约束。"""
    caps = {
        "conservative": {"single_max": 0.12, "portfolio_max": 0.70},
        "balanced": {"single_max": 0.18, "portfolio_max": 0.85},
        "aggressive": {"single_max": 0.25, "portfolio_max": 1.0},
    }.get(risk_profile, {"single_max": 0.18, "portfolio_max": 0.85})

    if regime == "bear":
        caps["single_max"] = max(0.08, caps["single_max"] - 0.05)
        caps["portfolio_max"] = max(0.45, caps["portfolio_max"] - 0.20)
    elif regime == "bull":
        caps["single_max"] = min(0.30, caps["single_max"] + 0.03)

    return {k: round(v, 4) for k, v in caps.items()}


def _build_portfolio_plan(candidates: List[Dict[str, Any]], risk_profile: str) -> Dict[str, Any]:
    """Phase-4: 根据候选池生成组合权重建议。"""
    if not candidates:
        return {"allocations": [], "summary": {"total_weight": 0.0, "count": 0}}

    regime = candidates[0].get("regime", "range")
    caps = _risk_caps(risk_profile, regime)

    scored = [c for c in candidates if c.get("composite_score", -999) > -998 and "error" not in c]
    scored = sorted(scored, key=lambda x: x.get("composite_score", -999), reverse=True)
    if not scored:
        return {"allocations": [], "summary": {"total_weight": 0.0, "count": 0, "caps": caps}}

    base = [max(0.0, _safe_float(x.get("composite_score", 0.0)) + 5.0) for x in scored]
    total = sum(base) or 1.0
    raw = [v / total for v in base]

    allocations: List[Dict[str, Any]] = []
    used = 0.0
    for item, w in zip(scored, raw):
        w1 = min(w, caps["single_max"])
        if used + w1 > caps["portfolio_max"]:
            w1 = max(0.0, caps["portfolio_max"] - used)
        if w1 <= 0:
            continue
        used += w1
        allocations.append({
            "code": item.get("code"),
            "weight": round(w1, 4),
            "signal": item.get("signal"),
            "composite_score": round(_safe_float(item.get("composite_score", 0.0)), 4),
            "confidence": round(_safe_float(item.get("confidence", 0.0)), 4),
        })
        if used >= caps["portfolio_max"] - 1e-9:
            break

    return {
        "allocations": allocations,
        "summary": {
            "count": len(allocations),
            "total_weight": round(sum(x["weight"] for x in allocations), 4),
            "cash_weight": round(1 - sum(x["weight"] for x in allocations), 4),
            "caps": caps,
            "regime": regime,
        },
    }


def _infer_sector_bucket(code: str) -> str:
    c = _normalize_code(code)
    if c.startswith(("600", "601", "603", "605")):
        return "main_sh"
    if c.startswith(("000", "001", "002")):
        return "main_sz"
    if c.startswith(("300", "301")):
        return "growth"
    if c.startswith(("688",)):
        return "star"
    return "other"


def _apply_bucket_caps(allocations: List[Dict[str, Any]], bucket_cap: float = 0.45) -> List[Dict[str, Any]]:
    """Phase-5: 简易行业/板块暴露约束（按代码桶近似）。"""
    by_bucket: Dict[str, float] = {}
    out: List[Dict[str, Any]] = []
    for a in allocations:
        bucket = _infer_sector_bucket(a["code"])
        used = by_bucket.get(bucket, 0.0)
        room = max(0.0, bucket_cap - used)
        w = min(a["weight"], room)
        if w <= 0:
            continue
        by_bucket[bucket] = used + w
        b = dict(a)
        b["weight"] = round(w, 4)
        b["bucket"] = bucket
        out.append(b)
    return out


def _optimize_with_covariance(candidates: List[Dict[str, Any]], cov: np.ndarray, caps: Dict[str, float]) -> List[Dict[str, Any]]:
    """Phase-5: 基于协方差的简化权重优化。"""
    if not candidates:
        return []
    scores = np.array([max(0.01, _safe_float(c.get("composite_score", 0.0)) + 5.0) for c in candidates], dtype=float)
    n = len(scores)
    cov = cov if cov.shape == (n, n) else np.eye(n)
    reg = 1e-3
    try:
        inv = np.linalg.pinv(cov + np.eye(n) * reg)
        raw = inv @ scores
    except Exception:
        raw = scores.copy()
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        raw = np.ones(n)
    raw = raw / raw.sum()

    allocs = []
    used = 0.0
    for i, c in enumerate(candidates):
        w = min(float(raw[i]), caps["single_max"])
        if used + w > caps["portfolio_max"]:
            w = max(0.0, caps["portfolio_max"] - used)
        if w <= 0:
            continue
        used += w
        allocs.append({
            "code": c.get("code"),
            "weight": round(w, 4),
            "signal": c.get("signal"),
            "composite_score": round(_safe_float(c.get("composite_score", 0.0)), 4),
            "confidence": round(_safe_float(c.get("confidence", 0.0)), 4),
        })
        if used >= caps["portfolio_max"] - 1e-9:
            break
    return allocs


async def _build_covariance_matrix(codes: List[str], lookback_days: int = 120) -> np.ndarray:
    """Phase-5: 构建收益率协方差矩阵。"""
    frames = []
    valid_codes = []
    for code in codes:
        df = await _get_price_data(code, lookback_days=lookback_days)
        if df is None or df.empty or "close" not in df.columns:
            continue
        s = pd.to_numeric(df["close"], errors="coerce").pct_change().dropna()
        if len(s) < 20:
            continue
        s.name = _normalize_code(code)
        frames.append(s)
        valid_codes.append(_normalize_code(code))

    if not frames:
        return np.eye(len(codes))

    ret = pd.concat(frames, axis=1).dropna(how="any")
    if ret.empty or ret.shape[1] == 0:
        return np.eye(len(codes))

    cov_small = ret.cov().values
    idx = {_normalize_code(c): i for i, c in enumerate(valid_codes)}
    n = len(codes)
    cov = np.eye(n)
    for i, c1 in enumerate(codes):
        for j, c2 in enumerate(codes):
            i1 = idx.get(_normalize_code(c1))
            i2 = idx.get(_normalize_code(c2))
            if i1 is not None and i2 is not None:
                cov[i, j] = cov_small[i1, i2]
    return cov


def _normalize_code_list(codes: List[str], limit: int = 200) -> List[str]:
    """标准化代码列表：去空、去重、截断。"""
    out: List[str] = []
    seen = set()
    for raw in codes:
        c = _normalize_code(raw)
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def _normalize_code(code: str) -> str:
    c = str(code).strip().lower()
    if c.startswith(("sh", "sz")) and len(c) >= 8:
        return c[-6:]
    return c[-6:] if len(c) >= 6 else c.zfill(6)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


async def _build_engine_result(analyzer_input: AnalyzerInput) -> Dict[str, Any]:
    code = _normalize_code(analyzer_input.code)
    engine = get_prediction_engine()

    price_data = await _get_price_data(code, analyzer_input.lookback_days)
    fundamental_data = await _get_fundamental_data(code)
    market_data = await _get_market_data()
    macro_data = await _get_macro_data()

    warnings: List[str] = []
    if price_data.empty:
        warnings.append("price_data_unavailable")
    if not fundamental_data:
        warnings.append("fundamental_data_unavailable")
    if not market_data:
        warnings.append("market_data_unavailable")
    if not macro_data:
        warnings.append("macro_data_unavailable")

    result = engine.predict(
        code=code,
        price_data=price_data,
        fundamental_data=fundamental_data,
        market_data=market_data,
        macro_data=macro_data,
    )
    component_bundle = _build_phase2_components(engine, price_data, fundamental_data, market_data, macro_data)
    regime = _detect_market_regime(market_data, macro_data)
    weights = _adaptive_weights(engine.weights, regime["regime"], analyzer_input.risk_profile)
    action_plan = _build_action_plan(result.signal, result.risk_level, result.confidence)

    risk_multiplier = {
        "conservative": 0.75,
        "balanced": 1.0,
        "aggressive": 1.2,
    }.get(analyzer_input.risk_profile, 1.0)

    adj_position = min(1.0, max(0.0, result.position_ratio * risk_multiplier))

    data_quality = DataQuality(
        has_price_data=not price_data.empty,
        has_market_data=bool(market_data),
        has_macro_data=bool(macro_data),
        has_fundamental_data=bool(fundamental_data),
        degraded=len(warnings) > 0,
        warnings=warnings,
    )

    protocol_response = {
        "protocol_version": "lu-analyzer-v4",
        "latest_portfolio_protocol": "lu-analyzer-v5",
        "quality_target": "9.5",
        "timestamp": datetime.now().isoformat(),
        "input": analyzer_input.model_dump(),
        "data_quality": data_quality.model_dump(),
        "analysis": {
            "signal": result.signal,
            "confidence": round(result.confidence, 4),
            "direction": round(result.direction, 4),
            "strength": int(result.strength),
            "risk_level": result.risk_level,
            "holding_period": int(result.holding_period),
            "target_price": round(_safe_float(result.target_price), 2) if result.target_price else None,
            "stop_loss": round(_safe_float(result.stop_loss), 2) if result.stop_loss else None,
            "position_ratio": round(adj_position, 4),
            "expected_return": round(_safe_float(result.expected_return), 4),
            "win_probability": round(_safe_float(result.win_probability), 4),
            "reasoning": result.reasoning if analyzer_input.include_reasoning else [],
            "component_scores": component_bundle["components"],
            "composite_score": component_bundle["composite_score"],
            "top_drivers": component_bundle["top_drivers"],
            "action_plan": action_plan,
            "market_regime": regime,
            "adaptive_weights": weights,
        },
        "compat": {
            "code": code,
            "signal": result.signal,
            "confidence": round(result.confidence, 2),
            "direction": round(result.direction, 2),
            "strength": result.strength,
            "target_price": round(result.target_price, 2) if result.target_price else None,
            "stop_loss": round(result.stop_loss, 2) if result.stop_loss else None,
            "position_ratio": round(adj_position, 2),
            "holding_period": result.holding_period,
            "reasoning": result.reasoning if analyzer_input.include_reasoning else [],
            "risk_level": result.risk_level,
            "expected_return": round(result.expected_return, 4),
            "win_probability": round(result.win_probability, 2),
            "timestamp": datetime.now().isoformat(),
        },
    }
    return protocol_response


@router.get("/predict/{code}")
async def predict_stock(code: str) -> Dict[str, Any]:
    """旧版预测接口（保持兼容）"""
    try:
        protocol_result = await _build_engine_result(
            AnalyzerInput(code=code, market="CN_A", timeframe="1d", lookback_days=240, risk_profile="balanced")
        )
        return protocol_result["compat"]
    except Exception as e:
        logger.error(f"预测失败 {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-v2")
async def predict_stock_v2(payload: AnalyzerInput) -> Dict[str, Any]:
    """新版预测接口（标准输入输出协议）"""
    try:
        if payload.market != "CN_A":
            raise HTTPException(status_code=400, detail="phase4 当前仅支持 CN_A")
        return await _build_engine_result(payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"predict-v2 failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch-predict")
async def batch_predict(codes: str = Query(..., description="股票代码，逗号分隔")) -> Dict:
    """批量预测"""
    code_list = [c.strip() for c in codes.split(",") if c.strip()]
    results = []

    for code in code_list[:10]:
        try:
            protocol_result = await _build_engine_result(AnalyzerInput(code=code))
            c = protocol_result["compat"]
            results.append(
                {
                    "code": c["code"],
                    "signal": c["signal"],
                    "confidence": c["confidence"],
                    "position_ratio": c["position_ratio"],
                    "win_probability": c["win_probability"],
                    "degraded": protocol_result["data_quality"]["degraded"],
                }
            )
        except Exception as e:
            results.append({"code": code, "error": str(e)})

    return {"results": results, "count": len(results)}


@router.post("/scan-v2")
async def scan_v2(codes: List[str], top_n: int = Query(5, ge=1, le=50), risk_profile: Literal["conservative", "balanced", "aggressive"] = "balanced") -> Dict[str, Any]:
    """Phase-3: 候选池扫描与排名"""
    norm_codes = _normalize_code_list(codes, limit=200)
    if not norm_codes:
        raise HTTPException(status_code=400, detail="codes 不能为空")

    rows: List[Dict[str, Any]] = []
    for code in norm_codes:
        try:
            payload = AnalyzerInput(code=code, risk_profile=risk_profile)
            out = await _build_engine_result(payload)
            ana = out.get("analysis", {})
            rows.append({
                "code": out.get("compat", {}).get("code", code),
                "signal": ana.get("signal"),
                "composite_score": _safe_float(ana.get("composite_score", 0.0)),
                "confidence": _safe_float(ana.get("confidence", 0.0)),
                "win_probability": _safe_float(ana.get("win_probability", 0.0)),
                "regime": ana.get("market_regime", {}).get("regime", "range"),
            })
        except Exception as e:
            rows.append({"code": code, "error": str(e), "composite_score": -999})

    ranked = sorted(rows, key=lambda x: x.get("composite_score", -999), reverse=True)
    return {
        "protocol_version": "lu-analyzer-v4",
        "count": len(ranked),
        "input_count": len(codes),
        "normalized_count": len(norm_codes),
        "top_n": top_n,
        "leaders": ranked[:top_n],
        "all": ranked,
    }


@router.post("/portfolio-v2")
async def portfolio_v2(codes: List[str], top_n: int = Query(8, ge=1, le=50), risk_profile: Literal["conservative", "balanced", "aggressive"] = "balanced") -> Dict[str, Any]:
    """Phase-4: 组合构建建议（含风控约束）"""
    norm_codes = _normalize_code_list(codes, limit=500)
    if not norm_codes:
        raise HTTPException(status_code=400, detail="codes 不能为空")
    scan = await scan_v2(codes=norm_codes, top_n=top_n, risk_profile=risk_profile)
    leaders = scan.get("leaders", [])
    plan = _build_portfolio_plan(leaders, risk_profile)
    return {
        "protocol_version": "lu-analyzer-v4",
        "risk_profile": risk_profile,
        "top_n": top_n,
        "leaders": leaders,
        "portfolio": plan,
        "diagnostics": {"input_count": len(codes), "normalized_count": len(norm_codes)},
    }


@router.post("/portfolio-v3")
async def portfolio_v3(codes: List[str], top_n: int = Query(8, ge=1, le=50), risk_profile: Literal["conservative", "balanced", "aggressive"] = "balanced", bucket_cap: float = Query(0.45, ge=0.2, le=0.8)) -> Dict[str, Any]:
    """Phase-5: 协方差优化 + 暴露约束组合建议。"""
    norm_codes = _normalize_code_list(codes, limit=500)
    if not norm_codes:
        raise HTTPException(status_code=400, detail="codes 不能为空")
    scan = await scan_v2(codes=norm_codes, top_n=top_n, risk_profile=risk_profile)
    leaders = scan.get("leaders", [])
    if not leaders:
        return {
            "protocol_version": "lu-analyzer-v5",
            "risk_profile": risk_profile,
            "top_n": top_n,
            "leaders": [],
            "portfolio": {"allocations": [], "summary": {"total_weight": 0.0, "cash_weight": 1.0}},
        }

    regime = leaders[0].get("regime", "range")
    caps = _risk_caps(risk_profile, regime)
    cov = await _build_covariance_matrix([x.get("code") for x in leaders], lookback_days=120)
    alloc = _optimize_with_covariance(leaders, cov, caps)
    alloc = _apply_bucket_caps(alloc, bucket_cap=bucket_cap)
    total_w = round(sum(a["weight"] for a in alloc), 4)

    return {
        "protocol_version": "lu-analyzer-v5",
        "risk_profile": risk_profile,
        "top_n": top_n,
        "leaders": leaders,
        "diagnostics": {"input_count": len(codes), "normalized_count": len(norm_codes)},
        "portfolio": {
            "allocations": alloc,
            "summary": {
                "count": len(alloc),
                "total_weight": total_w,
                "cash_weight": round(1 - total_w, 4),
                "caps": caps,
                "bucket_cap": bucket_cap,
                "regime": regime,
            },
        },
    }


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
        "phase": "phase5",
        "protocol_version": "lu-analyzer-v4",
        "phase4_scope": [
            "市场状态识别与自适应权重",
            "候选池扫描与排序",
            "组合权重建议与风控约束",
            "兼容层持续保留",
        ],
        "core_methods": [
            {
                "name": "矛盾分析法",
                "description": "识别市场主要矛盾，把握主要矛盾的主要方面",
                "factors": ["量价背离", "资金分歧", "行业分化", "政策预期"],
                "weight": 0.25,
            },
            {
                "name": "价值规律",
                "description": "价格围绕价值波动，识别背离机会",
                "factors": ["ROE", "PE", "PB", "增长率", "股息率"],
                "weight": 0.30,
            },
            {
                "name": "334法则",
                "description": "30%底仓 + 30%律动 + 40%预备队",
                "factors": ["底仓持有", "律动操作", "风险预备"],
                "weight": "仓位管理",
            },
            {
                "name": "律动操作",
                "description": "通过波段降低持仓成本",
                "factors": ["MACD金叉死叉", "支撑阻力", "成交量"],
                "weight": "操作策略",
            },
            {
                "name": "宏观周期",
                "description": "资本周转效率决定大方向",
                "factors": ["GDP/M2", "直接税占比", "基尼系数"],
                "weight": 0.20,
            },
        ],
        "signal_levels": {
            "强烈买入": "综合得分 >= 3.5",
            "买入": "综合得分 >= 2.5",
            "偏多": "综合得分 >= 1.5",
            "中性": "综合得分在 -0.5 到 0.5 之间",
            "偏空": "综合得分 <= -1.5",
            "卖出": "综合得分 <= -2.5",
            "强烈卖出": "综合得分 <= -3.5",
        },
        "risk_levels": ["低风险", "中低风险", "中风险", "中高风险", "高风险"],
        "advanced_phase5": {
            "portfolio_v3": "/api/lu-prediction/portfolio-v3",
            "features": ["covariance_optimization", "bucket_exposure_cap", "risk_profile_constraints"]
        },
        "examples": {
            "predict_v2": {
                "method": "POST",
                "path": "/api/lu-prediction/predict-v2",
                "body": {
                    "code": "600000",
                    "market": "CN_A",
                    "timeframe": "1d",
                    "lookback_days": 240,
                    "risk_profile": "balanced"
                }
            },
            "scan_v2": {
                "method": "POST",
                "path": "/api/lu-prediction/scan-v2?top_n=5&risk_profile=balanced",
                "body": ["600000", "000001", "300750"]
            },
            "portfolio_v2": {
                "method": "POST",
                "path": "/api/lu-prediction/portfolio-v2?top_n=8&risk_profile=balanced",
                "body": ["600000", "000001", "300750", "002415"]
            }
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
    if contradiction.contradiction_strength > 70:
        return f"市场存在{contradiction.primary_contradiction}，建议谨慎操作"
    if macro.capital_turnover < 0.45:
        return "资本周转效率较低，建议选择优质标的，避免追高"
    return "市场相对稳定，可按334法则逐步建仓"


async def _get_price_data(code: str, lookback_days: int = 240) -> pd.DataFrame:
    """获取价格数据"""
    try:
        import akshare as ak
        import asyncio

        code6 = _normalize_code(code)

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
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").tail(lookback_days)

        for col in ["open", "close", "high", "low", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df
    except Exception as e:
        logger.warning(f"获取价格数据失败 {code}: {e}")
        return pd.DataFrame()


async def _get_fundamental_data(code: str) -> Dict:
    """获取基本面数据"""
    return {
        "roe": 0.12,
        "pe": 15,
        "pb": 1.5,
        "growth_rate": 0.08,
        "dividend_yield": 0.02,
        "beta": 1.0,
    }


async def _get_market_data() -> Dict:
    """获取市场数据"""
    try:
        import akshare as ak
        import asyncio

        df = await asyncio.to_thread(ak.stock_zh_a_spot_em)

        if df is None or df.empty:
            return {}

        change_col = next((c for c in df.columns if "涨跌幅" in c), None)
        volume_col = next((c for c in df.columns if "成交量" in c), None)
        if change_col:
            up_count = len(df[df[change_col] > 0])
            down_count = len(df[df[change_col] < 0])
            limit_up = len(df[df[change_col] >= 9.9])
            limit_down = len(df[df[change_col] <= -9.9])
        else:
            up_count = down_count = limit_up = limit_down = 0

        volume = float(df[volume_col].fillna(0).sum()) if volume_col else 0.0
        avg_volume = float(df[volume_col].fillna(0).mean()) if volume_col else 1.0

        return {
            "up_count": up_count,
            "down_count": down_count,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "up_ratio": up_count / (up_count + down_count) if (up_count + down_count) > 0 else 0.5,
            "volume": volume,
            "avg_volume": max(avg_volume, 1.0),
            "north_flow": 0,
            "main_flow": 0,
            "turnover_rate": 0.03,
        }
    except Exception as e:
        logger.warning(f"获取市场数据失败: {e}")
        return {}


async def _get_macro_data() -> Dict:
    """获取宏观数据"""
    return {
        "gdp": 120,
        "m2": 200,
        "direct_tax_ratio": 0.35,
        "gini": 0.38,
    }


_position_manager = None


def _get_position_manager() -> PositionManager:
    """获取仓位管理器"""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager(total_capital=1000000)
    return _position_manager
