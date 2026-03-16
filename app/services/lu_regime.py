"""
lu_regime.py — 卢式分析器：市场状态识别 & 评分引擎
包含：市场状态识别、六维组件评分、动作建议生成
"""
from typing import Tuple, Dict
import pandas as pd

from app.services.lu_protocol import ADAPTIVE_WEIGHTS, RISK_CAPS, normalize_weights


def identify_market_regime(market_data: dict, price_data: pd.DataFrame) -> Tuple[str, float]:
    """
    识别市场状态：bull（牛市）/ bear（熊市）/ range（震荡）
    返回 (state, confidence)
    """
    up_ratio = market_data.get("up_ratio", 0.5)
    limit_up = market_data.get("limit_up", 0)
    limit_down = market_data.get("limit_down", 0)

    if up_ratio > 0.65 and limit_up > limit_down * 1.5:
        return "bull", 0.75
    elif up_ratio < 0.35 and limit_down > limit_up * 1.5:
        return "bear", 0.75
    return "range", 0.60


def build_component_scores(result, market_data: dict, fundamental_data: dict) -> Dict[str, dict]:
    """构建六维组件评分（contradiction/value/macro/technical/sentiment）"""

    def _score(signal: float, span: float = 5.0) -> float:
        return round(min(100, max(0, 50 + signal / span * 50)), 2)

    return {
        "contradiction": {
            "score": _score(result.direction * 0.3),
            "confidence": 0.65,
            "weight": 0.25,
            "evidence": "矛盾分析（量价关系/资金分歧）",
        },
        "value": {
            "score": _score(result.direction * 0.4),
            "confidence": 0.70,
            "weight": 0.30,
            "evidence": "ROE={:.0%} PE={:.1f} PB={:.1f}".format(
                fundamental_data.get("roe", 0.12),
                fundamental_data.get("pe", 15),
                fundamental_data.get("pb", 1.5),
            ),
        },
        "macro": {
            "score": _score(result.direction * 0.2),
            "confidence": 0.60,
            "weight": 0.20,
            "evidence": "宏观周期（资本周转/危机概率）",
        },
        "technical": {
            "score": _score(result.direction * 0.3),
            "confidence": 0.60,
            "weight": 0.15,
            "evidence": "MACD/RSI/均线排列",
        },
        "sentiment": {
            "score": _score(market_data.get("up_ratio", 0.5) * 10 - 5),
            "confidence": 0.50,
            "weight": 0.10,
            "evidence": "上涨比={:.0%}".format(market_data.get("up_ratio", 0.5)),
        },
    }


def calc_composite_score(component_scores: dict, market_regime: str) -> float:
    """计算综合评分（0-100）：使用自适应权重加权平均"""
    weights = normalize_weights(ADAPTIVE_WEIGHTS.get(market_regime, ADAPTIVE_WEIGHTS["range"]))
    numer = sum(
        v["score"] * weights.get(k, 0.2) * v.get("confidence", 0.6)
        for k, v in component_scores.items()
    )
    denom = max(
        sum(weights.get(k, 0.2) * v.get("confidence", 0.6) for k, v in component_scores.items()),
        1e-9,
    )
    return round(min(100, max(0, numer / denom)), 2)


def build_action_plan(
    score: float,
    market_regime: str,
    risk_pref: str,
    result,
    position_ratio: float,
) -> dict:
    """生成可执行建议卡片（动作 + 风险提示 + 建议仓位）"""
    if score >= 72:
        action = "可考虑分批建仓（334：首仓30%起）"
        risk_tip = "关注右肩风险，结合 MACD 节奏设好止损"
    elif score >= 58:
        action = "继续观察，等待确认信号后考虑首仓"
        risk_tip = "当前属于跟踪阶段，不宜追高"
    elif score >= 42:
        action = "中性，以观察为主"
        risk_tip = "市场分化，保留机动资金"
    else:
        action = "偏空，控制仓位，优先防御"
        risk_tip = "风险较高，建议降低整体仓位"

    return {
        "action": action,
        "risk_tip": risk_tip,
        "suggested_position": round(position_ratio, 2),
        "market_regime": market_regime,
        "risk_preference": risk_pref,
        "signal": result.signal,
        "confidence_level": (
            "高" if result.confidence > 0.75 else
            "中" if result.confidence > 0.55 else "低"
        ),
    }


def get_top_drivers(component_scores: dict, n: int = 3) -> list:
    """返回评分最高的 Top-N 驱动因子"""
    sorted_items = sorted(component_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return [
        {"factor": k, "score": v["score"], "evidence": v.get("evidence", "")}
        for k, v in sorted_items[:n]
    ]
