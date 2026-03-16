"""
lu_portfolio.py — 卢式分析器：组合优化引擎
包含：v2 规则分配、v3 协方差优化 + 桶约束
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from app.services.lu_protocol import RISK_CAPS


def allocate_v2(
    leaders: List[dict],
    risk_pref: str = "balanced",
) -> Tuple[Dict[str, float], dict]:
    """
    v2 规则分配：按 composite_score 比例分配，受 single_max 约束。
    返回 (weights, risk_summary)
    """
    caps = RISK_CAPS.get(risk_pref, RISK_CAPS["balanced"])
    single_max = caps["single_max"]
    portfolio_max = caps["portfolio_max"]

    total_score = sum(r.get("composite_score", 0) for r in leaders)
    weights: Dict[str, float] = {}

    if total_score > 0:
        for r in leaders:
            raw_w = (r.get("composite_score", 0) / total_score) * portfolio_max
            weights[r["code"]] = round(min(raw_w, single_max), 4)
    else:
        eq = round(min(portfolio_max / max(len(leaders), 1), single_max), 4)
        for r in leaders:
            weights[r["code"]] = eq

    total_weight = round(sum(weights.values()), 4)
    cash_weight = round(max(0.0, 1.0 - total_weight), 4)

    return weights, {
        "single_max": single_max,
        "portfolio_max": portfolio_max,
        "cash_weight": cash_weight,
        "risk_preference": risk_pref,
    }


def allocate_v3_cov(
    codes: List[str],
    price_map: Dict[str, pd.Series],
    risk_pref: str = "balanced",
) -> Tuple[Dict[str, float], dict]:
    """
    v3 协方差驱动的组合优化：简化风险平价 + 桶约束。
    返回 (weights, risk_summary)
    """
    caps = RISK_CAPS.get(risk_pref, RISK_CAPS["balanced"])
    single_max = caps["single_max"]
    portfolio_max = caps["portfolio_max"]
    bucket_cap = min(0.35, portfolio_max)

    valid_codes = [c for c in codes if c in price_map]
    weights: Dict[str, float] = {}

    if len(valid_codes) >= 2:
        returns = pd.DataFrame(
            {c: price_map[c].pct_change().dropna() for c in valid_codes}
        ).dropna()

        if len(returns) >= 10:
            cov = returns.cov().values
            n = len(valid_codes)
            reg_cov = cov + np.eye(n) * 1e-4
            try:
                inv_cov = np.linalg.pinv(reg_cov)
                raw_w = inv_cov @ np.ones(n)
                raw_w = np.maximum(raw_w, 0)
                total_raw = raw_w.sum()
                raw_w = raw_w / total_raw if total_raw > 0 else np.ones(n) / n
            except Exception:
                raw_w = np.ones(n) / n

            for i, code in enumerate(valid_codes):
                weights[code] = round(float(min(raw_w[i] * portfolio_max, single_max)), 4)
        else:
            eq = round(min(portfolio_max / len(valid_codes), single_max), 4)
            for c in valid_codes:
                weights[c] = eq
    elif len(valid_codes) == 1:
        weights[valid_codes[0]] = round(min(portfolio_max, single_max), 4)

    # 桶约束：代码前3位相同视为同板块
    bucket_totals: Dict[str, float] = {}
    for code, w in weights.items():
        bucket = code[:3]
        bucket_totals[bucket] = bucket_totals.get(bucket, 0) + w

    for bucket, bw in bucket_totals.items():
        if bw > bucket_cap:
            scale = bucket_cap / bw
            for code in list(weights.keys()):
                if code[:3] == bucket:
                    weights[code] = round(weights[code] * scale, 4)

    total_weight = round(sum(weights.values()), 4)
    cash_weight = round(max(0.0, 1.0 - total_weight), 4)

    return weights, {
        "single_max": single_max,
        "portfolio_max": portfolio_max,
        "bucket_cap": bucket_cap,
        "cash_weight": cash_weight,
        "risk_preference": risk_pref,
        "optimizer": "covariance_driven",
    }
