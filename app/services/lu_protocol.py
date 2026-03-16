"""
lu_protocol.py — 卢式分析器协议层
包含：输入模型、风险约束配置、自适应权重模板
"""
from typing import List
from pydantic import BaseModel

PROTOCOL_VERSION = "lu-analyzer-v5"


class AnalyzerInput(BaseModel):
    """Phase-4 标准化输入模型"""
    code: str
    market: str = "CN_A"
    period: str = "daily"
    lookback: int = 120
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


# 风险约束：single_max=单票上限，portfolio_max=总仓上限
RISK_CAPS = {
    "conservative": {
        "single_max": 0.10, "portfolio_max": 0.60,
        "regime_bull": 0.60, "regime_bear": 0.30, "regime_range": 0.45,
    },
    "balanced": {
        "single_max": 0.15, "portfolio_max": 0.75,
        "regime_bull": 0.75, "regime_bear": 0.40, "regime_range": 0.60,
    },
    "aggressive": {
        "single_max": 0.20, "portfolio_max": 0.90,
        "regime_bull": 0.90, "regime_bear": 0.55, "regime_range": 0.70,
    },
}

# 自适应权重：按市场状态切换
ADAPTIVE_WEIGHTS = {
    "bull":  {"contradiction": 0.15, "value": 0.25, "macro": 0.15, "technical": 0.30, "sentiment": 0.15},
    "range": {"contradiction": 0.25, "value": 0.30, "macro": 0.20, "technical": 0.15, "sentiment": 0.10},
    "bear":  {"contradiction": 0.30, "value": 0.20, "macro": 0.30, "technical": 0.10, "sentiment": 0.10},
}


def normalize_weights(w: dict) -> dict:
    total = sum(w.values())
    if total <= 0:
        return w
    return {k: round(v / total, 4) for k, v in w.items()}


def normalize_code(code: str) -> str:
    code = str(code).strip()
    if len(code) > 6:
        code = code[-6:]
    return code.zfill(6)


def normalize_code_list(codes: List[str]) -> List[str]:
    """标准化代码列表：去重、6位补零、截断至50条"""
    seen, result = set(), []
    for c in codes:
        c = normalize_code(str(c).strip())
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result[:50]
