# -*- coding: utf-8 -*-
"""
寻龙诀选股器 - 通达信公式转 Python 简化实现。
条件：前高、倍量、突破前高、涨幅与量能、涨停（昨日未涨停）。
"""
import pandas as pd
import numpy as np
from app.analysis.indicators import HHV, MA


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列为小写英文：open, high, low, close, volume。"""
    col_map = {
        "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume",
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    }
    out = df.copy()
    for cn, en in col_map.items():
        if cn in out.columns and en not in out.columns:
            out[en] = out[cn]
    return out


def _zt_price(ref_close: float, is_st: bool = False, is_cyb_kcb: bool = False) -> float:
    """涨停价：ST 5%，创业板/科创板 20%，主板 10%。"""
    if is_st:
        return round(ref_close * 1.05, 2)
    if is_cyb_kcb:
        return round(ref_close * 1.20, 2)
    return round(ref_close * 1.10, 2)


def xunlongjue_signal(df: pd.DataFrame, code: str = "") -> dict:
    """
    单只股票寻龙诀选股信号（最后一根K线是否满足）。
    df 需含 open, high, low, close, volume（或中文列名）。
    返回 {"pass": bool, "reason": str, "detail": dict}。
    """
    df = _ensure_columns(df)
    if df.empty or len(df) < 100:
        return {"pass": False, "reason": "数据不足", "detail": {}}
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    # 前高：9 日前的 19 日最高价（REF(HHV(H,19),9)）
    hhv19 = HHV(h, 19)
    prev_high = hhv19.shift(9).ffill()

    # 倍量、突破前高
    vol_ratio = v / v.shift(1)
    beiliang = (vol_ratio >= 1.1).iloc[-1]
    breakout = (c.iloc[-1] > prev_high.iloc[-1]) and (c.iloc[-2] <= prev_high.iloc[-2]) if len(df) >= 2 else False

    # 当日涨幅 >5%
    pct_up = (c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100 if len(df) >= 2 else 0
    x11 = pct_up > 5

    # 涨停近似（主板 10%）
    ref_c = c.iloc[-2] if len(df) >= 2 else c.iloc[-1]
    zt = _zt_price(ref_c, is_st=False, is_cyb_kcb=False)
    is_zt = c.iloc[-1] >= zt * 0.998
    # 昨日未涨停
    ref_c2 = c.iloc[-3] if len(df) >= 3 else c.iloc[-2]
    zt_prev = _zt_price(ref_c2, is_st=False, is_cyb_kcb=False)
    prev_not_zt = c.iloc[-2] < zt_prev * 0.998 if len(df) >= 3 else True

    # 放量阳线或跳空
    vol_up = vol_ratio.iloc[-1] > 1.2 if not pd.isna(vol_ratio.iloc[-1]) else False
    yang = c.iloc[-1] > o.iloc[-1]
    gap_up = l.iloc[-1] > h.iloc[-2] if len(df) >= 2 else False
    x27 = (vol_up and yang) or (gap_up and vol_up)

    # 综合：倍量 + 突破前高 + 涨幅>5% + (涨停或接近) + 昨日未涨停 + 量价配合
    pass_ = beiliang and breakout and x11 and is_zt and prev_not_zt and x27
    
    # === 宽松模式 (Loose Mode) ===
    # 条件：
    # 1. 突破前高 (必选)
    # 2. 涨幅 > 3% (放宽到 3%)
    # 3. 量比 > 0.8 (放宽)
    # 4. 均线多头 (ma5 > ma10 > ma20)
    ma5 = MA(c, 5).iloc[-1]
    ma10 = MA(c, 10).iloc[-1]
    ma20 = MA(c, 20).iloc[-1]
    trend_up = (ma5 > ma10) and (ma10 > ma20)
    
    pass_loose = breakout and (pct_up > 3.0) and (vol_ratio.iloc[-1] > 0.8) and trend_up
    
    reason_parts = []
    if pass_:
        reason_parts.append("【寻龙诀】倍量突破涨停")
    elif pass_loose:
        reason_parts.append("【强势观察】突破+多头趋势")
        
    if not pass_ and not pass_loose:
        if not beiliang: reason_parts.append("(缺倍量)")
        if not breakout: reason_parts.append("(未突破前高)")
        if not x11: reason_parts.append("(涨幅不足)")
    
    final_pass = pass_ or pass_loose
    reason = " ".join(reason_parts) if reason_parts else "不通过"
    
    detail = {
        "beiliang": bool(beiliang),
        "breakout": breakout,
        "pct_up": round(float(pct_up), 2),
        "is_zt": is_zt,
        "prev_not_zt": prev_not_zt,
        "x27": x27,
        "mode": "strict" if pass_ else ("loose" if pass_loose else "none")
    }
    return {"pass": bool(final_pass), "reason": reason, "detail": detail}


def screen_codes(df_list_by_code: dict) -> list:
    """
    对多只股票的结果 DataFrame 做寻龙诀筛选。
    df_list_by_code: { "600519": df, "000001": df, ... }
    返回通过的 code 列表及简要信息。
    """
    result = []
    for code, df in df_list_by_code.items():
        if df is None or df.empty:
            continue
        r = xunlongjue_signal(df, code=code)
        if r["pass"]:
            result.append({
                "code": code,
                "reason": r["reason"],
                "detail": r["detail"],
            })
    return result
