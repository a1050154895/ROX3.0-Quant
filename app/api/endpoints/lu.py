from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

import akshare as ak
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.api.endpoints.stock import _get_stock_basic_info, _normalize_code, calculate_fundamentals
from app.db import get_all_stocks_spot
from app.rox_quant.datasources import ashare_lite

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lu", tags=["卢式分析"])


async def _resolve_symbol(input_symbol: str) -> str:
    raw = str(input_symbol or "").strip()
    if not raw:
        return ""

    # 允许直接输入代码（含 sh600519 / sz000001 形式）
    if raw.lower().startswith(("sh", "sz")) and len(raw) >= 8:
        return _normalize_code(raw[-6:])
    if raw.isdigit():
        return _normalize_code(raw)

    # 名称输入时，优先从全市场快照中匹配
    try:
        df = await get_all_stocks_spot()
        if df is not None and not df.empty:
            cols = df.columns.tolist()
            code_col = next((c for c in cols if "代码" in c or "证券代码" in c or c.lower() in {"code", "symbol"}), "代码")
            name_col = next((c for c in cols if "名称" in c or "简称" in c or c.lower() in {"name", "display_name"}), "名称")
            df[code_col] = df[code_col].astype(str).str.zfill(6)
            exact = df[df[name_col].astype(str).str.strip() == raw]
            if not exact.empty:
                return str(exact.iloc[0][code_col])
            fuzzy = df[df[name_col].astype(str).str.contains(raw, case=False, na=False)].head(1)
            if not fuzzy.empty:
                return str(fuzzy.iloc[0][code_col])
    except Exception as e:
        logger.warning(f"名称解析失败 {raw}: {e}")

    # 进一步回退：静态股票列表匹配（支持离线）
    try:
        static_path = Path("app/static/stock_list.csv")
        if static_path.exists():
            stock_df = pd.read_csv(static_path, dtype=str)
            cols = [c.lower() for c in stock_df.columns]
            code_col = stock_df.columns[0] if stock_df.columns.size > 0 else None
            name_col = stock_df.columns[1] if stock_df.columns.size > 1 else None
            if code_col and name_col:
                exact = stock_df[stock_df[name_col].astype(str).str.strip() == raw]
                if not exact.empty:
                    return _normalize_code(str(exact.iloc[0][code_col]))
                fuzzy = stock_df[stock_df[name_col].astype(str).str.contains(raw, case=False, na=False)].head(1)
                if not fuzzy.empty:
                    return _normalize_code(str(fuzzy.iloc[0][code_col]))
    except Exception as e:
        logger.warning(f"静态列表名称解析失败 {raw}: {e}")

    # 回退：若字符串中包含6位数字则提取，否则返回空（避免将名称误当代码）
    digits = "".join(ch for ch in raw if ch.isdigit())
    return _normalize_code(digits[-6:]) if digits else ""


def _map_theme(industry: str) -> str:
    text = (industry or "").lower()
    mapping = [
        ("黄金", "黄金"),
        ("有色", "资源"),
        ("煤", "能源"),
        ("石油", "能源"),
        ("电力", "电力"),
        ("军工", "防御"),
        ("半导体", "科技"),
        ("软件", "科技"),
        ("通信", "科技"),
        ("银行", "普通风险资产"),
        ("证券", "普通风险资产"),
        ("消费", "普通风险资产"),
    ]
    for key, theme in mapping:
        if key in text:
            return theme
    return "普通风险资产"


async def _fetch_hist(code: str) -> pd.DataFrame:
    loop = asyncio.get_running_loop()
    try:
        df = await loop.run_in_executor(None, lambda: ashare_lite.get_price(code, count=240, frequency="1d"))
        if df is not None and not df.empty:
            return df.rename(columns={"open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量"})
    except Exception as e:
        logger.warning(f"Ashare Lite 拉取失败 {code}: {e}")

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=500)).strftime("%Y%m%d")
    try:
        return await loop.run_in_executor(
            None,
            lambda: ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq"),
        )
    except Exception as e:
        logger.warning(f"AkShare 拉取失败 {code}: {e}")

    local_path = Path(f"data/market_data/daily/{code}.csv")
    if local_path.exists():
        try:
            df = pd.read_csv(local_path)
            rename_map = {"open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量"}
            for k, v in rename_map.items():
                if k in df.columns and v not in df.columns:
                    df[v] = df[k]
            return df
        except Exception as e:
            logger.warning(f"本地CSV读取失败 {code}: {e}")

    return pd.DataFrame()


def _compute_lu_fields(hist: pd.DataFrame) -> Dict[str, Any]:
    if hist is None or hist.empty or len(hist) < 35:
        return {
            "direction_bias": "一般支持",
            "matrix_position": "观望区",
            "direction_note": "该方向判断仅作为辅助判断，不是自动结论。",
            "structure_stage": "结构未明",
            "macd_status": ["零下修复"],
            "discipline_advice": "观察为主",
            "summary": "方向信息有限，结构未明，建议先观察后再评估334节奏。",
        }

    close = hist["收盘"].astype(float)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist_bar = macd - signal

    latest = close.iloc[-1]
    ret20 = (latest / close.iloc[-20] - 1) * 100 if len(close) >= 20 and close.iloc[-20] else 0
    trend_score = 50
    if latest > ma20.iloc[-1]:
        trend_score += 10
    if ma20.iloc[-1] > ma60.iloc[-1]:
        trend_score += 10
    if ma60.iloc[-1] > ma120.iloc[-1] if pd.notna(ma120.iloc[-1]) else False:
        trend_score += 8
    if ret20 > 8:
        trend_score += 8
    elif ret20 < -8:
        trend_score -= 12

    if trend_score >= 75:
        direction_bias, matrix_position = "高度支持", "增强区"
    elif trend_score >= 62:
        direction_bias, matrix_position = "中度支持", "转强区"
    elif trend_score >= 50:
        direction_bias, matrix_position = "一般支持", "分化区"
    elif trend_score >= 40:
        direction_bias, matrix_position = "偏弱支持", "防御区"
    else:
        direction_bias, matrix_position = "暂不支持", "观望区"

    recent_low = close.tail(60).min()
    recent_high = close.tail(60).max()
    range_ratio = (latest - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
    if latest < ma60.iloc[-1] * 0.95 and ma20.iloc[-1] < ma60.iloc[-1]:
        structure_stage = "破位"
    elif latest > ma20.iloc[-1] > ma60.iloc[-1] and macd.iloc[-1] > signal.iloc[-1] and hist_bar.iloc[-1] > 0:
        structure_stage = "主升"
    elif latest > ma60.iloc[-1] and ma20.iloc[-1] >= ma60.iloc[-1]:
        structure_stage = "确认"
    elif latest < ma20.iloc[-1] and latest > ma60.iloc[-1]:
        structure_stage = "右肩风险"
    elif range_ratio < 0.3 and hist_bar.iloc[-1] > hist_bar.iloc[-2]:
        structure_stage = "左脚"
    elif macd.iloc[-1] > signal.iloc[-1]:
        structure_stage = "修复中"
    else:
        structure_stage = "结构未明"

    macd_status: List[str] = []
    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
        macd_status.append("金叉")
    elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
        macd_status.append("死叉")
    macd_status.append("零上强势" if macd.iloc[-1] >= 0 else "零下修复")
    macd_status.append("柱体放大" if abs(hist_bar.iloc[-1]) > abs(hist_bar.iloc[-2]) else "柱体缩短")

    if structure_stage == "主升" and matrix_position in {"增强区", "转强区"}:
        discipline_advice = "可考虑主升布局"
    elif structure_stage == "确认" and matrix_position in {"增强区", "转强区"}:
        discipline_advice = "可考虑二仓确认"
    elif structure_stage in {"左脚", "修复中"} and matrix_position in {"转强区", "分化区"}:
        discipline_advice = "可考虑首仓"
    elif structure_stage in {"破位", "右肩风险"} or matrix_position in {"防御区", "观望区"}:
        discipline_advice = "风险控制"
    else:
        discipline_advice = "观察为主"

    summary = f"方向{direction_bias}，结构处于{structure_stage}，当前以{discipline_advice}为宜。"
    return {
        "direction_bias": direction_bias,
        "matrix_position": matrix_position,
        "direction_note": "该方向判断仅作为辅助判断，不是自动结论。",
        "structure_stage": structure_stage,
        "macd_status": list(dict.fromkeys(macd_status)),
        "discipline_advice": discipline_advice,
        "summary": summary,
    }


@router.get("/analyze-symbol")
async def analyze_symbol(symbol: str = Query(..., description="股票代码或名称，例如 600519")) -> Dict[str, Any]:
    code6 = await _resolve_symbol(symbol)
    if not code6:
        raise HTTPException(status_code=404, detail="无法识别股票代码/名称")

    info, hist, fundamentals = await asyncio.gather(
        _get_stock_basic_info(code6),
        _fetch_hist(code6),
        asyncio.to_thread(calculate_fundamentals, code6),
    )
    metrics = (fundamentals or {}).get("metrics") or {}
    industry = metrics.get("industry") or "未知"
    theme = _map_theme(industry)
    lu_fields = _compute_lu_fields(hist)

    return {
        "symbol": code6,
        "name": (info or {}).get("name") or code6,
        "industry": industry,
        "theme": theme,
        "direction_bias": lu_fields["direction_bias"],
        "matrix_position": lu_fields["matrix_position"],
        "direction_note": lu_fields["direction_note"],
        "structure_stage": lu_fields["structure_stage"],
        "macd_status": lu_fields["macd_status"],
        "discipline_advice": lu_fields["discipline_advice"],
        "risk_note": "仓位由用户手工决定，系统仅做纪律提示。",
        "summary": lu_fields["summary"],
    }

