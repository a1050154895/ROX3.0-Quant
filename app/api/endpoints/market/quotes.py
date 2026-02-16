#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场API - 行情数据
处理实时行情、K线、分时等数据API
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import pandas as pd
import akshare as ak

from app.api.endpoints.market.services import (
    symbol_prefix, is_index_code, index_symbol,
    normalize_kline_df, fetch_batch_quotes, get_market_data_service
)
from app.analysis.indicators import MACD, KDJ, RSI, BOLL

logger = logging.getLogger(__name__)
router = APIRouter(tags=["quotes"])


@router.get("/spot")
async def api_spot(limit: int = 500, offset: int = 0):
    """获取A股实时行情"""
    try:
        df = await asyncio.to_thread(ak.stock_zh_a_spot_em)
        
        if df is None or df.empty:
            return {"items": [], "total": 0}
        
        total = len(df)
        df = df.iloc[offset:offset + limit]
        
        items = []
        for _, row in df.iterrows():
            items.append({
                "code": str(row.get("代码", "")),
                "name": row.get("名称", ""),
                "price": float(row.get("最新价", 0)),
                "change_pct": float(row.get("涨跌幅", 0)),
                "change": float(row.get("涨跌额", 0)),
                "volume": int(float(row.get("成交量", 0))),
                "amount": float(row.get("成交额", 0)),
                "high": float(row.get("最高", 0)),
                "low": float(row.get("最低", 0)),
                "open": float(row.get("今开", 0)),
                "prev_close": float(row.get("昨收", 0)),
            })
        
        return {"items": items, "total": total}
        
    except Exception as e:
        logger.error(f"获取实时行情失败: {e}")
        return {"items": [], "total": 0, "error": str(e)}


@router.get("/quotes")
async def api_market_quotes(codes: str):
    """批量获取股票行情"""
    if not codes:
        return {"quotes": {}}
    
    code_list = [c.strip() for c in codes.split(",") if c.strip()]
    quotes = await fetch_batch_quotes(code_list)
    
    return {"quotes": quotes}


@router.post("/fetch-realtime")
async def api_fetch_realtime(req: dict):
    """获取单只股票实时数据"""
    code = req.get("stock_name", "")
    if not code:
        return {"error": "缺少股票代码"}
    
    service = get_market_data_service()
    quote = await service.get_realtime_quote(code)
    
    return {
        "p_now": quote["price"],
        "p_change": quote["change"],
        "p_pct": quote["change_pct"]
    }


@router.get("/kline")
async def api_market_kline(
    code: str,
    period: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    count: Optional[int] = None
):
    """获取K线数据"""
    try:
        code = str(code).strip()
        
        if is_index_code(code):
            fetch_code = index_symbol(code)
            df = await asyncio.to_thread(ak.stock_zh_index_daily, symbol=fetch_code)
        else:
            code6 = code[-6:] if len(code) >= 6 else code.zfill(6)
            period_map = {"daily": "daily", "weekly": "weekly", "monthly": "monthly"}
            df = await asyncio.to_thread(
                ak.stock_zh_a_hist,
                symbol=code6,
                period=period_map.get(period, "daily"),
                adjust="qfq"
            )
        
        if df is None or df.empty:
            return {"items": [], "code": code}
        
        df = normalize_kline_df(df)
        
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        if count:
            df = df.tail(count)
        
        items = []
        for _, row in df.iterrows():
            items.append({
                "time": str(row.get("date", "")),
                "open": float(row.get("open", 0)),
                "close": float(row.get("close", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "volume": int(float(row.get("volume", 0))),
                "amount": float(row.get("amount", 0)) if "amount" in row else 0,
            })
        
        return {"items": items, "code": code, "count": len(items)}
        
    except Exception as e:
        logger.error(f"获取K线失败 {code}: {e}")
        return {"items": [], "code": code, "error": str(e)}


@router.get("/fenshi")
async def api_fenshi(code: str):
    """获取分时数据"""
    try:
        code = str(code).strip()
        code6 = code[-6:] if len(code) >= 6 else code.zfill(6)
        
        df = await asyncio.to_thread(ak.stock_zh_a_minute, symbol=code6, period="1")
        
        if df is None or df.empty:
            return {"items": [], "code": code}
        
        items = []
        for _, row in df.iterrows():
            items.append({
                "time": str(row.get("time", "")),
                "price": float(row.get("close", 0)),
                "volume": int(float(row.get("volume", 0))),
                "amount": float(row.get("amount", 0)) if "amount" in row else 0,
            })
        
        return {"items": items, "code": code}
        
    except Exception as e:
        logger.error(f"获取分时失败 {code}: {e}")
        return {"items": [], "code": code, "error": str(e)}


@router.get("/indicators")
async def api_indicators(code: str, period: str = "daily"):
    """获取技术指标"""
    try:
        kline_result = await api_market_kline(code, period, count=100)
        
        if "error" in kline_result or not kline_result.get("items"):
            return {"error": "无法获取K线数据"}
        
        items = kline_result["items"]
        df = pd.DataFrame(items)
        
        indicators = {}
        
        if len(df) >= 26:
            macd = MACD(df["close"])
            indicators["macd"] = {
                "dif": float(macd["dif"].iloc[-1]) if not macd.empty else 0,
                "dea": float(macd["dea"].iloc[-1]) if not macd.empty else 0,
                "macd": float(macd["macd"].iloc[-1]) if not macd.empty else 0,
            }
        
        if len(df) >= 9:
            kdj = KDJ(df["high"], df["low"], df["close"])
            indicators["kdj"] = {
                "k": float(kdj["k"].iloc[-1]) if not kdj.empty else 0,
                "d": float(kdj["d"].iloc[-1]) if not kdj.empty else 0,
                "j": float(kdj["j"].iloc[-1]) if not kdj.empty else 0,
            }
        
        if len(df) >= 14:
            rsi = RSI(df["close"])
            indicators["rsi"] = {
                "rsi6": float(rsi["rsi6"].iloc[-1]) if not rsi.empty else 0,
                "rsi12": float(rsi["rsi12"].iloc[-1]) if not rsi.empty else 0,
                "rsi24": float(rsi["rsi24"].iloc[-1]) if not rsi.empty else 0,
            }
        
        if len(df) >= 20:
            boll = BOLL(df["close"])
            indicators["boll"] = {
                "upper": float(boll["upper"].iloc[-1]) if not boll.empty else 0,
                "middle": float(boll["middle"].iloc[-1]) if not boll.empty else 0,
                "lower": float(boll["lower"].iloc[-1]) if not boll.empty else 0,
            }
        
        indicators["ma"] = {
            "ma5": float(df["close"].rolling(5).mean().iloc[-1]) if len(df) >= 5 else 0,
            "ma10": float(df["close"].rolling(10).mean().iloc[-1]) if len(df) >= 10 else 0,
            "ma20": float(df["close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else 0,
            "ma60": float(df["close"].rolling(60).mean().iloc[-1]) if len(df) >= 60 else 0,
        }
        
        return {"code": code, "indicators": indicators}
        
    except Exception as e:
        logger.error(f"获取技术指标失败 {code}: {e}")
        return {"code": code, "error": str(e)}


@router.get("/stock-suggest")
async def api_stock_suggest(q: str = "", limit: int = 10):
    """股票搜索建议"""
    if not q or len(q) < 1:
        return {"items": []}
    
    try:
        df = await asyncio.to_thread(ak.stock_info_a_code_name)
        
        if df is None or df.empty:
            return {"items": []}
        
        q_lower = q.lower()
        
        mask = (
            df["code"].str.lower().str.contains(q_lower) |
            df["name"].str.lower().str.contains(q_lower)
        )
        matched = df[mask].head(limit)
        
        items = []
        for _, row in matched.iterrows():
            items.append({
                "code": str(row["code"]),
                "name": row["name"],
            })
        
        return {"items": items}
        
    except Exception as e:
        logger.error(f"股票搜索失败: {e}")
        return {"items": []}
