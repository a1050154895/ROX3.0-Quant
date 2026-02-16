#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场API - 统计分析
处理市场统计、资金流向、龙虎榜等分析API
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
    get_market_data_service, get_sector_fund_flow, get_dragon_tiger
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])


@router.get("/indices")
async def api_market_indices():
    """获取主要市场指数"""
    service = get_market_data_service()
    indices = await service.get_indices()
    return {"indices": indices}


@router.get("/stats")
async def api_market_stats():
    """获取市场统计数据"""
    service = get_market_data_service()
    stats = await service.get_stats()
    return {"stats": stats}


@router.get("/rankings")
async def api_market_rankings(
    type: str = "change_pct",
    direction: str = "desc",
    limit: int = 50
):
    """获取股票排行榜"""
    try:
        df = await asyncio.to_thread(ak.stock_zh_a_spot_em)
        
        if df is None or df.empty:
            return {"items": []}
        
        column_map = {
            "change_pct": "涨跌幅",
            "turnover": "换手率",
            "amount": "成交额",
            "volume": "成交量",
        }
        
        sort_col = column_map.get(type, "涨跌幅")
        
        if sort_col not in df.columns:
            return {"items": [], "error": f"未找到排序列: {sort_col}"}
        
        ascending = direction == "asc"
        df = df.sort_values(by=sort_col, ascending=ascending).head(limit)
        
        items = []
        for _, row in df.iterrows():
            items.append({
                "code": str(row.get("代码", "")),
                "name": row.get("名称", ""),
                "price": float(row.get("最新价", 0)),
                "change_pct": float(row.get("涨跌幅", 0)),
                "volume": int(float(row.get("成交量", 0))),
                "amount": float(row.get("成交额", 0)),
                "turnover": float(row.get("换手率", 0)) if "换手率" in row else 0,
            })
        
        return {"items": items, "type": type, "direction": direction}
        
    except Exception as e:
        logger.error(f"获取排行榜失败: {e}")
        return {"items": [], "error": str(e)}


@router.get("/sector-fund-flow")
async def api_sector_fund_flow():
    """获取板块资金流向"""
    flows = await get_sector_fund_flow()
    return {"items": flows}


@router.get("/sector-flow")
async def api_sector_flow_alias():
    """板块资金流向别名"""
    return await api_sector_fund_flow()


@router.get("/rotation")
async def api_sector_rotation():
    """获取板块轮动数据"""
    try:
        df = await asyncio.to_thread(ak.stock_board_concept_name_em)
        
        if df is None or df.empty:
            return {"items": []}
        
        items = []
        for _, row in df.head(50).iterrows():
            items.append({
                "code": str(row.get("代码", "")),
                "name": row.get("板块名称", ""),
                "change_pct": float(row.get("涨跌幅", 0)) if "涨跌幅" in row else 0,
                "up_count": int(row.get("上涨家数", 0)) if "上涨家数" in row else 0,
                "down_count": int(row.get("下跌家数", 0)) if "下跌家数" in row else 0,
            })
        
        return {"items": items}
        
    except Exception as e:
        logger.error(f"获取板块轮动失败: {e}")
        return {"items": [], "error": str(e)}


@router.get("/dragon-tiger")
async def api_dragon_tiger(date: str = None):
    """获取龙虎榜数据"""
    items = await get_dragon_tiger(date)
    return {"items": items, "date": date}


@router.get("/dragon-tiger/{code}")
async def api_dragon_tiger_detail(code: str):
    """获取个股龙虎榜详情"""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        df = await asyncio.to_thread(
            ak.stock_lhb_detail_em,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            return {"items": [], "code": code}
        
        code6 = code[-6:] if len(code) >= 6 else code.zfill(6)
        df = df[df["代码"].astype(str).str.contains(code6)]
        
        items = []
        for _, row in df.iterrows():
            items.append({
                "date": str(row.get("日期", "")),
                "code": str(row.get("代码", "")),
                "name": row.get("名称", ""),
                "close_price": float(row.get("收盘价", 0)),
                "change_pct": float(row.get("涨跌幅", 0)),
                "reason": row.get("上榜原因", ""),
                "net_buy": float(row.get("龙虎榜净买额", 0)),
            })
        
        return {"items": items, "code": code}
        
    except Exception as e:
        logger.error(f"获取龙虎榜详情失败 {code}: {e}")
        return {"items": [], "code": code, "error": str(e)}


@router.get("/sentiment")
async def api_market_sentiment():
    """获取市场情绪指标"""
    try:
        service = get_market_data_service()
        stats = await service.get_stats()
        
        if not stats:
            return {"sentiment": {}}
        
        total = stats.get("total_stocks", 1)
        up = stats.get("up_count", 0)
        down = stats.get("down_count", 0)
        
        up_ratio = up / total if total > 0 else 0
        down_ratio = down / total if total > 0 else 0
        
        sentiment_score = int(up_ratio * 100)
        
        if sentiment_score >= 70:
            level = "极度乐观"
        elif sentiment_score >= 55:
            level = "乐观"
        elif sentiment_score >= 45:
            level = "中性"
        elif sentiment_score >= 30:
            level = "悲观"
        else:
            level = "极度悲观"
        
        return {
            "sentiment": {
                "score": sentiment_score,
                "level": level,
                "up_ratio": round(up_ratio * 100, 2),
                "down_ratio": round(down_ratio * 100, 2),
                "limit_up": stats.get("limit_up", 0),
                "limit_down": stats.get("limit_down", 0),
            }
        }
        
    except Exception as e:
        logger.error(f"获取市场情绪失败: {e}")
        return {"sentiment": {}, "error": str(e)}


@router.get("/overview")
async def get_market_overview():
    """获取市场概览"""
    try:
        service = get_market_data_service()
        
        indices = await service.get_indices()
        stats = await service.get_stats()
        flows = await service.get_fund_flow()
        
        return {
            "indices": indices,
            "stats": stats,
            "top_flows": flows[:10] if flows else [],
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
    except Exception as e:
        logger.error(f"获取市场概览失败: {e}")
        return {"error": str(e)}


@router.get("/concepts")
async def get_market_concepts(limit: int = 20):
    """获取概念板块"""
    try:
        df = await asyncio.to_thread(ak.stock_board_concept_name_em)
        
        if df is None or df.empty:
            return {"items": []}
        
        items = []
        for _, row in df.head(limit).iterrows():
            items.append({
                "code": str(row.get("代码", "")),
                "name": row.get("板块名称", ""),
                "change_pct": float(row.get("涨跌幅", 0)) if "涨跌幅" in row else 0,
                "up_count": int(row.get("上涨家数", 0)) if "上涨家数" in row else 0,
                "down_count": int(row.get("下跌家数", 0)) if "下跌家数" in row else 0,
                "total_count": int(row.get("总家数", 0)) if "总家数" in row else 0,
            })
        
        return {"items": items}
        
    except Exception as e:
        logger.error(f"获取概念板块失败: {e}")
        return {"items": [], "error": str(e)}


@router.get("/heatmap/data")
async def get_heatmap_data():
    """获取市场热力图数据"""
    try:
        df = await asyncio.to_thread(ak.stock_zh_a_spot_em)
        
        if df is None or df.empty:
            return {"items": []}
        
        items = []
        for _, row in df.head(200).iterrows():
            items.append({
                "code": str(row.get("代码", "")),
                "name": row.get("名称", ""),
                "change_pct": float(row.get("涨跌幅", 0)),
                "amount": float(row.get("成交额", 0)),
            })
        
        return {"items": items}
        
    except Exception as e:
        logger.error(f"获取热力图数据失败: {e}")
        return {"items": [], "error": str(e)}
