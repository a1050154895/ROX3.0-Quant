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
    try:
        service = get_market_data_service()
        indices = await service.get_indices()
        if indices:
            return {"indices": indices}
        else:
            # 提供示例数据
            sample_indices = [
                {"code": "000001.SH", "name": "上证指数", "price": 4144.08, "change": 0.0, "change_pct": 0.0},
                {"code": "399001.SZ", "name": "深证成指", "price": 13888.88, "change": -11.11, "change_pct": -0.08},
                {"code": "399006.SZ", "name": "创业板指", "price": 2777.77, "change": -22.22, "change_pct": -0.79},
                {"code": "399300.SZ", "name": "沪深300", "price": 4999.99, "change": -33.33, "change_pct": -0.66},
            ]
            return {"indices": sample_indices, "is_sample": True}
    except Exception as e:
        logger.error(f"获取市场指数失败: {e}")
        # 提供示例数据
        sample_indices = [
            {"code": "000001.SH", "name": "上证指数", "price": 4144.08, "change": 0.0, "change_pct": 0.0},
            {"code": "399001.SZ", "name": "深证成指", "price": 13888.88, "change": -11.11, "change_pct": -0.08},
            {"code": "399006.SZ", "name": "创业板指", "price": 2777.77, "change": -22.22, "change_pct": -0.79},
            {"code": "399300.SZ", "name": "沪深300", "price": 4999.99, "change": -33.33, "change_pct": -0.66},
        ]
        return {"indices": sample_indices, "error": "网络连接失败，显示示例数据", "is_sample": True}


@router.get("/stats")
async def api_market_stats():
    """获取市场统计数据"""
    try:
        service = get_market_data_service()
        stats = await service.get_stats()
        if stats:
            return {"stats": stats}
        else:
            # 提供示例数据
            sample_stats = {
                "total_stocks": 5000,
                "up_count": 2000,
                "down_count": 2500,
                "flat_count": 500,
                "limit_up": 50,
                "limit_down": 20,
                "total_volume": 500000000000,
                "total_amount": 8000000000000,
                "avg_change": -0.2,
            }
            return {"stats": sample_stats, "is_sample": True}
    except Exception as e:
        logger.error(f"获取市场统计失败: {e}")
        # 提供示例数据
        sample_stats = {
            "total_stocks": 5000,
            "up_count": 2000,
            "down_count": 2500,
            "flat_count": 500,
            "limit_up": 50,
            "limit_down": 20,
            "total_volume": 500000000000,
            "total_amount": 8000000000000,
            "avg_change": -0.2,
        }
        return {"stats": sample_stats, "error": "网络连接失败，显示示例数据", "is_sample": True}


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
            # 提供示例数据
            sample_items = [
                {"code": "600000", "name": "浦发银行", "price": 10.50, "change_pct": 10.0, "volume": 10000000, "amount": 105000000, "turnover": 5.2},
                {"code": "600036", "name": "招商银行", "price": 35.80, "change_pct": 8.5, "volume": 5000000, "amount": 179000000, "turnover": 3.1},
                {"code": "601318", "name": "中国平安", "price": 48.25, "change_pct": 7.2, "volume": 8000000, "amount": 386000000, "turnover": 2.5},
                {"code": "000001", "name": "平安银行", "price": 12.34, "change_pct": 6.8, "volume": 12345678, "amount": 152345678.90, "turnover": 4.8},
                {"code": "601166", "name": "兴业银行", "price": 18.60, "change_pct": 5.5, "volume": 6000000, "amount": 111600000, "turnover": 3.6},
            ]
            return {"items": sample_items, "type": type, "direction": direction, "is_sample": True}
        
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
        # 提供示例数据
        sample_items = [
            {"code": "600000", "name": "浦发银行", "price": 10.50, "change_pct": 10.0, "volume": 10000000, "amount": 105000000, "turnover": 5.2},
            {"code": "600036", "name": "招商银行", "price": 35.80, "change_pct": 8.5, "volume": 5000000, "amount": 179000000, "turnover": 3.1},
            {"code": "601318", "name": "中国平安", "price": 48.25, "change_pct": 7.2, "volume": 8000000, "amount": 386000000, "turnover": 2.5},
            {"code": "000001", "name": "平安银行", "price": 12.34, "change_pct": 6.8, "volume": 12345678, "amount": 152345678.90, "turnover": 4.8},
            {"code": "601166", "name": "兴业银行", "price": 18.60, "change_pct": 5.5, "volume": 6000000, "amount": 111600000, "turnover": 3.6},
        ]
        return {"items": sample_items, "type": type, "direction": direction, "error": "网络连接失败，显示示例数据", "is_sample": True}


@router.get("/sector-fund-flow")
async def api_sector_fund_flow():
    """获取板块资金流向"""
    try:
        flows = await get_sector_fund_flow()
        if flows:
            return {"items": flows}
        else:
            # 提供示例数据
            sample_flows = [
                {"code": "bank", "name": "银行", "main_inflow": 1000000000, "change_pct": 1.5},
                {"code": "tech", "name": "科技", "main_inflow": 800000000, "change_pct": 2.3},
                {"code": "consumer", "name": "消费", "main_inflow": 600000000, "change_pct": 0.8},
                {"code": "healthcare", "name": "医疗", "main_inflow": 400000000, "change_pct": 1.2},
                {"code": "energy", "name": "能源", "main_inflow": 200000000, "change_pct": 0.5},
            ]
            return {"items": sample_flows, "is_sample": True}
    except Exception as e:
        logger.error(f"获取板块资金流向失败: {e}")
        # 提供示例数据
        sample_flows = [
            {"code": "bank", "name": "银行", "main_inflow": 1000000000, "change_pct": 1.5},
            {"code": "tech", "name": "科技", "main_inflow": 800000000, "change_pct": 2.3},
            {"code": "consumer", "name": "消费", "main_inflow": 600000000, "change_pct": 0.8},
            {"code": "healthcare", "name": "医疗", "main_inflow": 400000000, "change_pct": 1.2},
            {"code": "energy", "name": "能源", "main_inflow": 200000000, "change_pct": 0.5},
        ]
        return {"items": sample_flows, "error": "网络连接失败，显示示例数据", "is_sample": True}


@router.get("/sector-flow")
async def api_sector_flow_alias():
    """板块资金流向别名"""
    return await api_sector_fund_flow()


@router.get("/rotation")
async def api_sector_rotation():
    """获取板块轮动数据"""
    try:
        from app.utils.data_fetcher import data_fetcher
        
        # 使用统一数据获取器
        sectors = await data_fetcher.get_sector_list()
        
        if sectors:
            # 转换数据格式
            items = []
            for sector in sectors[:50]:  # 限制返回数量
                items.append({
                    "code": sector.get("code", ""),
                    "name": sector.get("name", ""),
                    "change_pct": sector.get("change_pct", 0),
                    "up_count": sector.get("up_count", 0),
                    "down_count": sector.get("down_count", 0),
                })
            return {"items": items}
        else:
            # 提供示例数据
            sample_items = [
                {"code": "BK0475", "name": "半导体", "change_pct": 3.5, "up_count": 35, "down_count": 5},
                {"code": "BK0447", "name": "新能源车", "change_pct": 2.8, "up_count": 42, "down_count": 8},
                {"code": "BK0736", "name": "人工智能", "change_pct": 2.5, "up_count": 28, "down_count": 3},
                {"code": "BK0438", "name": "医药生物", "change_pct": 1.2, "up_count": 56, "down_count": 14},
                {"code": "BK0448", "name": "光伏产业", "change_pct": 1.0, "up_count": 22, "down_count": 6},
            ]
            return {"items": sample_items, "is_sample": True}
        
    except Exception as e:
        logger.error(f"获取板块轮动失败: {e}")
        # 提供示例数据
        sample_items = [
            {"code": "BK0475", "name": "半导体", "change_pct": 3.5, "up_count": 35, "down_count": 5},
            {"code": "BK0447", "name": "新能源车", "change_pct": 2.8, "up_count": 42, "down_count": 8},
            {"code": "BK0736", "name": "人工智能", "change_pct": 2.5, "up_count": 28, "down_count": 3},
            {"code": "BK0438", "name": "医药生物", "change_pct": 1.2, "up_count": 56, "down_count": 14},
            {"code": "BK0448", "name": "光伏产业", "change_pct": 1.0, "up_count": 22, "down_count": 6},
        ]
        return {"items": sample_items, "error": "网络连接失败，显示示例数据", "is_sample": True}


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
