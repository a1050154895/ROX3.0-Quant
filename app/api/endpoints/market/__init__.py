#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场API模块
整合所有市场相关的API端点

模块结构：
- models: 数据模型定义
- services: 数据服务层
- quotes: 行情数据API
- watchlist_alerts: 自选股和预警API
- analysis: 市场统计分析API
"""

from fastapi import APIRouter

# 导入子模块路由
from app.api.endpoints.market.quotes import router as quotes_router
from app.api.endpoints.market.watchlist_alerts import router as watchlist_alerts_router
from app.api.endpoints.market.analysis import router as analysis_router

# 创建主路由
router = APIRouter(prefix="/market")

# 包含子路由
router.include_router(quotes_router)
router.include_router(watchlist_alerts_router)
router.include_router(analysis_router)

# 导出服务和模型
from app.api.endpoints.market.services import (
    MarketDataService,
    get_market_data_service,
    symbol_prefix,
    is_index_code,
    index_symbol,
    normalize_kline_df,
)

from app.api.endpoints.market.models import (
    StockRequest,
    WatchlistItem,
    AlertCreate,
    KlineRequest,
    QuoteRequest,
    RealtimeQuote,
    KlineData,
    MarketStats,
    SectorInfo,
    NewsItem,
    FundFlow,
    DragonTigerItem,
    AlertItem,
)

__all__ = [
    # 路由
    "router",
    # 服务
    "MarketDataService",
    "get_market_data_service",
    "symbol_prefix",
    "is_index_code",
    "index_symbol",
    "normalize_kline_df",
    # 模型
    "StockRequest",
    "WatchlistItem",
    "AlertCreate",
    "KlineRequest",
    "QuoteRequest",
    "RealtimeQuote",
    "KlineData",
    "MarketStats",
    "SectorInfo",
    "NewsItem",
    "FundFlow",
    "DragonTigerItem",
    "AlertItem",
]
