#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场API - 数据模型
定义市场相关的请求和响应模型
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import date


class StockRequest(BaseModel):
    """股票请求模型"""
    stock_name: str


class WatchlistItem(BaseModel):
    """自选股项目模型"""
    stock_name: str
    stock_code: str
    sector: Optional[str] = None


class AlertCreate(BaseModel):
    """预警创建模型"""
    symbol: str
    name: str = ""
    alert_type: str = Field(..., description="price_above | price_below")
    value: float


class KlineRequest(BaseModel):
    """K线请求模型"""
    code: str
    period: str = "daily"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    count: Optional[int] = None


class QuoteRequest(BaseModel):
    """报价请求模型"""
    codes: List[str]


class FenshiRequest(BaseModel):
    """分时请求模型"""
    code: str


class DiagnosisRequest(BaseModel):
    """诊断请求模型"""
    code: str
    period: str = "daily"


class SectorFlowRequest(BaseModel):
    """板块资金流向请求模型"""
    date: Optional[str] = None


class RealtimeQuote(BaseModel):
    """实时报价模型"""
    code: str
    name: str = ""
    price: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: int = 0
    amount: float = 0.0
    timestamp: Optional[str] = None


class KlineData(BaseModel):
    """K线数据模型"""
    time: str
    open: float
    close: float
    high: float
    low: float
    volume: int
    amount: Optional[float] = None


class MarketStats(BaseModel):
    """市场统计模型"""
    total_stocks: int = 0
    up_count: int = 0
    down_count: int = 0
    flat_count: int = 0
    limit_up: int = 0
    limit_down: int = 0
    avg_change: float = 0.0
    turnover: float = 0.0


class SectorInfo(BaseModel):
    """板块信息模型"""
    code: str
    name: str
    change_pct: float = 0.0
    turnover: float = 0.0
    leading_stock: Optional[str] = None
    leading_change: float = 0.0


class NewsItem(BaseModel):
    """新闻条目模型"""
    title: str
    source: str = ""
    time: str = ""
    url: Optional[str] = None
    summary: Optional[str] = None


class MacroIndicator(BaseModel):
    """宏观指标模型"""
    name: str
    value: float
    unit: str = ""
    period: str = ""
    change: Optional[float] = None
    change_pct: Optional[float] = None


class FundFlow(BaseModel):
    """资金流向模型"""
    sector: str
    net_inflow: float
    net_inflow_pct: float = 0.0
    main_inflow: float = 0.0
    retail_inflow: float = 0.0


class DragonTigerItem(BaseModel):
    """龙虎榜条目模型"""
    code: str
    name: str
    close_price: float
    change_pct: float
    turnover_rate: float = 0.0
    reason: str = ""
    net_buy: float = 0.0
    institutions: List[str] = []


class AlertItem(BaseModel):
    """预警条目模型"""
    id: int
    symbol: str
    name: str = ""
    alert_type: str
    value: float
    status: str = "pending"
    created_at: Optional[str] = None
    triggered_at: Optional[str] = None
