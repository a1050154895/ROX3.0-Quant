#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据提供者模块
整合所有数据获取功能

模块结构：
- base: 基础类和配置
- history: 历史K线数据
- realtime: 实时行情数据
- funds_flow: 资金流向数据
- forex: 外汇数据
"""

from app.rox_quant.data_provider import DataProvider, PricePoint, get_data_provider

__all__ = [
    "DataProvider",
    "PricePoint",
    "get_data_provider",
]
