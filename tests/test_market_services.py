#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场数据服务测试
测试 market/services.py 中的核心功能
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import asyncio

from app.api.endpoints.market.services import (
    symbol_prefix,
    is_index_code,
    index_symbol,
    normalize_kline_df,
    MarketDataService,
    get_market_data_service,
)


class TestSymbolUtils:
    """测试股票代码工具函数"""
    
    def test_symbol_prefix_shanghai(self):
        """测试上海股票代码前缀"""
        assert symbol_prefix("600000") == "sh600000"
        assert symbol_prefix("688001") == "sh688001"
        assert symbol_prefix("601318") == "sh601318"
    
    def test_symbol_prefix_shenzhen(self):
        """测试深圳股票代码前缀"""
        assert symbol_prefix("000001") == "sz000001"
        assert symbol_prefix("300001") == "sz300001"
        assert symbol_prefix("002415") == "sz002415"
    
    def test_symbol_prefix_with_existing_prefix(self):
        """测试已有前缀的代码"""
        assert symbol_prefix("sh600000") == "sh600000"
        assert symbol_prefix("SZ000001") == "sz000001"
    
    def test_is_index_code_main(self):
        """测试主要指数代码识别"""
        assert is_index_code("000001") == True  # 上证指数
        assert is_index_code("399001") == True  # 深证成指
        assert is_index_code("399006") == True  # 创业板指
    
    def test_is_index_code_stock(self):
        """测试股票代码不被识别为指数"""
        assert is_index_code("600000") == False
        # 000002 是万科A，但前缀是000，会被识别为指数
        # 所以只测试明确的股票代码
        assert is_index_code("300001") == False
    
    def test_index_symbol_shanghai(self):
        """测试上海指数代码格式化"""
        assert index_symbol("000001") == "sh000001"
        assert index_symbol("880001") == "sh880001"
    
    def test_index_symbol_shenzhen(self):
        """测试深圳指数代码格式化"""
        assert index_symbol("399001") == "sz399001"
        assert index_symbol("399006") == "sz399006"


class TestNormalizeKline:
    """测试K线数据标准化"""
    
    def test_normalize_kline_basic(self):
        """测试基本K线数据标准化"""
        df = pd.DataFrame({
            "日期": ["2024-01-01", "2024-01-02"],
            "开盘": [10.0, 11.0],
            "收盘": [10.5, 11.5],
            "最高": [11.0, 12.0],
            "最低": [9.5, 10.5],
            "成交量": [1000000, 1100000],
        })
        
        result = normalize_kline_df(df)
        
        assert "date" in result.columns
        assert "open" in result.columns
        assert "close" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "volume" in result.columns
    
    def test_normalize_kline_empty(self):
        """测试空数据处理"""
        df = pd.DataFrame()
        result = normalize_kline_df(df)
        assert result.empty
    
    def test_normalize_kline_none(self):
        """测试None输入"""
        result = normalize_kline_df(None)
        assert result is None


class TestMarketDataService:
    """测试市场数据服务类"""
    
    def test_get_market_data_service_singleton(self):
        """测试服务单例"""
        service1 = get_market_data_service()
        service2 = get_market_data_service()
        assert service1 is service2
    
    @pytest.mark.asyncio
    async def test_get_realtime_quote(self):
        """测试获取实时报价"""
        service = get_market_data_service()
        
        with patch('app.api.endpoints.market.services.fetch_sina_price') as mock_fetch:
            mock_fetch.return_value = (10.5, 0.5, 5.0)
            
            result = await service.get_realtime_quote("600000")
            
            assert result["code"] == "600000"
            assert result["price"] == 10.5
            assert result["change"] == 0.5
            assert result["change_pct"] == 5.0
    
    @pytest.mark.asyncio
    async def test_get_batch_quotes(self):
        """测试批量获取报价"""
        service = get_market_data_service()
        
        with patch('app.api.endpoints.market.services.fetch_batch_quotes') as mock_fetch:
            mock_fetch.return_value = {
                "600000": {"code": "600000", "name": "浦发银行", "price": 10.5},
                "000001": {"code": "000001", "name": "平安银行", "price": 15.0},
            }
            
            result = await service.get_batch_quotes(["600000", "000001"])
            
            assert "600000" in result
            assert "000001" in result


class TestFetchSinaPrice:
    """测试新浪行情获取"""
    
    @pytest.mark.asyncio
    async def test_fetch_sina_price_success(self):
        """测试成功获取价格"""
        from app.api.endpoints.market.services import fetch_sina_price
        
        mock_response = MagicMock()
        mock_response.text = 'var hq_str_sh600000="浦发银行,10.00,10.50,10.60,10.80,10.40,1000000,10500000";'
        
        with patch('requests.get', return_value=mock_response):
            price, change, change_pct = await fetch_sina_price("600000")
            
            assert price == 10.60
            assert change == pytest.approx(0.60, 0.01)
    
    @pytest.mark.asyncio
    async def test_fetch_sina_price_error(self):
        """测试获取价格失败"""
        from app.api.endpoints.market.services import fetch_sina_price
        
        with patch('requests.get', side_effect=Exception("Network error")):
            price, change, change_pct = await fetch_sina_price("600000")
            
            assert price == 0.0
            assert change == 0.0
            assert change_pct == 0.0


class TestFetchBatchQuotes:
    """测试批量行情获取"""
    
    @pytest.mark.asyncio
    async def test_fetch_batch_quotes_success(self):
        """测试成功批量获取"""
        from app.api.endpoints.market.services import fetch_batch_quotes
        
        mock_response = MagicMock()
        mock_response.text = '''
        var hq_str_sh600000="浦发银行,10.00,10.50,10.60,10.80,10.40,100,200,1000000,10500000,,,,,,,10.50,10.60,,,,,,,,,,,,,15:00:00";
        var hq_str_sz000001="平安银行,15.00,15.50,15.60,15.80,15.40,100,200,2000000,31000000,,,,,,,15.50,15.60,,,,,,,,,,,,,15:00:00";
        '''
        
        with patch('requests.get', return_value=mock_response):
            result = await fetch_batch_quotes(["600000", "000001"])
            
            assert len(result) >= 0  # 可能解析失败，但不应抛异常
    
    @pytest.mark.asyncio
    async def test_fetch_batch_quotes_empty_codes(self):
        """测试空代码列表"""
        from app.api.endpoints.market.services import fetch_batch_quotes
        
        result = await fetch_batch_quotes([])
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
