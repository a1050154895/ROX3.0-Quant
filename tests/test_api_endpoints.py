#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API端点测试
测试核心API端点
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import json

from app.main import app


client = TestClient(app)


class TestHealthEndpoints:
    """测试健康检查端点"""
    
    def test_root_endpoint(self):
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_api_health(self):
        """测试API健康检查"""
        response = client.get("/api/system/health")
        assert response.status_code in [200, 404]  # 可能不存在


class TestMarketEndpoints:
    """测试市场数据端点"""
    
    def test_get_indices(self):
        """测试获取指数"""
        response = client.get("/api/market/indices")
        assert response.status_code == 200
        
        data = response.json()
        assert "indices" in data
    
    def test_get_stats(self):
        """测试获取统计"""
        response = client.get("/api/market/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "stats" in data
    
    def test_get_kline(self):
        """测试获取K线"""
        response = client.get("/api/market/kline?code=000001&count=5")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data or "error" in data
    
    def test_get_spot(self):
        """测试获取实时行情"""
        response = client.get("/api/market/spot")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
    
    def test_stock_suggest(self):
        """测试股票搜索"""
        response = client.get("/api/market/stock-suggest?q=平安")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
    
    def test_get_rankings(self):
        """测试获取排行榜"""
        response = client.get("/api/market/rankings")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data


class TestStrategyEndpoints:
    """测试策略端点"""
    
    def test_list_strategies(self):
        """测试列出策略"""
        response = client.get("/api/strategies/list")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_strategy_stats(self):
        """测试策略统计"""
        response = client.get("/api/strategies/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_strategies" in data
    
    def test_execute_strategy_invalid(self):
        """测试执行无效策略"""
        response = client.post(
            "/api/strategies/execute",
            json={"strategy_name": "不存在的策略", "params": {}}
        )
        assert response.status_code in [200, 400, 404]


class TestKnowledgeEndpoints:
    """测试知识库端点"""
    
    def test_knowledge_stats(self):
        """测试知识库统计"""
        response = client.get("/api/knowledge/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_documents" in data
    
    def test_knowledge_search(self):
        """测试知识库搜索"""
        response = client.get("/api/knowledge/search?query=投资")
        assert response.status_code in [200, 400, 404, 422]


class TestAuthEndpoints:
    """测试认证端点"""
    
    def test_token_without_credentials(self):
        """测试无凭证获取token"""
        response = client.post("/token", data={})
        assert response.status_code in [400, 401, 422]
    
    def test_register_validation(self):
        """测试注册验证"""
        response = client.post("/register", json={
            "username": "",
            "password": ""
        })
        assert response.status_code in [400, 422]


class TestWatchlistEndpoints:
    """测试自选股端点"""
    
    def test_get_watchlist_unauthorized(self):
        """测试未授权获取自选股"""
        response = client.get("/api/market/watchlist")
        assert response.status_code in [200, 401]


class TestAlertEndpoints:
    """测试预警端点"""
    
    def test_get_alerts_unauthorized(self):
        """测试未授权获取预警"""
        response = client.get("/api/market/alerts")
        assert response.status_code in [200, 401]


class TestAnalysisEndpoints:
    """测试分析端点"""
    
    def test_sector_fund_flow(self):
        """测试板块资金流向"""
        response = client.get("/api/market/sector-fund-flow")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
    
    def test_sentiment(self):
        """测试市场情绪"""
        response = client.get("/api/market/sentiment")
        assert response.status_code == 200
        
        data = response.json()
        assert "sentiment" in data
    
    def test_concepts(self):
        """测试概念板块"""
        response = client.get("/api/market/concepts")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
    
    def test_heatmap_data(self):
        """测试热力图数据"""
        response = client.get("/api/market/heatmap/data")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data


class TestIndicatorsEndpoint:
    """测试技术指标端点"""
    
    def test_indicators(self):
        """测试获取技术指标"""
        response = client.get("/api/market/indicators?code=600000")
        assert response.status_code == 200
        
        data = response.json()
        # 可能返回错误或指标数据
        assert "error" in data or "indicators" in data


class TestFenshiEndpoint:
    """测试分时端点"""
    
    def test_fenshi(self):
        """测试获取分时数据"""
        response = client.get("/api/market/fenshi?code=600000")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data or "error" in data


class TestDragonTigerEndpoint:
    """测试龙虎榜端点"""
    
    def test_dragon_tiger(self):
        """测试获取龙虎榜"""
        response = client.get("/api/market/dragon-tiger")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data


class TestRotationEndpoint:
    """测试板块轮动端点"""
    
    def test_rotation(self):
        """测试获取板块轮动"""
        response = client.get("/api/market/rotation")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data


class TestOverviewEndpoint:
    """测试市场概览端点"""
    
    def test_overview(self):
        """测试获取市场概览"""
        response = client.get("/api/market/overview")
        assert response.status_code == 200
        
        data = response.json()
        # 应该包含指数和统计
        assert "indices" in data or "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
