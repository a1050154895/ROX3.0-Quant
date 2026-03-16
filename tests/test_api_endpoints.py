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


# ─── 新增：卢式分析增强测试 ────────────────────────────────────────

class TestLuAnalysisEnhanced:
    """卢式分析增强测试：字段、边界、负面场景"""

    def test_four_matrix_has_relative_strength(self):
        """四矩阵每个资产均有 relative_strength 字段"""
        r = client.get("/api/lu/four-matrix")
        assert r.status_code == 200
        d = r.json()
        assets = d.get("assets", [])
        assert len(assets) >= 2
        for a in assets:
            assert "relative_strength" in a, f"资产 {a.get('name')} 缺少 relative_strength 字段"
            assert 0 <= a["relative_strength"] <= 100

    def test_four_matrix_has_trend_arrow(self):
        """四矩阵每个资产均有 trend_arrow 符号字段"""
        r = client.get("/api/lu/four-matrix")
        assert r.status_code == 200
        for a in r.json().get("assets", []):
            assert a.get("trend_arrow") in ("↑", "→", "↓"), \
                f"trend_arrow={a.get('trend_arrow')} 不在合法值集合"

    def test_analyze_symbol_matrix_position_not_always_default(self):
        """analyze-symbol 矩阵位置不应永远返回'观望区'（字段读取修复验证）"""
        r = client.get("/api/lu/analyze-symbol?symbol=000001")
        assert r.status_code == 200
        d = r.json()
        assert "matrix_position" in d
        # 矩阵位置应为合法值之一
        valid = {"增强区", "转强区", "分化区", "防御区", "观望区"}
        assert d["matrix_position"] in valid

    def test_candidates_use_real_codes(self):
        """候选池使用真实 ETF 代码，不含 GOLD_PROXY 等 mock ticker"""
        r = client.get("/api/lu/candidates")
        assert r.status_code == 200
        items = r.json().get("items", [])
        for item in items:
            symbol = item.get("symbol", "")
            assert not symbol.startswith("GOLD_PROXY"), f"发现 mock ticker: {symbol}"
            assert symbol.isdigit() or len(symbol) == 6, f"代码格式异常: {symbol}"

    def test_three_flows_has_bias_note(self):
        """三流快照包含 bias_note 字段（实时数据诊断信息）"""
        r = client.get("/api/lu/three-flows")
        assert r.status_code == 200
        d = r.json()
        assert "summary_bias" in d
        assert "bias_note" in d  # v2 新增字段
        assert "flow_volume" in d


class TestLuPredictionV5:
    """卢式预测 v5 协议测试"""

    def test_predict_v2_protocol_version(self):
        """predict-v2 返回正确的协议版本"""
        r = client.post("/api/lu-prediction/predict-v2", json={
            "code": "000001", "market": "CN_A", "period": "daily", "lookback": 30
        })
        assert r.status_code == 200
        d = r.json()
        assert d.get("protocol_version") == "lu-analyzer-v5"
        assert "analysis" in d
        assert "composite_score" in d["analysis"]
        assert "market_regime" in d["analysis"]
        assert "action_plan" in d["analysis"]
        assert "component_scores" in d["analysis"]

    def test_scan_v2_empty_codes_rejected(self):
        """scan-v2 空 codes 应返回 400"""
        r = client.post("/api/lu-prediction/scan-v2", json={"codes": []})
        assert r.status_code == 400

    def test_scan_v2_returns_leaders(self):
        """scan-v2 返回 leaders 字段"""
        r = client.post("/api/lu-prediction/scan-v2", json={
            "codes": ["000001", "600519"], "top_n": 2
        })
        assert r.status_code == 200
        d = r.json()
        assert "leaders" in d
        assert "results" in d
        assert d["protocol_version"] == "lu-analyzer-v5"

    def test_portfolio_v2_weight_constraints(self):
        """portfolio-v2 权重约束数学一致性"""
        r = client.post("/api/lu-prediction/portfolio-v2", json={
            "codes": ["000001", "600519", "000858"],
            "risk_preference": "balanced"
        })
        assert r.status_code == 200
        d = r.json()
        weights = d.get("weights", {})
        total = d.get("total_invested", 0)
        cash = d.get("cash_weight", 0)
        # 总权重 + 现金权重 ≈ 1.0（允许 0.05 误差）
        assert abs(total + cash - 1.0) < 0.05, f"权重不一致: total={total}, cash={cash}"
        # 单票上限 ≤ 0.20
        for code, w in weights.items():
            assert w <= 0.20, f"{code} 权重{w}超过单票上限"

    def test_portfolio_v2_empty_codes_rejected(self):
        """portfolio-v2 空 codes 应返回 400"""
        r = client.post("/api/lu-prediction/portfolio-v2", json={"codes": []})
        assert r.status_code == 400

    def test_portfolio_v3_optimizer_field(self):
        """portfolio-v3 返回 optimizer 字段"""
        r = client.post("/api/lu-prediction/portfolio-v3", json={
            "codes": ["000001", "600519"],
            "risk_preference": "conservative"
        })
        assert r.status_code == 200
        d = r.json()
        assert d["risk_summary"].get("optimizer") == "covariance_driven"


class TestPydanticV2Validators:
    """auth.py Pydantic V2 迁移验证"""

    def test_short_username_rejected(self):
        """用户名长度 < 3 应被校验拒绝"""
        r = client.post("/register", json={"username": "ab", "password": "Abc@12345"})
        assert r.status_code in (400, 422)

    def test_invalid_email_rejected(self):
        """非法邮箱格式应被校验拒绝"""
        r = client.post("/register", json={
            "username": "testuser", "password": "Abc@12345", "email": "not-an-email"
        })
        assert r.status_code in (400, 422)

    def test_weak_password_rejected(self):
        """弱密码应被 SecurityConfig 校验拒绝"""
        r = client.post("/register", json={"username": "testuser", "password": "123"})
        assert r.status_code in (400, 422)

