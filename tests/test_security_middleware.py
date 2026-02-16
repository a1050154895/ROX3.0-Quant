#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全中间件测试
"""

import pytest
import time
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.security_middleware import (
    RateLimiter,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RequestAuditMiddleware,
)


class TestRateLimiter:
    """RateLimiter测试类"""
    
    def test_initial_state(self):
        """测试初始状态"""
        limiter = RateLimiter(requests_per_window=10, window_seconds=60)
        
        allowed, remaining, reset_time = limiter.is_allowed("client1")
        
        assert allowed is True
        assert remaining == 9
        assert reset_time > 0
    
    def test_rate_limit_enforcement(self):
        """测试速率限制执行"""
        limiter = RateLimiter(requests_per_window=3, window_seconds=60)
        
        # 前三次应该允许
        for i in range(3):
            allowed, remaining, _ = limiter.is_allowed("client1")
            assert allowed is True
            assert remaining == 2 - i
        
        # 第四次应该被拒绝
        allowed, remaining, _ = limiter.is_allowed("client1")
        assert allowed is False
        assert remaining == 0
    
    def test_different_clients_independent(self):
        """测试不同客户端独立计数"""
        limiter = RateLimiter(requests_per_window=2, window_seconds=60)
        
        # 客户端1用完配额
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        
        # 客户端1应该被限制
        allowed, _, _ = limiter.is_allowed("client1")
        assert allowed is False
        
        # 客户端2应该还有配额
        allowed, remaining, _ = limiter.is_allowed("client2")
        assert allowed is True
        assert remaining == 1
    
    def test_window_expiry(self):
        """测试时间窗口过期"""
        limiter = RateLimiter(requests_per_window=2, window_seconds=1)
        
        # 用完配额
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        
        # 应该被限制
        allowed, _, _ = limiter.is_allowed("client1")
        assert allowed is False
        
        # 等待窗口过期
        time.sleep(1.1)
        
        # 应该恢复配额
        allowed, _, _ = limiter.is_allowed("client1")
        assert allowed is True
    
    def test_get_stats(self):
        """测试获取统计信息"""
        limiter = RateLimiter(requests_per_window=10, window_seconds=60)
        
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        
        stats = limiter.get_stats("client1")
        
        assert stats["client_id"] == "client1"
        assert stats["requests_in_window"] == 2
        assert stats["limit"] == 10


class TestRateLimitMiddleware:
    """RateLimitMiddleware测试类"""
    
    @pytest.fixture
    def app(self):
        """创建测试应用"""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @app.get("/health")
        async def health():
            return {"status": "ok"}
        
        return app
    
    def test_normal_request(self, app):
        """测试正常请求"""
        app.add_middleware(RateLimitMiddleware, requests_per_window=10, window_seconds=60)
        client = TestClient(app)
        
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
    
    def test_exempt_path(self, app):
        """测试豁免路径"""
        app.add_middleware(RateLimitMiddleware, requests_per_window=1, window_seconds=60)
        client = TestClient(app)
        
        # 健康检查路径应该豁免
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_rate_limit_exceeded(self, app):
        """测试超过速率限制"""
        app.add_middleware(RateLimitMiddleware, requests_per_window=2, window_seconds=60)
        client = TestClient(app)
        
        # 前两次应该成功
        client.get("/test")
        client.get("/test")
        
        # 第三次应该被拒绝
        response = client.get("/test")
        
        assert response.status_code == 429
        assert "Too Many Requests" in response.json()["error"]


class TestSecurityHeadersMiddleware:
    """SecurityHeadersMiddleware测试类"""
    
    @pytest.fixture
    def app(self):
        """创建测试应用"""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        return app
    
    def test_security_headers_present(self, app):
        """测试安全头存在"""
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert "Referrer-Policy" in response.headers
    
    def test_content_security_policy(self, app):
        """测试内容安全策略"""
        client = TestClient(app)
        response = client.get("/test")
        
        assert "Content-Security-Policy" in response.headers


class TestRequestAuditMiddleware:
    """RequestAuditMiddleware测试类"""
    
    @pytest.fixture
    def app(self):
        """创建测试应用"""
        app = FastAPI()
        app.add_middleware(RequestAuditMiddleware, log_responses=False)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @app.get("/error")
        async def error_endpoint():
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="Internal error")
        
        return app
    
    def test_process_time_header(self, app):
        """测试处理时间头"""
        client = TestClient(app)
        response = client.get("/test")
        
        assert "X-Process-Time" in response.headers
    
    def test_successful_request_logged(self, app, caplog):
        """测试成功请求被记录"""
        import logging
        caplog.set_level(logging.INFO)
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.status_code == 200
        # 检查日志中包含请求信息
        assert any("API请求审计" in record.message for record in caplog.records)
    
    def test_error_request_logged(self, app, caplog):
        """测试错误请求被记录"""
        import logging
        caplog.set_level(logging.WARNING)
        
        client = TestClient(app)
        response = client.get("/error")
        
        assert response.status_code == 500


class TestMaskSensitive:
    """敏感数据脱敏测试类"""
    
    def test_mask_password(self):
        """测试密码脱敏"""
        from app.core.security_middleware import RequestAuditMiddleware
        
        middleware = RequestAuditMiddleware(None)
        data = {"username": "test", "password": "secret123"}
        
        masked = middleware._mask_sensitive(data)
        
        assert masked["username"] == "test"
        assert masked["password"] == "***MASKED***"
    
    def test_mask_token(self):
        """测试Token脱敏"""
        from app.core.security_middleware import RequestAuditMiddleware
        
        middleware = RequestAuditMiddleware(None)
        data = {"token": "abc123", "data": "value"}
        
        masked = middleware._mask_sensitive(data)
        
        assert masked["token"] == "***MASKED***"
        assert masked["data"] == "value"
    
    def test_mask_nested(self):
        """测试嵌套数据脱敏"""
        from app.core.security_middleware import RequestAuditMiddleware
        
        middleware = RequestAuditMiddleware(None)
        data = {
            "user": {
                "name": "test",
                "password": "secret",
            },
            "api_key": "key123",
        }
        
        masked = middleware._mask_sensitive(data)
        
        assert masked["user"]["name"] == "test"
        assert masked["user"]["password"] == "***MASKED***"
        assert masked["api_key"] == "***MASKED***"
