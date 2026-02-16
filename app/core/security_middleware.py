#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API安全中间件
提供速率限制、请求审计等安全功能
"""

import time
import logging
from typing import Callable, Dict, Optional
from collections import defaultdict
from datetime import datetime
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    基于内存的速率限制器
    使用滑动窗口算法
    """
    
    def __init__(self, requests_per_window: int = 100, window_seconds: int = 60):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> tuple[bool, int, int]:
        """
        检查请求是否被允许
        
        Args:
            client_id: 客户端标识（通常是IP地址）
        
        Returns:
            (是否允许, 剩余请求数, 重置时间秒数)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # 清理过期请求
        self._requests[client_id] = [
            req_time for req_time in self._requests[client_id]
            if req_time > window_start
        ]
        
        current_count = len(self._requests[client_id])
        remaining = max(0, self.requests_per_window - current_count)
        reset_time = int(window_start + self.window_seconds - now)
        
        if current_count >= self.requests_per_window:
            return False, 0, reset_time
        
        # 记录新请求
        self._requests[client_id].append(now)
        return True, remaining - 1, reset_time
    
    def get_stats(self, client_id: str) -> Dict:
        """获取客户端请求统计"""
        now = time.time()
        window_start = now - self.window_seconds
        
        self._requests[client_id] = [
            req_time for req_time in self._requests[client_id]
            if req_time > window_start
        ]
        
        return {
            "client_id": client_id,
            "requests_in_window": len(self._requests[client_id]),
            "limit": self.requests_per_window,
            "window_seconds": self.window_seconds,
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    API速率限制中间件
    """
    
    # 不需要速率限制的路径
    EXEMPT_PATHS = {
        "/health",
        "/api/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }
    
    # 更宽松限制的路径（静态资源等）
    RELAXED_PATHS = {
        "/static",
        "/assets",
    }
    
    def __init__(self, app, requests_per_window: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(requests_per_window, window_seconds)
        self.relaxed_limiter = RateLimiter(requests_per_window * 10, window_seconds)
    
    def _get_client_id(self, request: Request) -> str:
        """
        获取客户端标识
        
        优先使用X-Forwarded-For（反向代理场景）
        然后使用X-Real-IP
        最后使用client.host
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _is_exempt(self, path: str) -> bool:
        """检查路径是否豁免速率限制"""
        for exempt_path in self.EXEMPT_PATHS:
            if path.startswith(exempt_path):
                return True
        return False
    
    def _is_relaxed(self, path: str) -> bool:
        """检查路径是否使用宽松限制"""
        for relaxed_path in self.RELAXED_PATHS:
            if path.startswith(relaxed_path):
                return True
        return False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求"""
        path = request.url.path
        
        # 豁免路径直接通过
        if self._is_exempt(path):
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        
        # 选择限制器
        limiter = self.relaxed_limiter if self._is_relaxed(path) else self.limiter
        
        # 检查速率限制
        allowed, remaining, reset_time = limiter.is_allowed(client_id)
        
        if not allowed:
            logger.warning(f"速率限制触发: client={client_id}, path={path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "请求过于频繁，请稍后再试",
                    "retry_after": reset_time,
                },
                headers={
                    "X-RateLimit-Limit": str(limiter.requests_per_window),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(reset_time),
                }
            )
        
        # 添加速率限制头
        response = await call_next(request)
        
        response.headers["X-RateLimit-Limit"] = str(limiter.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response


class RequestAuditMiddleware(BaseHTTPMiddleware):
    """
    请求审计中间件
    记录所有API请求的详细信息
    """
    
    # 敏感字段，记录时需要脱敏
    SENSITIVE_FIELDS = {"password", "token", "secret", "api_key", "apikey"}
    
    # 不记录响应体的路径
    NO_BODY_PATHS = {
        "/api/market/kline",
        "/api/market/realtime",
        "/api/stock/",
    }
    
    def __init__(self, app, log_responses: bool = True):
        super().__init__(app)
        self.log_responses = log_responses
    
    def _mask_sensitive(self, data: dict) -> dict:
        """脱敏敏感数据"""
        if not isinstance(data, dict):
            return data
        
        masked = {}
        for key, value in data.items():
            if key.lower() in self.SENSITIVE_FIELDS:
                masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = self._mask_sensitive(value)
            else:
                masked[key] = value
        
        return masked
    
    def _should_log_body(self, path: str) -> bool:
        """判断是否记录响应体"""
        for no_body_path in self.NO_BODY_PATHS:
            if path.startswith(no_body_path):
                return False
        return True
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求并记录审计日志"""
        start_time = time.time()
        
        # 获取客户端信息
        client_ip = request.headers.get("X-Forwarded-For", "")
        if client_ip:
            client_ip = client_ip.split(",")[0].strip()
        elif request.client:
            client_ip = request.client.host
        else:
            client_ip = "unknown"
        
        # 记录请求信息
        request_info = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params),
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", ""),
        }
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录响应信息
        audit_log = {
            **request_info,
            "status_code": response.status_code,
            "process_time_ms": round(process_time * 1000, 2),
        }
        
        # 根据状态码选择日志级别
        if response.status_code >= 500:
            logger.error(f"API请求审计: {audit_log}")
        elif response.status_code >= 400:
            logger.warning(f"API请求审计: {audit_log}")
        else:
            logger.info(f"API请求审计: {audit_log}")
        
        # 添加处理时间头
        response.headers["X-Process-Time"] = f"{process_time:.3f}s"
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    安全头中间件
    添加安全相关的HTTP头
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """添加安全头"""
        response = await call_next(request)
        
        # 防止点击劫持
        response.headers["X-Frame-Options"] = "DENY"
        
        # 防止MIME类型嗅探
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS保护
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # 引用策略
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # 内容安全策略（API服务可以宽松一些）
        if not request.url.path.startswith("/static"):
            response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


def setup_security_middleware(app):
    """
    配置安全中间件
    
    Args:
        app: FastAPI应用实例
    """
    from app.core.security_config import settings
    
    # 添加安全头中间件
    app.add_middleware(SecurityHeadersMiddleware)
    
    # 添加请求审计中间件
    app.add_middleware(RequestAuditMiddleware, log_responses=False)
    
    # 添加速率限制中间件
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=settings.API_RATE_LIMIT_REQUESTS,
        window_seconds=settings.API_RATE_LIMIT_WINDOW,
    )
    
    logger.info("✓ 安全中间件配置完成")
