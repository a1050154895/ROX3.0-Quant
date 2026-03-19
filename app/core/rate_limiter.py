"""
API速率限制中间件
用于防止API滥用和保护系统资源
"""

import time
from typing import Dict, Tuple
from fastapi import Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """API速率限制中间件"""
    
    def __init__(self, app, rate_limit: int = 100, window_seconds: int = 60):
        """
        初始化速率限制中间件
        
        Args:
            app: FastAPI应用实例
            rate_limit: 每个IP在时间窗口内的最大请求数
            window_seconds: 时间窗口大小（秒）
        """
        super().__init__(app)
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}  # ip -> [时间戳列表]
    
    async def dispatch(self, request: Request, call_next):
        """
        处理请求，应用速率限制
        """
        # 获取客户端IP
        client_ip = self._get_client_ip(request)
        
        # 检查速率限制
        if not self._check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Too Many Requests. Please try again later."
            )
        
        # 处理请求
        response = await call_next(request)
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """
        获取客户端IP地址
        """
        # 尝试从X-Forwarded-For获取
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        
        # 尝试从X-Real-IP获取
        x_real_ip = request.headers.get("X-Real-IP")
        if x_real_ip:
            return x_real_ip
        
        # 从request.client获取
        if request.client:
            return request.client.host
        
        # 默认值
        return "unknown"
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """
        检查速率限制
        
        Args:
            client_ip: 客户端IP地址
        
        Returns:
            bool: 是否允许请求
        """
        current_time = time.time()
        
        # 清理过期的请求记录
        if client_ip in self.requests:
            # 只保留时间窗口内的请求
            self.requests[client_ip] = [
                timestamp for timestamp in self.requests[client_ip]
                if current_time - timestamp < self.window_seconds
            ]
        else:
            self.requests[client_ip] = []
        
        # 检查是否超过限制
        if len(self.requests[client_ip]) >= self.rate_limit:
            return False
        
        # 记录当前请求
        self.requests[client_ip].append(current_time)
        return True


class RateLimiter:
    """速率限制器，用于特定路由的速率限制"""
    
    def __init__(self, rate_limit: int = 100, window_seconds: int = 60):
        """
        初始化速率限制器
        
        Args:
            rate_limit: 每个IP在时间窗口内的最大请求数
            window_seconds: 时间窗口大小（秒）
        """
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}  # ip -> [时间戳列表]
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """
        检查速率限制
        
        Args:
            client_ip: 客户端IP地址
        
        Returns:
            bool: 是否允许请求
        """
        current_time = time.time()
        
        # 清理过期的请求记录
        if client_ip in self.requests:
            # 只保留时间窗口内的请求
            self.requests[client_ip] = [
                timestamp for timestamp in self.requests[client_ip]
                if current_time - timestamp < self.window_seconds
            ]
        else:
            self.requests[client_ip] = []
        
        # 检查是否超过限制
        if len(self.requests[client_ip]) >= self.rate_limit:
            return False
        
        # 记录当前请求
        self.requests[client_ip].append(current_time)
        return True


# 全局速率限制器实例
global_rate_limiter = RateLimiter()


# 特定路由的速率限制器
route_rate_limiters = {
    "ai": RateLimiter(rate_limit=50, window_seconds=60),  # AI接口限制更严格
    "market": RateLimiter(rate_limit=200, window_seconds=60),  # 市场数据接口
    "trade": RateLimiter(rate_limit=100, window_seconds=60),  # 交易接口
    "analysis": RateLimiter(rate_limit=150, window_seconds=60),  # 分析接口
}


def get_rate_limiter_for_route(route_path: str) -> RateLimiter:
    """
    根据路由路径获取对应的速率限制器
    
    Args:
        route_path: 路由路径
    
    Returns:
        RateLimiter: 对应的速率限制器
    """
    for key in route_rate_limiters:
        if f"/{key}" in route_path:
            return route_rate_limiters[key]
    return global_rate_limiter