"""
QVeris API 包装器

功能:
1. 接入QVeris API作为A股数据备选数据源
2. 提供与akshare类似的接口
3. 处理API认证和错误

使用方法:
    from app.utils.qveris_wrapper import safe_qveris_call, async_qveris_call
    
    # 同步调用
    data = safe_qveris_call(lambda: qveris_client.get_stock_quote('600519'))
    
    # 异步调用
    data = await async_qveris_call(lambda: qveris_client.get_stock_quote('600519'))
"""
import os
import asyncio
import logging
import requests
from typing import Callable, TypeVar, Optional
from functools import wraps

from app.utils.retry import run_with_retry, RateLimiter
from app.utils.http_client import async_http_client

logger = logging.getLogger(__name__)

T = TypeVar('T')

# QVeris API 基础配置
QVERIS_API_URL = "https://api.qveris.ai/v1"
QVERIS_API_KEY = os.getenv('QVERIS_API_KEY')

# QVeris 专用限流器 - 10 QPS（根据QVeris API限制调整）
_qveris_limiter = RateLimiter(calls_per_second=10.0)


class QVerisClient:
    """
    QVeris API 客户端
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or QVERIS_API_KEY
        if not self.api_key:
            logger.warning("QVeris API key not configured. QVeris API will not be available.")
        self.base_url = QVERIS_API_URL
    
    def _request(self, endpoint: str, params: dict = None) -> dict:
        """
        发送API请求
        """
        if not self.api_key:
            raise ValueError("QVeris API key not configured")
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    
    async def _async_request(self, endpoint: str, params: dict = None) -> dict:
        """
        异步发送API请求
        """
        if not self.api_key:
            raise ValueError("QVeris API key not configured")
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        return await async_http_client.get(url, params, headers)
    
    def get_stock_quote(self, symbol: str) -> dict:
        """
        获取股票实时行情
        """
        try:
            data = self._request(f"/stock/quote", params={"symbol": symbol})
            return data.get("data", {})
        except Exception as e:
            logger.error(f"QVeris get_stock_quote failed: {e}")
            raise
    
    async def async_get_stock_quote(self, symbol: str) -> dict:
        """
        异步获取股票实时行情
        """
        try:
            data = await self._async_request(f"/stock/quote", params={"symbol": symbol})
            return data.get("data", {})
        except Exception as e:
            logger.error(f"QVeris async_get_stock_quote failed: {e}")
            raise
    
    def get_stock_kline(self, symbol: str, interval: str = "daily", limit: int = 100) -> list:
        """
        获取股票K线数据
        """
        try:
            data = self._request(f"/stock/kline", params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            })
            return data.get("data", [])
        except Exception as e:
            logger.error(f"QVeris get_stock_kline failed: {e}")
            raise
    
    async def async_get_stock_kline(self, symbol: str, interval: str = "daily", limit: int = 100) -> list:
        """
        异步获取股票K线数据
        """
        try:
            data = await self._async_request(f"/stock/kline", params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            })
            return data.get("data", [])
        except Exception as e:
            logger.error(f"QVeris async_get_stock_kline failed: {e}")
            raise
    
    def get_market_indices(self) -> list:
        """
        获取市场指数
        """
        try:
            data = self._request("/market/indices")
            return data.get("data", [])
        except Exception as e:
            logger.error(f"QVeris get_market_indices failed: {e}")
            raise
    
    async def async_get_market_indices(self) -> list:
        """
        异步获取市场指数
        """
        try:
            data = await self._async_request("/market/indices")
            return data.get("data", [])
        except Exception as e:
            logger.error(f"QVeris async_get_market_indices failed: {e}")
            raise
    
    def get_sector_list(self) -> list:
        """
        获取行业板块列表
        """
        try:
            data = self._request("/market/sectors")
            return data.get("data", [])
        except Exception as e:
            logger.error(f"QVeris get_sector_list failed: {e}")
            raise
    
    async def async_get_sector_list(self) -> list:
        """
        异步获取行业板块列表
        """
        try:
            data = await self._async_request("/market/sectors")
            return data.get("data", [])
        except Exception as e:
            logger.error(f"QVeris async_get_sector_list failed: {e}")
            raise


# 创建全局QVeris客户端实例
qveris_client = QVerisClient()


def safe_qveris_call(
    func: Callable[[], T],
    max_retries: int = 3,
    timeout: Optional[float] = 10.0,
    name: Optional[str] = None
) -> T:
    """
    安全的 QVeris 同步调用包装器
    
    Args:
        func: 要执行的 QVeris 函数（lambda 包装）
        max_retries: 最大重试次数
        timeout: 超时时间（秒），None 表示无超时
        name: 调用名称（用于日志）
    
    Returns:
        QVeris 函数返回值
    """
    if not QVERIS_API_KEY:
        raise ValueError("QVeris API key not configured")
    
    call_name = name or func.__name__ if hasattr(func, '__name__') else "qveris_call"
    
    def wrapped():
        _qveris_limiter.acquire()
        return func()
    
    try:
        result = run_with_retry(
            wrapped,
            max_attempts=max_retries,
            delays=(1, 2, 4),  # 指数退避
            on_retry=lambda e, attempt: logger.warning(
                f"QVeris {call_name} 重试 {attempt}/{max_retries}: {e}"
            )
        )
        return result
    except Exception as e:
        logger.error(f"QVeris {call_name} 失败: {e}")
        raise


async def async_qveris_call(
    func: Callable[[], T],
    max_retries: int = 3,
    timeout: float = 10.0,
    name: Optional[str] = None
) -> T:
    """
    安全的 QVeris 异步调用包装器
    
    Args:
        func: 要执行的 QVeris 函数（lambda 包装）
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
        name: 调用名称（用于日志）
    
    Returns:
        QVeris 函数返回值
    """
    if not QVERIS_API_KEY:
        raise ValueError("QVeris API key not configured")
    
    call_name = name or "qveris_async_call"
    loop = asyncio.get_event_loop()
    
    async def execute_with_retry():
        last_exc = None
        for attempt in range(max_retries):
            try:
                # 限流
                await _qveris_limiter.async_acquire()
                
                # 在线程池中执行同步 QVeris 调用
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func),
                    timeout=timeout
                )
                return result
                
            except asyncio.TimeoutError:
                last_exc = TimeoutError(f"QVeris {call_name} 超时 ({timeout}s)")
                logger.warning(f"QVeris {call_name} 超时，尝试 {attempt + 1}/{max_retries}")
                
            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                
                # 判断是否可重试
                retriable = any(kw in err_str for kw in [
                    "connection", "timeout", "remote disconnected",
                    "too many requests", "rate limit"
                ])
                
                if retriable and attempt < max_retries - 1:
                    delay = [1, 2, 4][min(attempt, 2)]
                    logger.warning(
                        f"QVeris {call_name} 重试 {attempt + 1}/{max_retries}: {e}, 等待 {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                    
                raise
        
        raise last_exc
    
    try:
        return await execute_with_retry()
    except Exception as e:
        logger.error(f"QVeris {call_name} 最终失败: {e}")
        raise


def qveris_cached(ttl: int = 60, name: Optional[str] = None):
    """
    QVeris 调用缓存装饰器
    
    结合限流、重试、缓存的完整解决方案
    
    Args:
        ttl: 缓存时间（秒）
        name: 缓存名称
    """
    from app.cache_utils import TTLCache
    
    cache = TTLCache(ttl=ttl, max_entries=10, name=name or "qveris_cache")
    
    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            
            # 尝试从缓存获取
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            
            # 调用并缓存
            result = safe_qveris_call(func, name=name or func.__name__)
            cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator
