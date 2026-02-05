"""
重试和限流工具模块

功能:
1. run_with_retry - 同步函数重试，支持指数退避
2. async_run_with_retry - 异步函数重试
3. RateLimiter - 请求限流器，防止 API 封禁
"""
import time
import asyncio
import logging
from functools import wraps
from typing import Callable, TypeVar, Optional

logger = logging.getLogger(__name__)

T = TypeVar('T')


def run_with_retry(
    func: Callable[[], T], 
    max_attempts: int = 3, 
    delays: tuple = (1, 2, 4),  # 指数退避
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> T:
    """
    带重试的函数执行器（同步版本）
    
    Args:
        func: 要执行的函数
        max_attempts: 最大尝试次数
        delays: 每次重试的延迟时间（秒），支持指数退避
        on_retry: 重试时的回调函数，接收(异常, 当前尝试次数)
    
    Returns:
        函数执行结果
        
    Raises:
        最后一次异常（如果所有重试都失败）
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            
            # 判断是否为可重试的错误
            retriable_errors = [
                "remote disconnected",
                "connection aborted", 
                "connection reset",
                "connection refused",
                "timeout",
                "timed out",
                "temporarily unavailable",
                "too many requests",
                "rate limit",
            ]
            
            is_retriable = any(err in err_str for err in retriable_errors)
            
            if is_retriable and attempt < max_attempts - 1:
                delay = delays[min(attempt, len(delays) - 1)]
                
                if on_retry:
                    on_retry(e, attempt + 1)
                else:
                    logger.warning(f"重试 {attempt + 1}/{max_attempts}: {e}, 等待 {delay}s")
                
                time.sleep(delay)
                continue
            raise
    raise last_exc


async def async_run_with_retry(
    func: Callable,
    max_attempts: int = 3,
    delays: tuple = (1, 2, 4),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> T:
    """
    带重试的函数执行器（异步版本）
    
    Args:
        func: 要执行的异步函数
        max_attempts: 最大尝试次数
        delays: 每次重试的延迟时间（秒）
        on_retry: 重试时的回调函数
    
    Returns:
        函数执行结果
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            
            retriable_errors = [
                "remote disconnected",
                "connection aborted",
                "connection reset",
                "timeout",
                "timed out",
                "too many requests",
                "rate limit",
            ]
            
            is_retriable = any(err in err_str for err in retriable_errors)
            
            if is_retriable and attempt < max_attempts - 1:
                delay = delays[min(attempt, len(delays) - 1)]
                
                if on_retry:
                    on_retry(e, attempt + 1)
                else:
                    logger.warning(f"异步重试 {attempt + 1}/{max_attempts}: {e}, 等待 {delay}s")
                
                await asyncio.sleep(delay)
                continue
            raise
    raise last_exc


class RateLimiter:
    """
    简单的请求限流器
    
    使用方法:
        limiter = RateLimiter(calls_per_second=5)
        limiter.acquire()  # 同步
        await limiter.async_acquire()  # 异步
    """
    
    def __init__(self, calls_per_second: float = 5.0):
        """
        Args:
            calls_per_second: 每秒允许的最大请求数
        """
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self._lock = asyncio.Lock()
    
    def acquire(self):
        """同步获取令牌（阻塞直到可以执行）"""
        now = time.time()
        wait_time = self.last_call + self.min_interval - now
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_call = time.time()
    
    async def async_acquire(self):
        """异步获取令牌"""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait_time = self.last_call + self.min_interval - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call = asyncio.get_event_loop().time()


# 全局限流器实例 - 用于 AkShare 请求
akshare_limiter = RateLimiter(calls_per_second=5.0)


def rate_limited(limiter: RateLimiter):
    """
    限流装饰器
    
    使用方法:
        @rate_limited(akshare_limiter)
        def fetch_data():
            return ak.some_api()
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire()
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            await limiter.async_acquire()
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator

