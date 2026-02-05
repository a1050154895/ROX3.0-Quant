"""
AkShare 统一调用包装器

功能:
1. 限流保护 - 防止请求过频导致 IP 封禁
2. 自动重试 - 处理临时网络错误
3. 超时控制 - 避免长时间阻塞
4. 统一日志 - 便于监控和调试

使用方法:
    from app.utils.akshare_wrapper import safe_ak_call, async_ak_call
    
    # 同步调用
    df = safe_ak_call(lambda: ak.stock_zh_a_spot_em())
    
    # 异步调用
    df = await async_ak_call(lambda: ak.stock_zh_a_spot_em())
"""
import asyncio
import logging
from typing import Callable, TypeVar, Optional
from functools import wraps

from app.utils.retry import run_with_retry, RateLimiter

logger = logging.getLogger(__name__)

T = TypeVar('T')

# AkShare 专用限流器 - 5 QPS（每秒5次请求）
_akshare_limiter = RateLimiter(calls_per_second=5.0)


def safe_ak_call(
    func: Callable[[], T],
    max_retries: int = 3,
    timeout: Optional[float] = 30.0,
    name: Optional[str] = None
) -> T:
    """
    安全的 AkShare 同步调用包装器
    
    Args:
        func: 要执行的 AkShare 函数（lambda 包装）
        max_retries: 最大重试次数
        timeout: 超时时间（秒），None 表示无超时
        name: 调用名称（用于日志）
    
    Returns:
        AkShare 函数返回值
        
    Example:
        df = safe_ak_call(lambda: ak.stock_zh_a_spot_em(), name="spot")
    """
    call_name = name or func.__name__ if hasattr(func, '__name__') else "ak_call"
    
    def wrapped():
        _akshare_limiter.acquire()
        return func()
    
    try:
        result = run_with_retry(
            wrapped,
            max_attempts=max_retries,
            delays=(1, 2, 4),  # 指数退避
            on_retry=lambda e, attempt: logger.warning(
                f"AkShare {call_name} 重试 {attempt}/{max_retries}: {e}"
            )
        )
        return result
    except Exception as e:
        logger.error(f"AkShare {call_name} 失败: {e}")
        raise


async def async_ak_call(
    func: Callable[[], T],
    max_retries: int = 3,
    timeout: float = 30.0,
    name: Optional[str] = None
) -> T:
    """
    安全的 AkShare 异步调用包装器
    
    Args:
        func: 要执行的 AkShare 函数（lambda 包装）
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
        name: 调用名称（用于日志）
    
    Returns:
        AkShare 函数返回值
        
    Example:
        df = await async_ak_call(lambda: ak.stock_zh_a_spot_em(), name="spot")
    """
    call_name = name or "ak_async_call"
    loop = asyncio.get_event_loop()
    
    async def execute_with_retry():
        last_exc = None
        for attempt in range(max_retries):
            try:
                # 限流
                await _akshare_limiter.async_acquire()
                
                # 在线程池中执行同步 AkShare 调用
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func),
                    timeout=timeout
                )
                return result
                
            except asyncio.TimeoutError:
                last_exc = TimeoutError(f"AkShare {call_name} 超时 ({timeout}s)")
                logger.warning(f"AkShare {call_name} 超时，尝试 {attempt + 1}/{max_retries}")
                
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
                        f"AkShare {call_name} 重试 {attempt + 1}/{max_retries}: {e}, 等待 {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                    
                raise
        
        raise last_exc
    
    try:
        return await execute_with_retry()
    except Exception as e:
        logger.error(f"AkShare {call_name} 最终失败: {e}")
        raise


def akshare_cached(ttl: int = 60, name: Optional[str] = None):
    """
    AkShare 调用缓存装饰器
    
    结合限流、重试、缓存的完整解决方案
    
    Args:
        ttl: 缓存时间（秒）
        name: 缓存名称
    
    Example:
        @akshare_cached(ttl=300, name="sector_list")
        def get_sectors():
            return ak.stock_board_industry_name_em()
    """
    from app.cache_utils import TTLCache
    
    cache = TTLCache(ttl=ttl, max_entries=10, name=name or "ak_cache")
    
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
            result = safe_ak_call(func, name=name or func.__name__)
            cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator


# 便捷函数：常用 AkShare 调用的预配置包装
class AkShareClient:
    """
    AkShare 客户端 - 提供常用接口的便捷封装
    
    Example:
        client = AkShareClient()
        df = await client.get_spot_data()
    """
    
    def __init__(self, rate_limit: float = 5.0):
        self._limiter = RateLimiter(calls_per_second=rate_limit)
    
    async def get_spot_data(self, timeout: float = 15.0):
        """获取 A 股实时行情"""
        import akshare as ak
        return await async_ak_call(
            lambda: ak.stock_zh_a_spot_em(),
            timeout=timeout,
            name="spot_data"
        )
    
    async def get_sector_list(self, timeout: float = 10.0):
        """获取行业板块列表"""
        import akshare as ak
        return await async_ak_call(
            lambda: ak.stock_board_industry_name_em(),
            timeout=timeout,
            name="sector_list"
        )
    
    async def get_index_daily(self, symbol: str, timeout: float = 10.0):
        """获取指数日线数据"""
        import akshare as ak
        return await async_ak_call(
            lambda: ak.stock_zh_index_daily(symbol=symbol),
            timeout=timeout,
            name=f"index_daily_{symbol}"
        )
    
    async def get_stock_hist(
        self, 
        symbol: str, 
        period: str = "daily",
        start_date: str = None,
        end_date: str = None,
        adjust: str = "qfq",
        timeout: float = 15.0
    ):
        """获取个股历史数据"""
        import akshare as ak
        return await async_ak_call(
            lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            ),
            timeout=timeout,
            name=f"stock_hist_{symbol}"
        )


# 全局客户端实例
akshare_client = AkShareClient()
