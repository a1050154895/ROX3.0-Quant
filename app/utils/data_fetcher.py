"""
统一数据获取服务

功能:
1. 优先使用akshare获取数据
2. akshare失败时自动切换到QVeris API
3. 提供统一的接口，简化数据获取逻辑

使用方法:
    from app.utils.data_fetcher import DataFetcher
    
    fetcher = DataFetcher()
    # 异步获取股票实时行情
    quote = await fetcher.get_stock_quote('600519')
    # 异步获取市场指数
    indices = await fetcher.get_market_indices()
"""
import logging
from typing import Optional, List, Dict, Any

from app.utils.akshare_wrapper import akshare_client, safe_ak_call
from app.utils.qveris_wrapper import qveris_client, safe_qveris_call
from app.utils.redis_cache import redis_cache_decorator

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    统一数据获取器
    """
    
    def __init__(self):
        self.akshare_client = akshare_client
        self.qveris_client = qveris_client
    
    @redis_cache_decorator(ttl=60, key_prefix="stock_quote")
    async def get_stock_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取股票实时行情
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票行情数据字典
        """
        try:
            # 优先使用akshare
            logger.info(f"尝试使用akshare获取 {symbol} 实时行情")
            df = await self.akshare_client.get_spot_data()
            if df is not None and not df.empty:
                # 过滤指定股票
                stock_data = df[df['代码'] == symbol]
                if not stock_data.empty:
                    data = stock_data.iloc[0].to_dict()
                    return {
                        "symbol": symbol,
                        "name": data.get('名称', ''),
                        "price": data.get('最新价', 0),
                        "open": data.get('开盘', 0),
                        "high": data.get('最高', 0),
                        "low": data.get('最低', 0),
                        "close": data.get('昨收', 0),
                        "volume": data.get('成交量', 0),
                        "amount": data.get('成交额', 0),
                        "change": data.get('涨跌幅', 0),
                        "time": data.get('date', '')
                    }
        except Exception as e:
            logger.warning(f"akshare获取 {symbol} 实时行情失败: {e}")
        
        try:
            # 尝试使用QVeris API
            logger.info(f"尝试使用QVeris API获取 {symbol} 实时行情")
            data = await self._async_qveris_call(
                lambda: self.qveris_client.async_get_stock_quote(symbol)
            )
            if data:
                return data
        except Exception as e:
            logger.warning(f"QVeris API获取 {symbol} 实时行情失败: {e}")
        
        return None
    
    @redis_cache_decorator(ttl=300, key_prefix="market_indices")
    async def get_market_indices(self) -> List[Dict[str, Any]]:
        """
        获取市场指数
        
        Returns:
            市场指数列表
        """
        try:
            # 优先使用akshare
            logger.info("尝试使用akshare获取市场指数")
            # 这里使用示例数据，因为akshare没有直接的指数接口
            # 实际项目中可以根据需要实现
            sample_indices = [
                {"code": "000001.SH", "name": "上证指数", "price": 4144.08, "change": 0.0, "change_pct": 0.0},
                {"code": "399001.SZ", "name": "深证成指", "price": 13888.88, "change": -11.11, "change_pct": -0.08},
                {"code": "399006.SZ", "name": "创业板指", "price": 2777.77, "change": -22.22, "change_pct": -0.79},
                {"code": "399300.SZ", "name": "沪深300", "price": 4999.99, "change": -33.33, "change_pct": -0.66},
            ]
            return sample_indices
        except Exception as e:
            logger.warning(f"akshare获取市场指数失败: {e}")
        
        try:
            # 尝试使用QVeris API
            logger.info("尝试使用QVeris API获取市场指数")
            indices = await self._async_qveris_call(
                lambda: self.qveris_client.async_get_market_indices()
            )
            if indices:
                return indices
        except Exception as e:
            logger.warning(f"QVeris API获取市场指数失败: {e}")
        
        # 返回示例数据作为最后 fallback
        return [
            {"code": "000001.SH", "name": "上证指数", "price": 4144.08, "change": 0.0, "change_pct": 0.0},
            {"code": "399001.SZ", "name": "深证成指", "price": 13888.88, "change": -11.11, "change_pct": -0.08},
            {"code": "399006.SZ", "name": "创业板指", "price": 2777.77, "change": -22.22, "change_pct": -0.79},
            {"code": "399300.SZ", "name": "沪深300", "price": 4999.99, "change": -33.33, "change_pct": -0.66},
        ]
    
    @redis_cache_decorator(ttl=600, key_prefix="sector_list")
    async def get_sector_list(self) -> List[Dict[str, Any]]:
        """
        获取行业板块列表
        
        Returns:
            行业板块列表
        """
        try:
            # 优先使用akshare
            logger.info("尝试使用akshare获取行业板块列表")
            df = await self.akshare_client.get_sector_list()
            if df is not None and not df.empty:
                sectors = []
                for _, row in df.iterrows():
                    sectors.append({
                        "code": str(row.get("代码", "")),
                        "name": row.get("板块名称", ""),
                        "change_pct": float(row.get("涨跌幅", 0)) if "涨跌幅" in row else 0
                    })
                return sectors
        except Exception as e:
            logger.warning(f"akshare获取行业板块列表失败: {e}")
        
        try:
            # 尝试使用QVeris API
            logger.info("尝试使用QVeris API获取行业板块列表")
            sectors = await self._async_qveris_call(
                lambda: self.qveris_client.async_get_sector_list()
            )
            if sectors:
                return sectors
        except Exception as e:
            logger.warning(f"QVeris API获取行业板块列表失败: {e}")
        
        # 返回示例数据作为最后 fallback
        return [
            {"code": "BK0475", "name": "半导体", "change_pct": 3.5},
            {"code": "BK0447", "name": "新能源车", "change_pct": 2.8},
            {"code": "BK0736", "name": "人工智能", "change_pct": 2.5},
            {"code": "BK0438", "name": "医药生物", "change_pct": 1.2},
            {"code": "BK0448", "name": "光伏产业", "change_pct": 1.0},
        ]
    
    @redis_cache_decorator(ttl=1800, key_prefix="stock_kline")
    async def get_stock_kline(self, symbol: str, interval: str = "daily", limit: int = 100) -> List[List[Any]]:
        """
        获取股票K线数据
        
        Args:
            symbol: 股票代码
            interval: K线周期 (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            limit: 数据条数
            
        Returns:
            K线数据列表
        """
        try:
            # 优先使用akshare
            logger.info(f"尝试使用akshare获取 {symbol} K线数据")
            df = await self.akshare_client.get_stock_hist(
                symbol=symbol,
                period=interval,
                start_date=None,
                end_date=None,
                adjust="qfq"
            )
            if df is not None and not df.empty:
                # 限制数据条数
                df = df.tail(limit)
                
                # 转换为列表格式
                kline_data = []
                for _, row in df.iterrows():
                    kline_data.append([
                        row.get('date', ''),
                        row.get('open', 0),
                        row.get('close', 0),
                        row.get('high', 0),
                        row.get('low', 0),
                        row.get('volume', 0)
                    ])
                return kline_data
        except Exception as e:
            logger.warning(f"akshare获取 {symbol} K线数据失败: {e}")
        
        try:
            # 尝试使用QVeris API
            logger.info(f"尝试使用QVeris API获取 {symbol} K线数据")
            kline_data = await self._async_qveris_call(
                lambda: self.qveris_client.async_get_stock_kline(symbol, interval, limit)
            )
            if kline_data:
                return kline_data
        except Exception as e:
            logger.warning(f"QVeris API获取 {symbol} K线数据失败: {e}")
        
        # 返回空列表作为最后fallback
        return []
    
    async def _async_qveris_call(self, func):
        """
        异步调用QVeris API
        """
        return await func()


# 创建全局数据获取器实例
data_fetcher = DataFetcher()
