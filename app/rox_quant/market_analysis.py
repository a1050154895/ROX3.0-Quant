import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional, Tuple

from app.utils.error_handling import safe_execute, retry

class MarketAnalyzer:
    def __init__(self):
        from app.rox_quant.data_source_manager import get_data_source_manager
        self.data_source_manager = get_data_source_manager()
        self.market_data_cache = {}
        self.cache_expiry = 60  # 缓存60秒
    
    @safe_execute
    async def get_market_data(self, symbol: str) -> Dict:
        """获取单个股票的市场数据"""
        cache_key = f"market_data_{symbol}"
        current_time = datetime.now().timestamp()
        
        if cache_key in self.market_data_cache:
            cached_data, timestamp = self.market_data_cache[cache_key]
            if current_time - timestamp < self.cache_expiry:
                return cached_data
        
        try:
            kline_data = self.data_source_manager.get_kline(symbol, "1m", 200)
            if not kline_data:
                return {}
            
            price_data = []
            for kline in kline_data:
                price_data.append({
                    "time": kline[0],
                    "open": kline[1],
                    "high": kline[2],
                    "low": kline[3],
                    "close": kline[4],
                    "volume": kline[5]
                })
            
            market_data = {
                "symbol": symbol,
                "price_data": price_data,
                "latest_price": price_data[-1]["close"] if price_data else 0,
                "volume": sum(p["volume"] for p in price_data[-20:]) if len(price_data) >=20 else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            self.market_data_cache[cache_key] = (market_data, current_time)
            return market_data
        except Exception as e:
            return {}
    
    @safe_execute
    async def get_broad_market_data(self) -> Dict:
        """获取大盘市场数据"""
        cache_key = "broad_market_data"
        current_time = datetime.now().timestamp()
        
        if cache_key in self.market_data_cache:
            cached_data, timestamp = self.market_data_cache[cache_key]
            if current_time - timestamp < self.cache_expiry:
                return cached_data
        
        try:
            indices = {
                "上证指数": await self.get_index_data("000001.SH"),
                "深证成指": await self.get_index_data("399001.SZ"),
                "创业板指": await self.get_index_data("399006.SZ")
            }
            
            advancing, declining = await self.get_market_breadth()
            
            market_data = {
                "indices": indices,
                "advancing": advancing,
                "declining": declining,
                "market_breadth": advancing / (advancing + declining) if (advancing + declining) > 0 else 0.5,
                "timestamp": datetime.now().isoformat()
            }
            
            self.market_data_cache[cache_key] = (market_data, current_time)
            return market_data
        except Exception as e:
            return {
                "indices": {},
                "advancing": 0,
                "declining": 0,
                "market_breadth": 0.5,
                "timestamp": datetime.now().isoformat()
            }
    
    @safe_execute
    async def get_index_data(self, symbol: str) -> Dict:
        """获取指数数据"""
        try:
            kline_data = await self.alltick_client.get_kline(symbol, "1m", 20)
            if not kline_data:
                return {}
            
            latest_close = kline_data[-1][4]
            previous_close = kline_data[0][1]
            change = (latest_close - previous_close) / previous_close * 100
            
            return {
                "symbol": symbol,
                "latest_price": latest_close,
                "change": change,
                "volume": sum(k[5] for k in kline_data)
            }
        except Exception as e:
            return {}
    
    @safe_execute
    async def get_market_breadth(self) -> Tuple[int, int]:
        """获取市场涨跌家数"""
        try:
            stocks = ["600519.SH", "000858.SZ", "000333.SZ", "000001.SZ", "601318.SH"]
            advancing = 0
            declining = 0
            
            for stock in stocks:
                data = await self.get_market_data(stock)
                if data:
                    price_data = data.get("price_data", [])
                    if len(price_data) >= 2:
                        if price_data[-1]["close"] > price_data[-2]["close"]:
                            advancing += 1
                        elif price_data[-1]["close"] < price_data[-2]["close"]:
                            declining += 1
            
            return advancing, declining
        except Exception as e:
            return 0, 0
    
    @safe_execute
    def analyze_volatility(self, price_data: List[Dict]) -> Dict:
        """分析波动率"""
        if len(price_data) < 20:
            return {
                "volatility": 0,
                "volatility_change": 0,
                "is_volatile": False
            }
        
        prices = [p["close"] for p in price_data]
        returns = np.diff(prices) / prices[:-1]
        
        current_vol = np.std(returns[-20:])
        historical_vol = np.std(returns[:-20]) if len(returns) > 20 else current_vol
        
        volatility_change = (current_vol - historical_vol) / historical_vol if historical_vol > 0 else 0
        
        return {
            "volatility": current_vol,
            "volatility_change": volatility_change,
            "is_volatile": current_vol > 0.02
        }
    
    @safe_execute
    def analyze_trend(self, price_data: List[Dict]) -> Dict:
        """分析趋势"""
        if len(price_data) < 50:
            return {
                "trend": "neutral",
                "trend_strength": 0,
                "support_level": 0,
                "resistance_level": 0
            }
        
        prices = [p["close"] for p in price_data]
        
        short_ma = np.mean(prices[-20:])
        long_ma = np.mean(prices[-50:])
        
        trend = "neutral"
        trend_strength = 0
        
        if short_ma > long_ma * 1.01:
            trend = "uptrend"
            trend_strength = min((short_ma / long_ma - 1) * 100, 1.0)
        elif short_ma < long_ma * 0.99:
            trend = "downtrend"
            trend_strength = min((1 - short_ma / long_ma) * 100, 1.0)
        
        support_level = min(prices[-50:])
        resistance_level = max(prices[-50:])
        
        return {
            "trend": trend,
            "trend_strength": trend_strength,
            "support_level": support_level,
            "resistance_level": resistance_level
        }
    
    @safe_execute
    def analyze_volume(self, price_data: List[Dict]) -> Dict:
        """分析成交量"""
        if len(price_data) < 20:
            return {
                "volume": 0,
                "volume_change": 0,
                "is_volume_increasing": False
            }
        
        volumes = [p["volume"] for p in price_data]
        current_volume = sum(volumes[-5:])
        historical_volume = sum(volumes[-20:-5]) if len(volumes) > 20 else current_volume
        
        volume_change = (current_volume - historical_volume) / historical_volume if historical_volume > 0 else 0
        
        return {
            "volume": current_volume,
            "volume_change": volume_change,
            "is_volume_increasing": volume_change > 0.2
        }
    
    @safe_execute
    async def get_sector_performance(self) -> Dict:
        """获取行业表现"""
        sectors = {
            "金融": ["601318.SH", "600036.SH", "600031.SH"],
            "科技": ["000063.SZ", "002415.SZ", "300750.SZ"],
            "消费": ["600519.SH", "000858.SZ", "601888.SH"],
            "医药": ["600276.SH", "600521.SH", "300015.SZ"],
            "能源": ["601857.SH", "600028.SH", "601088.SH"]
        }
        
        sector_performance = {}
        
        for sector, stocks in sectors.items():
            total_change = 0
            valid_stocks = 0
            
            for stock in stocks:
                data = await self.get_market_data(stock)
                if data:
                    price_data = data.get("price_data", [])
                    if len(price_data) >= 2:
                        close_prices = [p["close"] for p in price_data[-10:]]
                        if len(close_prices) >= 2:
                            change = (close_prices[-1] - close_prices[0]) / close_prices[0] * 100
                            total_change += change
                            valid_stocks += 1
            
            if valid_stocks > 0:
                sector_performance[sector] = {
                    "average_change": total_change / valid_stocks,
                    "stocks_count": valid_stocks
                }
            else:
                sector_performance[sector] = {
                    "average_change": 0,
                    "stocks_count": 0
                }
        
        return sector_performance
    
    @safe_execute
    async def get_market_opportunities(self) -> List[Dict]:
        """获取市场机会"""
        watchlist = ["600519.SH", "000858.SZ", "000333.SZ", "000001.SZ", "601318.SH", "000063.SZ", "600036.SH", "300750.SZ"]
        opportunities = []
        
        for symbol in watchlist:
            data = await self.get_market_data(symbol)
            if data:
                price_data = data.get("price_data", [])
                if len(price_data) >= 50:
                    trend_analysis = self.analyze_trend(price_data)
                    volume_analysis = self.analyze_volume(price_data)
                    volatility_analysis = self.analyze_volatility(price_data)
                    
                    if trend_analysis["trend"] == "uptrend" and trend_analysis["trend_strength"] > 0.3:
                        if volume_analysis["is_volume_increasing"]:
                            opportunities.append({
                                "symbol": symbol,
                                "opportunity_type": "strong_uptrend",
                                "trend_strength": trend_analysis["trend_strength"],
                                "volume_change": volume_analysis["volume_change"],
                                "latest_price": data.get("latest_price", 0),
                                "resistance_level": trend_analysis["resistance_level"]
                            })
                    elif trend_analysis["trend"] == "downtrend" and trend_analysis["trend_strength"] > 0.3:
                        opportunities.append({
                            "symbol": symbol,
                            "opportunity_type": "strong_downtrend",
                            "trend_strength": trend_analysis["trend_strength"],
                            "latest_price": data.get("latest_price", 0),
                            "support_level": trend_analysis["support_level"]
                        })
        
        return opportunities