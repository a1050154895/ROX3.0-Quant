#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时行情推送系统
支持WebSocket实时推送行情数据

功能：
1. 实时行情推送
2. 多股票订阅
3. 行情数据缓存
4. 断线重连
"""

import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class QuoteType(Enum):
    """行情类型"""
    REALTIME = "realtime"
    KLINE = "kline"
    DEPTH = "depth"
    TICK = "tick"


@dataclass
class RealtimeQuote:
    """实时行情数据"""
    code: str
    name: str
    price: float
    open: float
    high: float
    low: float
    prev_close: float
    volume: int
    amount: float
    bid1: float
    ask1: float
    bid1_vol: int
    ask1_vol: int
    time: str
    change: float = 0.0
    change_pct: float = 0.0
    
    def __post_init__(self):
        if self.prev_close > 0:
            self.change = self.price - self.prev_close
            self.change_pct = (self.change / self.prev_close) * 100


@dataclass
class Subscription:
    """订阅信息"""
    code: str
    quote_type: QuoteType
    callback: Optional[Callable] = None
    last_update: datetime = None


class RealtimeQuoteManager:
    """
    实时行情管理器
    
    功能：
    1. 管理行情订阅
    2. 推送实时数据
    3. 数据缓存
    """
    
    def __init__(self):
        self._subscriptions: Dict[str, Set[str]] = {}
        self._quote_cache: Dict[str, RealtimeQuote] = {}
        self._ws_clients: List[Any] = []
        self._running = False
        self._task = None
    
    async def start(self):
        """启动行情服务"""
        self._running = True
        self._task = asyncio.create_task(self._quote_loop())
        logger.info("实时行情服务已启动")
    
    async def stop(self):
        """停止行情服务"""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("实时行情服务已停止")
    
    async def _quote_loop(self):
        """行情推送循环"""
        while self._running:
            try:
                if self._subscriptions:
                    codes = list(self._subscriptions.keys())
                    quotes = await self._fetch_quotes(codes)
                    
                    for quote in quotes:
                        self._quote_cache[quote.code] = quote
                        await self._broadcast_quote(quote)
                
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"行情推送错误: {e}")
                await asyncio.sleep(5)
    
    async def _fetch_quotes(self, codes: List[str]) -> List[RealtimeQuote]:
        """获取实时行情"""
        quotes = []
        
        try:
            import requests
            
            prefix_codes = [self._add_prefix(code) for code in codes]
            codes_str = ",".join(prefix_codes)
            
            r = requests.get(
                f"http://hq.sinajs.cn/list={codes_str}",
                headers={"Referer": "http://finance.sina.com.cn/"},
                timeout=5
            )
            
            for line in r.text.split(";"):
                if not line.strip():
                    continue
                
                match = re.match(r'var hq_str_(\w+)="(.*)"', line)
                if not match:
                    continue
                
                full_code = match.group(1)
                data = match.group(2)
                
                if not data:
                    continue
                
                parts = data.split(",")
                if len(parts) < 32:
                    continue
                
                code = full_code[2:]
                
                try:
                    quote = RealtimeQuote(
                        code=code,
                        name=parts[0],
                        open=float(parts[1]) if parts[1] else 0,
                        prev_close=float(parts[2]) if parts[2] else 0,
                        price=float(parts[3]) if parts[3] else 0,
                        high=float(parts[4]) if parts[4] else 0,
                        low=float(parts[5]) if parts[5] else 0,
                        volume=int(float(parts[8])) if parts[8] else 0,
                        amount=float(parts[9]) if parts[9] else 0,
                        bid1=float(parts[10]) if parts[10] else 0,
                        ask1=float(parts[20]) if len(parts) > 20 and parts[20] else 0,
                        bid1_vol=int(float(parts[11])) if parts[11] else 0,
                        ask1_vol=int(float(parts[21])) if len(parts) > 21 and parts[21] else 0,
                        time=parts[31] if len(parts) > 31 else "",
                    )
                    quotes.append(quote)
                except (ValueError, IndexError) as e:
                    logger.warning(f"解析行情失败 {code}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"获取行情失败: {e}")
        
        return quotes
    
    def _add_prefix(self, code: str) -> str:
        """添加市场前缀"""
        code = str(code).strip()
        if code.startswith(("sh", "sz")):
            return code
        code6 = code[-6:] if len(code) >= 6 else code.zfill(6)
        if code6.startswith(("6", "5", "9")):
            return f"sh{code6}"
        return f"sz{code6}"
    
    async def _broadcast_quote(self, quote: RealtimeQuote):
        """广播行情数据"""
        message = json.dumps({
            "type": "quote",
            "data": {
                "code": quote.code,
                "name": quote.name,
                "price": quote.price,
                "change": quote.change,
                "change_pct": round(quote.change_pct, 2),
                "open": quote.open,
                "high": quote.high,
                "low": quote.low,
                "volume": quote.volume,
                "amount": quote.amount,
                "bid1": quote.bid1,
                "ask1": quote.ask1,
                "time": quote.time,
            }
        })
        
        for client in self._ws_clients:
            try:
                await client.send_text(message)
            except Exception as e:
                logger.warning(f"发送消息失败: {e}")
    
    def subscribe(self, code: str, client_id: str = "default"):
        """订阅股票行情"""
        if code not in self._subscriptions:
            self._subscriptions[code] = set()
        self._subscriptions[code].add(client_id)
        logger.info(f"订阅 {code} 成功")
    
    def unsubscribe(self, code: str, client_id: str = "default"):
        """取消订阅"""
        if code in self._subscriptions:
            self._subscriptions[code].discard(client_id)
            if not self._subscriptions[code]:
                del self._subscriptions[code]
        logger.info(f"取消订阅 {code}")
    
    def add_ws_client(self, client):
        """添加WebSocket客户端"""
        self._ws_clients.append(client)
    
    def remove_ws_client(self, client):
        """移除WebSocket客户端"""
        if client in self._ws_clients:
            self._ws_clients.remove(client)
    
    def get_cached_quote(self, code: str) -> Optional[RealtimeQuote]:
        """获取缓存的行情"""
        return self._quote_cache.get(code)
    
    def get_all_cached_quotes(self) -> Dict[str, RealtimeQuote]:
        """获取所有缓存行情"""
        return self._quote_cache.copy()


_quote_manager = None


def get_quote_manager() -> RealtimeQuoteManager:
    """获取行情管理器单例"""
    global _quote_manager
    if _quote_manager is None:
        _quote_manager = RealtimeQuoteManager()
    return _quote_manager
