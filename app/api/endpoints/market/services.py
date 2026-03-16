#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场API - 数据服务
提供市场数据获取的核心功能
"""

import re
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import akshare as ak
import requests as req_lib

from app.utils.cache import async_ttl_cache

logger = logging.getLogger(__name__)


def symbol_prefix(code: str) -> str:
    """根据股票代码添加市场前缀"""
    code = str(code).strip()
    
    if code.startswith(("sh", "sz", "SH", "SZ")):
        return code.lower()
    
    code6 = code[-6:] if len(code) >= 6 else code.zfill(6)
    
    if code6.startswith(("6", "5", "9")) or code6.startswith("688"):
        return f"sh{code6}"
    else:
        return f"sz{code6}"


def is_index_code(code: str) -> bool:
    """判断是否为指数代码"""
    code = str(code).strip().lower()
    code6 = code[-6:] if len(code) >= 6 else code.zfill(6)
    
    index_prefixes = ["000", "399", "880"]
    return any(code6.startswith(p) for p in index_prefixes)


def index_symbol(code: str) -> str:
    """获取指数的标准代码格式"""
    code = str(code).strip().lower()
    code6 = code[-6:] if len(code) >= 6 else code.zfill(6)
    
    if code6.startswith(("000", "880")):
        return f"sh{code6}"
    elif code6.startswith("399"):
        return f"sz{code6}"
    else:
        return f"sh{code6}"


async def fetch_sina_price(code: str) -> Tuple[float, float, float]:
    """从新浪获取实时价格"""
    try:
        prefix_code = symbol_prefix(code)
        
        def _fetch():
            return req_lib.get(
                f"http://hq.sinajs.cn/list={prefix_code}",
                headers={"Referer": "http://finance.sina.com.cn/"},
                timeout=3
            )
        
        r = await asyncio.to_thread(_fetch)
        
        m = re.search(r'"([^"]+)"', r.text)
        if not m or "," not in m.group(1):
            return 0.0, 0.0, 0.0
        
        parts = m.group(1).split(",")
        if len(parts) < 4:
            return 0.0, 0.0, 0.0
        
        open_price = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
        prev_close = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        price = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0

        baseline = open_price if open_price > 0 else prev_close
        if baseline > 0:
            change = price - baseline
            change_pct = (change / baseline) * 100
        else:
            change = 0.0
            change_pct = 0.0
        
        return price, change, change_pct
        
    except Exception as e:
        logger.warning(f"获取新浪价格失败 {code}: {e}")
        return 0.0, 0.0, 0.0


async def fetch_batch_quotes(codes: List[str]) -> Dict[str, Dict]:
    """批量获取股票行情"""
    results = {}
    
    if not codes:
        return results
    
    try:
        prefix_codes = [symbol_prefix(c) for c in codes]
        codes_str = ",".join(prefix_codes)
        
        def _fetch():
            return req_lib.get(
                f"http://hq.sinajs.cn/list={codes_str}",
                headers={"Referer": "http://finance.sina.com.cn/"},
                timeout=5
            )
        
        r = await asyncio.to_thread(_fetch)
        
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
                results[code] = {
                    "code": code,
                    "name": parts[0],
                    "open": float(parts[1]) if parts[1] else 0,
                    "prev_close": float(parts[2]) if parts[2] else 0,
                    "price": float(parts[3]) if parts[3] else 0,
                    "high": float(parts[4]) if parts[4] else 0,
                    "low": float(parts[5]) if parts[5] else 0,
                    "volume": int(float(parts[8])) if parts[8] else 0,
                    "amount": float(parts[9]) if parts[9] else 0,
                    "time": parts[31] if len(parts) > 31 else "",
                }
                
                if results[code]["prev_close"] > 0:
                    results[code]["change"] = results[code]["price"] - results[code]["prev_close"]
                    results[code]["change_pct"] = (results[code]["change"] / results[code]["prev_close"]) * 100
                else:
                    results[code]["change"] = 0
                    results[code]["change_pct"] = 0
                    
            except (ValueError, IndexError) as e:
                logger.warning(f"解析行情数据失败 {code}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"批量获取行情失败: {e}")
        return results


@async_ttl_cache(ttl=60)
async def get_market_indices() -> List[Dict]:
    """获取主要市场指数"""
    try:
        indices = [
            {"code": "sh000001", "name": "上证指数"},
            {"code": "sz399001", "name": "深证成指"},
            {"code": "sz399006", "name": "创业板指"},
            {"code": "sh000016", "name": "上证50"},
            {"code": "sh000300", "name": "沪深300"},
            {"code": "sh000905", "name": "中证500"},
            {"code": "sh000688", "name": "科创50"},
        ]
        
        codes = [idx["code"] for idx in indices]
        quotes = await fetch_batch_quotes(codes)
        
        for idx in indices:
            code6 = idx["code"][2:]
            if code6 in quotes:
                idx.update(quotes[code6])
        
        return indices
        
    except Exception as e:
        logger.error(f"获取市场指数失败: {e}")
        return []


@async_ttl_cache(ttl=300)
async def get_market_stats() -> Dict[str, Any]:
    """获取市场统计数据"""
    try:
        df = await asyncio.to_thread(ak.stock_zh_a_spot_em)
        
        if df is None or df.empty:
            return {}
        
        total = len(df)
        
        change_col = next((c for c in df.columns if "涨跌幅" in c), None)
        if change_col:
            up = len(df[df[change_col] > 0])
            down = len(df[df[change_col] < 0])
            flat = len(df[df[change_col] == 0])
            limit_up = len(df[df[change_col] >= 9.9])
            limit_down = len(df[df[change_col] <= -9.9])
            avg_change = df[change_col].mean()
        else:
            up = down = flat = limit_up = limit_down = 0
            avg_change = 0
        
        amount_col = next((c for c in df.columns if "成交额" in c), None)
        turnover = df[amount_col].sum() if amount_col else 0
        
        return {
            "total_stocks": total,
            "up_count": up,
            "down_count": down,
            "flat_count": flat,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "avg_change": round(avg_change, 2),
            "turnover": round(turnover, 2),
            "update_time": datetime.now().strftime("%H:%M:%S"),
        }
        
    except Exception as e:
        logger.error(f"获取市场统计失败: {e}")
        return {}


@async_ttl_cache(ttl=300)
async def get_sector_fund_flow() -> List[Dict]:
    """获取板块资金流向"""
    try:
        df = await asyncio.to_thread(ak.stock_board_concept_fund_flow_em)
        
        if df is None or df.empty:
            return []
        
        results = []
        for _, row in df.head(30).iterrows():
            results.append({
                "sector": row.get("板块", ""),
                "net_inflow": float(row.get("今日主力净流入-净额", 0)),
                "net_inflow_pct": float(row.get("今日主力净流入-净占比", 0)),
                "main_inflow": float(row.get("今日超大单净流入-净额", 0)),
                "retail_inflow": float(row.get("今日小单净流入-净额", 0)),
            })
        
        return results
        
    except Exception as e:
        logger.error(f"获取板块资金流向失败: {e}")
        return []


@async_ttl_cache(ttl=600)
async def get_dragon_tiger(date: str = None) -> List[Dict]:
    """获取龙虎榜数据"""
    try:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        df = await asyncio.to_thread(
            ak.stock_lhb_detail_em,
            start_date=date,
            end_date=date
        )
        
        if df is None or df.empty:
            return []
        
        results = []
        for _, row in df.iterrows():
            results.append({
                "code": str(row.get("代码", "")),
                "name": row.get("名称", ""),
                "close_price": float(row.get("收盘价", 0)),
                "change_pct": float(row.get("涨跌幅", 0)),
                "turnover_rate": float(row.get("换手率", 0)),
                "reason": row.get("上榜原因", ""),
                "net_buy": float(row.get("龙虎榜净买额", 0)),
            })
        
        return results
        
    except Exception as e:
        logger.error(f"获取龙虎榜失败: {e}")
        return []


def normalize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    """标准化K线数据格式"""
    if df is None or df.empty:
        return df
    
    column_mapping = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "change_pct",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    
    df = df.rename(columns=column_mapping)
    
    required_cols = ["open", "close", "high", "low", "volume"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df


class MarketDataService:
    """市场数据服务类"""
    
    def __init__(self):
        self._cache = {}
    
    async def get_realtime_quote(self, code: str) -> Dict:
        """获取实时报价"""
        price, change, change_pct = await fetch_sina_price(code)
        return {
            "code": code,
            "price": price,
            "change": change,
            "change_pct": change_pct,
        }
    
    async def get_batch_quotes(self, codes: List[str]) -> Dict[str, Dict]:
        """批量获取报价"""
        return await fetch_batch_quotes(codes)
    
    async def get_indices(self) -> List[Dict]:
        """获取市场指数"""
        return await get_market_indices()
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取市场统计"""
        return await get_market_stats()
    
    async def get_fund_flow(self) -> List[Dict]:
        """获取资金流向"""
        return await get_sector_fund_flow()
    
    async def get_dragon_tiger(self, date: str = None) -> List[Dict]:
        """获取龙虎榜"""
        return await get_dragon_tiger(date)


_market_data_service = None


def get_market_data_service() -> MarketDataService:
    """获取市场数据服务实例"""
    global _market_data_service
    if _market_data_service is None:
        _market_data_service = MarketDataService()
    return _market_data_service
