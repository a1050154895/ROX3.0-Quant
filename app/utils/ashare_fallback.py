# -*- coding: utf-8 -*-
# Ashare 备用数据源：腾讯/新浪日线，当 akshare 东方财富失败时使用
# 参考 https://github.com/mpquant/Ashare
import logging
import requests
import pandas as pd
from typing import Dict, List, Optional, Any

logger = logging.getLogger("rox-ashare-fallback")


def _code_to_ashare(symbol: str) -> str:
    """6 位代码转 Ashare 格式：sh600519 / sz000001"""
    code = str(symbol).strip()[:6].zfill(6)
    if code.startswith("6") or code.startswith("5"):
        return "sh" + code
    return "sz" + code


def get_daily_kline_ashare(symbol: str, count: int = 500, period: str = "daily") -> pd.DataFrame:
    """
    用 Ashare 数据源（腾讯/新浪）拉取日线/周线/月线，返回标准 DataFrame：date, open, close, high, low, volume。
    period: daily | weekly | monthly
    """
    code = _code_to_ashare(symbol)
    unit = "week" if period == "weekly" else "month" if period == "monthly" else "day"
    # 腾讯日线 API（前复权 qfq）
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},{unit},,,{count},qfq"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        st = r.json()
        if "data" not in st or code not in st["data"]:
            return pd.DataFrame()
        stk = st["data"][code]
        ms = "qfq" + unit
        buf = stk.get(ms) or stk.get(unit)
        if not buf:
            return pd.DataFrame()
        
        # FIX: Tencent sometimes returns more than 6 columns (e.g. turnover rate). 
        # We only need the first 6 columns.
        buf = [row[:6] for row in buf]

        df = pd.DataFrame(buf, columns=["time", "open", "close", "high", "low", "volume"])
        for c in ["open", "close", "high", "low", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["time"])
        df = df[["date", "open", "close", "high", "low", "volume"]]
        df = df.sort_values("date").tail(count)
        return df
    except Exception as e:
        logger.warning(f"Ashare Tencent failed for {symbol}: {e}")
    # 新浪日线备用
    try:
        freq_sina = "1d" if period == "daily" else "1w" if period == "weekly" else "1M"
        scale = 240  # 日线
        if period == "weekly":
            scale = 1200
        elif period == "monthly":
            scale = 7200
        url_sina = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale={scale}&ma=5&datalen={count}"
        r = requests.get(url_sina, timeout=10)
        r.raise_for_status()
        dstr = r.json()
        if not dstr:
            return pd.DataFrame()
        df = pd.DataFrame(dstr, columns=["day", "open", "high", "low", "close", "volume"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["day"])
        df = df[["date", "open", "close", "high", "low", "volume"]]
        df = df.sort_values("date").tail(count)
        return df
    except Exception as e:
        logger.warning(f"Ashare Sina failed for {symbol}: {e}")
    return pd.DataFrame()


def get_realtime_quotes_sina(codes: list) -> pd.DataFrame:
    """
    批量获取新浪实时行情 (list=sh600519,sz000001)
    返回 DataFrame columns: [代码, 名称, 最新价, 涨跌幅, 成交额, date]
    """
    if not codes:
        return pd.DataFrame()
        
    # Batching (Sina URL length limit)
    batch_size = 80
    results = []
    
    # Helper to format code
    fmt_codes = []
    for c in codes:
        try:
            fmt_codes.append(_code_to_ashare(str(c)))
        except:
            pass
    
    import requests
    from datetime import datetime
    
    for i in range(0, len(fmt_codes), batch_size):
        batch = fmt_codes[i:i+batch_size]
        url = "http://hq.sinajs.cn/list=" + ",".join(batch)
        try:
            headers = {"Referer": "http://finance.sina.com.cn/"}
            r = requests.get(url, headers=headers, timeout=5)
            # Parse response
            # var hq_str_sh600519="贵州茅台,1784.00,1781.99,1785.00,1799.00,1776.05,1784.50,1785.00,2485666,4444565258.00,1400,1784.50,1400,1784.49,100,1784.41,100,1784.40,100,1784.39,200,1785.00,100,1785.01,200,1785.03,500,1785.08,200,1785.09,2023-11-17,15:00:00,00,";
            lines = r.text.splitlines()
            for line in lines:
                if '="' in line:
                    parts = line.split('="')
                    if len(parts) < 2: continue
                    
                    symbol_part = parts[0].split('_')[-1] # sh600519
                    data_str = parts[1].strip('";')
                    if not data_str: continue
                    
                    fields = data_str.split(',')
                    if len(fields) < 30: continue
                    
                    name = fields[0]
                    # open_p = float(fields[1])
                    prev_close = float(fields[2])
                    price = float(fields[3])
                    
                    if prev_close > 0:
                        pct = (price - prev_close) / prev_close * 100
                    else:
                        pct = 0.0
                        
                    results.append({
                        "代码": symbol_part[2:], # Remove sh/sz
                        "名称": name,
                        "最新价": price,
                        "涨跌幅": round(pct, 2),
                        "成交额": float(fields[9])
                    })
        except Exception as e:
            logger.warning(f"Sina batch fetch failed: {e}")
            
    return pd.DataFrame(results)


def get_eastmoney_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """
    获取东方财富网实时行情
    
    Args:
        symbol: 股票代码
        
    Returns:
        实时行情数据字典
    """
    try:
        import akshare as ak
        # 使用 AkShare 获取东方财富网数据
        df = ak.stock_zh_a_spot_em()
        # 过滤指定股票
        stock_data = df[df['代码'] == symbol]
        if stock_data.empty:
            return None
        
        # 转换为字典
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
        logger.error(f"Error getting EastMoney quote: {e}")
        return None


def get_eastmoney_kline(symbol: str, interval: str, count: int) -> Optional[List[List[Any]]]:
    """
    获取东方财富网K线数据
    
    Args:
        symbol: 股票代码
        interval: K线周期 (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
        count: 数据条数
        
    Returns:
        K线数据列表
    """
    try:
        import akshare as ak
        # 转换周期
        period_map = {
            '1min': '1',
            '5min': '5',
            '15min': '15',
            '30min': '30',
            '60min': '60',
            'daily': 'daily',
            'weekly': 'weekly',
            'monthly': 'monthly'
        }
        
        period = period_map.get(interval, 'daily')
        
        if period in ['1', '5', '15', '30', '60']:
            # 分钟级数据
            df = ak.stock_zh_a_minute(symbol=symbol, period=period, adjust="qfq")
        else:
            # 日/周/月数据
            df = ak.stock_zh_a_hist(symbol=symbol, period=period, adjust="qfq")
        
        # 限制数据条数
        df = df.tail(count)
        
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
        logger.error(f"Error getting EastMoney kline: {e}")
        return None
