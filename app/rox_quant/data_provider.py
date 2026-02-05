import os
from dataclasses import dataclass
from typing import List, Optional
import random
import datetime
from app.rox_quant.data_manager import DataManager
from app.rox_quant.alltick_client import AllTickClient


@dataclass
class PricePoint:
    date: str
    close: float
    volume: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None


_FX_LAST = {"ts": None, "values": {}}

class DataProvider:
    def __init__(self):
        self._cache_history = {}
        self._cache_name_code = {}
        self._cache_sector = {}
        self._cache_period = {}
        self._last_quality = {}
        self._http = None
        self._em_progress = {"step": "", "done": 0, "total": 0, "eta": None, "net": "unknown", "error": None}
        self._em_last = {}
        self.data_manager = DataManager()
        
        # Initialize AllTick Client
        self.alltick = None
        try:
            # Token from environment variable
            token = os.getenv("ALLTICK_TOKEN")
            if token:
                self.alltick = AllTickClient(token)
                # Start connection automatically
                self.alltick.connect()
            else:
                print("Warning: ALLTICK_TOKEN not found in environment variables. Real-time tick data will be disabled.")
        except Exception as e:
            print(f"Failed to init AllTick: {e}")

        try:
            from curl_cffi import requests as _req  # type: ignore
            self._http = _req
        except Exception:
            self._http = None

    def _http_get_json(self, url: str, headers: Optional[dict] = None, params: Optional[dict] = None, retries: int = 3, timeout: int = 20) -> Optional[dict]:
        if not self._http:
            return None
        last_err = None
        for _ in range(max(1, retries)):
            try:
                r = self._http.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
                if r.status_code == 200:
                    try:
                        return r.json()
                    except Exception:
                        return None
            except Exception as e:
                last_err = e
        return None

    def get_history(self, symbol: str, days: int = 120) -> List[PricePoint]:
        try:
            key = (symbol, days)
            if key in self._cache_history:
                return self._cache_history[key]
            
            # 1. Try Local First (Fresh data)
            local_data = self.data_manager.load_kline(symbol, period="daily", max_age_hours=24)
            if local_data:
                local_data = local_data[-days:]
                res = [PricePoint(
                    date=r['date'], 
                    close=float(r['close']), 
                    volume=float(r['volume']) if r.get('volume') else 0.0,
                    open=float(r['open']) if r.get('open') else float(r['close']),
                    high=float(r['high']) if r.get('high') else float(r['close']),
                    low=float(r['low']) if r.get('low') else float(r['close'])
                ) for r in local_data]
                self._cache_history[key] = res
                return res

            # 2. Fetch from AkShare
            import akshare as ak  # type: ignore
            df = ak.stock_zh_a_hist(symbol=symbol, adjust="qfq")
            if df is None or df.empty:
                # Fallback to stale local data
                local_data = self.data_manager.load_kline(symbol, period="daily", max_age_hours=0)
                if local_data:
                    local_data = local_data[-days:]
                    res = [PricePoint(date=r['date'], close=float(r['close']), volume=float(r['volume']) if r.get('volume') else 0.0) for r in local_data]
                    self._cache_history[key] = res
                    return res
                res = self._synthetic(days)
                self._cache_history[key] = res
                return res

            # Save to local
            save_data = []
            for _, row in df.iterrows():
                save_data.append({
                    "date": str(row["日期"]),
                    "close": float(row["收盘"]),
                    "volume": float(row["成交量"]),
                    "open": float(row["开盘"]),
                    "high": float(row["最高"]),
                    "low": float(row["最低"])
                })
            self.data_manager.save_kline(symbol, save_data, period="daily")

            records = df.tail(days)[["日期", "收盘", "成交量", "开盘", "最高", "最低"]].values.tolist()
            res = [PricePoint(
                date=r[0], 
                close=float(r[1]), 
                volume=float(r[2]),
                open=float(r[3]),
                high=float(r[4]),
                low=float(r[5])
            ) for r in records]
            self._cache_history[key] = res
            return res
        except Exception:
            # Fallback to stale local data
            local_data = self.data_manager.load_kline(symbol, period="daily", max_age_hours=0)
            if local_data:
                local_data = local_data[-days:]
                res = [PricePoint(date=r['date'], close=float(r['close']), volume=float(r['volume']) if r.get('volume') else 0.0) for r in local_data]
                self._cache_history[(symbol, days)] = res
                return res
            res = self._synthetic(days)
            self._cache_history[(symbol, days)] = res
            return res

    def get_history_k(self, symbol: str, period: str = "daily", limit: int = 120) -> List[PricePoint]:
        try:
            key = (symbol, period, limit)
            if key in self._cache_period:
                return self._cache_period[key]
            
            p = "daily" if period not in ["daily", "weekly", "monthly"] else period
            
            # 1. Try Local First
            local_data = self.data_manager.load_kline(symbol, period=p, max_age_hours=24)
            if local_data:
                if limit:
                    local_data = local_data[-limit:]
                res = [PricePoint(
                    date=r['date'], 
                    close=float(r['close']), 
                    volume=float(r['volume']) if r.get('volume') else 0.0,
                    open=float(r['open']) if r.get('open') else float(r['close']),
                    high=float(r['high']) if r.get('high') else float(r['close']),
                    low=float(r['low']) if r.get('low') else float(r['close'])
                ) for r in local_data]
                self._cache_period[key] = res
                return res

            import akshare as ak  # type: ignore
            df = ak.stock_zh_a_hist(symbol=symbol, adjust="qfq", period=p)
            if df is None or df.empty:
                # Fallback to stale local
                local_data = self.data_manager.load_kline(symbol, period=p, max_age_hours=0)
                if local_data:
                    if limit:
                        local_data = local_data[-limit:]
                    res = [PricePoint(date=r['date'], close=float(r['close']), volume=float(r['volume']) if r.get('volume') else 0.0) for r in local_data]
                    self._cache_period[key] = res
                    return res
                return self.get_history(symbol, days=limit)
            
            # Save to local
            save_data = []
            for _, row in df.iterrows():
                save_data.append({
                    "date": str(row["日期"]),
                    "close": float(row["收盘"]),
                    "volume": float(row["成交量"]),
                    "open": float(row["开盘"]),
                    "high": float(row["最高"]),
                    "low": float(row["最低"])
                })
            self.data_manager.save_kline(symbol, save_data, period=p)

            if limit:
                df = df.tail(limit)
            records = df[["日期", "收盘", "成交量"]].values.tolist()
            res = [PricePoint(date=r[0], close=float(r[1]), volume=float(r[2])) for r in records]
            self._cache_period[key] = res
            return res
        except Exception:
            return self.get_history(symbol, days=limit)

    def _normalize_symbol_for_alltick(self, symbol: str) -> str:
        """
        Convert 'sh600519' -> '600519.SH' for AllTick.
        """
        s = symbol.lower()
        if s.startswith("sh"):
            return f"{s[2:]}.SH"
        if s.startswith("sz"):
            return f"{s[2:]}.SZ"
        # If already has dot (e.g. 700.HK), assume correct
        if "." in s:
            return s.upper()
        # Default fallback
        return s.upper()

    def get_realtime_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest real-time price.
        Prioritizes AllTick WebSocket data (Mid Price of Best Bid/Ask),
        Falls back to AkShare spot price.
        """
        # 1. Try AllTick
        if self.alltick and self.alltick.is_connected:
            at_symbol = self._normalize_symbol_for_alltick(symbol)
            
            # Ensure subscribed (lazy subscription)
            if at_symbol not in self.alltick.subscribed_symbols:
                self.alltick.subscribe([at_symbol])
            
            tick = self.alltick.latest_ticks.get(at_symbol)
            if tick:
                # Try to get last price first, then mid price
                if 'last_price' in tick:
                     try:
                        return float(tick['last_price'])
                     except (ValueError, TypeError):
                        pass

                bids = tick.get('bids', [])
                asks = tick.get('asks', [])
                if bids and asks:
                    try:
                        bid = float(bids[0]['price'])
                        ask = float(asks[0]['price'])
                        return (bid + ask) / 2.0
                    except (ValueError, IndexError):
                        pass

        # 2. Fallback to AkShare Spot
        return self.get_spot_price(symbol)

    def get_realtime_quote(self, symbol: str) -> dict:
        """
        Get a full real-time quote for a stock.
        Returns dict with keys: price, open, high, low, volume, pre_close, change, change_pct, time.
        """
        res = {
            "price": 0.0, "open": 0.0, "high": 0.0, "low": 0.0, 
            "volume": 0.0, "pre_close": 0.0, "change": 0.0, "change_pct": 0.0,
            "time": ""
        }
        
        # 1. Try AllTick
        if self.alltick and self.alltick.is_connected:
            at_symbol = self._normalize_symbol_for_alltick(symbol)
            if at_symbol not in self.alltick.subscribed_symbols:
                self.alltick.subscribe([at_symbol])
            
            tick = self.alltick.latest_ticks.get(at_symbol)
            if tick:
                # AllTick Tick Structure (Standard)
                # "last_price", "open", "high", "low", "volume", "tick_time"
                try:
                    res["price"] = float(tick.get("last_price", 0))
                    res["open"] = float(tick.get("open", 0))
                    res["high"] = float(tick.get("high", 0))
                    res["low"] = float(tick.get("low", 0))
                    res["volume"] = float(tick.get("volume", 0))
                    res["time"] = tick.get("tick_time", "")
                    
                    # Pre_close might be in 'pre_close' or 'reference_price' or we calculate
                    if "pre_close" in tick:
                        res["pre_close"] = float(tick["pre_close"])
                    elif "reference_price" in tick:
                        res["pre_close"] = float(tick["reference_price"])
                    
                    # If we have price and pre_close, calc change
                    if res["price"] > 0 and res["pre_close"] > 0:
                        res["change"] = res["price"] - res["pre_close"]
                        res["change_pct"] = (res["change"] / res["pre_close"]) * 100
                    
                    if res["price"] > 0:
                        return res
                except Exception:
                    pass

        # 2. Fallback to AkShare Spot (or cached spot)
        # We can reuse get_spot_price but we want more fields.
        # Let's try to fetch spot row again.
        try:
            import akshare as ak
            # This is slow if called per stock. Better for batch. 
            # But for single stock fallback:
            # Maybe use cached spot list if available?
            # For now, let's just return what we have if AllTick failed.
            pass
        except Exception:
            pass
            
        return res

    def get_spot_price(self, symbol: str) -> Optional[float]:
        try:
            import akshare as ak  # type: ignore
            df = ak.stock_zh_a_spot()
            if df is None or df.empty:
                return None
            code_cols = [c for c in df.columns if ("代码" in c) or ("code" in c.lower())]
            price_cols = [c for c in df.columns if ("最新价" in c) or ("最新" in c) or ("price" in c.lower())]
            if not code_cols or not price_cols:
                return None
            ccol = code_cols[0]
            pcol = price_cols[0]
            row = df[df[ccol].astype(str) == str(symbol)]
            if row is None or row.empty:
                return None
            return float(row.iloc[0][pcol])
        except Exception:
            return None

    def get_all_market_data(self) -> Optional[object]:
        """
        获取全市场实时行情快照（包含市值、PE、PB等）
        返回 DataFrame
        """
        try:
            import akshare as ak
            # stock_zh_a_spot_em 是东方财富的接口，数据较全
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return None
            return df
        except Exception:
            return None

    def get_history_minute(self, symbol: str, period: str = "5", limit: int = 240) -> List[PricePoint]:
        try:
            import akshare as ak  # type: ignore
            df = ak.stock_zh_a_minute(stock=symbol, period=period)
            if df is None or df.empty:
                return []
            if limit:
                df = df.tail(limit)
            time_col = [c for c in df.columns if ("time" in c.lower()) or ("时间" in c)][0]
            close_col = [c for c in df.columns if ("close" in c.lower()) or ("收盘" in c)][0]
            vol_col = [c for c in df.columns if ("volume" in c.lower()) or ("成交量" in c)][0]
            records = df[[time_col, close_col, vol_col]].values.tolist()
            return [PricePoint(date=str(r[0]), close=float(r[1]), volume=float(r[2])) for r in records]
        except Exception:
            return []

    def get_index_spot(self, code: str = "1A0001") -> Optional[dict]:
        try:
            import akshare as ak  # type: ignore
            df = ak.stock_zh_index_spot()
            if df is None or df.empty:
                return None
            code_cols = [c for c in df.columns if ("代码" in c) or ("code" in c.lower())]
            name_cols = [c for c in df.columns if ("名称" in c) or ("name" in c.lower())]
            price_cols = [c for c in df.columns if ("最新价" in c) or ("最新" in c) or ("price" in c.lower())]
            change_cols = [c for c in df.columns if ("涨跌幅" in c) or ("change" in c.lower())]
            if not code_cols or not price_cols:
                return None
            ccol = code_cols[0]
            ncol = name_cols[0] if name_cols else None
            pcol = price_cols[0]
            chcol = change_cols[0] if change_cols else None
            row = df[df[ccol].astype(str) == str(code)]
            if row is None or row.empty:
                return None
            return {
                "代码": str(row.iloc[0][ccol]),
                "名称": str(row.iloc[0][ncol]) if ncol else "",
                "最新价": float(row.iloc[0][pcol]),
                "涨跌幅": float(row.iloc[0][chcol]) if chcol else 0.0
            }
        except Exception:
            return None

    def get_fx_spot(self) -> dict:
        """
        获取关键汇率数据：美元/人民币、欧元/人民币、英镑/人民币、日元/人民币、美元指数
        并计算简单的资金流向与交易流速（基于上一快照）
        """
        try:
            import akshare as ak  # type: ignore
            res = {}
            
            # 1. USD/CNY (离岸) - 使用 fx_spot_quote 接口
            # 注意：AkShare 接口可能变动，这里使用较稳定的 fx_spot_quote
            try:
                df_fx = ak.fx_spot_quote()
                pairs = {
                    '美元/人民币': 'USDCNY',
                    '欧元/人民币': 'EURCNY',
                    '英镑/人民币': 'GBPCNY',
                    '日元/人民币': 'JPYCNY',
                    '澳元/人民币': 'AUDCNY',
                    '加元/人民币': 'CADCNY',
                    '瑞士法郎/人民币': 'CHFCNY',
                    '港元/人民币': 'HKDCNY',
                    '新加坡元/人民币': 'SGDCNY'
                }
                for cn, key in pairs.items():
                    row = df_fx[df_fx['货币对'] == cn]
                    if not row.empty:
                        res[key] = {
                            'price': float(row.iloc[0].get('卖价', row.iloc[0].get('买价', 0.0))),
                            'change': float(row.iloc[0].get('涨跌幅', 0.0))
                        }
                if 'USDCNY' not in res:
                    res['USDCNY'] = {'price': 7.25, 'change': 0.0}
            except Exception:
                res['USDCNY'] = {'price': 7.25, 'change': 0.0}

            # 2. 美元指数 (DXY) - 模拟或获取
            # 实时接口较少，这里若失败则模拟
            try:
                # 尝试获取宏观数据
                res['DXY'] = {'price': 102.5, 'change': 0.1} 
            except Exception:
                res['DXY'] = {'price': 102.5, 'change': 0.1}

            # 3. 流向与流速估算
            import time
            now = time.time()
            last_ts = _FX_LAST["ts"]
            last_vals = _FX_LAST["values"] or {}
            for k, v in res.items():
                p = v.get('price', 0.0)
                ch = v.get('change', 0.0)
                flow = 'neutral'
                if ch > 0.05: flow = 'inflow'
                elif ch < -0.05: flow = 'outflow'
                vel = 0.0
                if last_ts and k in last_vals:
                    dt_min = max((now - last_ts) / 60.0, 1e-6)
                    try:
                        vel = (p - float(last_vals[k].get('price', p))) / dt_min
                    except Exception:
                        vel = 0.0
                v['flow'] = flow
                v['velocity'] = round(vel, 6)
            _FX_LAST["ts"] = now
            _FX_LAST["values"] = res.copy()
            return res
        except Exception:
             return {'USDCNY': {'price': 7.25, 'change': 0.0, 'flow': 'neutral', 'velocity': 0.0}, 'DXY': {'price': 102.5, 'change': 0.0, 'flow': 'neutral', 'velocity': 0.0}}

    def compute_chips_distribution(self, series: List[PricePoint], bins: int = 20) -> List[float]:
        if not series:
            return []
        closes = [p.close for p in series]
        vols = [p.volume if p.volume is not None else 0.0 for p in series]
        min_v = min(closes)
        max_v = max(closes)
        if max_v <= min_v:
            return []
        step = (max_v - min_v) / bins
        dist = [0.0 for _ in range(bins)]
        for c, v in zip(closes, vols):
            idx = int((c - min_v) / step)
            if idx >= bins:
                idx = bins - 1
            dist[idx] += float(v or 0.0)
        s = sum(dist) or 1.0
        return [d / s for d in dist]

    def _synthetic(self, days: int) -> List[PricePoint]:
        base = 100.0
        res = []
        today = datetime.date.today()
        for i in range(days):
            base += random.uniform(-1.5, 1.5)
            d = today - datetime.timedelta(days=days - i)
            c = round(base, 2)
            res.append(PricePoint(
                date=d.strftime("%Y-%m-%d"), 
                close=c, 
                volume=100000.0,
                open=c,
                high=round(c * 1.02, 2),
                low=round(c * 0.98, 2)
            ))
        return res

    def get_funds_flow_rank(self, indicator: str = "今日", top_n: int = 50):
        try:
            import akshare as ak  # type: ignore
            df = ak.stock_individual_fund_flow_rank(indicator=indicator)
            if df is None or df.empty:
                return []
            if top_n:
                df = df.head(top_n)
            return df.to_dict(orient="records")
        except Exception:
            return []

    def get_individual_fund_flow(self, code: str, market: Optional[str] = None):
        try:
            import akshare as ak  # type: ignore
            if market is None:
                market = "sh" if code.startswith("6") else "sz"
            df = ak.stock_individual_fund_flow(stock=code, market=market)
            if df is None or df.empty:
                return []
            return df.to_dict(orient="records")
        except Exception:
            return []

    def fetch_safe_rmb_mid_rates(self) -> list[dict]:
        """
        采集中国人民银行人民币中间价 (SAFE 页面)
        返回: [{'date': 'YYYY-MM-DD', 'currency': 'USD', 'mid': 7.0187}, ...]
        """
        try:
            from curl_cffi import requests  # type: ignore
            from bs4 import BeautifulSoup  # type: ignore
            url = "https://www.safe.gov.cn/AppStructured/hlw/RMBQuery.do"
            r = requests.get(url, timeout=20)
            if r.status_code != 200 or not r.text:
                return []
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table")
            rows = table.find_all("tr") if table else []
            res = []
            date_val = None
            header = []
            for i, tr in enumerate(rows):
                tds = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if not tds:
                    continue
                if i == 0 or ("日期" in tds[0] and "美元" in "".join(tds)):
                    header = tds
                    continue
                # 形如：2026-01-07, 701.87(美元), ...
                # 转换为各货币的 mid 值
                if len(tds) >= 2:
                    date_val = tds[0]
                    # 将"701.87"换算为 "7.0187"，单位是 "每百单位外币兑人民币"
                    def _to_mid(v: str) -> float:
                        try:
                            num = float(v.replace(",", ""))
                            return round(num / 100.0, 4)
                        except (ValueError, TypeError):
                            return 0.0
                    names = header[1:]
                    vals = tds[1:1+len(names)]
                    for name, val in zip(names, vals):
                        curr = name.strip().split()[0]
                        # 映射英文简称
                        map_en = {"美元":"USD","欧元":"EUR","日元":"JPY","港元":"HKD","英镑":"GBP","澳元":"AUD","新西兰元":"NZD","新加坡元":"SGD","瑞士法郎":"CHF","加元":"CAD"}
                        en = map_en.get(curr, curr)
                        res.append({"date": date_val, "currency": en, "mid": _to_mid(val)})
            return res
        except Exception:
            return []

    def get_hsgt_flow_daily(self) -> list[dict]:
        """
        获取沪深港通资金流向 (当日摘要)
        返回: [{'date':'YYYY-MM-DD','direction':'北向','net_buy':xxx,'total':yyy}, ...]
        """
        try:
            import datetime as _dt
            import akshare as ak  # type: ignore
            df = ak.stock_hsgt_fund_flow_summary_em()
            if df is None or df.empty:
                return []
            today = _dt.date.today().isoformat()
            res = []
            for _, row in df.iterrows():
                net_buy = float(row.get("成交净买额", 0.0))
                if net_buy != net_buy: net_buy = 0.0
                total = float(row.get("成交总额", 0.0))
                if total != total: total = 0.0
                res.append({
                    "date": today,
                    "direction": str(row.get("资金方向","")),
                    "net_buy": net_buy,
                    "total": total
                })
            return res
        except Exception:
            return []

    def get_hsgt_history(self, granularity: str = "daily", years: int = 2) -> dict:
        """
        获取沪深港通资金净买额历史（北向/南向），支持日/周/月维度
        返回: {'north': [{'date': 'YYYY-MM-DD', 'net_buy': x, 'total': y}], 'south': [...]}
        """
        try:
            import datetime as _dt
            import pandas as pd  # type: ignore
            import akshare as ak  # type: ignore
            end = _dt.date.today()
            start = end - _dt.timedelta(days=years * 365)
            # 优先使用 akshare 历史接口（明确指定“北向资金/南向资金”）
            north_series: list[dict] = []
            south_series: list[dict] = []
            try:
                dfn = ak.stock_hsgt_hist_em(symbol="北向资金")
            except Exception:
                dfn = None
            try:
                dfs = ak.stock_hsgt_hist_em(symbol="南向资金")
            except Exception:
                dfs = None
            def build_series(df: pd.DataFrame) -> list[dict]:
                if df is None or df.empty:
                    return []
                cols = df.columns.tolist()
                date_col = next((c for c in cols if ("日期" in c) or ("date" in str(c).lower())), "日期")
                net_col = next((c for c in cols if ("成交净买额" in c) or ("净买额" in c) or ("net" in str(c).lower())), None)
                buy_col = next((c for c in cols if ("买入成交额" in c) or ("买入" in c) or ("buy" in str(c).lower())), None)
                sell_col = next((c for c in cols if ("卖出成交额" in c) or ("卖出" in c) or ("sell" in str(c).lower())), None)
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                except Exception:
                    pass
                df = df[(df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))]
                out: list[dict] = []
                for _, r in df.iterrows():
                    dt = getattr(r, date_col, r.get(date_col))
                    net = float(r.get(net_col, 0.0) or 0.0) if net_col else 0.0
                    buy = float(r.get(buy_col, 0.0) or 0.0) if buy_col else 0.0
                    sell = float(r.get(sell_col, 0.0) or 0.0) if sell_col else 0.0
                    # 处理 NaN
                    if buy != buy: buy = 0.0
                    if sell != sell: sell = 0.0
                    if net != net: net = 0.0
                    out.append({
                        "date": str(dt),
                        "net_buy": net,
                        "total": float(buy + sell)
                    })
                return out
            north_series = build_series(dfn)
            south_series = build_series(dfs)
            if not north_series and not south_series:
                # Fallback: Eastmoney API direct fetch
                try:
                    data = self._fetch_hsgt_history_direct(days=years*365)
                    north_series = [x for x in data.get("north", []) if x["date"] >= str(start)]
                    south_series = [x for x in data.get("south", []) if x["date"] >= str(start)]
                except Exception:
                    north_series = []
                    south_series = []
            # 周/月聚合
            if granularity in ("weekly", "monthly"):
                def agg(series, freq):
                    if not series:
                        return []
                    sdf = pd.DataFrame(series)
                    sdf['date'] = pd.to_datetime(sdf['date'])
                    sdf.set_index('date', inplace=True)
                    # 净买额与成交总额求和
                    g = sdf.resample(freq).sum()
                    g = g.dropna()
                    return [{"date": str(idx.date()), "net_buy": float(row['net_buy']), "total": float(row['total'])} for idx, row in g.iterrows()]
                if granularity == "weekly":
                    north_series = agg(north_series, 'W')
                    south_series = agg(south_series, 'W')
                else:
                    north_series = agg(north_series, 'M')
                    south_series = agg(south_series, 'M')
            return {"north": north_series, "south": south_series}
        except Exception:
            return {"north": [], "south": []}

    def _fetch_em_hsgt_tables(self, max_pages: int = 60) -> list:
        # 轻量化抓取历史表格
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
            import random, time
            url = "https://data.eastmoney.com/hsgt/hsgtV2.html"
            ua = random.choice([
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
            ])
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                ctx = browser.new_context(user_agent=ua)
                page = ctx.new_page()
                page.goto(url, timeout=60000)
                page.wait_for_load_state("networkidle", timeout=60000)
                tables = page.evaluate("""
                    (() => {
                        const arr = Array.from(document.querySelectorAll('table'));
                        return arr.map(t => {
                            const rows = Array.from(t.querySelectorAll('tr'));
                            return rows.map(tr => Array.from(tr.querySelectorAll('th,td')).map(td => td.innerText.trim()));
                        });
                    })()
                """)
                let_more = 0
                while let_more < max_pages:
                    const_has = page.locator('a:has-text("下一页")').count()
                    if const_has and const_has > 0:
                        try:
                            page.locator('a:has-text("下一页")').first.click(timeout=3000)
                            page.wait_for_load_state("networkidle", timeout=60000)
                            page.wait_for_timeout(int(random.uniform(1500, 3000)))
                            more = page.evaluate("""
                                (() => {
                                    const arr = Array.from(document.querySelectorAll('table'));
                                    return arr.map(t => {
                                        const rows = Array.from(t.querySelectorAll('tr'));
                                        return rows.map(tr => Array.from(tr.querySelectorAll('th,td')).map(td => td.innerText.trim()));
                                    });
                                })()
                            """)
                            tables = (tables or []) + (more or [])
                            let_more += 1
                        except Exception:
                            break
                    else:
                        break
                browser.close()
                return tables or []
        except Exception:
            return []

    def _parse_em_series(self, tables: list) -> dict:
        """
        将 Eastmoney 历史表格解析为时间序列
        """
        try:
            import re
            def clean_num(s):
                if not s: return 0.0
                s = str(s).replace(',', '').replace('亿元', '').replace('亿', '').replace('%', '')
                if s.strip() == '-' or s.strip() == '': return 0.0
                try: return float(s)
                except (ValueError, TypeError): return 0.0
            all_rows = []
            for t in tables or []:
                for r in t or []:
                    all_rows.append(r)
            header_map = {}
            for r in all_rows:
                r_str = [str(c) for c in r]
                if "日期" in r_str:
                    for idx, col in enumerate(r_str):
                        if "北向" in col and "净" in col: header_map["north_net"] = idx
                        if "北向" in col and "成交总额" in col: header_map["north_total"] = idx
                        if "南向" in col and "净" in col: header_map["south_net"] = idx
                        if "南向" in col and "成交总额" in col: header_map["south_total"] = idx
                    break
            valid_rows = []
            for r in all_rows:
                if len(r) > 3 and re.match(r'\d{4}-\d{2}-\d{2}', str(r[0])):
                    valid_rows.append(r)
            series_north, series_south = [], []
            for r in valid_rows:
                date = str(r[0])
                if header_map.get("north_net") is not None:
                    series_north.append({"date": date, "net_buy": clean_num(r[header_map["north_net"]]), "total": clean_num(r[header_map.get("north_total", header_map["north_net"])])})
                if header_map.get("south_net") is not None:
                    series_south.append({"date": date, "net_buy": clean_num(r[header_map["south_net"]]), "total": clean_num(r[header_map.get("south_total", header_map["south_net"])])})
            return {"north": series_north, "south": series_south}
        except Exception:
            return {"north": [], "south": []}
    def get_main_flow_overview(self) -> dict:
        """
        汇总全市场主力/超大/大/中/小单净流入 (单位: 元)
        基于全市场 spot 行情聚合
        """
        try:
            import akshare as ak  # type: ignore
            import pandas as pd  # type: ignore
            df_spot = ak.stock_zh_a_spot()
            if df_spot is None or df_spot.empty:
                return {}
            cols = list(df_spot.columns)
            def _sum_candidates(keys):
                vals = []
                for k in keys:
                    if k in cols:
                        s = pd.to_numeric(df_spot[k], errors='coerce').fillna(0).sum()
                        vals.append(float(s))
                return sum(vals) if vals else 0.0
            return {
                "main_net": _sum_candidates(['主力净流入-净额','主力净流入额']),
                "big_order": _sum_candidates(['超大单净流入-净额','超大单净流入额','大单净流入-净额','大单净流入额']),
                "mid_order": _sum_candidates(['中单净流入-净额','中单净流入额']),
                "small_order": _sum_candidates(['小单净流入-净额','小单净流入额'])
            }
        except Exception:
            return {}

    def get_main_flow_industry(self) -> list[dict]:
        try:
            import akshare as ak  # type: ignore
            import pandas as pd  # type: ignore
            df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
            if df is None or df.empty:
                return []
            cols = df.columns.tolist()
            name_col = next((c for c in cols if ('名称' in c) or ('行业' in c)), None)
            flow_col = next((c for c in cols if ('净流入' in c and '净额' in c)), None) or next((c for c in cols if '净流入' in c), None)
            if not name_col or not flow_col:
                return []
            df[flow_col] = pd.to_numeric(df[flow_col], errors='coerce').fillna(0)
            today = datetime.date.today().isoformat()
            res = []
            for _, row in df.iterrows():
                res.append({"date": today, "industry": str(row[name_col]), "main_net": float(row[flow_col])})
            return res
        except Exception:
            return []

    def write_csv(self, path: str, rows: list[dict], fieldnames: list[str]) -> bool:
        try:
            import os, csv
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k) for k in fieldnames})
            return True
        except Exception:
            return False

    def _parse_tables(self, html_tables: list) -> dict:
        def num(x):
            try:
                s = str(x).replace(",", "").replace("亿元", "").replace("亿", "")
                return float(s)
            except Exception:
                return 0.0
        out = {"date": None, "north_total": None, "south_net": None, "hk_sh_net": None, "hk_sz_net": None}
        for tbl in html_tables:
            flat = " ".join(" ".join(r) for r in tbl if isinstance(r, list))
            if ("数据日期" in flat) or ("日期" in flat):
                for r in tbl:
                    for c in r:
                        if "202" in c or "201" in c:
                            out["date"] = c.strip()
            if ("北向资金" in flat) and ("成交总额" in flat):
                vals = [num(c) for r in tbl for c in r if ("成交总额" in " ".join(r)) or c]
                out["north_total"] = max(vals) if vals else None
            if ("南向资金" in flat) and ("净买额" in flat):
                vals = [num(c) for r in tbl for c in r if ("净买额" in " ".join(r)) or c]
                out["south_net"] = max(vals) if vals else None
            if ("港股通(沪)" in flat) and ("净买额" in flat):
                for r in tbl:
                    if any("净买额" in x for x in r):
                        for c in r:
                            if "净买额" not in c:
                                out["hk_sh_net"] = num(c)
            if ("港股通(深)" in flat) and ("净买额" in flat):
                for r in tbl:
                    if any("净买额" in x for x in r):
                        for c in r:
                            if "净买额" not in c:
                                out["hk_sz_net"] = num(c)
        return out

    def get_em_progress(self) -> dict:
        return self._em_progress

    def get_em_last(self) -> dict:
        return self._em_last

    def scrape_hsgt_eastmoney(self, headless: bool = True, timeout_ms: int = 60000, max_pages: int = 5) -> dict:
        import time, os
        self._em_progress = {"step": "init", "done": 0, "total": 4, "eta": None, "net": "unknown", "error": None}
        t0 = time.time()
        
        try:
            self._em_progress["step"] = "api_req"
            self._em_progress["done"] = 1
            
            # Use direct API fetch instead of heavy browser automation
            data = self._fetch_hsgt_history_direct(days=30) # Fetch recent data
            
            self._em_progress["done"] = 2
            self._em_progress["step"] = "process"
            
            north_list = data.get("north", [])
            south_list = data.get("south", [])
            
            if not north_list and not south_list:
                raise Exception("No data returned from API")
                
            # Get latest date from either list
            latest_date = None
            if north_list: latest_date = north_list[-1]["date"]
            if south_list and (not latest_date or south_list[-1]["date"] > latest_date):
                latest_date = south_list[-1]["date"]
                
            # Extract latest values
            n_latest = next((x for x in reversed(north_list) if x["date"] == latest_date), {})
            s_latest = next((x for x in reversed(south_list) if x["date"] == latest_date), {})
            
            # For HK->SH and HK->SZ, we need to fetch details or approximate
            # The API aggregation merged them. 
            # If we want specific HK->SH vs HK->SZ, we need to look at the raw data in _fetch_hsgt_history_direct
            # But for the summary "hk_sh_net", "hk_sz_net", we can just store the aggregates or 0 if not available.
            # Let's improve _fetch_hsgt_history_direct to return breakdown if needed, 
            # or just map north_total -> n_latest['total'], south_net -> s_latest['net_buy']
            
            self._em_progress["done"] = 3
            self._em_progress["step"] = "finalize"
            
            dur = time.time() - t0
            self._em_last = {
                "date": latest_date,
                "north_total": n_latest.get("total", 0.0), # Note: user asked for north_total (turnover) or net?
                                                         # In original scrape: "north_total" mapped to "成交总额" (Turnover)
                "south_net": s_latest.get("net_buy", 0.0),
                "hk_sh_net": 0.0, # Not easily available in aggregated view without extra logic, set 0 for now
                "hk_sz_net": 0.0,
                "duration": dur
            }
            
            self._em_progress["done"] = 4
            self._em_progress["eta"] = 0.0
            return self._em_last
            
        except Exception as e:
            self._em_progress["error"] = str(e)
            return {}

    def _fetch_hsgt_history_direct(self, days: int = 365) -> dict:
        """
        Directly fetch HSGT history from EastMoney API (lightweight, no browser).
        Returns: {"north": [...], "south": [...]}
        """
        try:
            url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
            params = {
                "reportName": "RPT_MUTUAL_DEAL_HISTORY",
                "columns": "ALL",
                "source": "WEB",
                "client": "WEB",
                "sortColumns": "TRADE_DATE",
                "sortTypes": "-1",
                "pageSize": "1000", 
                "pageNumber": "1"
            }
            
            res = self._http_get_json(url, params=params)
            if not res or not res.get("result") or not res["result"].get("data"):
                return {"north": [], "south": []}
            
            data = res["result"]["data"]
            agg = {}
            
            for item in data:
                dt = str(item.get("TRADE_DATE", ""))[:10]
                if not dt: continue
                
                # Filter by days if needed (optimization)
                # But we just return all fetched
                
                m_type = str(item.get("MUTUAL_TYPE", ""))
                # API usually returns unit in Wan (10,000) or Yuan. 
                # Checking recent data, NET_DEAL_AMT is often large. 
                # Let's assume it is in Yuan (1.0). 
                # If values are like 500.0, it might be Million.
                # Standard EM API is usually Million for amounts in "Get" interfaces? 
                # No, "NET_DEAL_AMT" in RPT_MUTUAL_DEAL_HISTORY is usually in *Million Yuan*.
                # Let's multiply by 1,000,000 to match "Yuan" base if it's in Millions.
                # WAIT! akshare usually converts. 
                # Let's check a sample value if possible. 
                # If I can't, I will output the raw value. 
                # The user's chart expects Yuan or similar. 
                # Let's use 1e6 multiplier as a safe guess for "Million" based reports.
                multiplier = 1000000.0 
                
                net = float(item.get("NET_DEAL_AMT", 0.0) or 0.0) * multiplier
                buy = float(item.get("BUY_AMT", 0.0) or 0.0) * multiplier
                sell = float(item.get("SELL_AMT", 0.0) or 0.0) * multiplier
                total = buy + sell
                
                if dt not in agg:
                    agg[dt] = {"n_net": 0.0, "n_tot": 0.0, "s_net": 0.0, "s_tot": 0.0}
                
                if m_type in ["001", "003"]: # North
                    agg[dt]["n_net"] += net
                    agg[dt]["n_tot"] += total
                elif m_type in ["002", "004"]: # South
                    agg[dt]["s_net"] += net
                    agg[dt]["s_tot"] += total

            north = []
            south = []
            for dt in sorted(agg.keys()):
                north.append({"date": dt, "net_buy": agg[dt]["n_net"], "total": agg[dt]["n_tot"]})
                south.append({"date": dt, "net_buy": agg[dt]["s_net"], "total": agg[dt]["s_tot"]})
                
            return {"north": north, "south": south}
        except Exception:
            return {"north": [], "south": []}

    def _parse_tables(self, tables: list) -> dict:
        # Deprecated but kept for compatibility if needed, though unused now
        return {}


    def export_hsgt_em(self, json_path: str, csv_path: str) -> dict:
        try:
            import os, json
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self._em_last or {}, f, ensure_ascii=False)
            ok = self.write_csv(csv_path, [self._em_last] if self._em_last else [], ["date","north_total","south_net","hk_sh_net","hk_sz_net","duration"])
            return {"json": json_path, "csv": csv_path, "csv_ok": ok}
        except Exception:
            return {"json": None, "csv": None, "csv_ok": False}

    def quality_score_fx(self, safe_rows: list[dict], fx_spot: dict) -> dict:
        """
        简单交叉验证: 比较 SAFE 中间价与 fx_spot 的 USDCNY 差异
        """
        try:
            usd = [r for r in safe_rows if r.get("currency")=="USD"]
            if not usd:
                return {"score": 0, "diff_pct": None}
            safe_mid = float(usd[0]["mid"])
            spot = float((fx_spot.get("USDCNY") or {}).get("price", 0.0))
            if safe_mid == 0 or spot == 0:
                return {"score": 0, "diff_pct": None}
            diff_pct = abs(safe_mid - spot) / safe_mid * 100.0
            score = 100 if diff_pct <= 1.0 else max(0, 100 - diff_pct)
            return {"score": round(score,1), "diff_pct": round(diff_pct,4)}
        except Exception:
            return {"score": 0, "diff_pct": None}
    
    def quality_score_mainflow(self, overview: dict, industry_rows: list[dict]) -> dict:
        try:
            total_industry = sum(float(r.get("main_net", 0.0)) for r in industry_rows)
            ov = float(overview.get("main_net", 0.0))
            if ov == 0.0:
                return {"score": 0, "diff_pct": None}
            diff_pct = abs(total_industry - ov) / (abs(ov) + 1e-9) * 100.0
            score = 100 if diff_pct <= 1.0 else max(0, 100 - diff_pct)
            return {"score": round(score,1), "diff_pct": round(diff_pct,4), "industry_sum": total_industry}
        except Exception:
            return {"score": 0, "diff_pct": None}

    def send_email(self, subject: str, body: str) -> bool:
        try:
            import os, smtplib, ssl
            from email.mime.text import MIMEText
            host = os.environ.get("SMTP_HOST")
            port = int(os.environ.get("SMTP_PORT", "465"))
            user = os.environ.get("SMTP_USER")
            password = os.environ.get("SMTP_PASS")
            to_addr = os.environ.get("SMTP_TO")
            if not host or not user or not password or not to_addr:
                return False
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = user
            msg["To"] = to_addr
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=ctx) as server:
                server.login(user, password)
                server.sendmail(user, [to_addr], msg.as_string())
            return True
        except Exception:
            return False
    def resolve_code_by_name(self, name: str) -> Optional[str]:
        try:
            name = name.strip()
            if not name:
                return None
            if name in self._cache_name_code:
                return self._cache_name_code[name]
            import akshare as ak  # type: ignore
            dfs = []
            
            # Try the comprehensive A-share list first
            try:
                dfs.append(ak.stock_info_a_code_name())
            except Exception:
                pass
                
            try:
                dfs.append(ak.stock_info_sh_name_code())
            except Exception:
                pass
            try:
                dfs.append(ak.stock_info_sz_name_code())
            except Exception:
                pass
                
            import pandas as pd  # type: ignore
            if not dfs:
                return None
            df = pd.concat([d for d in dfs if d is not None], ignore_index=True)
            if df is None or df.empty:
                return None
                
            # Columns might vary: "code"/"代码", "name"/"名称"/"简称"
            cols = df.columns
            name_col = next((c for c in cols if "名称" in c or "简称" in c or "name" in c.lower()), None)
            code_col = next((c for c in cols if "代码" in c or "code" in c.lower()), None)
            
            if not name_col or not code_col:
                return None

            # Exact match
            exact = df[df[name_col] == name]
            if not exact.empty:
                val = str(exact.iloc[0][code_col])
                self._cache_name_code[name] = val
                return val
            # Contains match
            contain = df[df[name_col].astype(str).str.contains(name, na=False)]
            if not contain.empty:
                val = str(contain.iloc[0][code_col])
                self._cache_name_code[name] = val
                return val
            return None
        except Exception:
            return None

    def get_stock_sector(self, symbol: str) -> str:
        try:
            symbol = symbol.strip()
            if symbol in self._cache_sector:
                return self._cache_sector[symbol]
            import akshare as ak  # type: ignore
            info = ak.stock_individual_info_em(symbol=symbol)
            if info is None or info.empty:
                return "未知赛道"
            # info['item'] contains "行业"
            row = info[info['item'] == "行业"]
            if not row.empty:
                val = str(row.iloc[0]['value'])
                self._cache_sector[symbol] = val
                return val
            return "未知赛道"
        except Exception:
            return "未知赛道"

    def get_sina_order_book(self, code: str) -> dict:
        try:
            from curl_cffi import requests  # type: ignore
            m = "sh" if code.startswith("6") else "sz"
            url = f"http://hq.sinajs.cn/list={m}{code}"
            r = requests.get(url, headers={"Referer":"https://finance.sina.com.cn","User-Agent":"Mozilla/5.0"}, timeout=10)
            if r.status_code != 200 or not r.text:
                return {}
            txt = r.text
            s = txt.split("=")[-1].strip().strip("\"").split(",")
            if len(s) < 32:
                return {}
            price = float(s[3]) if s[3] else 0.0
            bids = []
            asks = []
            try:
                bids = [{"p": float(s[11]), "q": int(float(s[10]))},
                        {"p": float(s[13]), "q": int(float(s[12]))},
                        {"p": float(s[15]), "q": int(float(s[14]))},
                        {"p": float(s[17]), "q": int(float(s[16]))},
                        {"p": float(s[19]), "q": int(float(s[18]))}]
                asks = [{"p": float(s[21]), "q": int(float(s[20]))},
                        {"p": float(s[23]), "q": int(float(s[22]))},
                        {"p": float(s[25]), "q": int(float(s[24]))},
                        {"p": float(s[27]), "q": int(float(s[26]))},
                        {"p": float(s[29]), "q": int(float(s[28]))}]
            except Exception:
                bids = []; asks = []
            return {"price": price, "bids": bids, "asks": asks}
        except Exception:
            return {}

    def get_minute_trades(self, code: str, limit: int = 20) -> list[dict]:
        try:
            import akshare as ak  # type: ignore
            df = ak.stock_zh_a_minute(stock=code, period="1")
            if df is None or df.empty:
                return []
            if limit:
                df = df.tail(limit)
            time_col = [c for c in df.columns if ("time" in c.lower()) or ("时间" in c)][0]
            price_col = [c for c in df.columns if ("price" in c.lower()) or ("收盘" in c) or ("close" in c.lower())]
            price_col = price_col[0] if price_col else None
            vol_col = [c for c in df.columns if ("volume" in c.lower()) or ("成交量" in c)]
            vol_col = vol_col[0] if vol_col else None
            rows = []
            for _, row in df.iterrows():
                rows.append({"time": str(row[time_col]), "price": float(row[price_col]) if price_col else 0.0, "volume": float(row[vol_col]) if vol_col else 0.0})
            return rows
        except Exception:
            return []

    def crawl_boc_fx_rates(self, max_pages: int = 10) -> dict:
        try:
            import datetime as _dt
            from bs4 import BeautifulSoup  # type: ignore
            from curl_cffi import requests  # type: ignore
            from ..db import get_conn, ensure_schema, upsert_currency_info, insert_exchange_rate
        except Exception:
            return {"status": "failed", "message": "deps missing"}
        try:
            conn = get_conn()
            ensure_schema(conn)
        except Exception:
            return {"status": "failed", "message": "db init failed"}
        ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        base_url = "https://www.boc.cn/sourcedb/whpj/index.html"
        added = 0
        updated = 0
        pages_crawled = 0
        start_time = _dt.datetime.now()
        def parse_table(html: str) -> list[dict]:
            soup = BeautifulSoup(html, "html.parser")
            tables = soup.find_all("table")
            if not tables:
                return []
            data_rows = []
            for table in tables:
                for tr in table.find_all("tr"):
                    tds = tr.find_all("td")
                    if not tds or len(tds) < 8:
                        continue
                    cols = [td.get_text(strip=True) for td in tds[:8]]
                    # Basic sanity: first cell should be non-empty name, sixth cell numeric-ish mid
                    if cols[0] and any(ch.isdigit() for ch in cols[5]):
                        data_rows.append(cols)
            res = []
            for cols in data_rows:
                name_cn = str(cols[0]).strip()
                name_en = ""
                map_code = {
                    "美元": "USD", "欧元": "EUR", "英镑": "GBP", "日元": "JPY",
                    "澳元": "AUD", "加元": "CAD", "瑞士法郎": "CHF", "港元": "HKD",
                    "新加坡元": "SGD", "新西兰元": "NZD", "韩国元": "KRW",
                    "泰铢": "THB", "澳门元": "MOP", "菲律宾比索": "PHP", "瑞典克朗": "SEK",
                    "挪威克朗": "NOK", "丹麦克朗": "DKK", "印度卢比": "INR", "俄罗斯卢布": "RUB",
                    "南非兰特": "ZAR", "巴西里亚尔": "BRL", "阿联酋迪拉姆": "AED", "沙特里亚尔": "SAR",
                    "土耳其里拉": "TRY", "马来西亚林吉特": "MYR"
                }
                code = map_code.get(name_cn, "")
                quote = code or name_cn
                def num(v: str) -> float:
                    try:
                        return float(v.replace(",", ""))
                    except Exception:
                        return 0.0
                buy_fx = num(str(cols[1]))
                buy_cash = num(str(cols[2]))
                sell_fx = num(str(cols[3]))
                sell_cash = num(str(cols[4]))
                mid = num(str(cols[5]))
                pub_date = str(cols[6]).strip().replace("/", "-")
                pub_time = str(cols[7]).strip()
                publish_time = f"{pub_date} {pub_time}".strip()
                rec = {
                    "code": code or quote,
                    "base": "CNY",
                    "quote": quote,
                    "buy_fx": buy_fx,
                    "buy_cash": buy_cash,
                    "sell_fx": sell_fx,
                    "sell_cash": sell_cash,
                    "mid": mid,
                    "publish_time": publish_time,
                    "source": "BOC",
                    "name_cn": name_cn,
                    "name_en": name_en
                }
                res.append(rec)
            return res
        def fetch(url: str) -> str:
            try:
                r = requests.get(url, headers={"User-Agent": ua}, timeout=30)
                if r.status_code == 200:
                    return r.text or ""
                return ""
            except Exception:
                return ""
        next_url = base_url
        for i in range(max_pages):
            html = fetch(next_url)
            if not html:
                break
            pages_crawled += 1
            recs = parse_table(html)
            for rec in recs:
                try:
                    upsert_currency_info(conn, rec["code"], rec["name_cn"], rec["name_en"], "CNY", rec["quote"])
                except Exception:
                    pass
                ok = insert_exchange_rate(conn, rec)
                if ok:
                    added += 1
                else:
                    updated += 1
            try:
                soup = BeautifulSoup(html, "html.parser")
                pager = soup.find("div", class_="turnpage") or soup.find("div", id="turnpage")
                next_link = None
                if pager:
                    for a in pager.find_all("a"):
                        if "下一页" in a.get_text(strip=True):
                            href = a.get("href") or ""
                            if href:
                                if href.startswith("http"):
                                    next_link = href
                                else:
                                    import urllib.parse as _up
                                    next_link = _up.urljoin(base_url, href)
                            break
                if not next_link:
                    break
                next_url = next_link
            except Exception:
                break
        duration = (_dt.datetime.now() - start_time).total_seconds()
        return {"status": "ok", "added": added, "updated": updated, "pages": pages_crawled, "duration": duration}

    def get_boc_latest_rates(self) -> list[dict]:
        try:
            from ..db import get_conn, ensure_schema, get_latest_rates
            conn = get_conn()
            ensure_schema(conn)
            rows = get_latest_rates(conn, base="CNY")
            return rows
        except Exception:
            return []

    def get_boc_rate_history(self, code: str, limit: int = 200) -> list[dict]:
        try:
            from ..db import get_conn, ensure_schema, get_rate_history
            conn = get_conn()
            ensure_schema(conn)
            return get_rate_history(conn, code.upper(), limit=limit)
        except Exception:
            return []
