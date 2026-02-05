import abc
import ctypes
import os
import platform
import logging
from typing import Dict, Any, List, Optional
import random
import datetime
from ctypes import byref, POINTER, c_char_p, c_int, cast, c_void_p
import pandas as pd

logger = logging.getLogger(__name__)

# Import binding
from app.quant.emquant_binding import (
    EQLOGININFO, EQDATA, EQVARIENT, EQErr, LogCallback,
    EQERR_SUCCESS, get_eqvarient_value
)

class DataProvider(abc.ABC):
    @abc.abstractmethod
    def get_history(self, code: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        pass

    @abc.abstractmethod
    def get_snapshot(self, code: str) -> Dict:
        pass

    @abc.abstractmethod
    def subscribe(self, codes: List[str]):
        pass

class MockDataProvider(DataProvider):
    def get_history(self, code: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        # Simulate data
        data = []
        base_price = 100.0
        try:
            if start_date:
                current_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            else:
                current_date = datetime.datetime.now() - datetime.timedelta(days=365)
                
            if end_date:
                end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end = datetime.datetime.now()
        except:
            current_date = datetime.datetime.now() - datetime.timedelta(days=30)
            end = datetime.datetime.now()
        
        while current_date <= end:
            if current_date.weekday() < 5: # Weekdays only
                change = random.uniform(-0.05, 0.05)
                base_price *= (1 + change)
                data.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "open": base_price * (1 - random.uniform(0, 0.02)),
                    "high": base_price * (1 + random.uniform(0, 0.02)),
                    "low": base_price * (1 - random.uniform(0, 0.02)),
                    "close": base_price,
                    "volume": int(random.uniform(1000, 100000))
                })
            current_date += datetime.timedelta(days=1)
        return data

    def get_snapshot(self, code: str) -> Dict:
        price = random.uniform(10, 100)
        return {
            "code": code,
            "price": price,
            "open": price * 0.99,
            "high": price * 1.02,
            "low": price * 0.98,
            "volume": 100000,
            "ask1": price + 0.01,
            "bid1": price - 0.01
        }

    def subscribe(self, codes: List[str]):
        logger.debug(f"Mock subscribed to {codes}")

class EMQuantDataProvider(DataProvider):
    def __init__(self, dll_path: str = None):
        self.lib = None
        self.connected = False
        
        if not dll_path:
            # Default path based on user's env
            dll_path = "/Users/mac/Downloads/EMQuantAPI_CPP_Mac/x64/bin/libEMQuantAPIx64.dylib"
        
        if os.path.exists(dll_path):
            try:
                self.lib = ctypes.CDLL(dll_path)
                logger.info(f"Loaded EMQuant library from {dll_path}")
                self._setup_functions()
                self._try_login()
            except Exception as e:
                logger.warning(f"Failed to load EMQuant lib: {e}")
                self.lib = None
        else:
            logger.info(f"EMQuant lib not found at {dll_path}")

    def _setup_functions(self):
        # Setup function signatures
        
        # EQErr start(EQLOGININFO* pLoginInfo, const char* options, logcallback pfnCallback);
        self.lib.start.argtypes = [POINTER(EQLOGININFO), c_char_p, LogCallback]
        self.lib.start.restype = c_int # EQErr

        # EQErr stop();
        self.lib.stop.argtypes = []
        self.lib.stop.restype = c_int

        # EQErr csd(const char* codes, const char* indicators, const char* startDate, const char* endDate, const char* options, EQDATA*& pEQData);
        self.lib.csd.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, POINTER(POINTER(EQDATA))]
        self.lib.csd.restype = c_int

        # EQErr css(const char* codes, const char* indicators, const char* options, EQDATA*& pEQData);
        self.lib.css.argtypes = [c_char_p, c_char_p, c_char_p, POINTER(POINTER(EQDATA))]
        self.lib.css.restype = c_int

        # EQErr releasedata(void* pEQData);
        self.lib.releasedata.argtypes = [c_void_p]
        self.lib.releasedata.restype = c_int

    def _log_callback(self, msg):
        # print(f"[EMQuant Log] {msg.decode('utf-8')}")
        return 0

    def _try_login(self):
        # Create callback
        self.cb_func = LogCallback(self._log_callback)
        
        # Login info (can be empty for auto-login if configured)
        login_info = EQLOGININFO()
        
        # Try to login
        ret = self.lib.start(byref(login_info), b"ForceLogin=1", self.cb_func)
        if ret == EQERR_SUCCESS:
            logger.info("EMQuant login successful")
            self.connected = True
        else:
            logger.warning(f"EMQuant login failed with error code: {ret}")

    def get_history(self, code: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        if not self.connected:
            return []
        
        # Defaults
        if not start_date: start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date: end_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Standardize code format (e.g., 000001.SZ)
        # Assuming input is already formatted or we pass it directly
        c_codes = code.encode('utf-8')
        c_indicators = b"open,close,high,low,volume"
        c_start = start_date.replace("-", "").encode('utf-8')
        c_end = end_date.replace("-", "").encode('utf-8')
        c_options = b"Period=1,Adjustflag=1" # Daily, Forward Adjusted

        p_eq_data = POINTER(EQDATA)()
        
        ret = self.lib.csd(c_codes, c_indicators, c_start, c_end, c_options, byref(p_eq_data))
        
        result = []
        if ret == EQERR_SUCCESS and p_eq_data:
            data = p_eq_data.contents
            # Parse data
            # Structure: [Date][Code][Indicator]
            # Since we ask for 1 code, it's just [Date][0][Indicator]
            
            n_dates = data.dateArray.nSize
            n_codes = data.codeArray.nSize # Should be 1
            n_indicators = data.indicatorArray.nSize # Should be 5
            
            # Access helper
            # index = n_codes * n_indicators * date_idx + n_indicators * code_idx + ind_idx
            
            # Get pointers to arrays for faster access
            dates_ptr = data.dateArray.pChArray
            values_ptr = data.valueArray.pEQVarient
            
            for d in range(n_dates):
                date_str = dates_ptr[d].pChar.decode('utf-8')
                # Reformat date to YYYY-MM-DD
                if len(date_str) == 8:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                
                # Base index for this date and code 0
                base_idx = n_codes * n_indicators * d + n_indicators * 0
                
                # open, close, high, low, volume
                try:
                    open_val = get_eqvarient_value(values_ptr[base_idx + 0])
                    close_val = get_eqvarient_value(values_ptr[base_idx + 1])
                    high_val = get_eqvarient_value(values_ptr[base_idx + 2])
                    low_val = get_eqvarient_value(values_ptr[base_idx + 3])
                    vol_val = get_eqvarient_value(values_ptr[base_idx + 4])
                    
                    item = {
                        "date": date_str,
                        "open": open_val,
                        "close": close_val,
                        "high": high_val,
                        "low": low_val,
                        "volume": vol_val
                    }
                    result.append(item)
                except Exception as e:
                    logger.error(f"Error parsing row {d}: {e}")
            
            # Release data
            self.lib.releasedata(cast(p_eq_data, c_void_p))
            
        else:
            logger.warning(f"csd failed: {ret}")
            
        return result

    def get_snapshot(self, code: str) -> Dict:
        if not self.connected:
            return {}
            
        c_codes = code.encode('utf-8')
        c_indicators = b"open,now,high,low,volume,bid1,ask1" # now is close/current
        c_options = b""
        
        p_eq_data = POINTER(EQDATA)()
        ret = self.lib.css(c_codes, c_indicators, c_options, byref(p_eq_data))
        
        res_dict = {}
        if ret == EQERR_SUCCESS and p_eq_data:
            data = p_eq_data.contents
            if data.codeArray.nSize > 0:
                values_ptr = data.valueArray.pEQVarient
                # 1 code, 1 date (usually), N indicators
                # css result structure is usually similar, but date dimension might be 1
                
                try:
                    res_dict["code"] = code
                    res_dict["open"] = get_eqvarient_value(values_ptr[0])
                    res_dict["price"] = get_eqvarient_value(values_ptr[1]) # now
                    res_dict["high"] = get_eqvarient_value(values_ptr[2])
                    res_dict["low"] = get_eqvarient_value(values_ptr[3])
                    res_dict["volume"] = get_eqvarient_value(values_ptr[4])
                    res_dict["bid1"] = get_eqvarient_value(values_ptr[5])
                    res_dict["ask1"] = get_eqvarient_value(values_ptr[6])
                except Exception as e:
                    logger.error(f"Error parsing snapshot: {e}")
            
            self.lib.releasedata(cast(p_eq_data, c_void_p))
            
        return res_dict

    def subscribe(self, codes: List[str]):
        # EMQuant uses csq for subscription usually, or just polling css
        pass

class AkShareDataProvider(DataProvider):
    def __init__(self):
        try:
            # Fix Proxy issues in some environments
            os.environ['no_proxy'] = '*'
            import akshare as ak
            self.ak = ak
            logger.info("Initialized AkShareDataProvider")
        except ImportError:
            logger.warning("AkShare not installed")
            self.ak = None

    def get_history(self, code: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        if not self.ak:
            return []
        
        try:
            # AkShare expects YYYYMMDD
            if start_date:
                s_date = start_date.replace("-", "")
            else:
                s_date = "20200101" # Default start
                
            if end_date:
                e_date = end_date.replace("-", "")
            else:
                e_date = datetime.datetime.now().strftime("%Y%m%d")
            
            clean_code = ''.join(filter(str.isdigit, code))
            
            df = self.ak.stock_zh_a_hist(symbol=clean_code, start_date=s_date, end_date=e_date, adjust="qfq")
            
            if df is None or df.empty:
                return []
            
            result = []
            for _, row in df.iterrows():
                result.append({
                    "date": str(row["日期"]),
                    "open": float(row["开盘"]),
                    "close": float(row["收盘"]),
                    "high": float(row["最高"]),
                    "low": float(row["最低"]),
                    "volume": float(row["成交量"])
                })
            return result
        except Exception as e:
            logger.warning(f"AkShare get_history failed: {e}")
            return []

    def get_snapshot(self, code: str) -> Dict:
        if not self.ak:
            return {}
        
        clean_code = ''.join(filter(str.isdigit, code))
        
        # 1. Try Spot API
        try:
            df = self.ak.stock_zh_a_spot_em()
            row = df[df['代码'] == clean_code]
            if not row.empty:
                r = row.iloc[0]
                price = float(r['最新价'])
                return {
                    "code": code,
                    "price": price,
                    "open": float(r['今开']),
                    "high": float(r['最高']),
                    "low": float(r['最低']),
                    "volume": float(r['成交量']),
                    "bid1": price,
                    "ask1": price
                }
        except Exception as e:
            # print(f"AkShare spot failed, trying fallback: {e}")
            pass

        # 2. Fallback to History (Last Close)
        try:
            import datetime
            end = datetime.datetime.now().strftime("%Y%m%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y%m%d")
            df = self.ak.stock_zh_a_hist(symbol=clean_code, start_date=start, end_date=end, adjust="qfq")
            
            if df is not None and not df.empty:
                r = df.iloc[-1]
                price = float(r['收盘'])
                return {
                    "code": code,
                    "price": price,
                    "open": float(r['开盘']),
                    "high": float(r['最高']),
                    "low": float(r['最低']),
                    "volume": float(r['成交量']),
                    "bid1": price,
                    "ask1": price
                }
        except Exception as e:
            logger.warning(f"AkShare snapshot fallback failed: {e}")
            
        return {}

    def subscribe(self, codes: List[str]):
        pass

# Factory
def get_data_provider(use_mock=False) -> DataProvider:
    if use_mock:
        return MockDataProvider()
    
    # 1. Try EMQuant (Priority if connected)
    provider = EMQuantDataProvider()
    if provider.connected:
        return provider
    
    # 2. Try AkShare (Free Real Data)
    ak_provider = AkShareDataProvider()
    if ak_provider.ak:
        return ak_provider

    logger.info("Fallback to MockDataProvider")
    return MockDataProvider()
