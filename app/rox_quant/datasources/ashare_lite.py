#-*- coding:utf-8 -*-
# A-share Data Fetcher (Based on Ashare: https://github.com/mpquant/Ashare)
import json
import requests
import datetime
import pandas as pd

# --- Global Config ---
TIMEOUT = 10

# --- Tencent Day K-Line ---
def get_price_day_tx(code, end_date='', count=10, frequency='1d'):
    """Fetch daily K-line from Tencent."""
    unit = 'week' if frequency in '1w' else 'month' if frequency in '1M' else 'day'
    
    if end_date:
        end_date = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime.date) else end_date.split(' ')[0]
    
    # If end_date is today, make it empty to fetch latest
    if end_date == datetime.datetime.now().strftime('%Y-%m-%d'):
        end_date = ''
        
    url = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},{unit},,{end_date},{count},qfq'
    
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        st = json.loads(resp.content)
        ms = 'qfq' + unit
        stk = st['data'][code]
        
        # Index returns 'day' not 'qfqday'
        buf = stk[ms] if ms in stk else stk[unit]
        
        df = pd.DataFrame(buf, columns=['time', 'open', 'close', 'high', 'low', 'volume'], dtype='float')
        df.time = pd.to_datetime(df.time)
        df.set_index(['time'], inplace=True)
        df.index.name = ''
        return df
    except Exception as e:
        print(f"Error fetching day K-line (TX) for {code}: {e}")
        return pd.DataFrame()

# --- Tencent Minute K-Line ---
def get_price_min_tx(code, end_date=None, count=10, frequency='1d'):
    """Fetch minute K-line from Tencent."""
    # Parse frequency: 1m, 5m, 15m, 30m, 60m
    ts = int(frequency[:-1]) if frequency[:-1].isdigit() else 1
    
    if end_date:
        end_date = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime.date) else end_date.split(' ')[0]
        
    url = f'http://ifzq.gtimg.cn/appstock/app/kline/mkline?param={code},m{ts},,{count}'
    
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        st = json.loads(resp.content)
        buf = st['data'][code]['m'+str(ts)]
        
        df = pd.DataFrame(buf, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'n1', 'n2'])
        df = df[['time', 'open', 'close', 'high', 'low', 'volume']]
        
        # Convert types
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = df[col].astype('float')
            
        df.time = pd.to_datetime(df.time)
        df.set_index(['time'], inplace=True)
        df.index.name = ''
        
        # Update latest close with real-time data if available
        try:
            latest_price = float(st['data'][code]['qt'][code][3])
            df.iloc[-1, df.columns.get_loc('close')] = latest_price
        except:
            pass
            
        return df
    except Exception as e:
        print(f"Error fetching min K-line (TX) for {code}: {e}")
        return pd.DataFrame()

# --- Sina K-Line (Support minutes) ---
def get_price_sina(code, end_date='', count=10, frequency='60m'):
    """Fetch K-line from Sina (supports 5m, 15m, 30m, 60m)."""
    # Map frequency to minutes
    freq_map = {'1d': '240m', '1w': '1200m', '1M': '7200m'}
    mapped_freq = freq_map.get(frequency, frequency)
    
    mcount = count
    ts = int(mapped_freq[:-1]) if mapped_freq[:-1].isdigit() else 1
    
    if end_date != '' and mapped_freq in ['240m', '1200m', '7200m']:
        end_date = pd.to_datetime(end_date) if not isinstance(end_date, datetime.date) else end_date
        
        unit = 4 if mapped_freq == '1200m' else 29 if mapped_freq == '7200m' else 1
        days_diff = (datetime.datetime.now() - end_date).days
        count = count + days_diff // unit
        
    url = f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale={ts}&ma=5&datalen={count}'
    
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        dstr = json.loads(resp.content)
        
        df = pd.DataFrame(dstr, columns=['day', 'open', 'high', 'low', 'close', 'volume'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        df.day = pd.to_datetime(df.day)
        df.set_index(['day'], inplace=True)
        df.index.name = ''
        
        # Filter by end_date if provided
        if end_date != '' and mapped_freq in ['240m', '1200m', '7200m']:
            return df[df.index <= end_date][-mcount:]
            
        return df
    except Exception as e:
        print(f"Error fetching K-line (Sina) for {code}: {e}")
        raise e  # Let caller handle or fallback

# --- Main API ---
def get_price(code, end_date='', count=10, frequency='1d', fields=[]):
    """
    Unified API to get price data.
    code: 'sh000001' or '000001.XSHG'
    frequency: '1d', '1w', '1M', '1m', '5m', '15m', '30m', '60m'
    """
    # Standardize code
    xcode = code.replace('.XSHG', '').replace('.XSHE', '')
    if 'XSHG' in code:
        xcode = 'sh' + xcode
    elif 'XSHE' in code:
        xcode = 'sz' + xcode
    elif not (code.startswith('sh') or code.startswith('sz')):
        # Simple heuristic if no prefix provided
        if code.startswith('6') or code == '000001': # 000001 (Index)
             # Note: 000001 is ambiguous (Index vs PingAn). Assuming Index if 000001 and context implies index? 
             # Ashare logic usually expects explicit 'sh'/'sz' for best results.
             # If passed 6 digits, we need to guess.
             if code.startswith('6'): xcode = 'sh' + code
             elif code.startswith('0') or code.startswith('3'): xcode = 'sz' + code
             elif code.startswith('8') or code.startswith('4'): xcode = 'bj' + code
    
    # 1. Daily/Weekly/Monthly
    if frequency in ['1d', '1w', '1M']:
        try:
            return get_price_sina(xcode, end_date=end_date, count=count, frequency=frequency)
        except:
            return get_price_day_tx(xcode, end_date=end_date, count=count, frequency=frequency)
            
    # 2. Minute Lines
    if frequency in ['1m', '5m', '15m', '30m', '60m']:
        if frequency == '1m':
            return get_price_min_tx(xcode, end_date=end_date, count=count, frequency=frequency)
        try:
            return get_price_sina(xcode, end_date=end_date, count=count, frequency=frequency)
        except:
            return get_price_min_tx(xcode, end_date=end_date, count=count, frequency=frequency)
            
    return pd.DataFrame()
