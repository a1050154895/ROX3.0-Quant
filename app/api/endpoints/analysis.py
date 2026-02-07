from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
import akshare as ak
import pandas as pd
import numpy as np
from app.analysis.hot_money import calculate_hot_money_indicator
from app.utils.retry import run_with_retry
from app.analysis.three_color_resonance import calculate_three_color_resonance
from app.analysis.kang_long_you_hui import calculate_kang_long_you_hui
from app.analysis.dark_pool_fund import calculate_dark_pool_fund
from app.analysis.precise_trading import get_precise_trading_signals

from app.services.dashboard_analyzer import dashboard_analyzer
from app.utils.akshare_wrapper import akshare_client
from app.quant.data_provider import DataProvider
import pandas as pd

@router.get("/dashboard/{symbol}")
async def get_dashboard_analysis(symbol: str):
    """
    Get Deep Analysis Dashboard for a stock (A-Share)
    """
    # 1. Get History Data (for Technical Analysis)
    provider = DataProvider()
    history = provider.get_history(symbol, days=120) 
    # Convert to DataFrame
    if not history:
        return {"error": "No history data found"}
        
    df = pd.DataFrame([vars(p) for p in history])
    
    # 2. Get Realtime & Chip Data (Parallel if poss, but sequential is fine for now)
    # Note: symbol might need adjustment for akshare (6 digits)
    ak_symbol = symbol
    if symbol.startswith('sh') or symbol.startswith('sz'): 
        ak_symbol = symbol
    
    # Chip Distribution (A-Share Only)
    chip_data = None
    realtime = None
    stock_name = symbol
    
    try:
        # Simple heuristic for A-Share
        is_ashare = symbol.startswith('sh6') or symbol.startswith('sz0') or symbol.startswith('sz3') or (symbol.isdigit() and len(symbol)==6)
        
        if is_ashare:
            chip_res = await akshare_client.get_chip_distribution(ak_symbol)
            # Process chip data to summary dict if needed, or pass raw if small
            # Actually akshare returns DataFrame usually. We need to parse it.
            # Convert DF to list/dict
            if hasattr(chip_res, 'to_dict'):
                chip_data = chip_res.to_dict(orient='records')
            
            # Realtime Indicators
            # For specific stock, we might just rely on provider's get_realtime_quote
            # But we added get_realtime_indicators (all market) in wrapper. Let's send basic realtime quote from provider for now to save time/bandwidth
            rt_quote = provider.data_manager.get_realtime_quote(symbol) # Returns dict
            if rt_quote:
                realtime = rt_quote
                stock_name = rt_quote.get('name', symbol)
                
    except Exception as e:
        print(f"Extra data fetch failed: {e}")
        
    # 3. Analyze
    result = await dashboard_analyzer.analyze(
        symbol=symbol,
        stock_name=stock_name,
        df=df,
        chip_data=chip_data,
        realtime=realtime
    )
    
    return result


@router.get("/hot-money/{stock_code}", tags=["Analysis"])
async def get_hot_money_analysis(stock_code: str):
    """
    Provides the 'Hot Money Dark Pool' analysis for a given stock code.
    
    - **stock_code**: The 6-digit stock code (e.g., '600519').
    """
    try:
        # Fetch daily historical data for the stock
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
        if stock_zh_a_hist_df.empty:
            raise HTTPException(status_code=404, detail="Could not fetch historical data for the stock code.")

        # The column names from akshare are in Chinese, rename them to English for processing
        stock_zh_a_hist_df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        }, inplace=True)
        
        # Ensure date is in the correct format
        stock_zh_a_hist_df['date'] = pd.to_datetime(stock_zh_a_hist_df['date'])
        stock_zh_a_hist_df.set_index('date', inplace=True)

        # Calculate the indicator
        analysis_df = calculate_hot_money_indicator(stock_zh_a_hist_df)

        # Replace NaN/inf/-inf with None for JSON compatibility
        analysis_df.replace([pd.NA, np.nan, np.inf, -np.inf], None, inplace=True)
        
        # Reset index to include date in the output
        analysis_df.reset_index(inplace=True)

        # Convert DataFrame to a list of dictionaries for the JSON response
        result = analysis_df.to_dict(orient='records')
        
        return result

    except Exception as e:
        err_str = str(e).lower()
        if "remote disconnected" in err_str or "connection aborted" in err_str:
            raise HTTPException(status_code=503, detail="行情数据源暂时不可用，请稍后重试")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/precise-trading/{stock_code}", tags=["Analysis"])
async def get_precise_trading_analysis(stock_code: str):
    """
    Provides the 'Precise Trading' analysis for a given stock code.
    
    - **stock_code**: The 6-digit stock code (e.g., '600519').
    """
    try:
        end_d = datetime.now()
        start_d = end_d - timedelta(days=365 * 2)
        start_date = start_d.strftime("%Y%m%d")
        end_date = end_d.strftime("%Y%m%d")
        df = run_with_retry(
            lambda: ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        )
        if df.empty:
            raise HTTPException(status_code=404, detail="Could not fetch historical data.")

        # Rename columns for consistency
        df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        }, inplace=True)
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Calculate the indicator
        analysis_results = get_precise_trading_signals(df)

        # Combine original data with analysis results
        # The frontend needs OHLC data plus the indicator values
        combined_df = df.copy()
        for key, value in analysis_results.items():
            combined_df[key] = value

        # Replace NaN/inf/-inf with None for JSON compatibility
        combined_df.replace([pd.NA, np.nan, np.inf, -np.inf], None, inplace=True)
        
        # Reset index to include date in the output
        combined_df.reset_index(inplace=True)

        # Convert to list of records
        result = combined_df.to_dict(orient='records')
        
        return result

    except Exception as e:
        err_str = str(e).lower()
        if "remote disconnected" in err_str or "connection aborted" in err_str:
            raise HTTPException(status_code=503, detail="行情数据源暂时不可用，请稍后重试")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/three-color-resonance/{stock_code}", tags=["Analysis"])
async def get_three_color_resonance_analysis(stock_code: str):
    """
    Provides the 'Three-Color Fund Resonance' analysis for a given stock code.
    
    - **stock_code**: The 6-digit stock code (e.g., '600519').
    """
    try:
        end_d = datetime.now()
        start_d = end_d - timedelta(days=365 * 2)
        start_date = start_d.strftime("%Y%m%d")
        end_date = end_d.strftime("%Y%m%d")
        stock_zh_a_hist_df = run_with_retry(
            lambda: ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        )
        if stock_zh_a_hist_df.empty:
            raise HTTPException(status_code=404, detail="Could not fetch historical data for the stock code.")

        # Rename columns for consistency
        stock_zh_a_hist_df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        }, inplace=True)
        
        # Set date as index
        stock_zh_a_hist_df['date'] = pd.to_datetime(stock_zh_a_hist_df['date'])
        stock_zh_a_hist_df.set_index('date', inplace=True)

        # Calculate the indicator
        analysis_df = calculate_three_color_resonance(stock_zh_a_hist_df)

        # Clean data for JSON response
        analysis_df.replace([pd.NA, np.nan, np.inf, -np.inf], None, inplace=True)
        
        # Reset index to include date in the output
        analysis_df.reset_index(inplace=True)

        # Convert to list of records
        result = analysis_df.to_dict(orient='records')
        
        return result

    except Exception as e:
        err_str = str(e).lower()
        if "remote disconnected" in err_str or "connection aborted" in err_str:
            raise HTTPException(status_code=503, detail="行情数据源暂时不可用，请稍后重试")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/kang-long-you-hui/{stock_code}", tags=["Analysis"])
async def get_kang_long_you_hui_analysis(stock_code: str):
    """
    Provides the 'Kang Long You Hui' analysis for a given stock code.
    
    - **stock_code**: The 6-digit stock code (e.g., '600519').
    """
    try:
        end_d = datetime.now()
        start_d = end_d - timedelta(days=365 * 2)
        start_date = start_d.strftime("%Y%m%d")
        end_date = end_d.strftime("%Y%m%d")
        stock_zh_a_hist_df = run_with_retry(
            lambda: ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        )
        if stock_zh_a_hist_df.empty:
            raise HTTPException(status_code=404, detail="Could not fetch historical data for the stock code.")

        # Rename columns
        stock_zh_a_hist_df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        }, inplace=True)
        
        # Set date as index
        stock_zh_a_hist_df['date'] = pd.to_datetime(stock_zh_a_hist_df['date'])
        stock_zh_a_hist_df.set_index('date', inplace=True)

        # Calculate the indicator
        analysis_df = calculate_kang_long_you_hui(stock_zh_a_hist_df)

        # Clean data for JSON response
        analysis_df.replace([pd.NA, np.nan, np.inf, -np.inf], None, inplace=True)
        
        # Reset index to include date in the output
        analysis_df.reset_index(inplace=True)

        # Convert to list of records
        result = analysis_df.to_dict(orient='records')
        
        return result

    except Exception as e:
        err_str = str(e).lower()
        if "remote disconnected" in err_str or "connection aborted" in err_str:
            raise HTTPException(status_code=503, detail="行情数据源暂时不可用，请稍后重试")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dark-pool-fund/{stock_code}", tags=["Analysis"])
async def get_dark_pool_fund_analysis(stock_code: str):
    """
    Provides the 'Dark Pool Fund' analysis for a given stock code.
    This requires both OHLC and money flow data.
    """
    try:
        end_d = datetime.now()
        start_d = end_d - timedelta(days=365 * 2)
        start_date = start_d.strftime("%Y%m%d")
        end_date = end_d.strftime("%Y%m%d")
        stock_zh_a_hist_df = run_with_retry(
            lambda: ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        )
        if stock_zh_a_hist_df.empty:
            raise HTTPException(status_code=404, detail="Could not fetch OHLC data.")

        # 2. Fetch money flow data
        money_flow_df = ak.stock_individual_fund_flow(stock=stock_code, market="sh") # Assuming SH, can be improved
        if money_flow_df.empty:
            raise HTTPException(status_code=404, detail="Could not fetch money flow data.")

        # 3. Merge the two dataframes
        # Ensure date columns are of the same type
        stock_zh_a_hist_df['日期'] = pd.to_datetime(stock_zh_a_hist_df['日期']).dt.date
        money_flow_df['日期'] = pd.to_datetime(money_flow_df['日期']).dt.date
        
        merged_df = pd.merge(stock_zh_a_hist_df, money_flow_df, on='日期', how='inner')

        # 4. Calculate the indicator
        analysis_df = calculate_dark_pool_fund(merged_df)

        # 5. Clean and format for response
        analysis_df.replace([pd.NA, np.nan, np.inf, -np.inf], None, inplace=True)
        result = analysis_df.to_dict(orient='records')
        
        return result

    except Exception as e:
        err_str = str(e).lower()
        if "remote disconnected" in err_str or "connection aborted" in err_str:
            raise HTTPException(status_code=503, detail="行情数据源暂时不可用，请稍后重试")
        raise HTTPException(status_code=500, detail=str(e))

