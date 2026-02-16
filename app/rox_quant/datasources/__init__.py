#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据源模块初始化
提供统一的数据源接口
"""

from .ashare_lite import get_price_day_tx, get_price_min_tx

__all__ = [
    'get_price_day_tx',
    'get_price_min_tx'
]

def get_data_source():
    """
    获取默认数据源
    """
    class DataSource:
        """默认数据源实现"""
        
        def get_bars(self, symbol, start_date=None, end_date=None, count=None, frequency='daily'):
            """获取K线数据"""
            if frequency == 'daily':
                return get_price_day_tx(symbol, end_date=end_date, count=count)
            else:
                return get_price_min_tx(symbol, end_date=end_date, count=count, frequency=frequency)
        
        def get_all_stocks(self):
            """获取所有股票"""
            return []
        
        def get_trade_days(self, start_date=None, end_date=None):
            """获取交易日"""
            from datetime import datetime, timedelta
            import pandas as pd
            
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365)
            
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            return dates[dates.weekday < 5]
        
        def get_fundamentals(self, table, statDate=None, limit=None, columns=None):
            """获取基本面数据"""
            import pandas as pd
            return pd.DataFrame()
        
        def get_index_stocks(self, index_code):
            """获取指数成分股"""
            return []
    
    return DataSource()