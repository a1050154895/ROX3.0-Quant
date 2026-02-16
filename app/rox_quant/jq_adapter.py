#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚宽API适配器
将聚宽API调用转换为ROX 3.0的数据接口
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class JQDataMock:
    """聚宽数据API的模拟实现"""
    
    def __init__(self, data_source=None):
        self.data_source = data_source
        self._cache = {}
        
    def get_all_securities(self, types=['stock', 'etf']):
        """获取所有证券信息"""
        if self.data_source:
            try:
                if hasattr(self.data_source, 'get_all_stocks'):
                    stocks = self.data_source.get_all_stocks()
                    df = pd.DataFrame(stocks)
                    df = df.reset_index().rename(columns={'index': 'code'})
                    return df
            except Exception as e:
                logger.warning(f"无法从数据源获取证券信息: {e}")
        
        return pd.DataFrame(columns=['code', 'display_name', 'name', 'start_date', 'end_date', 'type'])
    
    def get_price(self, code, end_date=None, count=None, frequency='daily', fields=None):
        """获取价格数据"""
        if fields is None:
            fields = ['open', 'close', 'high', 'low', 'volume', 'money']
        
        if self.data_source:
            try:
                if hasattr(self.data_source, 'get_bars'):
                    if end_date:
                        if isinstance(end_date, str):
                            end_date = pd.to_datetime(end_date)
                        elif isinstance(end_date, date):
                            end_date = pd.to_datetime(end_date)
                    
                    df = self.data_source.get_bars(
                        symbol=code,
                        start_date=None,
                        end_date=end_date,
                        count=count,
                        frequency=frequency
                    )
                    
                    if not df.empty:
                        return df
            except Exception as e:
                logger.warning(f"无法从数据源获取价格数据: {e}")
        
        return pd.DataFrame(columns=fields)
    
    def get_fundamentals(self, table, statDate=None, limit=None, columns=None):
        """获取基本面数据"""
        return pd.DataFrame()
    
    def get_trade_days(self, start_date=None, end_date=None):
        """获取交易日"""
        if self.data_source:
            try:
                if hasattr(self.data_source, 'get_trade_days'):
                    return self.data_source.get_trade_days(start_date, end_date)
            except Exception as e:
                logger.warning(f"无法获取交易日: {e}")
        
        return pd.DatetimeIndex([])
    
    def get_index_stocks(self, index_code):
        """获取指数成分股"""
        if self.data_source:
            try:
                if hasattr(self.data_source, 'get_index_stocks'):
                    return self.data_source.get_index_stocks(index_code)
            except Exception as e:
                logger.warning(f"无法获取指数成分股: {e}")
        
        return []
    
    def get_industry(self, code):
        """获取股票行业"""
        return {}
    
    def get_concept(self, code):
        """获取股票概念"""
        return {}

class JQStrategyAdapter:
    """聚宽策略适配器"""
    
    def __init__(self, data_source=None):
        self.jq_data = JQDataMock(data_source)
        self.context = None
        self.portfolio = None
        
    def setup_environment(self):
        """设置聚宽环境"""
        import sys
        sys.modules['jqdata'] = type(sys)('jqdata')
        sys.modules['jqdata'].get_all_securities = self.jq_data.get_all_securities
        sys.modules['jqdata'].get_price = self.jq_data.get_price
        sys.modules['jqdata'].get_fundamentals = self.jq_data.get_fundamentals
        sys.modules['jqdata'].get_trade_days = self.jq_data.get_trade_days
        sys.modules['jqdata'].get_index_stocks = self.jq_data.get_index_stocks
        sys.modules['jqdata'].get_industry = self.jq_data.get_industry
        sys.modules['jqdata'].get_concept = self.jq_data.get_concept
        
        logger.info("聚宽API环境已设置")
    
    def load_strategy(self, strategy_file: str):
        """加载聚宽策略文件"""
        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                code = f.read()
            return code
        except Exception as e:
            logger.error(f"加载策略文件失败: {e}")
            return None
    
    def execute_strategy(self, strategy_code: str, params: Dict[str, Any] = None):
        """执行聚宽策略"""
        self.setup_environment()
        
        if params is None:
            params = {}
        
        try:
            exec_globals = {
                '__name__': '__main__',
                'pd': pd,
                'np': np,
                'datetime': datetime,
                'date': date,
                **params
            }
            
            exec(strategy_code, exec_globals)
            
            return {
                'success': True,
                'message': '策略执行成功',
                'results': exec_globals.get('results', {})
            }
        except Exception as e:
            logger.error(f"策略执行失败: {e}")
            return {
                'success': False,
                'message': f'策略执行失败: {str(e)}',
                'results': {}
            }
    
    def list_strategies(self, strategy_dir: str = None) -> List[Dict[str, Any]]:
        """列出所有可用的聚宽策略"""
        if strategy_dir is None:
            strategy_dir = Path(__file__).parent / 'jq_strategies'
        else:
            strategy_dir = Path(strategy_dir)
        
        strategies = []
        
        if not strategy_dir.exists():
            return strategies
        
        for file in strategy_dir.glob('*.py'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                strategy_info = {
                    'name': file.stem,
                    'file': str(file),
                    'size': len(content),
                    'modified': datetime.fromtimestamp(file.stat().st_mtime)
                }
                
                strategies.append(strategy_info)
            except Exception as e:
                logger.warning(f"无法读取策略文件 {file}: {e}")
        
        return sorted(strategies, key=lambda x: x['name'])
    
    def get_strategy_info(self, strategy_file: str) -> Dict[str, Any]:
        """获取策略详细信息"""
        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            description = ''
            for line in lines[:20]:
                if line.strip().startswith('#'):
                    description += line.strip()[1:].strip() + ' '
                elif description:
                    break
            
            return {
                'name': Path(strategy_file).stem,
                'file': strategy_file,
                'description': description.strip(),
                'size': len(content),
                'lines': len(lines)
            }
        except Exception as e:
            logger.error(f"获取策略信息失败: {e}")
            return {}

_jq_adapter_instance = None

def get_jq_adapter(data_source=None):
    """获取聚宽适配器单例"""
    global _jq_adapter_instance
    if _jq_adapter_instance is None:
        _jq_adapter_instance = JQStrategyAdapter(data_source)
    return _jq_adapter_instance