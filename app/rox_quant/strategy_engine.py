#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROX策略引擎 - 聚宽策略适配层
将聚宽策略深度内化为ROX系统的一部分
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, date, timedelta
import logging
import inspect
import importlib.util
import sys
from pathlib import Path
from dataclasses import dataclass

from app.core.config import settings
from app.rox_quant.datasources import get_data_source
from app.rox_quant.context import Portfolio

logger = logging.getLogger(__name__)

@dataclass
class SecurityInfo:
    """证券信息"""
    code: str
    display_name: str
    name: str
    start_date: date
    end_date: date
    type: str

class ROXStrategyEngine:
    """
    ROX策略执行引擎
    深度内化聚宽策略，使其成为ROX系统的一部分
    """
    
    def __init__(self):
        self.data_source = get_data_source()
        self.strategies_dir = Path(__file__).parent / "jq_strategies"
        self.execution_history = []
        self.strategy_cache = {}
        self.file_storage = {}
        
        # 初始化回测引擎
        self.backtest_engine = None
        
        logger.info("ROX策略引擎初始化完成")
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """
        获取策略详细信息
        """
        strategy_file = self.strategies_dir / f"{strategy_name}.py"
        
        if not strategy_file.exists():
            logger.error(f"策略文件不存在: {strategy_file}")
            return {}
        
        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            description = ''
            dependencies = []
            
            for line in lines[:30]:
                if line.strip().startswith('#'):
                    description += line.strip()[1:].strip() + ' '
                elif 'import' in line:
                    dependencies.append(line.strip())
                elif description and not line.strip():
                    break
            
            return {
                'name': strategy_name,
                'file': str(strategy_file),
                'description': description.strip(),
                'size': len(content),
                'lines': len(lines),
                'dependencies': dependencies,
                'modified': datetime.fromtimestamp(strategy_file.stat().st_mtime)
            }
        except Exception as e:
            logger.error(f"获取策略信息失败: {e}")
            return {}
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """
        列出所有可用的策略
        """
        strategies = []
        
        if not self.strategies_dir.exists():
            logger.warning(f"策略目录不存在: {self.strategies_dir}")
            return strategies
        
        for file in self.strategies_dir.glob('*.py'):
            try:
                strategy_name = file.stem
                info = self.get_strategy_info(strategy_name)
                if info:
                    strategies.append(info)
            except Exception as e:
                logger.warning(f"处理策略文件失败: {file}, 错误: {e}")
        
        return sorted(strategies, key=lambda x: x['name'])
    
    def execute_strategy(self, strategy_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行策略
        """
        if params is None:
            params = {}
        
        start_time = datetime.now()
        execution_id = f"{strategy_name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"开始执行策略: {strategy_name}, 执行ID: {execution_id}")
            
            # 加载策略
            strategy_file = self.strategies_dir / f"{strategy_name}.py"
            if not strategy_file.exists():
                raise FileNotFoundError(f"策略文件不存在: {strategy_file}")
            
            # 读取策略代码
            with open(strategy_file, 'r', encoding='utf-8') as f:
                strategy_code = f.read()
            
            # 创建ROX执行环境
            exec_globals = self._create_execution_environment(params)
            
            # 执行策略
            exec(strategy_code, exec_globals)
            
            # 获取执行结果
            results = {
                'execution_id': execution_id,
                'strategy_name': strategy_name,
                'params': params,
                'start_time': start_time,
                'end_time': datetime.now(),
                'success': True,
                'message': '策略执行成功',
                'output': exec_globals.get('output', {}),
                'results': exec_globals.get('results', {}),
                'portfolio': exec_globals.get('portfolio', None)
            }
            
            # 记录执行历史
            self.execution_history.append(results)
            
            logger.info(f"策略执行成功: {strategy_name}, 耗时: {results['end_time'] - start_time}")
            return results
            
        except Exception as e:
            error_message = f"策略执行失败: {str(e)}"
            logger.error(error_message)
            
            results = {
                'execution_id': execution_id,
                'strategy_name': strategy_name,
                'params': params,
                'start_time': start_time,
                'end_time': datetime.now(),
                'success': False,
                'message': error_message,
                'error': str(e),
                'traceback': str(sys.exc_info())
            }
            
            # 记录执行历史
            self.execution_history.append(results)
            
            return results
    
    def _create_execution_environment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建ROX策略执行环境
        """
        # 基础环境
        exec_globals = {
            '__name__': '__main__',
            '__file__': '',
            'pd': pd,
            'np': np,
            'math': math,
            'sqrt': math.sqrt,
            'log': math.log,
            'exp': math.exp,
            'pow': math.pow,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sorted': sorted,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'datetime': datetime,
            'date': date,
            'timedelta': timedelta,
            'logging': logging,
            'logger': logger,
            **params
        }
        
        # ROX系统集成
        exec_globals['ROX'] = {
            'data_source': self.data_source,
            'settings': settings,
            'portfolio': Portfolio(cash=100000),
            'get_strategy_engine': lambda: self
        }
        
        # 聚宽API适配
        exec_globals['jqdata'] = self._create_jqdata_adapter()
        
        # ROX专用函数
        exec_globals['get_price'] = self._get_price
        exec_globals['get_all_securities'] = self._get_all_securities
        exec_globals['get_fundamentals'] = self._get_fundamentals
        exec_globals['get_trade_days'] = self._get_trade_days
        exec_globals['get_index_stocks'] = self._get_index_stocks
        exec_globals['get_security_info'] = self._get_security_info
        exec_globals['write_file'] = self._write_file
        exec_globals['read_file'] = self._read_file
        
        # 结果存储
        exec_globals['output'] = {}
        exec_globals['results'] = {}
        exec_globals['portfolio'] = Portfolio(cash=100000)
        
        return exec_globals
    
    def _create_jqdata_adapter(self) -> Any:
        """
        创建聚宽API适配器
        """
        class JQDataAdapter:
            """聚宽API适配器"""
            
            def __init__(self, engine):
                self.engine = engine
            
            def get_all_securities(self, types=['stock', 'etf']):
                return self.engine._get_all_securities(types)
            
            def get_price(self, code, end_date=None, count=None, frequency='daily', fields=None):
                return self.engine._get_price(code, end_date, count, frequency, fields)
            
            def get_fundamentals(self, table, statDate=None, limit=None, columns=None):
                return self.engine._get_fundamentals(table, statDate, limit, columns)
            
            def get_trade_days(self, start_date=None, end_date=None):
                return self.engine._get_trade_days(start_date, end_date)
            
            def get_index_stocks(self, index_code):
                return self.engine._get_index_stocks(index_code)
            
            def get_industry(self, code):
                return self.engine._get_industry(code)
            
            def get_concept(self, code):
                return self.engine._get_concept(code)
            
            def get_security_info(self, code):
                return self.engine._get_security_info(code)
        
        return JQDataAdapter(self)
    
    def _get_price(self, code: str, end_date: Optional[Any] = None, 
                   count: Optional[int] = None, frequency: str = 'daily', 
                   fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取价格数据 - ROX实现
        """
        if fields is None:
            fields = ['open', 'close', 'high', 'low', 'volume', 'money']
        
        try:
            if self.data_source:
                # 转换日期格式
                if end_date:
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)
                    elif isinstance(end_date, date):
                        end_date = pd.to_datetime(end_date)
                
                # 使用ROX数据接口
                df = self.data_source.get_bars(
                    symbol=code,
                    start_date=None,
                    end_date=end_date,
                    count=count,
                    frequency=frequency
                )
                
                if not df.empty:
                    # 确保返回正确的字段
                    for field in fields:
                        if field not in df.columns:
                            df[field] = 0
                    return df[fields]
        except Exception as e:
            logger.warning(f"获取价格数据失败: {e}")
        
        return pd.DataFrame(columns=fields)
    
    def _get_all_securities(self, types: List[str] = ['stock', 'etf']) -> pd.DataFrame:
        """
        获取所有证券信息 - ROX实现
        """
        try:
            if self.data_source and hasattr(self.data_source, 'get_all_stocks'):
                stocks = self.data_source.get_all_stocks()
                df = pd.DataFrame(stocks)
                df = df.reset_index().rename(columns={'index': 'code'})
                
                # 添加必要的字段
                if 'display_name' not in df.columns:
                    df['display_name'] = df.get('name', '')
                if 'start_date' not in df.columns:
                    df['start_date'] = date(2000, 1, 1)
                if 'end_date' not in df.columns:
                    df['end_date'] = date(2100, 12, 31)
                if 'type' not in df.columns:
                    df['type'] = 'stock'
                
                return df
        except Exception as e:
            logger.warning(f"获取证券信息失败: {e}")
        
        return pd.DataFrame(columns=['code', 'display_name', 'name', 'start_date', 'end_date', 'type'])
    
    def _get_security_info(self, code: str) -> SecurityInfo:
        """
        获取单个证券信息 - ROX实现
        """
        try:
            # 从代码中提取基本信息
            if 'XSHE' in code:
                display_name = f"深圳股票 {code}"
                sec_type = 'stock'
            elif 'XSHG' in code:
                display_name = f"上海股票 {code}"
                sec_type = 'stock'
            elif 'ETF' in code or code.startswith('5') or code.startswith('15'):
                display_name = f"ETF {code}"
                sec_type = 'etf'
            else:
                display_name = f"证券 {code}"
                sec_type = 'stock'
            
            return SecurityInfo(
                code=code,
                display_name=display_name,
                name=display_name,
                start_date=date(2000, 1, 1),
                end_date=date(2100, 12, 31),
                type=sec_type
            )
        except Exception as e:
            logger.warning(f"获取证券信息失败: {e}")
            return SecurityInfo(
                code=code,
                display_name=code,
                name=code,
                start_date=date(2000, 1, 1),
                end_date=date(2100, 12, 31),
                type='stock'
            )
    
    def _get_fundamentals(self, table: str, statDate: Optional[str] = None, 
                         limit: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取基本面数据 - ROX实现
        """
        try:
            if self.data_source and hasattr(self.data_source, 'get_fundamentals'):
                return self.data_source.get_fundamentals(table, statDate, limit, columns)
        except Exception as e:
            logger.warning(f"获取基本面数据失败: {e}")
        
        return pd.DataFrame()
    
    def _get_trade_days(self, start_date: Optional[Any] = None, 
                       end_date: Optional[Any] = None) -> pd.DatetimeIndex:
        """
        获取交易日 - ROX实现
        """
        try:
            if self.data_source and hasattr(self.data_source, 'get_trade_days'):
                return self.data_source.get_trade_days(start_date, end_date)
        except Exception as e:
            logger.warning(f"获取交易日失败: {e}")
        
        return pd.DatetimeIndex([])
    
    def _get_index_stocks(self, index_code: str) -> List[str]:
        """
        获取指数成分股 - ROX实现
        """
        try:
            if self.data_source and hasattr(self.data_source, 'get_index_stocks'):
                return self.data_source.get_index_stocks(index_code)
        except Exception as e:
            logger.warning(f"获取指数成分股失败: {e}")
        
        return []
    
    def _get_industry(self, code: str) -> Dict[str, Any]:
        """
        获取股票行业 - ROX实现
        """
        return {}
    
    def _get_concept(self, code: str) -> Dict[str, Any]:
        """
        获取股票概念 - ROX实现
        """
        return {}
    
    def _write_file(self, name: str, content: str):
        """
        写入文件 - ROX实现
        """
        self.file_storage[name] = content
        logger.info(f"文件已写入: {name}")
    
    def _read_file(self, name: str) -> str:
        """
        读取文件 - ROX实现
        """
        return self.file_storage.get(name, '')
    
    def backtest_strategy(self, strategy_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        回测聚宽策略
        """
        if params is None:
            params = {}
        
        start_time = datetime.now()
        
        try:
            # 执行策略并收集结果
            execution_result = self.execute_strategy(strategy_name, params)
            
            if execution_result['success']:
                # 简化的回测结果
                backtest_result = {
                    'success': True,
                    'message': '回测完成',
                    'execution_result': execution_result,
                    'metrics': {
                        'total_return': '待实现',
                        'sharpe_ratio': '待实现',
                        'max_drawdown': '待实现'
                    }
                }
                
                return {
                    'execution_id': execution_result['execution_id'],
                    'strategy_name': strategy_name,
                    'params': params,
                    'start_time': start_time,
                    'end_time': datetime.now(),
                    'success': True,
                    'execution_result': execution_result,
                    'backtest_result': backtest_result
                }
            else:
                return execution_result
                
        except Exception as e:
            error_message = f"回测策略失败: {str(e)}"
            logger.error(error_message)
            
            return {
                'strategy_name': strategy_name,
                'params': params,
                'start_time': start_time,
                'end_time': datetime.now(),
                'success': False,
                'message': error_message,
                'error': str(e)
            }
    
    def get_strategy_categories(self) -> Dict[str, int]:
        """
        获取策略分类统计
        """
        strategies = self.list_strategies()
        
        categories = {
            '小市值策略': 0,
            'ETF策略': 0,
            '打板策略': 0,
            '机器学习': 0,
            '价值投资': 0,
            '其他': 0
        }
        
        for strategy in strategies:
            name = strategy['name'].lower()
            if '小市值' in name or 'small' in name or '小盘' in name:
                categories['小市值策略'] += 1
            elif 'etf' in name:
                categories['ETF策略'] += 1
            elif '板' in name or '涨停' in name:
                categories['打板策略'] += 1
            elif '机器学习' in name or 'machine' in name or '学习' in name:
                categories['机器学习'] += 1
            elif '价值' in name or '投资' in name or '股息' in name:
                categories['价值投资'] += 1
            else:
                categories['其他'] += 1
        
        return categories
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取执行历史
        """
        return self.execution_history[-limit:]
    
    def clear_execution_history(self):
        """
        清空执行历史
        """
        self.execution_history = []
        logger.info("执行历史已清空")

# 全局策略引擎实例
_rox_strategy_engine = None

def get_strategy_engine() -> ROXStrategyEngine:
    """
    获取ROX策略引擎实例
    """
    global _rox_strategy_engine
    if _rox_strategy_engine is None:
        _rox_strategy_engine = ROXStrategyEngine()
    return _rox_strategy_engine

# 兼容旧API
def get_jq_adapter(data_source=None) -> ROXStrategyEngine:
    """
    兼容旧API，返回ROX策略引擎
    """
    return get_strategy_engine()