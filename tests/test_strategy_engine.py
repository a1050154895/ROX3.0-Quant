#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略引擎模块测试
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from app.rox_quant.strategy_engine import (
    ROXStrategyEngine,
    SecurityInfo,
    get_strategy_engine,
)


class TestSecurityInfo:
    """SecurityInfo测试类"""
    
    def test_security_info_creation(self):
        """测试证券信息创建"""
        info = SecurityInfo(
            code="000001.XSHE",
            display_name="平安银行",
            name="平安银行",
            start_date="2000-01-01",
            end_date="2100-12-31",
            type="stock"
        )
        
        assert info.code == "000001.XSHE"
        assert info.display_name == "平安银行"
        assert info.type == "stock"


class TestROXStrategyEngine:
    """ROXStrategyEngine测试类"""
    
    @pytest.fixture
    def engine(self, tmp_path):
        """创建测试引擎实例"""
        # 创建临时策略目录
        strategies_dir = tmp_path / "jq_strategies"
        strategies_dir.mkdir()
        
        # 创建测试策略文件
        test_strategy = strategies_dir / "test_strategy.py"
        test_strategy.write_text("""
# Test Strategy
import pandas as pd

def test_function():
    return "success"
""")
        
        with patch.object(ROXStrategyEngine, '__init__', lambda self: None):
            engine = ROXStrategyEngine()
            engine.strategies_dir = strategies_dir
            engine.data_source = MagicMock()
            engine.execution_history = []
            engine.strategy_cache = {}
            engine.file_storage = {}
            engine.backtest_engine = None
        
        return engine
    
    def test_list_strategies(self, engine):
        """测试策略列表"""
        strategies = engine.list_strategies()
        
        assert isinstance(strategies, list)
        assert len(strategies) == 1
        assert strategies[0]['name'] == 'test_strategy'
    
    def test_get_strategy_info(self, engine):
        """测试获取策略信息"""
        info = engine.get_strategy_info('test_strategy')
        
        assert info['name'] == 'test_strategy'
        assert info['lines'] == 8
        assert 'file' in info
    
    def test_get_strategy_info_not_found(self, engine):
        """测试获取不存在的策略"""
        info = engine.get_strategy_info('nonexistent')
        assert info == {}
    
    def test_execute_strategy_success(self, engine):
        """测试策略执行成功"""
        result = engine.execute_strategy('test_strategy')
        
        assert result['success'] is True
        assert 'execution_id' in result
        assert result['strategy_name'] == 'test_strategy'
    
    def test_execute_strategy_not_found(self, engine):
        """测试执行不存在的策略"""
        result = engine.execute_strategy('nonexistent')
        
        assert result['success'] is False
        assert '不存在' in result['message']
    
    def test_execution_history(self, engine):
        """测试执行历史"""
        # 执行策略
        engine.execute_strategy('test_strategy')
        
        # 获取历史
        history = engine.get_execution_history()
        
        assert len(history) == 1
        assert history[0]['strategy_name'] == 'test_strategy'
    
    def test_clear_execution_history(self, engine):
        """测试清空执行历史"""
        engine.execute_strategy('test_strategy')
        engine.clear_execution_history()
        
        history = engine.get_execution_history()
        assert len(history) == 0
    
    def test_get_strategy_categories(self, engine):
        """测试策略分类"""
        categories = engine.get_strategy_categories()
        
        assert isinstance(categories, dict)
        assert '小市值策略' in categories
        assert 'ETF策略' in categories


class TestStrategyExecutionEnvironment:
    """策略执行环境测试类"""
    
    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        with patch.object(ROXStrategyEngine, '__init__', lambda self: None):
            engine = ROXStrategyEngine()
            engine.data_source = MagicMock()
            engine.execution_history = []
            engine.strategy_cache = {}
            engine.file_storage = {}
            return engine
    
    def test_create_execution_environment(self, engine):
        """测试创建执行环境"""
        env = engine._create_execution_environment({})
        
        # 检查基础模块
        assert 'pd' in env
        assert 'np' in env
        assert 'datetime' in env
        
        # 检查ROX系统集成
        assert 'ROX' in env
        assert 'jqdata' in env
        
        # 检查函数
        assert 'get_price' in env
        assert 'get_security_info' in env
        assert 'write_file' in env
        assert 'read_file' in env
    
    def test_get_security_info(self, engine):
        """测试获取证券信息"""
        info = engine._get_security_info("000001.XSHE")
        
        assert isinstance(info, SecurityInfo)
        assert info.code == "000001.XSHE"
    
    def test_get_security_info_etf(self, engine):
        """测试获取ETF信息"""
        info = engine._get_security_info("510300.XSHG")
        
        assert info.type == "etf"
    
    def test_write_read_file(self, engine):
        """测试文件读写"""
        engine._write_file("test", "content")
        content = engine._read_file("test")
        
        assert content == "content"
    
    def test_read_nonexistent_file(self, engine):
        """测试读取不存在的文件"""
        content = engine._read_file("nonexistent")
        assert content == ""


class TestDataAdapter:
    """数据适配器测试类"""
    
    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        with patch.object(ROXStrategyEngine, '__init__', lambda self: None):
            engine = ROXStrategyEngine()
            engine.data_source = MagicMock()
            engine.execution_history = []
            return engine
    
    def test_get_price(self, engine):
        """测试获取价格数据"""
        import pandas as pd
        
        # 模拟数据源返回
        mock_df = pd.DataFrame({
            'open': [10, 11],
            'close': [11, 12],
            'high': [12, 13],
            'low': [9, 10],
            'volume': [1000, 2000],
        })
        engine.data_source.get_bars.return_value = mock_df
        
        result = engine._get_price("000001.XSHE", count=10)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
    
    def test_get_price_empty(self, engine):
        """测试获取空价格数据"""
        import pandas as pd
        
        engine.data_source.get_bars.return_value = pd.DataFrame()
        
        result = engine._get_price("000001.XSHE", count=10)
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_get_all_securities(self, engine):
        """测试获取所有证券"""
        import pandas as pd
        
        engine.data_source.get_all_stocks.return_value = []
        
        result = engine._get_all_securities()
        
        assert isinstance(result, pd.DataFrame)
    
    def test_get_trade_days(self, engine):
        """测试获取交易日"""
        import pandas as pd
        
        engine.data_source.get_trade_days.return_value = pd.DatetimeIndex([])
        
        result = engine._get_trade_days()
        
        assert isinstance(result, pd.DatetimeIndex)


class TestGlobalEngine:
    """全局引擎测试类"""
    
    def test_get_strategy_engine_singleton(self):
        """测试单例模式"""
        import app.rox_quant.strategy_engine as module
        
        # 重置单例
        module._rox_strategy_engine = None
        
        engine1 = get_strategy_engine()
        engine2 = get_strategy_engine()
        
        assert engine1 is engine2
