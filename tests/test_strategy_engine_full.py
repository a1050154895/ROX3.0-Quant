#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略引擎测试
测试 strategy_engine.py 中的核心功能
"""

import pytest
from unittest.mock import patch, MagicMock
import os

from app.rox_quant.strategy_engine import (
    ROXStrategyEngine,
    SecurityInfo,
    get_strategy_engine,
)


class TestSecurityInfo:
    """测试证券信息数据类"""
    
    def test_security_info_creation(self):
        """测试创建证券信息"""
        info = SecurityInfo(
            code="600000",
            display_name="浦发银行"
        )
        
        assert info.code == "600000"
        assert info.display_name == "浦发银行"


class TestStrategyEngine:
    """测试策略引擎"""
    
    def test_get_strategy_engine_singleton(self):
        """测试引擎单例"""
        engine1 = get_strategy_engine()
        engine2 = get_strategy_engine()
        assert engine1 is engine2
    
    def test_list_strategies(self):
        """测试列出策略"""
        engine = get_strategy_engine()
        strategies = engine.list_strategies()
        assert isinstance(strategies, list)
    
    def test_get_strategy_info(self):
        """测试获取策略信息"""
        engine = get_strategy_engine()
        info = engine.get_strategy_info("不存在的策略")
        assert info is None or isinstance(info, dict)


class TestStrategyExecution:
    """测试策略执行"""
    
    def test_engine_exists(self):
        """测试引擎存在"""
        engine = get_strategy_engine()
        assert engine is not None
        assert isinstance(engine, ROXStrategyEngine)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
