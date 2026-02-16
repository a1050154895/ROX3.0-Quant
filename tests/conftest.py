#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置
提供测试 fixtures 和通用测试工具
"""

import pytest
import os
import sys
from pathlib import Path
from typing import Generator
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置测试环境变量
os.environ["ENVIRONMENT"] = "test"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only-32chars"
os.environ["DATABASE_URL"] = "sqlite:///./test_data/test.db"
os.environ["USE_REDIS"] = "False"


@pytest.fixture(scope="session")
def test_settings():
    """测试配置"""
    from app.core.security_config import Settings
    return Settings()


@pytest.fixture
def client() -> Generator:
    """创建测试客户端"""
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_data_source():
    """模拟数据源"""
    mock = MagicMock()
    mock.get_bars.return_value = MagicMock()
    mock.get_all_stocks.return_value = []
    return mock


@pytest.fixture
def sample_stock_data():
    """示例股票数据"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.uniform(10, 20, 100),
        'close': np.random.uniform(10, 20, 100),
        'high': np.random.uniform(15, 25, 100),
        'low': np.random.uniform(5, 15, 100),
        'volume': np.random.randint(1000000, 10000000, 100),
    }, index=dates)
    
    return data


@pytest.fixture
def auth_headers() -> dict:
    """认证头"""
    from app.auth import create_access_token
    
    token = create_access_token(data={"sub": "test_user"})
    return {"Authorization": f"Bearer {token}"}


def assert_response_success(response, expected_status: int = 200):
    """断言响应成功"""
    assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}: {response.text}"


def assert_response_error(response, expected_status: int = 400):
    """断言响应错误"""
    assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}"


class MockResponse:
    """模拟HTTP响应"""
    
    def __init__(self, json_data, status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")
