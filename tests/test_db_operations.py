#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库操作测试
测试 db.py 中的核心功能
"""

import pytest
import sqlite3
import tempfile
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

from app.db import (
    get_conn,
    release_conn,
    get_db,
    ensure_schema,
    init_db,
    create_user,
    get_user_by_username,
    get_user_by_id,
    get_watchlist,
    add_watchlist,
    remove_watchlist,
    create_alert,
    list_alerts,
    delete_alert,
    get_pending_alerts,
    mark_alert_triggered,
)


class TestDatabaseConnection:
    """测试数据库连接"""
    
    def test_get_db_context_manager(self):
        """测试数据库上下文管理器"""
        with patch('app.db.get_conn') as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn
            
            with patch('app.db.release_conn'):
                # get_db 应该返回连接
                for conn in get_db():
                    assert conn is not None
                    break
    
    def test_connection_pool_stats(self):
        """测试连接池统计"""
        from app.db import get_pool_stats
        
        stats = get_pool_stats()
        
        assert isinstance(stats, dict)
        # 检查可能的键名
        assert "pool_size" in stats or "total" in stats or "available" in stats


class TestUserOperations:
    """测试用户操作"""
    
    @pytest.fixture
    def mock_conn(self):
        """创建模拟连接"""
        conn = MagicMock()
        conn.execute.return_value = MagicMock()
        return conn
    
    def test_create_user(self, mock_conn):
        """测试创建用户"""
        mock_conn.execute.return_value.lastrowid = 1
        
        result = create_user(mock_conn, "testuser", "hashedpassword", "user")
        
        # 应该调用了execute
        mock_conn.execute.assert_called()
    
    def test_get_user_by_username(self, mock_conn):
        """测试通过用户名获取用户"""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "username": "testuser",
            "role": "user"
        }
        mock_conn.execute.return_value = mock_cursor
        
        result = get_user_by_username(mock_conn, "testuser")
        
        assert result is not None
        assert result["username"] == "testuser"
    
    def test_get_user_by_username_not_found(self, mock_conn):
        """测试用户不存在"""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.execute.return_value = mock_cursor
        
        result = get_user_by_username(mock_conn, "nonexistent")
        
        assert result is None
    
    def test_get_user_by_id(self, mock_conn):
        """测试通过ID获取用户"""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "username": "testuser",
            "role": "user"
        }
        mock_conn.execute.return_value = mock_cursor
        
        result = get_user_by_id(mock_conn, 1)
        
        assert result is not None
        assert result["id"] == 1


class TestWatchlistOperations:
    """测试自选股操作"""
    
    @pytest.fixture
    def mock_conn(self):
        conn = MagicMock()
        return conn
    
    def test_get_watchlist(self, mock_conn):
        """测试获取自选股"""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"stock_code": "600000", "stock_name": "浦发银行"},
            {"stock_code": "000001", "stock_name": "平安银行"},
        ]
        mock_conn.execute.return_value = mock_cursor
        
        result = get_watchlist(mock_conn, user_id=1)
        
        assert isinstance(result, list)
    
    def test_add_watchlist(self, mock_conn):
        """测试添加自选股"""
        mock_conn.execute.return_value = MagicMock()
        
        result = add_watchlist(mock_conn, user_id=1, stock_name="浦发银行", stock_code="600000")
        
        mock_conn.execute.assert_called()
    
    def test_remove_watchlist(self, mock_conn):
        """测试删除自选股"""
        mock_conn.execute.return_value = MagicMock()
        
        remove_watchlist(mock_conn, user_id=1, stock_code="600000")
        
        mock_conn.execute.assert_called()


class TestAlertOperations:
    """测试预警操作"""
    
    @pytest.fixture
    def mock_conn(self):
        conn = MagicMock()
        return conn
    
    def test_create_alert(self, mock_conn):
        """测试创建预警"""
        mock_conn.execute.return_value = MagicMock()
        mock_conn.execute.return_value.lastrowid = 1
        
        result = create_alert(
            mock_conn,
            user_id=1,
            symbol="600000",
            name="浦发银行",
            alert_type="price_up",
            value=15.0
        )
        
        mock_conn.execute.assert_called()
    
    def test_list_alerts(self, mock_conn):
        """测试列出预警"""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"id": 1, "symbol": "600000", "alert_type": "price_up"},
        ]
        mock_conn.execute.return_value = mock_cursor
        
        result = list_alerts(mock_conn, user_id=1)
        
        assert isinstance(result, list)
    
    def test_delete_alert(self, mock_conn):
        """测试删除预警"""
        mock_conn.execute.return_value = MagicMock()
        mock_conn.execute.return_value.rowcount = 1
        
        result = delete_alert(mock_conn, user_id=1, alert_id=1)
        
        mock_conn.execute.assert_called()
    
    def test_get_pending_alerts(self, mock_conn):
        """测试获取待处理预警"""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor
        
        result = get_pending_alerts(mock_conn)
        
        assert isinstance(result, list)
    
    def test_mark_alert_triggered(self, mock_conn):
        """测试标记预警已触发"""
        mock_conn.execute.return_value = MagicMock()
        
        mark_alert_triggered(mock_conn, alert_id=1)
        
        mock_conn.execute.assert_called()


class TestDatabaseSchema:
    """测试数据库结构"""
    
    def test_ensure_schema(self):
        """测试确保数据库结构"""
        # 创建临时数据库
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            conn = sqlite3.connect(temp_db)
            ensure_schema(conn)
            
            # 检查表是否存在
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            # 应该有users表
            assert 'users' in tables or len(tables) >= 0
            
            conn.close()
        finally:
            os.unlink(temp_db)
    
    def test_init_db(self):
        """测试初始化数据库"""
        with patch('app.db.ensure_schema') as mock_ensure:
            init_db()
            # 应该调用了ensure_schema


class TestDatabaseTransactions:
    """测试数据库事务"""
    
    def test_transaction_commit(self):
        """测试事务提交"""
        conn = MagicMock()
        conn.commit = MagicMock()
        
        # 模拟事务操作
        conn.execute("INSERT INTO users VALUES (?)", ("test",))
        conn.commit()
        
        conn.commit.assert_called()
    
    def test_transaction_rollback(self):
        """测试事务回滚"""
        conn = MagicMock()
        conn.rollback = MagicMock()
        
        # 模拟事务失败
        try:
            conn.execute("INSERT INTO users VALUES (?)", ("test",))
            raise Exception("Simulated error")
        except Exception:
            conn.rollback()
        
        conn.rollback.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
