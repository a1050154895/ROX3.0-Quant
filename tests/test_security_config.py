#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全配置模块测试
"""

import pytest
import os
import tempfile
from pathlib import Path

from app.core.security_config import (
    SecurityConfig,
    Settings,
    generate_secret_key,
    get_or_create_secret_key,
)


class TestSecurityConfig:
    """SecurityConfig测试类"""
    
    def test_password_validation_valid(self):
        """测试有效密码验证"""
        valid_passwords = [
            "Password123",
            "SecurePass1",
            "MyP@ssw0rd",
        ]
        
        for password in valid_passwords:
            is_valid, msg = SecurityConfig.validate_password(password)
            assert is_valid, f"密码 '{password}' 应该有效: {msg}"
    
    def test_password_validation_too_short(self):
        """测试密码长度不足"""
        is_valid, msg = SecurityConfig.validate_password("Pass1")
        assert not is_valid
        assert "长度" in msg
    
    def test_password_validation_no_uppercase(self):
        """测试缺少大写字母"""
        is_valid, msg = SecurityConfig.validate_password("password123")
        assert not is_valid
        assert "大写" in msg
    
    def test_password_validation_no_lowercase(self):
        """测试缺少小写字母"""
        is_valid, msg = SecurityConfig.validate_password("PASSWORD123")
        assert not is_valid
        assert "小写" in msg
    
    def test_password_validation_no_digit(self):
        """测试缺少数字"""
        is_valid, msg = SecurityConfig.validate_password("Password")
        assert not is_valid
        assert "数字" in msg
    
    def test_jwt_config(self):
        """测试JWT配置"""
        assert SecurityConfig.JWT_ALGORITHM == "HS256"
        assert SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES > 0
    
    def test_rate_limit_config(self):
        """测试速率限制配置"""
        assert SecurityConfig.API_RATE_LIMIT_REQUESTS > 0
        assert SecurityConfig.API_RATE_LIMIT_WINDOW > 0


class TestSecretKeyGeneration:
    """密钥生成测试类"""
    
    def test_generate_secret_key_length(self):
        """测试密钥长度"""
        key = generate_secret_key(32)
        assert len(key) == 64  # 32 bytes = 64 hex chars
        
        key = generate_secret_key(64)
        assert len(key) == 128  # 64 bytes = 128 hex chars
    
    def test_generate_secret_key_uniqueness(self):
        """测试密钥唯一性"""
        keys = [generate_secret_key() for _ in range(100)]
        assert len(set(keys)) == 100  # 所有密钥应该唯一
    
    def test_generate_secret_key_format(self):
        """测试密钥格式"""
        key = generate_secret_key()
        assert all(c in '0123456789abcdef' for c in key)


class TestGetOrCreateSecretKey:
    """密钥获取/创建测试类"""
    
    def test_from_environment(self):
        """测试从环境变量获取"""
        test_key = "test-env-secret-key-12345678"
        os.environ["TEST_SECRET_KEY"] = test_key
        
        key = get_or_create_secret_key(env_var="TEST_SECRET_KEY")
        assert key == test_key
        
        del os.environ["TEST_SECRET_KEY"]
    
    def test_from_file(self, monkeypatch):
        """测试从文件获取"""
        # 确保没有环境变量影响
        monkeypatch.delenv("SECRET_KEY", raising=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / ".test_key"
            test_key = "test-file-secret-key-12345678"
            key_file.write_text(test_key)
            
            key = get_or_create_secret_key(key_file=str(key_file))
            assert key == test_key
    
    def test_generate_new_key(self, monkeypatch):
        """测试生成新密钥"""
        # 确保没有环境变量影响
        monkeypatch.delenv("SECRET_KEY", raising=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / ".new_key"
            
            key = get_or_create_secret_key(key_file=str(key_file))
            assert len(key) > 0
            assert key_file.exists()
            assert key_file.read_text().strip() == key


class TestSettings:
    """Settings测试类"""
    
    def test_settings_creation(self):
        """测试配置创建"""
        settings = Settings()
        assert settings.PROJECT_NAME == "ROX Quant Trading System"
        assert settings.API_V1_STR == "/api"
    
    def test_settings_directories_created(self):
        """测试必要目录创建"""
        settings = Settings()
        assert os.path.exists(settings.LOG_DIR)
        assert os.path.exists(settings.DATA_DIR)
    
    def test_settings_secret_key_not_default(self):
        """测试SECRET_KEY不是默认值"""
        settings = Settings()
        assert settings.SECRET_KEY != "change-me-in-production"
    
    def test_environment_properties(self):
        """测试环境属性"""
        settings = Settings()
        
        # 测试环境判断
        if settings.ENVIRONMENT == "production":
            assert settings.is_production
            assert not settings.is_development
        else:
            assert not settings.is_production
    
    def test_cors_origins(self):
        """测试CORS配置"""
        settings = Settings()
        assert isinstance(settings.ALLOWED_ORIGINS, list)
        assert len(settings.ALLOWED_ORIGINS) > 0
    
    def test_rate_limit_settings(self):
        """测试速率限制设置"""
        settings = Settings()
        assert settings.API_RATE_LIMIT_REQUESTS > 0
        assert settings.API_RATE_LIMIT_WINDOW > 0


class TestSettingsValidation:
    """配置验证测试类"""
    
    def test_production_validation(self, monkeypatch):
        """测试生产环境验证"""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEBUG", "false")
        
        # 应该不抛出异常
        settings = Settings()
        assert settings.ENVIRONMENT == "production"
