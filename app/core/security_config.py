#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全配置模块
提供安全的配置管理和密钥生成功能
"""

import os
import secrets
import logging
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_secret_key(length: int = 64) -> str:
    """
    生成安全的随机密钥
    
    Args:
        length: 密钥长度（字节数）
    
    Returns:
        十六进制格式的安全密钥
    """
    return secrets.token_hex(length)


def get_or_create_secret_key(env_var: str = "SECRET_KEY", 
                             key_file: str = ".secret_key") -> str:
    """
    获取或创建SECRET_KEY
    
    优先级：
    1. 环境变量
    2. 密钥文件
    3. 自动生成并保存
    
    Args:
        env_var: 环境变量名
        key_file: 密钥文件名
    
    Returns:
        SECRET_KEY字符串
    """
    # 1. 检查环境变量
    key = os.getenv(env_var)
    if key and key != "change-me-in-production":
        logger.info(f"从环境变量 {env_var} 加载SECRET_KEY")
        return key
    
    # 2. 检查密钥文件
    key_path = Path(key_file)
    if key_path.exists():
        key = key_path.read_text().strip()
        if key:
            logger.info(f"从密钥文件 {key_file} 加载SECRET_KEY")
            return key
    
    # 3. 生成新密钥并保存
    new_key = generate_secret_key()
    
    try:
        # 确保目录存在
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(new_key)
        key_path.chmod(0o600)  # 仅所有者可读写
        logger.warning(f"已生成新的SECRET_KEY并保存到 {key_file}")
        logger.warning("请将此密钥妥善保管，或设置环境变量 SECRET_KEY")
    except Exception as e:
        logger.error(f"无法保存密钥文件: {e}")
    
    return new_key


class SecurityConfig:
    """安全配置类"""
    
    # 密码哈希算法
    PASSWORD_HASH_ALGORITHM = "bcrypt"
    
    # JWT配置
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7天
    
    # API安全配置
    API_RATE_LIMIT_REQUESTS = 100  # 每分钟请求数
    API_RATE_LIMIT_WINDOW = 60  # 窗口时间（秒）
    
    # 密码策略
    PASSWORD_MIN_LENGTH = 8
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_DIGIT = True
    PASSWORD_REQUIRE_SPECIAL = False
    
    # 会话配置
    SESSION_COOKIE_SECURE = True  # 仅HTTPS
    SESSION_COOKIE_HTTPONLY = True  # 防止XSS
    SESSION_COOKIE_SAMESITE = "lax"  # 防止CSRF
    
    @classmethod
    def validate_password(cls, password: str) -> tuple[bool, str]:
        """
        验证密码强度
        
        Args:
            password: 待验证的密码
        
        Returns:
            (是否有效, 错误信息)
        """
        if len(password) < cls.PASSWORD_MIN_LENGTH:
            return False, f"密码长度至少需要 {cls.PASSWORD_MIN_LENGTH} 个字符"
        
        if cls.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            return False, "密码需要包含至少一个大写字母"
        
        if cls.PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            return False, "密码需要包含至少一个小写字母"
        
        if cls.PASSWORD_REQUIRE_DIGIT and not any(c.isdigit() for c in password):
            return False, "密码需要包含至少一个数字"
        
        if cls.PASSWORD_REQUIRE_SPECIAL:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return False, "密码需要包含至少一个特殊字符"
        
        return True, ""


class Settings:
    """应用配置，支持环境变量和.env文件"""
    
    def __init__(self):
        # ============ 应用设置 ============
        self.PROJECT_NAME: str = "ROX Quant Trading System"
        self.API_V1_STR: str = "/api"
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
        self.DEBUG: bool = self.ENVIRONMENT == "development"
        
        # ============ 安全设置 ============
        self.SECRET_KEY: str = get_or_create_secret_key()
        self.ALGORITHM: str = SecurityConfig.JWT_ALGORITHM
        self.ACCESS_TOKEN_EXPIRE_MINUTES: int = SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        
        # ============ 路径设置 ============
        base_dir_env = os.getenv("BASE_DIR", "")
        if base_dir_env:
            self.BASE_DIR: str = base_dir_env
        else:
            # 计算项目根目录 (app/core/security_config.py -> rox3.0/)
            self.BASE_DIR: str = str(Path(__file__).parent.parent.parent)
        self.DATA_DIR: str = os.path.join(self.BASE_DIR, "data")
        self.DB_PATH: str = os.path.join(self.DATA_DIR, "docs.db")
        self.LOG_DIR: str = os.path.join(self.BASE_DIR, "logs")
        
        # ============ 数据库设置 ============
        self.DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{self.DB_PATH}")
        
        # ============ 缓存设置 ============
        self.REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))
        self.USE_REDIS: bool = os.getenv("USE_REDIS", "False").lower() == "true"
        
        # ============ 市场数据设置 ============
        self.AKSHARE_TIMEOUT: int = int(os.getenv("AKSHARE_TIMEOUT", "20"))
        self.BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
        
        # ============ 日志设置 ============
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG" if self.DEBUG else "INFO")
        self.LOG_FILE: str = os.getenv("LOG_FILE", os.path.join(self.LOG_DIR, "rox_quant.log"))
        
        # ============ CORS设置 ============
        self.ALLOWED_ORIGINS: List[str] = self._get_cors_origins()
        
        # ============ 桌面应用设置 ============
        self.DESKTOP_HOST: str = os.getenv("DESKTOP_HOST", "127.0.0.1")
        self.DESKTOP_PORT: int = int(os.getenv("DESKTOP_PORT", "8008"))
        
        # ============ 量化设置 ============
        self.BACKTEST_PARALLEL: bool = os.getenv("BACKTEST_PARALLEL", "True").lower() == "true"
        self.MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
        
        # ============ AI设置 ============
        self.AI_PROVIDER: str = os.getenv("AI_PROVIDER", "default")
        self.AI_API_KEY: str = os.getenv("AI_API_KEY", "").strip()
        self.AI_BASE_URL: str = os.getenv("AI_BASE_URL", "https://tb.api.mkeai.com").strip()
        self.AI_DEFAULT_MODEL: str = os.getenv("AI_DEFAULT_MODEL", "deepseek-chat")
        
        # ============ API安全设置 ============
        self.API_RATE_LIMIT_REQUESTS: int = int(os.getenv("API_RATE_LIMIT_REQUESTS", "100"))
        self.API_RATE_LIMIT_WINDOW: int = int(os.getenv("API_RATE_LIMIT_WINDOW", "60"))
        
        # 初始化
        self._validate_and_setup()
    
    def _get_cors_origins(self) -> List[str]:
        """获取CORS允许的源"""
        origins = os.getenv("CORS_ORIGINS", "")
        if origins:
            return [o.strip() for o in origins.split(",")]
        
        return [
            "http://localhost:8080",
            "http://127.0.0.1:8081",
            "http://localhost:8081",
            "http://127.0.0.1:8500",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    
    def _validate_and_setup(self):
        """验证配置并创建必要目录"""
        # 创建必要目录
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        
        # 生产环境安全检查
        if self.ENVIRONMENT == "production":
            self._validate_production_config()
    
    def _validate_production_config(self):
        """验证生产环境配置"""
        warnings = []
        
        if self.DEBUG:
            warnings.append("生产环境不应启用DEBUG模式")
        
        if not self.AI_API_KEY:
            warnings.append("生产环境建议配置AI_API_KEY")
        
        if not self.USE_REDIS:
            warnings.append("生产环境建议启用Redis缓存")
        
        for warning in warnings:
            logger.warning(f"⚠️ 配置警告: {warning}")
    
    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.ENVIRONMENT == "development"


# 单例实例
settings = Settings()
