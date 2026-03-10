import os
from typing import List


class Settings:
    """应用配置，支持环境变量和.env文件"""
    
    # ============ 应用设置 ============
    PROJECT_NAME: str = "ROX Quant Trading System"
    API_V1_STR: str = "/api"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # ============ 安全设置 ============
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))
    
    # ============ 路径设置 ============
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    DB_PATH: str = os.path.join(DATA_DIR, "docs.db")
    LOG_DIR: str = os.path.join(BASE_DIR, "logs")
    
    # ============ 数据库设置 ============
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'docs.db')}")
    
    # ============ 缓存设置 ============
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))
    USE_REDIS: bool = os.getenv("USE_REDIS", "False").lower() == "true"
    
    # ============ 市场数据设置 ============
    AKSHARE_TIMEOUT: int = int(os.getenv("AKSHARE_TIMEOUT", "20"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    
    # ============ 日志设置 ============
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO" if os.getenv("ENVIRONMENT", "development") == "production" else "DEBUG")
    LOG_FILE: str = os.getenv("LOG_FILE", os.path.join(LOG_DIR, "rox_quant.log"))
    
    # ============ CORS设置 ============
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8080",
        "http://127.0.0.1:8081",
        "http://localhost:8081",
        "http://127.0.0.1:8500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # ============ 桌面应用设置 ============
    DESKTOP_HOST: str = os.getenv("DESKTOP_HOST", "127.0.0.1")
    DESKTOP_PORT: int = int(os.getenv("DESKTOP_PORT", "8008"))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    
    # ============ 量化设置 ============
    BACKTEST_PARALLEL: bool = os.getenv("BACKTEST_PARALLEL", "True").lower() == "true"
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))

    # ============ AI 多模型/多平台（参考 go-stock） ============
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "default")
    AI_API_KEY: str = os.getenv("AI_API_KEY", "").strip()
    AI_BASE_URL: str = os.getenv("AI_BASE_URL", "https://tb.api.mkeai.com").strip()
    AI_DEFAULT_MODEL: str = os.getenv("AI_DEFAULT_MODEL", "deepseek-chat")

    # ============ TradingAgents-CN 集成配置 ============
    TRADING_AGENTS_ENABLED: bool = os.getenv("TRADING_AGENTS_ENABLED", "False").lower() == "true"
    TRADING_AGENTS_BASE_URL: str = os.getenv("TRADING_AGENTS_BASE_URL", "").strip()
    TRADING_AGENTS_API_KEY: str = os.getenv("TRADING_AGENTS_API_KEY", "").strip()
    TRADING_AGENTS_TIMEOUT: int = int(os.getenv("TRADING_AGENTS_TIMEOUT", "30"))
    TRADING_AGENTS_ENDPOINT: str = os.getenv("TRADING_AGENTS_ENDPOINT", "/analyze").strip() or "/analyze"
    TRADING_AGENTS_RETRY_COUNT: int = int(os.getenv("TRADING_AGENTS_RETRY_COUNT", "2"))
    TRADING_AGENTS_RETRY_BACKOFF: float = float(os.getenv("TRADING_AGENTS_RETRY_BACKOFF", "0.5"))
    TRADING_AGENTS_FALLBACK_LOCAL: bool = os.getenv("TRADING_AGENTS_FALLBACK_LOCAL", "True").lower() == "true"
    VERBOSE_ROUTE_LOGGING: bool = os.getenv("VERBOSE_ROUTE_LOGGING", "False").lower() == "true"

    def __init__(self):
        """验证配置并创建必要目录"""
        # 安全检查：生产环境必须更改SECRET_KEY
        if self.SECRET_KEY == "change-me-in-production":
            if self.ENVIRONMENT == "production":
                raise ValueError("⚠️ 生产环境必须设置安全的SECRET_KEY")
            else:
                import logging
                logging.warning("⚠️ 当前使用默认的不安全 SECRET_KEY。在生产环境中请务必通过环境变量设置。")
        
        # 创建日志目录
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)


# 单例实例
settings = Settings()
