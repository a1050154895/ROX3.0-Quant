import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.core.config import settings
from app.core.app_factory import AppFactory
from app.ai.decision_assistant import AIDecisionAssistant
from app.rox_quant.market_analysis import MarketAnalyzer

# ============ 日志配置 ============
def setup_logging():
    """环境感知的日志配置"""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # 创建日志目录
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    
    # 日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 文件处理器
    file_handler = logging.FileHandler(
        settings.LOG_FILE,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    # 根日志配置
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger("rox-backend")

logger = setup_logging()

# 初始化AI决策助手和市场分析器
ai_assistant = AIDecisionAssistant()
market_analyzer = MarketAnalyzer()

# 使用AppFactory创建FastAPI应用
app = AppFactory.create_app()

if __name__ == "__main__":
    import uvicorn
    
    # 根据环境选择日志级别
    uvicorn_log_level = "debug" if settings.ENVIRONMENT == "development" else "info"
    
    logger.info(f"启动 FastAPI 服务器: {settings.DESKTOP_HOST}:{settings.DESKTOP_PORT}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.DESKTOP_HOST,
        port=settings.DESKTOP_PORT,
        log_level=uvicorn_log_level,
        access_log=settings.ENVIRONMENT == "development",
        reload=settings.ENVIRONMENT == "development"
    )
