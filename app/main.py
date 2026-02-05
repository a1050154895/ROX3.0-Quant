import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.core.config import settings
from app.db import init_db
from app.api.api import api_router

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


# ============ Lifespan 事件管理器 (FastAPI 0.95+ 推荐) ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器：使用现代 lifespan 模式替代废弃的 on_event 装饰器
    FastAPI 0.95+ 推荐使用此模式
    """
    # ========== STARTUP ==========
    logger.info(f"=== ROX Quant 应用启动 (环境: {settings.ENVIRONMENT}) ===")
    
    # 打印所有注册的路由
    logger.info("=" * 80)
    logger.info(" " * 25 + "REGISTERED ROUTES")
    logger.info("=" * 80)
    for route in app.routes:
        if hasattr(route, "methods"):
            logger.info(f"Path: {route.path}, Name: {route.name}, Methods: {route.methods}")
        else:
            logger.info(f"Path: {route.path}, Name: {route.name}")
    logger.info("=" * 80)
    
    # 初始化数据库
    try:
        init_db()
        logger.info("✓ 数据库初始化完成")
    except Exception as e:
        logger.error(f"✗ 数据库初始化失败: {e}")
        raise
    
    # 可选：启动调度器
    scheduler = None
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        logger.info("✓ 调度器启动成功")
    except ImportError:
        logger.debug("APScheduler未安装，跳过调度器启动")
    except Exception as e:
        logger.warning(f"⚠️  调度器启动失败: {e}")
        
    # 初始化事件总线和交易监听
    event_bus = None
    try:
        from app.core.event_bus import EventBus
        from app.api.endpoints.trade import setup_trade_listeners
        
        setup_trade_listeners()
        event_bus = EventBus()
        await event_bus.start_listening()
        logger.info("✓ 后端事件总线启动成功")
    except Exception as e:
        logger.warning(f"⚠️  事件总线启动失败: {e}")
    
    logger.info("✓ 系统启动完毕")
    
    yield  # 应用运行中
    
    # ========== SHUTDOWN ==========
    logger.info("=== ROX Quant 应用关闭 ===")
    
    # 清理资源
    if scheduler:
        try:
            scheduler.shutdown(wait=False)
            logger.info("✓ 调度器已关闭")
        except Exception as e:
            logger.warning(f"⚠️  调度器关闭失败: {e}")


# ============ FastAPI应用初始化 ============
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="3.0.0",
    description="ROX Quant 量化投研系统 API：市场数据、个股诊断、交易、知识库、策略回测等。",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan,  # 使用现代 lifespan 模式
)

# ============ 统一错误响应 ============
from app.core.errors import register_exception_handlers
register_exception_handlers(app)

# ============ API路由 ============
app.include_router(api_router)

# ============ CORS配置 ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 静态文件和模板配置 ============
static_path = os.path.join(settings.BASE_DIR, "app/static")
templates_path = os.path.join(settings.BASE_DIR, "app/templates")

if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

if os.path.exists(templates_path):
    templates = Jinja2Templates(directory=templates_path)
else:
    templates = None

# ============ 前端根路由 ============
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """默认首页：ROX 2.0 全新 UI"""
    if templates is None:
        return "<h1>Rox Quant</h1><p>模板未找到</p>"
    return templates.TemplateResponse("index_rox2.html", {"request": request})

@app.get("/classic", response_class=HTMLResponse)
async def read_classic(request: Request):
    """经典版：1.0 风格"""
    if templates is None:
        return "<h1>Rox Quant</h1><p>模板未找到</p>"
    return templates.TemplateResponse("index_rox1.html", {"request": request})

@app.get("/builder", response_class=HTMLResponse)
async def read_builder(request: Request):
    """可视化策略工坊"""
    if templates is None:
        return "<h1>Rox Quant</h1><p>模板未找到</p>"
    return templates.TemplateResponse("strategy_builder.html", {"request": request})


@app.get("/pro", response_class=HTMLResponse)
async def read_pro(request: Request):
    """专业版：3.0 紧凑三栏布局"""
    if templates is None:
        return "<h1>Rox Quant</h1><p>模板未找到</p>"
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/map", response_class=HTMLResponse)
async def read_market_map(request: Request):
    """全市场热力图"""
    if templates is None:
        return "<h1>Rox Quant</h1><p>模板未找到</p>"
    return templates.TemplateResponse("market_map.html", {"request": request})

# ============ 健康检查 ============
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "ok",
        "environment": settings.ENVIRONMENT,
        "version": "3.0.0"
    }

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
