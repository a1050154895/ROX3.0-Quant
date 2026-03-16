import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.service_registry import service_registry
from app.api.api import api_router
from app.db import init_db

class AppFactory:
    """应用工厂类，负责创建和配置FastAPI应用"""
    
    @staticmethod
    def create_app() -> FastAPI:
        """创建FastAPI应用实例"""
        app = FastAPI(
            title=settings.PROJECT_NAME,
            version="3.0.0",
            description="ROX Quant 量化投研系统 API：市场数据、个股诊断、交易、知识库、策略回测等。",
            docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
            redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
            lifespan=AppFactory.lifespan,
        )
        
        AppFactory._configure_middleware(app)
        AppFactory._register_routes(app)
        AppFactory._setup_static_files(app)
        AppFactory._setup_templates(app)
        
        return app
    
    @staticmethod
    def _configure_middleware(app: FastAPI):
        """配置中间件"""
        # 注册异常处理器
        register_exception_handlers(app)
        
        # 配置CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 可以在这里添加其他中间件，如安全中间件、请求日志中间件等
    
    @staticmethod
    def _register_routes(app: FastAPI):
        """注册路由"""
        # 注册API路由
        app.include_router(api_router)
        
        # 注册前端根路由
        AppFactory._register_frontend_routes(app)
        
        # 注册健康检查路由
        @app.get("/health")
        async def health_check():
            """健康检查端点"""
            return {
                "status": "ok",
                "environment": settings.ENVIRONMENT,
                "version": "3.0.0"
            }
    
    @staticmethod
    def _register_frontend_routes(app: FastAPI):
        """注册前端路由"""
        from fastapi.responses import HTMLResponse
        from fastapi import Request
        
        static_path = os.path.join(settings.BASE_DIR, "app/static")
        templates_path = os.path.join(settings.BASE_DIR, "app/templates")
        
        templates = None
        if os.path.exists(templates_path):
            templates = Jinja2Templates(directory=templates_path)
        
        @app.get("/", response_class=HTMLResponse)
        async def read_root(request: Request):
            """默认首页：ROX 3.0 全新落地页"""
            if templates is None:
                return "<h1>Rox Quant</h1><p>模板未找到</p>"
            return templates.TemplateResponse("landing.html", {"request": request})
        
        @app.get("/home", response_class=HTMLResponse)
        async def read_home(request: Request):
            """ROX 2.0 全新 UI"""
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
        
        @app.get("/knowledge", response_class=HTMLResponse)
        async def read_knowledge_center(request: Request):
            """知识中心"""
            if templates is None:
                return "<h1>Rox Quant</h1><p>模板未找到</p>"
            return templates.TemplateResponse("knowledge_center.html", {"request": request})
        
        @app.get("/strategies", response_class=HTMLResponse)
        async def read_strategy_center(request: Request):
            """策略中心"""
            if templates is None:
                return "<h1>Rox Quant</h1><p>模板未找到</p>"
            return templates.TemplateResponse("strategy_center.html", {"request": request})

        @app.get("/lu", response_class=HTMLResponse)
        async def read_lu_war_room(request: Request):
            """卢式作战室：半自动辅助决策系统"""
            if templates is None:
                return "<h1>Rox Quant</h1><p>模板未找到</p>"
            return templates.TemplateResponse("lu_dashboard.html", {"request": request})
    
    @staticmethod
    def _setup_static_files(app: FastAPI):
        """设置静态文件"""
        static_path = os.path.join(settings.BASE_DIR, "app/static")
        if os.path.exists(static_path):
            app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    @staticmethod
    def _setup_templates(app: FastAPI):
        """设置模板"""
        templates_path = os.path.join(settings.BASE_DIR, "app/templates")
        if os.path.exists(templates_path):
            app.state.templates = Jinja2Templates(directory=templates_path)
        else:
            app.state.templates = None
    
    @staticmethod
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        应用生命周期管理器：使用现代 lifespan 模式替代废弃的 on_event 装饰器
        FastAPI 0.95+ 推荐使用此模式
        """
        # ========== STARTUP ==========
        logger = logging.getLogger("rox-backend")
        logger.info(f"=== ROX Quant 应用启动 (环境: {settings.ENVIRONMENT}) ===")
        
        # 可选：打印所有注册路由（默认关闭，避免启动日志噪音）
        if settings.VERBOSE_ROUTE_LOGGING:
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
        
        # 注册和启动核心服务
        try:
            AppFactory._register_core_services()
            logger.info("✓ 核心服务注册完成")
            
            # 启动所有注册的服务
            service_results = await service_registry.start_all()
            logger.info(f"✓ 服务启动结果: {service_results}")
        except Exception as e:
            logger.error(f"✗ 服务注册或启动失败: {e}")
            raise
        
        logger.info("✓ 系统启动完毕")
        
        yield  # 应用运行中
        
        # ========== SHUTDOWN ==========
        logger.info("=== ROX Quant 应用关闭 ===")
        
        # 停止所有服务
        try:
            service_stop_results = await service_registry.stop_all()
            logger.info(f"✓ 服务停止结果: {service_stop_results}")
        except Exception as e:
            logger.error(f"✗ 服务停止失败: {e}")
    
    @staticmethod
    def _register_core_services():
        """注册核心服务"""
        logger = logging.getLogger("rox-backend")
        
        # 注册调度器服务
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            
            class SchedulerService:
                def __init__(self):
                    self._scheduler = None
                    self._is_running = False
                
                async def start(self) -> bool:
                    try:
                        self._scheduler = AsyncIOScheduler()
                        self._scheduler.start()
                        self._is_running = True
                        return True
                    except Exception as e:
                        logger.error(f"调度器服务启动失败: {e}")
                        return False
                
                async def stop(self) -> bool:
                    try:
                        if self._scheduler:
                            self._scheduler.shutdown(wait=False)
                            self._is_running = False
                            return True
                        return False
                    except Exception as e:
                        logger.error(f"调度器服务停止失败: {e}")
                        return False
                
                @property
                def name(self) -> str:
                    return "scheduler"
                
                @property
                def is_running(self) -> bool:
                    return self._is_running
                
                @property
                def scheduler(self):
                    return self._scheduler
            
            scheduler_service = SchedulerService()
            service_registry.register(scheduler_service)
        except ImportError:
            logger.debug("APScheduler未安装，跳过调度器服务注册")
        except Exception as e:
            logger.warning(f"⚠️  调度器服务注册失败: {e}")
        
        # 注册事件总线服务
        try:
            from app.core.event_bus import EventBus
            from app.api.endpoints.trade import setup_trade_listeners
            
            class EventBusService:
                def __init__(self):
                    self._event_bus = None
                    self._is_running = False
                
                async def start(self) -> bool:
                    try:
                        setup_trade_listeners()
                        self._event_bus = EventBus()
                        await self._event_bus.start_listening()
                        self._is_running = True
                        return True
                    except Exception as e:
                        logger.error(f"事件总线服务启动失败: {e}")
                        return False
                
                async def stop(self) -> bool:
                    try:
                        if self._event_bus:
                            # 这里可以添加事件总线的停止逻辑
                            self._is_running = False
                            return True
                        return False
                    except Exception as e:
                        logger.error(f"事件总线服务停止失败: {e}")
                        return False
                
                @property
                def name(self) -> str:
                    return "event_bus"
                
                @property
                def is_running(self) -> bool:
                    return self._is_running
                
                @property
                def event_bus(self):
                    return self._event_bus
            
            event_bus_service = EventBusService()
            service_registry.register(event_bus_service)
        except Exception as e:
            logger.warning(f"⚠️  事件总线服务注册失败: {e}")
        
        # 可以在这里注册其他核心服务，如缓存服务、数据同步服务等
