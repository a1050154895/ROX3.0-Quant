#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局错误处理和系统监控模块

增强系统容错能力，减少崩溃风险，提供自动恢复机制
"""

import logging
import traceback
import sys
import time
import threading
try:
    import psutil
except ImportError:
    psutil = None
from datetime import datetime
from typing import Callable, Optional, Dict, Any, List
from functools import wraps

logger = logging.getLogger(__name__)


class SystemHealthMonitor:
    """
    系统健康监控器
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.error_count = 0
        self.warning_count = 0
        self.info_count = 0
        self.recovery_count = 0
        self.system_status = "healthy"
        self.last_health_check = datetime.now()
        self.resource_usage = {}
        self._lock = threading.RLock()
        self._health_check_interval = 60  # 秒
        self._health_check_thread = None
        self._start_health_check()
    
    def _start_health_check(self):
        """启动健康检查线程"""
        if not self._health_check_thread or not self._health_check_thread.is_alive():
            self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self._health_check_thread.start()
    
    def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                self.check_system_health()
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
            time.sleep(self._health_check_interval)
    
    def check_system_health(self):
        """检查系统健康状态"""
        with self._lock:
            # 检查系统资源
            if psutil is not None:
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent
            else:
                cpu_usage = 0.0
                memory_usage = 0.0
                disk_usage = 0.0
            
            self.resource_usage = {
                "cpu": cpu_usage,
                "memory": memory_usage,
                "disk": disk_usage,
                "timestamp": datetime.now().isoformat()
            }
            
            # 检查系统状态
            if cpu_usage > 90 or memory_usage > 90 or disk_usage > 95:
                self.system_status = "critical"
                logger.warning(f"系统资源紧张: CPU={cpu_usage}%, 内存={memory_usage}%, 磁盘={disk_usage}%")
            elif cpu_usage > 70 or memory_usage > 70 or disk_usage > 85:
                self.system_status = "warning"
                logger.warning(f"系统资源警告: CPU={cpu_usage}%, 内存={memory_usage}%, 磁盘={disk_usage}%")
            else:
                self.system_status = "healthy"
            
            self.last_health_check = datetime.now()
    
    def record_error(self, error: Exception, context: str = ""):
        """记录错误"""
        with self._lock:
            self.error_count += 1
            logger.error(f"错误记录 [{context}]: {error}")
            logger.debug(traceback.format_exc())
    
    def record_warning(self, message: str, context: str = ""):
        """记录警告"""
        with self._lock:
            self.warning_count += 1
            logger.warning(f"警告记录 [{context}]: {message}")
    
    def record_info(self, message: str, context: str = ""):
        """记录信息"""
        with self._lock:
            self.info_count += 1
            logger.info(f"信息记录 [{context}]: {message}")
    
    def record_recovery(self, error: Exception, context: str = ""):
        """记录恢复"""
        with self._lock:
            self.recovery_count += 1
            logger.info(f"错误恢复 [{context}]: {error}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self._lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            return {
                "status": self.system_status,
                "uptime": uptime,
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "info_count": self.info_count,
                "recovery_count": self.recovery_count,
                "last_health_check": self.last_health_check.isoformat(),
                "resource_usage": self.resource_usage,
                "start_time": self.start_time.isoformat()
            }


class ErrorHandler:
    """
    错误处理器
    """
    
    def __init__(self, health_monitor: SystemHealthMonitor):
        self.health_monitor = health_monitor
        self.error_handlers = {}
        self.register_default_handlers()
    
    def register_default_handlers(self):
        """注册默认错误处理器"""
        self.register_handler(Exception, self.default_exception_handler)
        self.register_handler(ValueError, self.value_error_handler)
        self.register_handler(TypeError, self.type_error_handler)
        self.register_handler(IOError, self.io_error_handler)
        self.register_handler(ConnectionError, self.connection_error_handler)
    
    def register_handler(self, error_type: type, handler: Callable):
        """
        注册错误处理器
        
        Args:
            error_type: 错误类型
            handler: 错误处理函数
        """
        self.error_handlers[error_type] = handler
    
    def handle_error(self, error: Exception, context: str = "") -> Optional[Any]:
        """
        处理错误
        
        Args:
            error: 错误对象
            context: 错误上下文
        
        Returns:
            错误处理结果
        """
        self.health_monitor.record_error(error, context)
        
        # 查找合适的错误处理器
        for error_type, handler in self.error_handlers.items():
            if isinstance(error, error_type):
                try:
                    result = handler(error, context)
                    self.health_monitor.record_recovery(error, context)
                    return result
                except Exception as e:
                    logger.error(f"错误处理器失败: {e}")
                    break
        
        # 使用默认处理器
        try:
            result = self.default_exception_handler(error, context)
            self.health_monitor.record_recovery(error, context)
            return result
        except Exception as e:
            logger.error(f"默认错误处理器失败: {e}")
            return None
    
    def default_exception_handler(self, error: Exception, context: str = "") -> None:
        """
        默认异常处理器
        
        Args:
            error: 错误对象
            context: 错误上下文
        """
        logger.error(f"未处理的异常 [{context}]: {error}")
        logger.debug(traceback.format_exc())
    
    def value_error_handler(self, error: ValueError, context: str = "") -> None:
        """
        值错误处理器
        
        Args:
            error: 错误对象
            context: 错误上下文
        """
        logger.warning(f"值错误 [{context}]: {error}")
    
    def type_error_handler(self, error: TypeError, context: str = "") -> None:
        """
        类型错误处理器
        
        Args:
            error: 错误对象
            context: 错误上下文
        """
        logger.warning(f"类型错误 [{context}]: {error}")
    
    def io_error_handler(self, error: IOError, context: str = "") -> None:
        """
        IO错误处理器
        
        Args:
            error: 错误对象
            context: 错误上下文
        """
        logger.warning(f"IO错误 [{context}]: {error}")
    
    def connection_error_handler(self, error: ConnectionError, context: str = "") -> None:
        """
        连接错误处理器
        
        Args:
            error: 错误对象
            context: 错误上下文
        """
        logger.warning(f"连接错误 [{context}]: {error}")
        # 可以在这里添加自动重连逻辑


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        delay: 初始延迟（秒）
        backoff: 延迟增长因子
    
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(f"函数 {func.__name__} 执行失败，已达到最大重试次数: {e}")
                        raise
                    
                    logger.warning(f"函数 {func.__name__} 执行失败，{attempts}/{max_attempts}，{current_delay}秒后重试: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

def safe_execute(default: Any = None, context: str = ""):
    """
    安全执行装饰器
    
    Args:
        default: 出错时的默认返回值
        context: 错误上下文
    
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                error_handler.handle_error(e, f"{context}:{func.__name__}")
                return default
        return wrapper
    return decorator

def setup_global_error_handler():
    """
    设置全局错误处理器
    """
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        """处理未捕获的异常"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical(f"未捕获的异常: {exc_value}")
        logger.debug(''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        
        # 记录到健康监控
        health_monitor = get_health_monitor()
        health_monitor.record_error(exc_value, "全局异常")
    
    sys.excepthook = handle_uncaught_exception


# 全局实例
_health_monitor = None
_error_handler = None


def get_health_monitor() -> SystemHealthMonitor:
    """
    获取健康监控器实例
    
    Returns:
        健康监控器实例
    """
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = SystemHealthMonitor()
    return _health_monitor


def get_error_handler() -> ErrorHandler:
    """
    获取错误处理器实例
    
    Returns:
        错误处理器实例
    """
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(get_health_monitor())
    return _error_handler


def initialize_error_handling():
    """
    初始化错误处理系统
    """
    setup_global_error_handler()
    # 启动健康监控
    get_health_monitor()
    logger.info("错误处理系统初始化完成")


def get_system_status() -> Dict[str, Any]:
    """
    获取系统状态
    
    Returns:
        系统状态
    """
    health_monitor = get_health_monitor()
    return health_monitor.get_status()
