from typing import Dict, Any, Optional, Type
from abc import ABC, abstractmethod
import logging

class Service(ABC):
    """服务基类"""
    
    @abstractmethod
    async def start(self) -> bool:
        """启动服务"""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """停止服务"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """服务名称"""
        pass
    
    @property
    @abstractmethod
    def is_running(self) -> bool:
        """服务是否正在运行"""
        pass

class ServiceRegistry:
    """服务注册表，用于管理应用中的各种服务"""
    
    def __init__(self):
        self._services: Dict[str, Service] = {}
        self._logger = logging.getLogger("rox-backend.service-registry")
    
    def register(self, service: Service) -> bool:
        """注册服务"""
        if service.name in self._services:
            self._logger.warning(f"服务 {service.name} 已注册，跳过")
            return False
        
        self._services[service.name] = service
        self._logger.info(f"服务 {service.name} 注册成功")
        return True
    
    def unregister(self, service_name: str) -> bool:
        """注销服务"""
        if service_name not in self._services:
            self._logger.warning(f"服务 {service_name} 未注册，跳过")
            return False
        
        service = self._services[service_name]
        if service.is_running:
            self._logger.warning(f"服务 {service_name} 正在运行，先停止服务")
            import asyncio
            asyncio.create_task(service.stop())
        
        del self._services[service_name]
        self._logger.info(f"服务 {service_name} 注销成功")
        return True
    
    def get(self, service_name: str) -> Optional[Service]:
        """获取服务"""
        return self._services.get(service_name)
    
    async def start_all(self) -> Dict[str, bool]:
        """启动所有服务"""
        results = {}
        self._logger.info("开始启动所有服务")
        
        for service_name, service in self._services.items():
            try:
                success = await service.start()
                results[service_name] = success
                if success:
                    self._logger.info(f"服务 {service_name} 启动成功")
                else:
                    self._logger.error(f"服务 {service_name} 启动失败")
            except Exception as e:
                self._logger.error(f"服务 {service_name} 启动异常: {e}")
                results[service_name] = False
        
        self._logger.info("所有服务启动完成")
        return results
    
    async def stop_all(self) -> Dict[str, bool]:
        """停止所有服务"""
        results = {}
        self._logger.info("开始停止所有服务")
        
        for service_name, service in self._services.items():
            try:
                success = await service.stop()
                results[service_name] = success
                if success:
                    self._logger.info(f"服务 {service_name} 停止成功")
                else:
                    self._logger.error(f"服务 {service_name} 停止失败")
            except Exception as e:
                self._logger.error(f"服务 {service_name} 停止异常: {e}")
                results[service_name] = False
        
        self._logger.info("所有服务停止完成")
        return results
    
    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """列出所有服务的状态"""
        status = {}
        for service_name, service in self._services.items():
            status[service_name] = {
                "is_running": service.is_running,
                "service": service
            }
        return status
    
    def get_service_count(self) -> int:
        """获取服务数量"""
        return len(self._services)

# 创建全局服务注册表实例
service_registry = ServiceRegistry()
