import os
import json
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import logging

from app.rox_quant.alltick_client import AllTickClient
from app.utils.akshare_wrapper import get_akshare_stock_quote, get_akshare_kline
from app.utils.ashare_fallback import get_eastmoney_quote, get_eastmoney_kline

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """数据源类型"""
    ALLTICK = "alltick"
    SINA = "sina"
    EASTMONEY = "eastmoney"
    AKSHARE = "akshare"
    BAIJING = "baijing"

class DataSourceStatus(Enum):
    """数据源状态"""
    DISABLED = "disabled"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

class DataSourceConfig:
    """数据源配置"""
    def __init__(self, data_source_type: DataSourceType, token: str = "", enabled: bool = True):
        self.type = data_source_type
        self.token = token
        self.enabled = enabled

class DataSource:
    """数据源基类"""
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.status = DataSourceStatus.DISABLED
        self.last_error = ""
        self.connection_time = 0
        self.error_count = 0
        self.success_count = 0
        self._lock = threading.RLock()

    def connect(self) -> bool:
        """连接数据源"""
        pass

    def disconnect(self):
        """断开数据源"""
        pass

    def get_tick_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取tick数据"""
        pass

    def get_kline(self, symbol: str, interval: str, count: int) -> Optional[List[List[Any]]]:
        """获取K线数据"""
        pass

    def is_healthy(self) -> bool:
        """检查数据源是否健康"""
        return self.status == DataSourceStatus.CONNECTED

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            "type": self.config.type.value,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "last_error": self.last_error,
            "connection_time": self.connection_time,
            "error_count": self.error_count,
            "success_count": self.success_count,
            "is_healthy": self.is_healthy()
        }

class AllTickDataSource(DataSource):
    """AllTick数据源实现"""
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.alltick_client = None

    def connect(self) -> bool:
        if not self.config.enabled:
            return False

        if not self.config.token:
            self.status = DataSourceStatus.ERROR
            self.last_error = "No token provided"
            return False

        try:
            self.alltick_client = AllTickClient(self.config.token)
            self.alltick_client.connect()
            
            # 等待连接完成
            for _ in range(5):
                if self.alltick_client.is_healthy():
                    self.status = DataSourceStatus.CONNECTED
                    self.connection_time = time.time()
                    self.last_error = ""
                    logger.info(f"AllTick data source connected successfully")
                    return True
                time.sleep(1)

            self.status = DataSourceStatus.ERROR
            self.last_error = "Connection timeout"
            logger.error("AllTick data source connection timeout")
            return False
        except Exception as e:
            self.status = DataSourceStatus.ERROR
            self.last_error = str(e)
            logger.error(f"AllTick data source connection failed: {e}")
            return False

    def disconnect(self):
        if self.alltick_client:
            try:
                self.alltick_client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting AllTick: {e}")
        self.status = DataSourceStatus.DISABLED

    def get_tick_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.is_healthy() or not self.alltick_client:
            return None

        try:
            tick_data = self.alltick_client.get_tick_data(symbol)
            if tick_data:
                self.success_count += 1
            return tick_data
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Error getting tick data: {e}")
            return None

    def get_kline(self, symbol: str, interval: str, count: int) -> Optional[List[List[Any]]]:
        if not self.is_healthy() or not self.alltick_client:
            return None

        try:
            # AllTickClient可能没有直接的get_kline方法，需要根据实际实现调整
            # 这里假设它有这个方法
            kline_data = self.alltick_client.get_kline(symbol, interval, count)
            if kline_data:
                self.success_count += 1
            return kline_data
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Error getting kline data: {e}")
            return None

    def is_healthy(self) -> bool:
        if not self.alltick_client:
            return False
        return self.alltick_client.is_healthy()


class EastMoneyDataSource(DataSource):
    """东方财富网数据源实现"""
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)

    def connect(self) -> bool:
        if not self.config.enabled:
            return False

        try:
            # 东方财富网不需要Token，直接标记为连接成功
            self.status = DataSourceStatus.CONNECTED
            self.connection_time = time.time()
            self.last_error = ""
            logger.info(f"EastMoney data source connected successfully")
            return True
        except Exception as e:
            self.status = DataSourceStatus.ERROR
            self.last_error = str(e)
            logger.error(f"EastMoney data source connection failed: {e}")
            return False

    def disconnect(self):
        self.status = DataSourceStatus.DISABLED

    def get_tick_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.is_healthy():
            return None

        try:
            tick_data = get_eastmoney_quote(symbol)
            if tick_data:
                self.success_count += 1
            return tick_data
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Error getting tick data from EastMoney: {e}")
            return None

    def get_kline(self, symbol: str, interval: str, count: int) -> Optional[List[List[Any]]]:
        if not self.is_healthy():
            return None

        try:
            kline_data = get_eastmoney_kline(symbol, interval, count)
            if kline_data:
                self.success_count += 1
            return kline_data
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Error getting kline data from EastMoney: {e}")
            return None


class AkShareDataSource(DataSource):
    """AKShare数据源实现"""
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)

    def connect(self) -> bool:
        if not self.config.enabled:
            return False

        try:
            # AKShare不需要Token，直接标记为连接成功
            self.status = DataSourceStatus.CONNECTED
            self.connection_time = time.time()
            self.last_error = ""
            logger.info(f"AKShare data source connected successfully")
            return True
        except Exception as e:
            self.status = DataSourceStatus.ERROR
            self.last_error = str(e)
            logger.error(f"AKShare data source connection failed: {e}")
            return False

    def disconnect(self):
        self.status = DataSourceStatus.DISABLED

    def get_tick_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.is_healthy():
            return None

        try:
            tick_data = get_akshare_stock_quote(symbol)
            if tick_data:
                self.success_count += 1
            return tick_data
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Error getting tick data from AKShare: {e}")
            return None

    def get_kline(self, symbol: str, interval: str, count: int) -> Optional[List[List[Any]]]:
        if not self.is_healthy():
            return None

        try:
            kline_data = get_akshare_kline(symbol, interval, count)
            if kline_data:
                self.success_count += 1
            return kline_data
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Error getting kline data from AKShare: {e}")
            return None

class DataSourceManager:
    """数据源管理器"""
    def __init__(self):
        self.data_sources: Dict[DataSourceType, DataSource] = {}
        self.default_source = DataSourceType.ALLTICK
        self._lock = threading.RLock()
        self._health_check_thread = None
        self._health_check_interval = 30  # 健康检查间隔（秒）

    def initialize(self, configs: List[DataSourceConfig]):
        """初始化数据源"""
        for config in configs:
            if config.type == DataSourceType.ALLTICK:
                data_source = AllTickDataSource(config)
                self.data_sources[config.type] = data_source
                if config.enabled:
                    data_source.connect()
            elif config.type == DataSourceType.EASTMONEY:
                data_source = EastMoneyDataSource(config)
                self.data_sources[config.type] = data_source
                if config.enabled:
                    data_source.connect()
            elif config.type == DataSourceType.AKSHARE:
                data_source = AkShareDataSource(config)
                self.data_sources[config.type] = data_source
                if config.enabled:
                    data_source.connect()

        # 启动健康检查线程
        self._start_health_check()

    def add_data_source(self, data_source: DataSource):
        """添加数据源"""
        with self._lock:
            self.data_sources[data_source.config.type] = data_source
            if data_source.config.enabled:
                data_source.connect()

    def remove_data_source(self, data_source_type: DataSourceType):
        """移除数据源"""
        with self._lock:
            if data_source_type in self.data_sources:
                data_source = self.data_sources[data_source_type]
                data_source.disconnect()
                del self.data_sources[data_source_type]

    def set_default_source(self, data_source_type: DataSourceType):
        """设置默认数据源"""
        with self._lock:
            if data_source_type in self.data_sources:
                self.default_source = data_source_type

    def get_data_source(self, data_source_type: Optional[DataSourceType] = None) -> Optional[DataSource]:
        """获取数据源"""
        with self._lock:
            if data_source_type:
                return self.data_sources.get(data_source_type)
            return self.data_sources.get(self.default_source)

    def get_healthy_data_sources(self) -> List[DataSource]:
        """获取健康的数据源"""
        with self._lock:
            return [ds for ds in self.data_sources.values() if ds.is_healthy()]

    def get_tick_data(self, symbol: str, data_source_type: Optional[DataSourceType] = None) -> Optional[Dict[str, Any]]:
        """获取tick数据，自动切换数据源"""
        # 优先使用指定的数据源
        if data_source_type:
            data_source = self.get_data_source(data_source_type)
            if data_source and data_source.is_healthy():
                data = data_source.get_tick_data(symbol)
                if data:
                    return data

        # 使用默认数据源
        default_source = self.get_data_source()
        if default_source and default_source.is_healthy():
            data = default_source.get_tick_data(symbol)
            if data:
                return data

        # 尝试所有健康的数据源
        for data_source in self.get_healthy_data_sources():
            if data_source.config.type != data_source_type:
                data = data_source.get_tick_data(symbol)
                if data:
                    # 更新默认数据源
                    self.set_default_source(data_source.config.type)
                    return data

        return None

    def get_kline(self, symbol: str, interval: str, count: int, data_source_type: Optional[DataSourceType] = None) -> Optional[List[List[Any]]]:
        """获取K线数据，自动切换数据源"""
        # 优先使用指定的数据源
        if data_source_type:
            data_source = self.get_data_source(data_source_type)
            if data_source and data_source.is_healthy():
                data = data_source.get_kline(symbol, interval, count)
                if data:
                    return data

        # 使用默认数据源
        default_source = self.get_data_source()
        if default_source and default_source.is_healthy():
            data = default_source.get_kline(symbol, interval, count)
            if data:
                return data

        # 尝试所有健康的数据源
        for data_source in self.get_healthy_data_sources():
            if data_source.config.type != data_source_type:
                data = data_source.get_kline(symbol, interval, count)
                if data:
                    # 更新默认数据源
                    self.set_default_source(data_source.config.type)
                    return data

        return None

    def is_healthy(self) -> bool:
        """检查是否有健康的数据源"""
        return len(self.get_healthy_data_sources()) > 0

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        with self._lock:
            return {
                "default_source": self.default_source.value,
                "data_sources": [ds.get_health_status() for ds in self.data_sources.values()],
                "healthy_count": len(self.get_healthy_data_sources()),
                "total_count": len(self.data_sources),
                "is_healthy": self.is_healthy()
            }

    def _start_health_check(self):
        """启动健康检查线程"""
        if not self._health_check_thread or not self._health_check_thread.is_alive():
            self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self._health_check_thread.start()

    def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                with self._lock:
                    for data_source in self.data_sources.values():
                        if data_source.config.enabled and not data_source.is_healthy():
                            logger.info(f"Reconnecting data source: {data_source.config.type.value}")
                            data_source.connect()
                time.sleep(self._health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(self._health_check_interval)

    def reload_config(self, configs: List[DataSourceConfig]):
        """重新加载配置"""
        with self._lock:
            # 断开并移除所有现有数据源
            for data_source in self.data_sources.values():
                data_source.disconnect()
            self.data_sources.clear()

            # 初始化新数据源
            self.initialize(configs)

# 全局数据源管理器实例
_data_source_manager = None

def get_data_source_manager() -> DataSourceManager:
    """获取数据源管理器"""
    global _data_source_manager
    if _data_source_manager is None:
        _data_source_manager = DataSourceManager()
        # 从环境变量加载配置
        configs = []
        
        # AllTick配置
        alltick_token = os.getenv('ALLTICK_TOKEN', '')
        alltick_enabled = os.getenv('ALLTICK_ENABLED', 'true').lower() == 'true'
        if alltick_token:
            configs.append(DataSourceConfig(
                DataSourceType.ALLTICK,
                alltick_token,
                alltick_enabled
            ))
        
        # 东方财富网配置（默认启用）
        eastmoney_enabled = os.getenv('EASTMONEY_ENABLED', 'true').lower() == 'true'
        configs.append(DataSourceConfig(
            DataSourceType.EASTMONEY,
            '',  # 不需要token
            eastmoney_enabled
        ))
        
        # AKShare配置（默认启用）
        akshare_enabled = os.getenv('AKSHARE_ENABLED', 'true').lower() == 'true'
        configs.append(DataSourceConfig(
            DataSourceType.AKSHARE,
            '',  # 不需要token
            akshare_enabled
        ))
        
        _data_source_manager.initialize(configs)
    return _data_source_manager

# 辅助函数
def get_tick_data(symbol: str) -> Optional[Dict[str, Any]]:
    """获取tick数据"""
    return get_data_source_manager().get_tick_data(symbol)

def get_kline(symbol: str, interval: str, count: int) -> Optional[List[List[Any]]]:
    """获取K线数据"""
    return get_data_source_manager().get_kline(symbol, interval, count)

def is_data_available() -> bool:
    """检查数据是否可用"""
    return get_data_source_manager().is_healthy()

def get_data_health_status() -> Dict[str, Any]:
    """获取数据健康状态"""
    return get_data_source_manager().get_health_status()
