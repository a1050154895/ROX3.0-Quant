import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, List

logger = logging.getLogger("rox-trader")

class BaseTrader(ABC):
    """
    交易执行器基类 (Abstract Base Class)
    定义了实盘/模拟盘的标准接口
    """
    
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """连接交易账户"""
        pass

    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """获取资金状况: {total, available, market_value}"""
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """获取持仓列表"""
        pass

    @abstractmethod
    def buy(self, security: str, price: float, amount: int) -> Dict:
        """买入接口"""
        pass

    @abstractmethod
    def sell(self, security: str, price: float, amount: int) -> Dict:
        """卖出接口"""
        pass

class EasyTraderAdapter(BaseTrader):
    """
    EasyTrader 适配器
    封装 easytrader 的调用逻辑
    """
    def __init__(self, broker: str = 'ths', read_only: bool = True, max_retries: int = 3):
        self.user = None
        self.broker = broker
        self.read_only = read_only
        self._is_connected = False
        self.max_retries = max_retries
        self.retry_delay = 2  # 重试延迟（秒）
        self.client_path = None
        self.connect_kwargs = {}

    def connect(self, client_path: str = None, **kwargs) -> bool:
        self.client_path = client_path
        self.connect_kwargs = kwargs
        return self._connect_with_retry()

    def _connect_with_retry(self, retry_count: int = 0) -> bool:
        try:
            import easytrader
            # 支持: ths (同花顺), yh (银河), xq (雪球), etc.
            self.user = easytrader.use(self.broker)
            
            # 连接客户端
            if self.broker == 'xq':
                # 雪球需要 user.prepare('user.json') 或 user.prepare(user='...', password='...')
                # 为了安全，建议使用 cookie 文件
                # kwargs: {'config_path': 'xq.json'}
                config = self.connect_kwargs.get('config_path', 'xq.json')
                self.user.prepare(config)
            elif self.client_path:
                self.user.connect(self.client_path)
            else:
                # 尝试自动查找或无需路径
                self.user.connect()
                
            self._is_connected = True
            logger.info(f"EasyTrader connected to {self.broker} (ReadOnly: {self.read_only})")
            return True
        except ImportError:
            logger.error("easytrader module not found. Please pip install easytrader")
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            if retry_count < self.max_retries:
                import time
                logger.info(f"Retrying connection ({retry_count + 1}/{self.max_retries})...")
                time.sleep(self.retry_delay)
                return self._connect_with_retry(retry_count + 1)
            return False

    def _ensure_connected(self) -> bool:
        """确保连接状态，如果断开则尝试重连"""
        if not self._is_connected:
            logger.warning("Connection lost, attempting to reconnect...")
            return self._connect_with_retry()
        return True

    def get_balance(self) -> Dict[str, float]:
        if not self._ensure_connected(): return {}
        try:
            # easytrader 返回格式通常为列表或字典
            balance = self.user.balance
            # 标准化返回格式
            if isinstance(balance, list) and len(balance) > 0:
                data = balance[0]
                return {
                    "total": float(data.get('资产总值', 0)),
                    "available": float(data.get('可用金额', 0)),
                    "market_value": float(data.get('证券市值', 0))
                }
            return {}
        except Exception as e:
            logger.error(f"Get balance error: {e}")
            self._is_connected = False  # 标记连接已断开
            return {}

    def get_positions(self) -> List[Dict]:
        if not self._ensure_connected(): return []
        try:
            return self.user.position
        except Exception as e:
            logger.error(f"Get position error: {e}")
            self._is_connected = False  # 标记连接已断开
            return []

    def buy(self, security: str, price: float, amount: int) -> Dict:
        if not self._ensure_connected(): return {"status": "error", "msg": "Not connected", "code": "CONNECTION_ERROR"}
        if self.read_only:
            logger.warning(f"[READ-ONLY] Blocked BUY request: {security}, {amount} @ {price}")
            return {"status": "blocked", "msg": "Read-only mode enabled", "code": "READ_ONLY"}
        
        # 验证交易参数
        if not security or not isinstance(security, str):
            return {"status": "error", "msg": "Invalid security code", "code": "INVALID_PARAMETER"}
        if price <= 0:
            return {"status": "error", "msg": "Invalid price", "code": "INVALID_PARAMETER"}
        if amount <= 0 or amount % 100 != 0:  # 假设A股交易单位为100股
            return {"status": "error", "msg": "Invalid amount", "code": "INVALID_PARAMETER"}
            
        for retry in range(self.max_retries):
            try:
                logger.info(f"Executing BUY: {security}, {amount} shares @ {price} (Attempt {retry + 1}/{self.max_retries})")
                # easytrader 接口: user.buy('162411', price=0.55, amount=100)
                res = self.user.buy(security, price=price, amount=amount)
                logger.info(f"BUY execution result: {res}")
                return {"status": "sent", "raw": res, "order_id": self._extract_order_id(res)}
            except Exception as e:
                error_msg = str(e)
                logger.error(f"BUY execution failed: {error_msg}")
                if retry < self.max_retries - 1:
                    import time
                    logger.info(f"Retrying BUY ({retry + 2}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                else:
                    error_code = self._map_error_code(error_msg)
                    return {"status": "error", "msg": error_msg, "code": error_code}

    def sell(self, security: str, price: float, amount: int) -> Dict:
        if not self._ensure_connected(): return {"status": "error", "msg": "Not connected", "code": "CONNECTION_ERROR"}
        if self.read_only:
            logger.warning(f"[READ-ONLY] Blocked SELL request: {security}, {amount} @ {price}")
            return {"status": "blocked", "msg": "Read-only mode enabled", "code": "READ_ONLY"}
        
        # 验证交易参数
        if not security or not isinstance(security, str):
            return {"status": "error", "msg": "Invalid security code", "code": "INVALID_PARAMETER"}
        if price <= 0:
            return {"status": "error", "msg": "Invalid price", "code": "INVALID_PARAMETER"}
        if amount <= 0 or amount % 100 != 0:  # 假设A股交易单位为100股
            return {"status": "error", "msg": "Invalid amount", "code": "INVALID_PARAMETER"}
            
        for retry in range(self.max_retries):
            try:
                logger.info(f"Executing SELL: {security}, {amount} shares @ {price} (Attempt {retry + 1}/{self.max_retries})")
                res = self.user.sell(security, price=price, amount=amount)
                logger.info(f"SELL execution result: {res}")
                return {"status": "sent", "raw": res, "order_id": self._extract_order_id(res)}
            except Exception as e:
                error_msg = str(e)
                logger.error(f"SELL execution failed: {error_msg}")
                if retry < self.max_retries - 1:
                    import time
                    logger.info(f"Retrying SELL ({retry + 2}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                else:
                    error_code = self._map_error_code(error_msg)
                    return {"status": "error", "msg": error_msg, "code": error_code}

    def _extract_order_id(self, response) -> Optional[str]:
        """从响应中提取订单ID"""
        try:
            if isinstance(response, dict):
                return response.get('order_id') or response.get('订单号') or response.get('委托编号')
            elif isinstance(response, str):
                # 尝试从字符串中提取订单号
                import re
                match = re.search(r'(订单号|委托编号):?\s*([\d]+)', response)
                if match:
                    return match.group(2)
            return None
        except Exception:
            return None

    def _map_error_code(self, error_msg: str) -> str:
        """将错误信息映射到标准错误代码"""
        error_msg_lower = error_msg.lower()
        if 'not connected' in error_msg_lower or 'connection' in error_msg_lower:
            return 'CONNECTION_ERROR'
        elif 'insufficient' in error_msg_lower or '资金' in error_msg or '余额' in error_msg:
            return 'INSUFFICIENT_FUNDS'
        elif 'position' in error_msg_lower or '持仓' in error_msg:
            return 'INSUFFICIENT_POSITIONS'
        elif 'price' in error_msg_lower or '价格' in error_msg:
            return 'INVALID_PRICE'
        elif 'amount' in error_msg_lower or '数量' in error_msg:
            return 'INVALID_AMOUNT'
        elif 'market' in error_msg_lower or '交易时间' in error_msg or '闭市' in error_msg:
            return 'MARKET_CLOSED'
        else:
            return 'UNKNOWN_ERROR'

class MockTrader(BaseTrader):
    """
    模拟交易器 (用于开发测试)
    """
    def connect(self, **kwargs):
        return True
        
    def get_balance(self):
        return {"total": 100000.0, "available": 50000.0, "market_value": 50000.0}
        
    def get_positions(self):
        return [{"stock_code": "000001", "stock_name": "平安银行", "current_amount": 1000, "cost_price": 10.5, "market_value": 10800}]
        
    def buy(self, security, price, amount):
        logger.info(f"[MOCK] Buying {security}: {amount} @ {price}")
        return {"status": "mock_success"}
        
    def sell(self, security, price, amount):
        logger.info(f"[MOCK] Selling {security}: {amount} @ {price}")
        return {"status": "mock_success"}

# 工厂模式
def get_trader(mode: str = 'mock', **kwargs) -> BaseTrader:
    if mode == 'real':
        # Load from environment variables if not provided
        import os
        read_only = kwargs.get('read_only', os.getenv('EASYTRADER_READ_ONLY', 'True').lower() == 'true')
        broker = kwargs.get('broker', os.getenv('EASYTRADER_BROKER', 'ths'))
        
        # Log configuration status
        logger.info(f"Initializing Real Trader: Broker={broker}, ReadOnly={read_only}")
        
        return EasyTraderAdapter(broker=broker, read_only=read_only)
    return MockTrader()
