import logging
import random
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SimulatedExchange:
    """
    模拟交易所类
    
    功能:
    1. 处理订单匹配
    2. 执行交易
    3. 维护市场数据
    4. 计算交易费用
    """
    
    def __init__(self):
        self.order_book = {}
        self.stock_prices = {}
        self.indices = {}
        self.sectors = {}
        self.trade_history = []
        self.fee_rate = 0.0003  # 交易费率
    
    def update_index_price(self, index_code: str, price: float):
        """
        更新指数价格
        
        Args:
            index_code: 指数代码
            price: 指数价格
        """
        self.indices[index_code] = {
            "price": price,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_sector_change(self, sector_code: str, change_pct: float):
        """
        更新行业板块涨跌幅
        
        Args:
            sector_code: 板块代码
            change_pct: 涨跌幅
        """
        self.sectors[sector_code] = {
            "change_pct": change_pct,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_stock_price(self, symbol: str, price: float):
        """
        更新股票价格
        
        Args:
            symbol: 股票代码
            price: 股票价格
        """
        self.stock_prices[symbol] = {
            "price": price,
            "timestamp": datetime.now().isoformat()
        }
    
    def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        提交订单
        
        Args:
            order: 订单信息
        
        Returns:
            交易结果
        """
        try:
            symbol = order["symbol"]
            side = order["side"]
            quantity = order["quantity"]
            price = order["price"]
            trader_id = order.get("trader_id", "unknown")
            
            # 生成成交价格（模拟市场波动）
            executed_price = price * (0.995 + 0.01 * random.random())
            
            # 计算交易费用
            fee = executed_price * quantity * self.fee_rate
            
            # 计算交易金额
            total_amount = executed_price * quantity
            
            # 记录交易
            trade = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": executed_price,
                "total_amount": total_amount,
                "fee": fee,
                "trader_id": trader_id,
                "timestamp": datetime.now().isoformat()
            }
            
            self.trade_history.append(trade)
            
            # 更新股票价格（模拟交易对价格的影响）
            price_change = (executed_price - price) / price * 100
            new_price = executed_price * (1 + 0.001 * random.random() * (1 if side == "buy" else -1))
            self.update_stock_price(symbol, new_price)
            
            # 生成交易结果
            result = {
                "status": "filled",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": executed_price,
                "total_amount": total_amount,
                "fee": fee,
                "timestamp": trade["timestamp"],
                "pnl": 0  # 简化计算，实际应该根据持仓成本计算
            }
            
            logger.info(f"订单执行成功: {result}")
            return result
            
        except Exception as e:
            logger.error(f"订单执行失败: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_stock_prices(self) -> List[Dict[str, Any]]:
        """
        获取所有股票价格
        
        Returns:
            股票价格列表
        """
        prices = []
        for symbol, data in self.stock_prices.items():
            prices.append({
                "symbol": symbol,
                "price": data["price"],
                "timestamp": data["timestamp"]
            })
        return prices
    
    def get_indices(self) -> List[Dict[str, Any]]:
        """
        获取所有指数
        
        Returns:
            指数列表
        """
        indices = []
        for code, data in self.indices.items():
            indices.append({
                "code": code,
                "price": data["price"],
                "timestamp": data["timestamp"]
            })
        return indices
    
    def get_sectors(self) -> List[Dict[str, Any]]:
        """
        获取所有行业板块
        
        Returns:
            行业板块列表
        """
        sectors = []
        for code, data in self.sectors.items():
            sectors.append({
                "code": code,
                "change_pct": data["change_pct"],
                "timestamp": data["timestamp"]
            })
        return sectors
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取交易历史
        
        Args:
            limit: 限制数量
        
        Returns:
            交易历史列表
        """
        return self.trade_history[-limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取交易所状态
        
        Returns:
            交易所状态
        """
        return {
            "stock_count": len(self.stock_prices),
            "index_count": len(self.indices),
            "sector_count": len(self.sectors),
            "trade_count": len(self.trade_history),
            "fee_rate": self.fee_rate,
            "timestamp": datetime.now().isoformat()
        }
    
    def reset(self):
        """
        重置交易所状态
        """
        self.order_book = {}
        self.stock_prices = {}
        self.indices = {}
        self.sectors = {}
        self.trade_history = []
        logger.info("交易所状态已重置")