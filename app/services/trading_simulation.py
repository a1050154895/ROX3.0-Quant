import asyncio
import logging
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.services.ai_traders import AITrader
from app.services.simulated_exchange import SimulatedExchange
from app.services.feedback_system import FeedbackSystem
from app.utils.data_fetcher import data_fetcher

logger = logging.getLogger(__name__)

class TradingSimulation:
    """
    交易模拟引擎
    
    功能:
    1. 管理多个AI交易员
    2. 运行交易模拟
    3. 收集交易结果
    4. 提供模拟数据
    """
    
    def __init__(self):
        self.exchange = SimulatedExchange()
        self.feedback_system = FeedbackSystem()
        self.traders: List[AITrader] = []
        self.running = False
        self.simulation_id = f"sim_{int(time.time())}"
    
    def init_traders(self):
        """
        初始化AI交易员
        """
        # 定义10个不同个性和策略的AI交易员
        traders_config = [
            {
                "name": "保守型投资者",
                "strategy": "value",
                "risk_level": 1,
                "personality": "conservative",
                "emotion": "calm",
                "initial_capital": 100000
            },
            {
                "name": "激进型投资者",
                "strategy": "momentum",
                "risk_level": 5,
                "personality": "aggressive",
                "emotion": "excited",
                "initial_capital": 50000
            },
            {
                "name": "技术分析专家",
                "strategy": "technical",
                "risk_level": 3,
                "personality": "analytical",
                "emotion": "focused",
                "initial_capital": 80000
            },
            {
                "name": "基本面分析师",
                "strategy": "fundamental",
                "risk_level": 2,
                "personality": "methodical",
                "emotion": "calm",
                "initial_capital": 90000
            },
            {
                "name": "短线交易者",
                "strategy": "scalping",
                "risk_level": 4,
                "personality": "impulsive",
                "emotion": "nervous",
                "initial_capital": 60000
            },
            {
                "name": "长线投资者",
                "strategy": "long_term",
                "risk_level": 1,
                "personality": "patient",
                "emotion": "calm",
                "initial_capital": 100000
            },
            {
                "name": "波段交易者",
                "strategy": "swing",
                "risk_level": 3,
                "personality": "balanced",
                "emotion": "confident",
                "initial_capital": 70000
            },
            {
                "name": "逆向投资者",
                "strategy": "contrarian",
                "risk_level": 4,
                "personality": "contrarian",
                "emotion": "skeptical",
                "initial_capital": 65000
            },
            {
                "name": "量化交易员",
                "strategy": "quantitative",
                "risk_level": 3,
                "personality": "analytical",
                "emotion": "focused",
                "initial_capital": 85000
            },
            {
                "name": "初学者",
                "strategy": "random",
                "risk_level": 2,
                "personality": "naive",
                "emotion": "anxious",
                "initial_capital": 40000
            }
        ]
        
        for config in traders_config:
            trader = AITrader(
                name=config["name"],
                strategy=config["strategy"],
                risk_level=config["risk_level"],
                personality=config["personality"],
                emotion=config["emotion"],
                initial_capital=config["initial_capital"]
            )
            self.traders.append(trader)
        
        logger.info(f"初始化了 {len(self.traders)} 个AI交易员")
    
    async def run_simulation(self, duration_seconds: int = 3600):
        """
        运行交易模拟
        
        Args:
            duration_seconds: 模拟持续时间（秒）
        """
        if not self.traders:
            self.init_traders()
        
        self.running = True
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        logger.info(f"开始交易模拟，持续时间: {duration_seconds}秒")
        
        while time.time() < end_time and self.running:
            try:
                # 更新市场数据
                await self.update_market_data()
                
                # 每个交易员执行交易决策
                for trader in self.traders:
                    await self.execute_trader_decision(trader)
                
                # 等待一段时间
                await asyncio.sleep(5)  # 每5秒执行一次
                
            except Exception as e:
                logger.error(f"模拟执行错误: {e}")
                await asyncio.sleep(1)
        
        self.running = False
        logger.info("交易模拟结束")
    
    async def update_market_data(self):
        """
        更新市场数据
        """
        try:
            # 获取市场指数
            indices = await data_fetcher.get_market_indices()
            if indices:
                for index in indices:
                    self.exchange.update_index_price(index["code"], index["price"])
            
            # 获取行业板块数据
            sectors = await data_fetcher.get_sector_list()
            if sectors:
                for sector in sectors:
                    self.exchange.update_sector_change(sector["code"], sector["change_pct"])
            
        except Exception as e:
            logger.warning(f"更新市场数据失败: {e}")
    
    async def execute_trader_decision(self, trader: AITrader):
        """
        执行交易员的交易决策
        
        Args:
            trader: AI交易员
        """
        try:
            # 获取市场数据
            market_data = self.get_market_snapshot()
            
            # 交易员做出决策
            decision = await trader.make_decision(market_data)
            
            if decision:
                # 执行交易
                order = {
                    "symbol": decision["symbol"],
                    "side": decision["side"],
                    "quantity": decision["quantity"],
                    "price": decision["price"],
                    "trader_id": trader.id
                }
                
                # 提交订单到交易所
                result = self.exchange.submit_order(order)
                
                if result.get("status") == "filled":
                    # 更新交易员的持仓和资金
                    trader.update_position(
                        symbol=decision["symbol"],
                        side=decision["side"],
                        quantity=decision["quantity"],
                        price=result["price"],
                        fee=result.get("fee", 0)
                    )
                    
                    # 更新交易员情绪
                    trader.update_emotion(result)
                    
                    # 记录交易结果
                    self.feedback_system.record_trade(
                        trader_id=trader.id,
                        symbol=decision["symbol"],
                        side=decision["side"],
                        quantity=decision["quantity"],
                        price=result["price"],
                        result=result
                    )
                    
        except Exception as e:
            logger.error(f"执行交易决策失败: {e}")
    
    def get_market_snapshot(self) -> Dict[str, Any]:
        """
        获取市场快照
        
        Returns:
            市场快照数据
        """
        return {
            "indices": self.exchange.get_indices(),
            "sectors": self.exchange.get_sectors(),
            "stocks": self.exchange.get_stock_prices(),
            "time": datetime.now().isoformat()
        }
    
    def get_traders_status(self) -> List[Dict[str, Any]]:
        """
        获取所有交易员的状态
        
        Returns:
            交易员状态列表
        """
        status_list = []
        for trader in self.traders:
            status_list.append({
                "id": trader.id,
                "name": trader.name,
                "strategy": trader.strategy,
                "risk_level": trader.risk_level,
                "personality": trader.personality,
                "emotion": trader.emotion,
                "capital": trader.capital,
                "total_assets": trader.total_assets,
                "positions": trader.positions,
                "trades": trader.trade_history[-5:]  # 最近5笔交易
            })
        return status_list
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """
        获取模拟状态
        
        Returns:
            模拟状态
        """
        return {
            "simulation_id": self.simulation_id,
            "running": self.running,
            "trader_count": len(self.traders),
            "market_status": self.get_market_snapshot(),
            "exchange_status": self.exchange.get_status()
        }
    
    def stop_simulation(self):
        """
        停止模拟
        """
        self.running = False
        logger.info("交易模拟已停止")
    
    def reset_simulation(self):
        """
        重置模拟
        """
        self.exchange = SimulatedExchange()
        self.feedback_system = FeedbackSystem()
        self.traders = []
        self.running = False
        self.simulation_id = f"sim_{int(time.time())}"
        logger.info("交易模拟已重置")

# 创建全局交易模拟实例
trading_simulation = TradingSimulation()