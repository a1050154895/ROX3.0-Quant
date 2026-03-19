import uuid
import random
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AITrader:
    """
    AI交易员类
    
    功能:
    1. 根据策略和个性做出交易决策
    2. 管理资金和持仓
    3. 记录交易历史
    4. 根据交易结果更新情绪
    """
    
    def __init__(self, name: str, strategy: str, risk_level: int, 
                 personality: str, emotion: str, initial_capital: float):
        self.id = str(uuid.uuid4())
        self.name = name
        self.strategy = strategy
        self.risk_level = risk_level
        self.personality = personality
        self.emotion = emotion
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.risk_appetite = self._calculate_risk_appetite()
    
    def _calculate_risk_appetite(self) -> float:
        """
        计算风险偏好
        
        Returns:
            风险偏好值 (0-1)
        """
        # 基于风险等级和个性计算风险偏好
        base_risk = self.risk_level / 5.0
        
        # 根据个性调整风险偏好
        personality_factor = {
            "conservative": 0.5,
            "aggressive": 1.5,
            "analytical": 0.8,
            "methodical": 0.6,
            "impulsive": 1.2,
            "patient": 0.4,
            "balanced": 1.0,
            "contrarian": 1.3,
            "naive": 0.7
        }
        
        factor = personality_factor.get(self.personality, 1.0)
        return min(1.0, max(0.1, base_risk * factor))
    
    async def make_decision(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        做出交易决策
        
        Args:
            market_data: 市场数据
        
        Returns:
            交易决策
        """
        try:
            # 选择股票
            symbol = self._select_stock(market_data)
            if not symbol:
                return None
            
            # 获取股票价格
            stock_price = self._get_stock_price(symbol, market_data)
            if not stock_price:
                return None
            
            # 决定交易方向
            side = self._decide_trade_side(symbol, market_data)
            
            # 计算交易数量
            quantity = self._calculate_trade_quantity(stock_price)
            if quantity <= 0:
                return None
            
            # 生成交易决策
            decision = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": stock_price,
                "trader_id": self.id,
                "strategy": self.strategy,
                "personality": self.personality,
                "emotion": self.emotion
            }
            
            logger.info(f"交易员 {self.name} 做出决策: {decision}")
            return decision
            
        except Exception as e:
            logger.error(f"交易员 {self.name} 决策失败: {e}")
            return None
    
    def _select_stock(self, market_data: Dict[str, Any]) -> str:
        """
        选择股票
        
        Args:
            market_data: 市场数据
        
        Returns:
            股票代码
        """
        # 示例股票列表
        stocks = ["600519", "000001", "600036", "000002", "601318", 
                 "600276", "601888", "601398", "600031", "601628"]
        
        # 根据策略选择股票
        if self.strategy == "value":
            # 价值投资：选择低估值股票
            return random.choice(stocks[:5])
        elif self.strategy == "momentum":
            # 动量策略：选择近期表现好的股票
            return random.choice(stocks[5:])
        elif self.strategy == "technical":
            # 技术分析：随机选择
            return random.choice(stocks)
        elif self.strategy == "fundamental":
            # 基本面分析：选择蓝筹股
            return random.choice(["600519", "600036", "601318"])
        elif self.strategy == "scalping":
            # 短线交易：选择波动大的股票
            return random.choice(["600276", "601628"])
        elif self.strategy == "long_term":
            # 长线投资：选择蓝筹股
            return random.choice(["600519", "600036", "601398"])
        elif self.strategy == "swing":
            # 波段交易：随机选择
            return random.choice(stocks)
        elif self.strategy == "contrarian":
            # 逆向投资：选择近期表现差的股票
            return random.choice(stocks[:5])
        elif self.strategy == "quantitative":
            # 量化策略：基于简单指标选择
            return random.choice(stocks)
        else:
            # 随机策略：完全随机选择
            return random.choice(stocks)
    
    def _get_stock_price(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """
        获取股票价格
        
        Args:
            symbol: 股票代码
            market_data: 市场数据
        
        Returns:
            股票价格
        """
        # 从市场数据中获取价格
        if "stocks" in market_data:
            for stock in market_data["stocks"]:
                if stock["symbol"] == symbol:
                    return stock["price"]
        
        # 如果没有市场数据，生成随机价格
        base_prices = {
            "600519": 1500,
            "000001": 15,
            "600036": 35,
            "000002": 12,
            "601318": 45,
            "600276": 25,
            "601888": 18,
            "601398": 5,
            "600031": 28,
            "601628": 32
        }
        
        base_price = base_prices.get(symbol, 100)
        # 添加一些随机波动
        return base_price * (0.95 + 0.1 * random.random())
    
    def _decide_trade_side(self, symbol: str, market_data: Dict[str, Any]) -> str:
        """
        决定交易方向
        
        Args:
            symbol: 股票代码
            market_data: 市场数据
        
        Returns:
            交易方向 (buy/sell)
        """
        # 检查是否已经持有该股票
        if symbol in self.positions and self.positions[symbol]["quantity"] > 0:
            # 有持仓，可能卖出
            if random.random() < 0.3:  # 30%的概率卖出
                return "sell"
        
        # 决定买入
        return "buy"
    
    def _calculate_trade_quantity(self, price: float) -> int:
        """
        计算交易数量
        
        Args:
            price: 股票价格
        
        Returns:
            交易数量
        """
        # 基于风险偏好和情绪计算交易金额
        base_amount = self.capital * self.risk_appetite * 0.1  # 每次交易使用10%的资金
        
        # 根据情绪调整
        emotion_factor = {
            "calm": 1.0,
            "excited": 1.5,
            "focused": 1.2,
            "nervous": 0.8,
            "anxious": 0.6,
            "confident": 1.3,
            "skeptical": 0.9
        }
        
        factor = emotion_factor.get(self.emotion, 1.0)
        trade_amount = base_amount * factor
        
        # 计算数量
        quantity = int(trade_amount / price)
        return max(100, quantity)  # 最少100股
    
    def update_position(self, symbol: str, side: str, quantity: int, price: float, fee: float):
        """
        更新持仓
        
        Args:
            symbol: 股票代码
            side: 交易方向
            quantity: 交易数量
            price: 交易价格
            fee: 交易费用
        """
        total_cost = quantity * price + fee
        
        if side == "buy":
            # 买入
            if symbol in self.positions:
                # 已有持仓，更新
                current_quantity = self.positions[symbol]["quantity"]
                current_cost = self.positions[symbol]["cost"]
                new_quantity = current_quantity + quantity
                new_cost = current_cost + total_cost
                self.positions[symbol] = {
                    "quantity": new_quantity,
                    "cost": new_cost,
                    "average_price": new_cost / new_quantity
                }
            else:
                # 新持仓
                self.positions[symbol] = {
                    "quantity": quantity,
                    "cost": total_cost,
                    "average_price": price
                }
            # 减少资金
            self.capital -= total_cost
        else:
            # 卖出
            if symbol in self.positions and self.positions[symbol]["quantity"] >= quantity:
                # 计算卖出收益
                sell_amount = quantity * price - fee
                # 更新持仓
                current_quantity = self.positions[symbol]["quantity"]
                current_cost = self.positions[symbol]["cost"]
                new_quantity = current_quantity - quantity
                if new_quantity > 0:
                    # 还有持仓
                    new_cost = current_cost * (new_quantity / current_quantity)
                    self.positions[symbol] = {
                        "quantity": new_quantity,
                        "cost": new_cost,
                        "average_price": new_cost / new_quantity
                    }
                else:
                    # 清空持仓
                    del self.positions[symbol]
                # 增加资金
                self.capital += sell_amount
        
        # 记录交易
        self.trade_history.append({
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "fee": fee,
            "total_cost": total_cost if side == "buy" else quantity * price - fee,
            "timestamp": market_data.get("time", "")
        })
    
    def update_emotion(self, trade_result: Dict[str, Any]):
        """
        根据交易结果更新情绪
        
        Args:
            trade_result: 交易结果
        """
        # 基于交易结果更新情绪
        if trade_result.get("status") == "filled":
            # 交易成功
            if trade_result.get("pnl", 0) > 0:
                # 盈利，情绪变好
                good_emotions = ["excited", "confident", "focused"]
                self.emotion = random.choice(good_emotions)
            else:
                # 亏损，情绪变差
                bad_emotions = ["nervous", "anxious", "skeptical"]
                self.emotion = random.choice(bad_emotions)
    
    @property
    def total_assets(self) -> float:
        """
        计算总资产
        
        Returns:
            总资产
        """
        # 计算持仓价值
        positions_value = 0
        for symbol, position in self.positions.items():
            # 假设当前价格等于平均成本（简化计算）
            positions_value += position["quantity"] * position["average_price"]
        
        return self.capital + positions_value