from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
from app.services.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

class TradeProvider(ABC):
    """
    Abstract Base Class for Trading Providers.
    Standardizes interface for Paper Trading (Simulation) and Real Trading (Brokerage).
    """
    
    @abstractmethod
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account balance and equity."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        pass

    @abstractmethod
    def place_order(self, symbol: str, action: str, price: float, quantity: int, name: str = "") -> Dict[str, Any]:
        """
        Place an order.
        action: 'buy' or 'sell'
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass


class PaperTradeProvider(TradeProvider):
    """
    Simulation Trading Provider using local PortfolioManager.
    """
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.pm = PortfolioManager(user_id=user_id)

    def get_account_summary(self) -> Dict[str, Any]:
        return self.pm.get_account_summary() or {"balance": 0.0, "asset": 0.0}

    def get_positions(self) -> List[Dict[str, Any]]:
        return self.pm.get_positions()

    def place_order(self, symbol: str, action: str, price: float, quantity: int, name: str = "") -> Dict[str, Any]:
        # PortfolioManager.execute_order returns bool
        # We wrap it to return a dict standard
        try:
            success = self.pm.execute_order(symbol, name, action, price, quantity, reason="Manual Order")
            if success:
                return {"status": "success", "order_id": f"sim-{symbol}-{quantity}", "message": "Order filled"}
            else:
                return {"status": "failed", "message": "Insufficient funds or holdings"}
        except Exception as e:
            logger.error(f"Paper trade failed: {e}")
            return {"status": "error", "message": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        # Paper trading fills instantly, so cancel is not really supported in this simple version
        return False


class RealTradeProvider(TradeProvider):
    """
    Real Trading Provider (Stub).
    Connects to CTP (A-Share) or IB/Tiger (US/HK).
    """
    def __init__(self, user_id: int):
        self.user_id = user_id
        # In future: load broker config from DB/Env
        self.configured = False 

    def get_account_summary(self) -> Dict[str, Any]:
        # Return mock real data or error if not connected
        if not self.configured:
            return {
                "balance": 0.0,
                "asset": 0.0,
                "message": "Real Trading not configured. Please contact admin."
            }
        return {"balance": 0.0, "asset": 0.0}

    def get_positions(self) -> List[Dict[str, Any]]:
        if not self.configured:
            return []
        return []

    def place_order(self, symbol: str, action: str, price: float, quantity: int, name: str = "") -> Dict[str, Any]:
        return {
            "status": "failed", 
            "message": "Real Trading Interface is not connected. This is a placeholder for CTP/IB integration."
        }

    def cancel_order(self, order_id: str) -> bool:
        return False

def get_trade_provider(user_id: int, mode: str = "sim") -> TradeProvider:
    if mode == "real":
        return RealTradeProvider(user_id)
    return PaperTradeProvider(user_id)
