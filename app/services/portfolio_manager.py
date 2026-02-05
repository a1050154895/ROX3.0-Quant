
import logging
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from app.db import get_conn, release_conn

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    模拟交易账户管理器 (Paper Trading Portfolio)
    
    核心功能:
    1. 账户管理: 初始化、查询资产
    2. 订单执行: 买入/卖出、更新余额和持仓
    3. 资产估值: 根据最新价格更新持仓市值
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.account_type = 'sim'
        self.account_id = None
        self._ensure_account_exists()
    
    def _ensure_account_exists(self):
        """确保模拟账户存在，不存在则创建"""
        conn = get_conn()
        try:
            # 1. 尝试查询现有账户
            cursor = conn.execute(
                "SELECT id FROM accounts WHERE user_id = ? AND type = ?", 
                (self.user_id, self.account_type)
            )
            row = cursor.fetchone()
            
            if row:
                self.account_id = row[0]
                logger.info(f"Loaded existing simulation account for user {self.user_id}, ID: {self.account_id}")
            else:
                # 2. 不存在则创建
                logger.info(f"Creating new simulation account for user {self.user_id}")
                
                try:
                    conn.execute(
                        """
                        INSERT INTO accounts (user_id, type, initial_capital, balance, total_assets)
                        VALUES (?, ?, 1000000.0, 1000000.0, 1000000.0)
                        """,
                        (self.user_id, self.account_type)
                    )
                    conn.commit()
                    self.account_id = cursor.lastrowid # This might be wrong if cursor was not from execute
                    
                    # 获取 lastrowid 的正确方式
                    # cursor = conn.execute(...) -> cursor.lastrowid
                    # 但上一句是 conn.execute
                    
                except sqlite3.IntegrityError as e:
                    logger.warning(f"Insert failed (likely exists): {e}")
                
                # 3. 再次查询获取 ID
                cursor = conn.execute(
                    "SELECT id FROM accounts WHERE user_id = ? AND type = ?", 
                    (self.user_id, self.account_type)
                )
                row = cursor.fetchone()
                if row:
                    self.account_id = row[0]
                    logger.info(f"Initialized new simulation account for user {self.user_id}, ID: {self.account_id}")
                else:
                    logger.error(f"Failed to create or retrieve account for user {self.user_id}")
                    raise ValueError("Account initialization failed")

            if not self.account_id:
                raise ValueError("Account ID is None after initialization")
                
        except Exception as e:
            logger.error(f"Failed to ensure account exists: {e}")
            raise
        finally:
            release_conn(conn)
            
    def get_account_summary(self) -> Dict:
        """获取账户概览 (余额、总资产、当日盈亏)"""
        conn = get_conn()
        try:
            cursor = conn.execute(
                "SELECT balance, total_assets, day_pnl, initial_capital FROM accounts WHERE id = ?",
                (self.account_id,)
            )
            row = cursor.fetchone()
            if not row:
                return {}
            
            return {
                "balance": row[0],
                "total_assets": row[1],
                "day_pnl": row[2],
                "total_pnl": row[1] - row[3],
                "total_pnl_pct": (row[1] - row[3]) / row[3] * 100
            }
        finally:
            release_conn(conn)
            
    def get_positions(self) -> List[Dict]:
        """获取当前持仓"""
        conn = get_conn()
        try:
            cursor = conn.execute(
                """
                SELECT symbol, name, quantity, average_cost, current_price, 
                       market_value, unrealized_pnl, unrealized_pnl_pct 
                FROM positions WHERE account_id = ? AND quantity > 0
                """,
                (self.account_id,)
            )
            columns = [col[0] for col in cursor.description]
            positions = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return positions
        finally:
            release_conn(conn)

    def execute_order(self, symbol: str, name: str, side: str, price: float, quantity: int, reason: str = "") -> bool:
        """
        执行订单 (模拟撮合)
        side: 'buy' or 'sell'
        """
        if quantity <= 0 or price <= 0:
            logger.error(f"Invalid order parameters: {symbol} {side} {quantity} @ {price}")
            return False
            
        conn = get_conn()
        try:
            # 1. 获取账户余额
            cursor = conn.execute("SELECT balance FROM accounts WHERE id = ?", (self.account_id,))
            row = cursor.fetchone()
            if not row:
                logger.error(f"Account {self.account_id} not found during execution")
                return False
                
            balance = row[0]
            
            total_cost = price * quantity
            fee = self._calculate_commission(total_cost) # 模拟手续费
                        
            if side == 'buy':
                if balance < (total_cost + fee):
                    logger.warning(f"Insufficient funds for buy order: Need {total_cost + fee}, Have {balance}")
                    return False
                
                # 扣款
                new_balance = balance - total_cost - fee
                conn.execute("UPDATE accounts SET balance = ? WHERE id = ?", (new_balance, self.account_id))
                
                # 更新持仓
                self._update_position_buy(conn, symbol, name, quantity, price)
                
            elif side == 'sell':
                # 检查持仓是否足够
                cursor = conn.execute("SELECT quantity FROM positions WHERE account_id = ? AND symbol = ?", (self.account_id, symbol))
                row = cursor.fetchone()
                current_qty = row[0] if row else 0
                
                if current_qty < quantity:
                    logger.warning(f"Insufficient quantity for sell order: Need {quantity}, Have {current_qty}")
                    return False
                
                # 入账
                new_balance = balance + total_cost - fee
                conn.execute("UPDATE accounts SET balance = ? WHERE id = ?", (new_balance, self.account_id))
                
                # 更新持仓
                self._update_position_sell(conn, symbol, quantity, price)
            
            # 记录交易日志
            conn.execute(
                """
                INSERT INTO trades (user_id, account_type, symbol, name, side, open_price, open_quantity, status, strategy_note)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'filled', ?)
                """,
                (self.user_id, self.account_type, symbol, name, side, price, quantity, reason)
            )
            
            conn.commit()
            logger.info(f"Order executed: {side} {symbol} {quantity} @ {price}")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Order execution failed: {e}")
            return False
        finally:
            release_conn(conn)

    def _update_position_buy(self, conn, symbol, name, quantity, price):
        """买入时更新持仓: 加权平均成本"""
        cursor = conn.execute(
            "SELECT quantity, average_cost FROM positions WHERE account_id = ? AND symbol = ?",
            (self.account_id, symbol)
        )
        row = cursor.fetchone()
        
        if row:
            old_qty = row[0]
            old_cost = row[1]
            new_qty = old_qty + quantity
            new_cost = (old_qty * old_cost + quantity * price) / new_qty
            
            conn.execute(
                """
                UPDATE positions SET quantity = ?, average_cost = ?, current_price = ?, updated_at = CURRENT_TIMESTAMP
                WHERE account_id = ? AND symbol = ?
                """,
                (new_qty, new_cost, price, self.account_id, symbol)
            )
        else:
            conn.execute(
                """
                INSERT INTO positions (account_id, symbol, name, quantity, average_cost, current_price, market_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (self.account_id, symbol, name, quantity, price, price, quantity * price)
            )

    def _update_position_sell(self, conn, symbol, quantity, price):
        """卖出时更新持仓"""
        cursor = conn.execute(
            "SELECT quantity FROM positions WHERE account_id = ? AND symbol = ?",
            (self.account_id, symbol)
        )
        row = cursor.fetchone()
        current_qty = row[0]
        
        new_qty = current_qty - quantity
        
        if new_qty > 0:
            conn.execute(
                """
                UPDATE positions SET quantity = ?, current_price = ?, updated_at = CURRENT_TIMESTAMP
                WHERE account_id = ? AND symbol = ?
                """,
                (new_qty, price, self.account_id, symbol)
            )
        else:
            # 清仓，可以选择删除记录或保留 quantity=0
            conn.execute("DELETE FROM positions WHERE account_id = ? AND symbol = ?", (self.account_id, symbol))

    def _calculate_commission(self, amount: float) -> float:
        """模拟佣金: 万分之2.5，最低5元"""
        fee = amount * 0.00025
        return max(fee, 5.0)

    def update_market_values(self, current_prices: Dict[str, float]):
        """
        根据最新行情更新持仓市值和账户总资产
        current_prices: { '600519': 1800.0, ... }
        """
        conn = get_conn()
        try:
            positions = self.get_positions()
            total_market_value = 0.0
            
            for pos in positions:
                symbol = pos['symbol']
                if symbol in current_prices:
                    price = current_prices[symbol]
                    qty = pos['quantity']
                    cost = pos['average_cost']
                    
                    market_val = price * qty
                    unrealized_pnl = market_val - (cost * qty)
                    unrealized_pnl_pct = (price - cost) / cost * 100 if cost > 0 else 0
                    
                    conn.execute(
                        """
                        UPDATE positions SET current_price = ?, market_value = ?, 
                                             unrealized_pnl = ?, unrealized_pnl_pct = ?
                        WHERE account_id = ? AND symbol = ?
                        """,
                        (price, market_val, unrealized_pnl, unrealized_pnl_pct, self.account_id, symbol)
                    )
                    total_market_value += market_val
                else:
                    # 如果没有最新价，使用旧市值
                    total_market_value += pos.get('market_value', 0)
            
            # 更新账户总资产
            summary = self.get_account_summary()
            balance = summary.get('balance', 0)
            new_total_assets = balance + total_market_value
            
            # 简单计算当日盈亏 (实际应记录昨收资产，这里简化处理)
            # 假设 initial_capital 不变，这里只是更新总资产
            conn.execute(
                "UPDATE accounts SET total_assets = ? WHERE id = ?",
                (new_total_assets, self.account_id)
            )
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to update market values: {e}")
            conn.rollback()
        finally:
            release_conn(conn)

    def get_trades_history(self, symbol: str = None) -> List[Dict]:
        """获取交易历史 (用于复盘)"""
        self._ensure_account_exists()
        
        sql = """
            SELECT symbol, name, side, open_price, open_quantity, status, open_time as created_at, strategy_note as reason
            FROM trades 
            WHERE user_id = ?
        """
        params = [self.user_id]
        
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
            
        sql += " ORDER BY open_time DESC"
        
        conn = get_conn()
        try:
            cursor = conn.execute(sql, params)
            columns = [col[0] for col in cursor.description]
            trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return trades
        finally:
            release_conn(conn)
