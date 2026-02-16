import logging
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class AccountManager:
    """
    账户管理服务
    负责多账户的创建、查询、更新和管理
    """
    
    def __init__(self, db_path: str = "rox.db"):
        self.db_path = db_path
    
    def create_account(self, user_id: int, account_type: str, name: str, initial_balance: float = 100000.0, currency: str = "CNY") -> Optional[int]:
        """
        创建新账户
        
        Args:
            user_id: 用户ID
            account_type: 账户类型 (sim/real)
            name: 账户名称
            initial_balance: 初始余额
            currency: 货币类型
            
        Returns:
            账户ID，创建失败返回None
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 检查账户名称是否已存在
            if self._account_name_exists(conn, user_id, name):
                logger.warning(f"Account name {name} already exists for user {user_id}")
                return None
            
            # 创建账户
            cursor = conn.execute(
                "INSERT INTO accounts (user_id, type, name, initial_capital, balance, currency) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, account_type, name, initial_balance, initial_balance, currency)
            )
            conn.commit()
            
            account_id = cursor.lastrowid
            logger.info(f"Created new account {account_id} for user {user_id}: {name} ({account_type})")
            return account_id
            
        except Exception as e:
            logger.error(f"Error creating account: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_user_accounts(self, user_id: int) -> List[Dict]:
        """
        获取用户的所有账户
        
        Args:
            user_id: 用户ID
            
        Returns:
            账户列表
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT id, type, name, initial_capital, balance, currency, created_at "
                "FROM accounts WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            )
            
            accounts = []
            for row in cursor.fetchall():
                accounts.append({
                    "id": row[0],
                    "type": row[1],
                    "name": row[2],
                    "initial_capital": row[3],
                    "balance": row[4],
                    "currency": row[5],
                    "created_at": row[6]
                })
            
            return accounts
            
        except Exception as e:
            logger.error(f"Error getting user accounts: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_account(self, account_id: int, user_id: int) -> Optional[Dict]:
        """
        获取单个账户详情
        
        Args:
            account_id: 账户ID
            user_id: 用户ID（用于验证所有权）
            
        Returns:
            账户详情，不存在返回None
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT id, type, name, initial_capital, balance, currency, created_at "
                "FROM accounts WHERE id = ? AND user_id = ?",
                (account_id, user_id)
            )
            
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "type": row[1],
                    "name": row[2],
                    "initial_capital": row[3],
                    "balance": row[4],
                    "currency": row[5],
                    "created_at": row[6]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def update_account(self, account_id: int, user_id: int, name: Optional[str] = None, balance: Optional[float] = None) -> bool:
        """
        更新账户信息
        
        Args:
            account_id: 账户ID
            user_id: 用户ID（用于验证所有权）
            name: 账户名称
            balance: 账户余额
            
        Returns:
            更新成功返回True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 检查账户是否存在且属于该用户
            if not self._account_exists(conn, account_id, user_id):
                logger.warning(f"Account {account_id} not found for user {user_id}")
                return False
            
            # 构建更新语句
            updates = []
            params = []
            
            if name:
                # 检查新名称是否已存在
                if self._account_name_exists(conn, user_id, name, exclude_account_id=account_id):
                    logger.warning(f"Account name {name} already exists for user {user_id}")
                    return False
                updates.append("name = ?")
                params.append(name)
            
            if balance is not None:
                updates.append("balance = ?")
                params.append(balance)
            
            if not updates:
                return True
            
            params.extend([account_id, user_id])
            query = f"UPDATE accounts SET {', '.join(updates)} WHERE id = ? AND user_id = ?"
            
            cursor = conn.execute(query, params)
            conn.commit()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error updating account: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def delete_account(self, account_id: int, user_id: int) -> bool:
        """
        删除账户
        
        Args:
            account_id: 账户ID
            user_id: 用户ID（用于验证所有权）
            
        Returns:
            删除成功返回True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 检查账户是否存在且属于该用户
            if not self._account_exists(conn, account_id, user_id):
                logger.warning(f"Account {account_id} not found for user {user_id}")
                return False
            
            # 检查账户是否有未平仓交易
            if self._has_open_trades(conn, account_id):
                logger.warning(f"Cannot delete account {account_id} with open trades")
                return False
            
            # 删除账户
            cursor = conn.execute(
                "DELETE FROM accounts WHERE id = ? AND user_id = ?",
                (account_id, user_id)
            )
            conn.commit()
            
            logger.info(f"Deleted account {account_id} for user {user_id}")
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error deleting account: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def transfer_funds(self, from_account_id: int, to_account_id: int, user_id: int, amount: float) -> bool:
        """
        在账户之间转移资金
        
        Args:
            from_account_id: 转出账户ID
            to_account_id: 转入账户ID
            user_id: 用户ID（用于验证所有权）
            amount: 转移金额
            
        Returns:
            转移成功返回True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("BEGIN TRANSACTION")
            
            # 检查两个账户是否都存在且属于该用户
            from_account = self.get_account(from_account_id, user_id)
            to_account = self.get_account(to_account_id, user_id)
            
            if not from_account or not to_account:
                logger.warning("One or both accounts not found")
                conn.execute("ROLLBACK")
                return False
            
            # 检查转出账户余额是否足够
            if from_account["balance"] < amount:
                logger.warning("Insufficient balance in source account")
                conn.execute("ROLLBACK")
                return False
            
            # 检查货币类型是否相同
            if from_account["currency"] != to_account["currency"]:
                logger.warning("Currency mismatch between accounts")
                conn.execute("ROLLBACK")
                return False
            
            # 执行资金转移
            conn.execute(
                "UPDATE accounts SET balance = balance - ? WHERE id = ?",
                (amount, from_account_id)
            )
            
            conn.execute(
                "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                (amount, to_account_id)
            )
            
            # 记录转账记录
            conn.execute(
                "INSERT INTO transfers (from_account_id, to_account_id, user_id, amount, currency, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (from_account_id, to_account_id, user_id, amount, from_account["currency"], datetime.utcnow())
            )
            
            conn.execute("COMMIT")
            logger.info(f"Transferred {amount} from account {from_account_id} to {to_account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error transferring funds: {e}")
            if conn:
                conn.execute("ROLLBACK")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_account_performance(self, account_id: int, user_id: int) -> Dict[str, Any]:
        """
        获取账户性能统计
        
        Args:
            account_id: 账户ID
            user_id: 用户ID（用于验证所有权）
            
        Returns:
            性能统计数据
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 检查账户是否存在且属于该用户
            account = self.get_account(account_id, user_id)
            if not account:
                return {"error": "Account not found"}
            
            # 获取账户交易数据
            cursor = conn.execute(
                "SELECT side, open_price, close_price, open_quantity "
                "FROM trades WHERE account_id = ? AND close_price IS NOT NULL",
                (account_id,)
            )
            
            # 计算性能指标
            total_trades = 0
            total_pnl = 0
            wins = 0
            losses = 0
            total_win = 0
            total_loss = 0
            
            for row in cursor.fetchall():
                side, open_price, close_price, quantity = row
                quantity = float(quantity)
                
                if side.lower() == "buy":
                    pnl = (float(close_price) - float(open_price)) * quantity
                else:
                    pnl = (float(open_price) - float(close_price)) * quantity
                
                total_trades += 1
                total_pnl += pnl
                
                if pnl > 0:
                    wins += 1
                    total_win += pnl
                elif pnl < 0:
                    losses += 1
                    total_loss += abs(pnl)
            
            # 计算指标
            win_rate = wins / total_trades if total_trades > 0 else 0
            average_win = total_win / wins if wins > 0 else 0
            average_loss = total_loss / losses if losses > 0 else 0
            profit_factor = total_win / total_loss if total_loss > 0 else 0
            total_return = (account["balance"] - account["initial_capital"]) / account["initial_capital"] * 100
            
            return {
                "account": account,
                "performance": {
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                    "average_win": average_win,
                    "average_loss": average_loss,
                    "profit_factor": profit_factor,
                    "total_pnl": total_pnl,
                    "total_return": total_return,
                    "current_balance": account["balance"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting account performance: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()
    
    def get_combined_performance(self, user_id: int) -> Dict[str, Any]:
        """
        获取用户所有账户的组合性能统计
        
        Args:
            user_id: 用户ID
            
        Returns:
            组合性能统计数据
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 获取所有账户
            accounts = self.get_user_accounts(user_id)
            if not accounts:
                return {"error": "No accounts found"}
            
            # 计算总初始资金和总当前余额
            total_initial_capital = sum(a["initial_capital"] for a in accounts)
            total_current_balance = sum(a["balance"] for a in accounts)
            total_return = (total_current_balance - total_initial_capital) / total_initial_capital * 100
            
            # 获取所有账户的交易数据
            account_ids = [a["id"] for a in accounts]
            placeholders = ",".join(["?"] * len(account_ids))
            cursor = conn.execute(
                f"SELECT side, open_price, close_price, open_quantity "
                f"FROM trades WHERE account_id IN ({placeholders}) AND close_price IS NOT NULL",
                account_ids
            )
            
            # 计算组合性能指标
            total_trades = 0
            total_pnl = 0
            wins = 0
            losses = 0
            total_win = 0
            total_loss = 0
            
            for row in cursor.fetchall():
                side, open_price, close_price, quantity = row
                quantity = float(quantity)
                
                if side.lower() == "buy":
                    pnl = (float(close_price) - float(open_price)) * quantity
                else:
                    pnl = (float(open_price) - float(close_price)) * quantity
                
                total_trades += 1
                total_pnl += pnl
                
                if pnl > 0:
                    wins += 1
                    total_win += pnl
                elif pnl < 0:
                    losses += 1
                    total_loss += abs(pnl)
            
            # 计算指标
            win_rate = wins / total_trades if total_trades > 0 else 0
            average_win = total_win / wins if wins > 0 else 0
            average_loss = total_loss / losses if losses > 0 else 0
            profit_factor = total_win / total_loss if total_loss > 0 else 0
            
            # 账户分布
            account_distribution = {}
            for account in accounts:
                account_type = account["type"]
                account_distribution[account_type] = account_distribution.get(account_type, 0) + 1
            
            return {
                "summary": {
                    "total_accounts": len(accounts),
                    "total_initial_capital": total_initial_capital,
                    "total_current_balance": total_current_balance,
                    "total_return": total_return,
                    "total_trades": total_trades
                },
                "performance": {
                    "win_rate": win_rate,
                    "average_win": average_win,
                    "average_loss": average_loss,
                    "profit_factor": profit_factor,
                    "total_pnl": total_pnl
                },
                "account_distribution": account_distribution,
                "accounts": accounts
            }
            
        except Exception as e:
            logger.error(f"Error getting combined performance: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()
    
    def _account_exists(self, conn: sqlite3.Connection, account_id: int, user_id: int) -> bool:
        """
        检查账户是否存在且属于该用户
        """
        cursor = conn.execute(
            "SELECT 1 FROM accounts WHERE id = ? AND user_id = ?",
            (account_id, user_id)
        )
        return cursor.fetchone() is not None
    
    def _account_name_exists(self, conn: sqlite3.Connection, user_id: int, name: str, exclude_account_id: Optional[int] = None) -> bool:
        """
        检查账户名称是否已存在
        """
        if exclude_account_id:
            cursor = conn.execute(
                "SELECT 1 FROM accounts WHERE user_id = ? AND name = ? AND id != ?",
                (user_id, name, exclude_account_id)
            )
        else:
            cursor = conn.execute(
                "SELECT 1 FROM accounts WHERE user_id = ? AND name = ?",
                (user_id, name)
            )
        return cursor.fetchone() is not None
    
    def _has_open_trades(self, conn: sqlite3.Connection, account_id: int) -> bool:
        """
        检查账户是否有未平仓交易
        """
        cursor = conn.execute(
            "SELECT 1 FROM trades WHERE account_id = ? AND close_price IS NULL",
            (account_id,)
        )
        return cursor.fetchone() is not None

# 全局账户管理实例
_account_manager_instance = None
_account_manager_lock = threading.Lock()

def get_account_manager() -> AccountManager:
    """获取账户管理实例"""
    global _account_manager_instance
    
    with _account_manager_lock:
        if _account_manager_instance is None:
            _account_manager_instance = AccountManager()
        
    return _account_manager_instance
