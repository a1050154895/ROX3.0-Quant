import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sqlite3
import asyncio

from app.rox_quant.risk_management_advanced import RiskManager as AdvancedRiskManager
from app.rox_quant.data_provider import DataProvider
from app.rox_quant.trade_executor import get_trader

logger = logging.getLogger(__name__)

class RiskMonitor:
    """
    实时风险监控服务
    负责监控持仓风险、执行止损止盈、评估整体账户风险、系统状态监控
    """
    
    def __init__(self, db_path: str = "rox.db"):
        self.db_path = db_path
        self.risk_manager = AdvancedRiskManager()
        self.data_provider = DataProvider()
        self.is_running = False
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._check_interval = 5  # 检查间隔（秒）
        self._risk_alerts = []
        self._system_alerts = []
        self._system_status = {
            "last_check": None,
            "db_connected": False,
            "api_connected": False,
            "websocket_connected": False,
            "system_load": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0
        }
        # 导入系统监控相关模块
        try:
            import psutil
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
    
    def start(self):
        """启动风险监控服务"""
        with self._lock:
            if self.is_running:
                logger.warning("Risk monitor already running")
                return
            
            self.is_running = True
            logger.info("Starting risk monitor service")
            
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop(self):
        """停止风险监控服务"""
        with self._lock:
            if not self.is_running:
                logger.warning("Risk monitor not running")
                return
            
            logger.info("Stopping risk monitor service")
            self.is_running = False
            
            if self._monitor_thread:
                self._monitor_thread.join(timeout=10)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 检查所有用户的风险
                self._check_all_users_risk()
                
                # 检查系统状态
                self._check_system_status()
                
                # 清理过期警报
                self._cleanup_alerts()
                self._cleanup_system_alerts()
                
            except Exception as e:
                logger.error(f"Error in risk monitor loop: {e}")
            
            # 等待下一次检查
            for _ in range(self._check_interval):
                if not self.is_running:
                    break
                time.sleep(1)
    
    def _check_system_status(self):
        """检查系统状态"""
        try:
            # 更新系统状态
            self._system_status["last_check"] = datetime.now().isoformat()
            
            # 检查数据库连接
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute("SELECT 1")
                self._system_status["db_connected"] = True
            except Exception:
                self._system_status["db_connected"] = False
                self._add_system_alert("数据库连接失败", "error")
            
            # 检查API连接
            try:
                # 这里可以添加具体的API连接检查逻辑
                # 例如，尝试获取一个简单的市场数据
                self._system_status["api_connected"] = True
            except Exception:
                self._system_status["api_connected"] = False
                self._add_system_alert("API连接失败", "error")
            
            # 检查WebSocket连接
            try:
                # 这里可以添加具体的WebSocket连接检查逻辑
                self._system_status["websocket_connected"] = True
            except Exception:
                self._system_status["websocket_connected"] = False
                self._add_system_alert("WebSocket连接失败", "error")
            
            # 检查系统资源使用情况
            if self.psutil_available:
                import psutil
                
                # 系统负载
                if hasattr(psutil, 'cpu_percent'):
                    self._system_status["system_load"] = psutil.cpu_percent(interval=0.1)
                    if self._system_status["system_load"] > 80:
                        self._add_system_alert(f"系统负载过高: {self._system_status['system_load']}%", "warning")
                
                # 内存使用
                if hasattr(psutil, 'virtual_memory'):
                    memory = psutil.virtual_memory()
                    self._system_status["memory_usage"] = memory.percent
                    if self._system_status["memory_usage"] > 80:
                        self._add_system_alert(f"内存使用过高: {self._system_status['memory_usage']}%", "warning")
                
                # 磁盘使用
                if hasattr(psutil, 'disk_usage'):
                    disk = psutil.disk_usage('/')
                    self._system_status["disk_usage"] = disk.percent
                    if self._system_status["disk_usage"] > 80:
                        self._add_system_alert(f"磁盘使用过高: {self._system_status['disk_usage']}%", "warning")
            
        except Exception as e:
            logger.error(f"Error checking system status: {e}")
    
    def _check_all_users_risk(self):
        """检查所有用户的风险"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 获取所有活跃用户
            users = self._get_active_users(conn)
            
            for user in users:
                try:
                    self._check_user_risk(conn, user)
                except Exception as e:
                    logger.error(f"Error checking risk for user {user}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in _check_all_users_risk: {e}")
        finally:
            if conn:
                conn.close()
    
    def _get_active_users(self, conn: sqlite3.Connection) -> List[int]:
        """获取所有活跃用户"""
        try:
            cursor = conn.execute("SELECT id FROM users WHERE is_active = 1")
            return [row[0] for row in cursor.fetchall()]
        except Exception:
            # 如果表不存在或查询失败，返回空列表
            return []
    
    def _check_user_risk(self, conn: sqlite3.Connection, user_id: int):
        """检查单个用户的风险"""
        # 1. 检查持仓风险（止损止盈）
        self._check_position_risk(conn, user_id)
        
        # 2. 检查账户整体风险
        self._check_account_risk(conn, user_id)
    
    def _check_position_risk(self, conn: sqlite3.Connection, user_id: int):
        """检查持仓风险，执行止损止盈"""
        # 获取带风险参数的开仓记录
        open_trades = self._get_open_trades_with_risk(conn, user_id)
        
        for trade in open_trades:
            try:
                symbol = trade.get("symbol")
                trade_id = trade.get("id")
                
                if not symbol or not trade_id:
                    continue
                
                # 获取当前价格
                current_price = self.data_provider.get_realtime_price(symbol)
                if current_price is None:
                    logger.warning(f"No price data for {symbol}, skipping risk check")
                    continue
                
                # 获取止损止盈参数
                open_price = float(trade.get("open_price", 0))
                stop_loss = trade.get("stop_loss")
                take_profit = trade.get("take_profit")
                account_type = trade.get("account_type", "sim")
                
                # 检查是否触发止损止盈
                should_stop = False
                stop_reason = ""
                
                if stop_loss is not None:
                    sl_price = float(stop_loss)
                    if current_price <= sl_price:
                        should_stop = True
                        stop_reason = "stop_loss"
                
                if take_profit is not None and not should_stop:
                    tp_price = float(take_profit)
                    if current_price >= tp_price:
                        should_stop = True
                        stop_reason = "take_profit"
                
                # 检查时间止盈
                if not should_stop:
                    bars_held = self._calculate_bars_held(trade.get("created_at"))
                    unrealized_pnl_pct = (current_price - open_price) / open_price
                    
                    if self.risk_manager.should_time_stop(bars_held, unrealized_pnl_pct):
                        should_stop = True
                        stop_reason = "time_stop"
                
                # 执行平仓
                if should_stop:
                    logger.info(f"Triggering {stop_reason} for trade {trade_id}: {symbol} at {current_price}")
                    
                    # 平仓
                    if self._close_trade(conn, user_id, trade_id, current_price):
                        # 如果是实盘，执行真实交易
                        if account_type == "real":
                            self._execute_real_close(symbol, current_price, trade.get("open_quantity", 0))
                        
                        # 添加警报
                        self._add_alert(user_id, f"{symbol} 触发{stop_reason}，已平仓", "info")
                        
            except Exception as e:
                logger.error(f"Error checking position risk for trade {trade.get('id')}: {e}")
    
    def _check_account_risk(self, conn: sqlite3.Connection, user_id: int):
        """检查账户整体风险"""
        # 获取账户信息
        account_info = self._get_account_info(conn, user_id)
        
        if not account_info:
            return
        
        # 计算风险指标
        risk_metrics = self._calculate_account_risk_metrics(conn, user_id, account_info)
        
        # 检查风险阈值
        self._check_risk_thresholds(user_id, risk_metrics)
    
    def _get_open_trades_with_risk(self, conn: sqlite3.Connection, user_id: int) -> List[Dict]:
        """获取带风险参数的开仓记录"""
        try:
            cursor = conn.execute(
                "SELECT id, symbol, open_price, open_quantity, stop_loss, take_profit, account_type, created_at "
                "FROM trades WHERE user_id = ? AND close_price IS NULL",
                (user_id,)
            )
            
            trades = []
            for row in cursor.fetchall():
                trades.append({
                    "id": row[0],
                    "symbol": row[1],
                    "open_price": row[2],
                    "open_quantity": row[3],
                    "stop_loss": row[4],
                    "take_profit": row[5],
                    "account_type": row[6],
                    "created_at": row[7]
                })
            
            return trades
        except Exception as e:
            logger.error(f"Error getting open trades: {e}")
            return []
    
    def _calculate_bars_held(self, created_at: str) -> int:
        """计算持仓时间（按bar数）"""
        try:
            created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            now = datetime.utcnow()
            minutes_held = (now - created).total_seconds() / 60
            # 假设每个bar为1分钟
            return int(minutes_held)
        except Exception:
            return 0
    
    def _close_trade(self, conn: sqlite3.Connection, user_id: int, trade_id: int, close_price: float) -> bool:
        """平仓交易"""
        try:
            cursor = conn.execute(
                "UPDATE trades SET close_price = ?, close_time = CURRENT_TIMESTAMP, status = 'closed' "
                "WHERE id = ? AND user_id = ? AND close_price IS NULL",
                (close_price, trade_id, user_id)
            )
            
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            return False
    
    def _execute_real_close(self, symbol: str, price: float, quantity: int):
        """执行实盘平仓"""
        try:
            trader = get_trader('real')
            if trader._ensure_connected():
                res = trader.sell(symbol, price, quantity)
                logger.info(f"Real close executed: {res}")
            else:
                logger.warning("Cannot execute real close: trader not connected")
        except Exception as e:
            logger.error(f"Error executing real close: {e}")
    
    def _get_account_info(self, conn: sqlite3.Connection, user_id: int) -> Optional[Dict]:
        """获取账户信息"""
        try:
            cursor = conn.execute(
                "SELECT total_balance, available_balance FROM accounts WHERE user_id = ?",
                (user_id,)
            )
            
            row = cursor.fetchone()
            if row:
                return {
                    "total_balance": float(row[0]) if row[0] else 0,
                    "available_balance": float(row[1]) if row[1] else 0
                }
            
            return None
        except Exception:
            return None
    
    def _calculate_account_risk_metrics(self, conn: sqlite3.Connection, user_id: int, account_info: Dict) -> Dict:
        """计算账户风险指标"""
        metrics = {
            "total_balance": account_info.get("total_balance", 0),
            "available_balance": account_info.get("available_balance", 0),
            "position_count": 0,
            "total_position_value": 0,
            "total_pnl": 0,
            "max_drawdown": 0,
            "margin_usage": 0
        }
        
        try:
            # 获取持仓信息
            cursor = conn.execute(
                "SELECT symbol, open_price, open_quantity, close_price FROM trades WHERE user_id = ?",
                (user_id,)
            )
            
            for row in cursor.fetchall():
                symbol, open_price, quantity, close_price = row
                
                if close_price is None:
                    # 未平仓
                    metrics["position_count"] += 1
                    current_price = self.data_provider.get_realtime_price(symbol)
                    if current_price:
                        position_value = current_price * quantity
                        metrics["total_position_value"] += position_value
                        pnl = (current_price - float(open_price)) * quantity
                        metrics["total_pnl"] += pnl
                else:
                    # 已平仓
                    pnl = (float(close_price) - float(open_price)) * quantity
                    metrics["total_pnl"] += pnl
            
            # 计算风险指标
            if metrics["total_balance"] > 0:
                metrics["margin_usage"] = metrics["total_position_value"] / metrics["total_balance"]
            
        except Exception as e:
            logger.error(f"Error calculating account risk metrics: {e}")
        
        return metrics
    
    def _check_risk_thresholds(self, user_id: int, metrics: Dict):
        """检查风险阈值"""
        # 检查持仓数量
        if metrics["position_count"] > self.risk_manager.params.max_concurrent_positions:
            self._add_alert(
                user_id, 
                f"持仓数量超限: {metrics['position_count']} > {self.risk_manager.params.max_concurrent_positions}",
                "warning"
            )
        
        # 检查保证金使用率
        if metrics["margin_usage"] > 0.8:
            self._add_alert(
                user_id, 
                f"保证金使用率过高: {metrics['margin_usage']:.2%}",
                "warning"
            )
        
        # 检查总亏损
        if metrics["total_pnl"] < -metrics["total_balance"] * 0.1:
            self._add_alert(
                user_id, 
                f"总亏损过大: {metrics['total_pnl']:.2f}",
                "danger"
            )
    
    def _add_alert(self, user_id: int, message: str, level: str = "info"):
        """添加风险警报"""
        alert = {
            "user_id": user_id,
            "message": message,
            "level": level,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with self._lock:
            self._risk_alerts.append(alert)
            logger.info(f"Risk alert for user {user_id}: {message}")
    
    def _cleanup_alerts(self):
        """清理过期警报"""
        with self._lock:
            # 只保留最近100条警报
            if len(self._risk_alerts) > 100:
                self._risk_alerts = self._risk_alerts[-100:]
    
    def _add_system_alert(self, message: str, level: str = "info"):
        """添加系统告警"""
        alert = {
            "message": message,
            "level": level,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with self._lock:
            self._system_alerts.append(alert)
            logger.info(f"System alert: [{level.upper()}] {message}")
    
    def _cleanup_system_alerts(self):
        """清理过期系统警报"""
        with self._lock:
            # 只保留最近100条系统警报
            if len(self._system_alerts) > 100:
                self._system_alerts = self._system_alerts[-100:]
    
    def get_alerts(self, user_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
        """获取风险警报"""
        with self._lock:
            alerts = self._risk_alerts
            
            if user_id:
                alerts = [alert for alert in alerts if alert["user_id"] == user_id]
            
            # 按时间倒序排序
            alerts.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return alerts[:limit]
    
    def get_risk_summary(self, user_id: int) -> Dict:
        """获取用户风险摘要"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 获取账户信息
            account_info = self._get_account_info(conn, user_id)
            if not account_info:
                return {"error": "Account not found"}
            
            # 计算风险指标
            metrics = self._calculate_account_risk_metrics(conn, user_id, account_info)
            
            # 获取未平仓交易
            open_trades = self._get_open_trades_with_risk(conn, user_id)
            
            # 构建摘要
            summary = {
                "account": {
                    "total_balance": metrics["total_balance"],
                    "available_balance": metrics["available_balance"],
                    "total_pnl": metrics["total_pnl"]
                },
                "risk": {
                    "position_count": metrics["position_count"],
                    "margin_usage": metrics["margin_usage"],
                    "max_position_limit": self.risk_manager.params.max_concurrent_positions
                },
                "positions": [],
                "alerts": self.get_alerts(user_id, limit=10)
            }
            
            # 添加持仓信息
            for trade in open_trades:
                symbol = trade.get("symbol")
                current_price = self.data_provider.get_realtime_price(symbol)
                
                if current_price:
                    pnl = (current_price - float(trade.get("open_price", 0))) * trade.get("open_quantity", 0)
                    pnl_pct = (current_price - float(trade.get("open_price", 0))) / float(trade.get("open_price", 1))
                    
                    summary["positions"].append({
                        "symbol": symbol,
                        "open_price": trade.get("open_price"),
                        "current_price": current_price,
                        "quantity": trade.get("open_quantity"),
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "stop_loss": trade.get("stop_loss"),
                        "take_profit": trade.get("take_profit")
                    })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        with self._lock:
            return self._system_status.copy()
    
    def get_system_alerts(self, limit: int = 50) -> List[Dict]:
        """获取系统告警"""
        with self._lock:
            # 按时间倒序排序
            alerts = sorted(self._system_alerts, key=lambda x: x["timestamp"], reverse=True)
            return alerts[:limit]
    
    def get_system_summary(self) -> Dict:
        """获取系统摘要"""
        return {
            "status": self.get_system_status(),
            "alerts": self.get_system_alerts(limit=10)
        }

# 全局风险监控实例
_risk_monitor_instance = None
_risk_monitor_lock = threading.Lock()

def get_risk_monitor() -> RiskMonitor:
    """获取风险监控实例"""
    global _risk_monitor_instance
    
    with _risk_monitor_lock:
        if _risk_monitor_instance is None:
            _risk_monitor_instance = RiskMonitor()
        
    return _risk_monitor_instance

def start_risk_monitor():
    """启动风险监控服务"""
    monitor = get_risk_monitor()
    monitor.start()

def stop_risk_monitor():
    """停止风险监控服务"""
    monitor = get_risk_monitor()
    monitor.stop()
