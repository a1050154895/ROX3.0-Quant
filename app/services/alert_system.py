import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from app.rox_quant.data_provider import DataProvider
from app.analysis.enhanced_signals_v2 import TechnicalIndicators

logger = logging.getLogger(__name__)

class AlertSystem:
    """
    综合预警系统
    支持价格、指标、成交量、趋势等多维度预警
    """
    
    def __init__(self, db_path: str = "rox.db"):
        self.db_path = db_path
        self.data_provider = DataProvider()
        self.tech_indicators = TechnicalIndicators()
        self._lock = threading.RLock()
        self._ensure_tables()
    
    def _ensure_tables(self):
        """
        确保预警相关表存在
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 预警规则表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    alert_type TEXT NOT NULL,
                    indicator_type TEXT,
                    value REAL NOT NULL,
                    operator TEXT NOT NULL,  -- '>', '<', '>=', '<=', '=='
                    time_period INTEGER,      -- 时间周期（如 5, 15, 30, 60, 240 分钟）
                    active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_triggered_at TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            
            # 预警触发记录表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id INTEGER,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    trigger_value REAL NOT NULL,
                    current_value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(alert_id) REFERENCES alerts(id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error ensuring alert tables: {e}")
        finally:
            if conn:
                conn.close()
    
    def create_alert(self, user_id: int, symbol: str, name: str, alert_type: str, 
                    indicator_type: Optional[str] = None, value: float = 0, 
                    operator: str = ">", time_period: int = 60) -> Optional[int]:
        """
        创建预警规则
        
        Args:
            user_id: 用户ID
            symbol: 股票代码
            name: 预警名称
            alert_type: 预警类型 (price, indicator, volume, trend, technical)
            indicator_type: 指标类型 (如 macd, rsi, kdj, bollinger, ma, volume, adx, williams_r, cci, obv, vwap)
            value: 预警阈值
            operator: 操作符 (>, <, >=, <=, ==)
            time_period: 时间周期（分钟）
            
        Returns:
            预警ID，创建失败返回None
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.execute(
                "INSERT INTO alerts (user_id, symbol, name, alert_type, indicator_type, value, operator, time_period) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, symbol, name, alert_type, indicator_type, value, operator, time_period)
            )
            conn.commit()
            
            alert_id = cursor.lastrowid
            logger.info(f"Created alert {alert_id} for user {user_id}: {name} ({alert_type})")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_user_alerts(self, user_id: int, active_only: bool = True) -> List[Dict]:
        """
        获取用户的预警规则
        
        Args:
            user_id: 用户ID
            active_only: 是否只返回激活的预警
            
        Returns:
            预警规则列表
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            if active_only:
                cursor = conn.execute(
                    "SELECT id, symbol, name, alert_type, indicator_type, value, operator, time_period, active, created_at, last_triggered_at "
                    "FROM alerts WHERE user_id = ? AND active = 1 ORDER BY created_at DESC",
                    (user_id,)
                )
            else:
                cursor = conn.execute(
                    "SELECT id, symbol, name, alert_type, indicator_type, value, operator, time_period, active, created_at, last_triggered_at "
                    "FROM alerts WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,)
                )
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    "id": row[0],
                    "symbol": row[1],
                    "name": row[2],
                    "alert_type": row[3],
                    "indicator_type": row[4],
                    "value": row[5],
                    "operator": row[6],
                    "time_period": row[7],
                    "active": bool(row[8]),
                    "created_at": row[9],
                    "last_triggered_at": row[10]
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting user alerts: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def update_alert(self, alert_id: int, user_id: int, **kwargs) -> bool:
        """
        更新预警规则
        
        Args:
            alert_id: 预警ID
            user_id: 用户ID
            **kwargs: 要更新的字段
            
        Returns:
            更新成功返回True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 检查预警是否存在且属于该用户
            cursor = conn.execute(
                "SELECT 1 FROM alerts WHERE id = ? AND user_id = ?",
                (alert_id, user_id)
            )
            if not cursor.fetchone():
                return False
            
            # 构建更新语句
            updates = []
            params = []
            
            for key, value in kwargs.items():
                if key in ['name', 'alert_type', 'indicator_type', 'value', 'operator', 'time_period', 'active']:
                    updates.append(f"{key} = ?")
                    params.append(value)
            
            if not updates:
                return True
            
            params.extend([alert_id, user_id])
            query = f"UPDATE alerts SET {', '.join(updates)} WHERE id = ? AND user_id = ?"
            
            cursor = conn.execute(query, params)
            conn.commit()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error updating alert: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def delete_alert(self, alert_id: int, user_id: int) -> bool:
        """
        删除预警规则
        
        Args:
            alert_id: 预警ID
            user_id: 用户ID
            
        Returns:
            删除成功返回True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 检查预警是否存在且属于该用户
            cursor = conn.execute(
                "SELECT 1 FROM alerts WHERE id = ? AND user_id = ?",
                (alert_id, user_id)
            )
            if not cursor.fetchone():
                return False
            
            # 删除预警
            cursor = conn.execute(
                "DELETE FROM alerts WHERE id = ? AND user_id = ?",
                (alert_id, user_id)
            )
            conn.commit()
            
            logger.info(f"Deleted alert {alert_id} for user {user_id}")
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error deleting alert: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def check_alerts(self, user_id: Optional[int] = None) -> List[Dict]:
        """
        检查并触发预警
        
        Args:
            user_id: 用户ID，None表示检查所有用户的预警
            
        Returns:
            触发的预警列表
        """
        conn = None
        triggered_alerts = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 获取激活的预警
            if user_id:
                cursor = conn.execute(
                    "SELECT id, user_id, symbol, name, alert_type, indicator_type, value, operator, time_period "
                    "FROM alerts WHERE active = 1 AND user_id = ?",
                    (user_id,)
                )
            else:
                cursor = conn.execute(
                    "SELECT id, user_id, symbol, name, alert_type, indicator_type, value, operator, time_period "
                    "FROM alerts WHERE active = 1"
                )
            
            alerts = cursor.fetchall()
            
            for alert in alerts:
                alert_id, alert_user_id, symbol, name, alert_type, indicator_type, 
                threshold, operator, time_period = alert
                
                try:
                    # 检查预警是否触发
                    triggered, current_value = self._check_alert_condition(
                        symbol, alert_type, indicator_type, threshold, operator, time_period
                    )
                    
                    if triggered:
                        # 记录触发事件
                        self._record_alert_trigger(conn, alert_id, alert_user_id, symbol, 
                                                alert_type, threshold, current_value)
                        
                        # 标记预警为已触发
                        conn.execute(
                            "UPDATE alerts SET last_triggered_at = ? WHERE id = ?",
                            (datetime.now().isoformat(), alert_id)
                        )
                        conn.commit()
                        
                        triggered_alerts.append({
                            "alert_id": alert_id,
                            "user_id": alert_user_id,
                            "symbol": symbol,
                            "name": name,
                            "alert_type": alert_type,
                            "indicator_type": indicator_type,
                            "threshold": threshold,
                            "operator": operator,
                            "current_value": current_value,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    logger.error(f"Error checking alert {alert_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in check_alerts: {e}")
        finally:
            if conn:
                conn.close()
        
        return triggered_alerts
    
    def _check_alert_condition(self, symbol: str, alert_type: str, indicator_type: Optional[str], 
                             threshold: float, operator: str, time_period: int) -> tuple:
        """
        检查预警条件是否满足
        
        Args:
            symbol: 股票代码
            alert_type: 预警类型
            indicator_type: 指标类型
            threshold: 阈值
            operator: 操作符
            time_period: 时间周期
            
        Returns:
            (是否触发, 当前值)
        """
        if alert_type == "price":
            # 价格预警
            current_price = self.data_provider.get_realtime_price(symbol)
            if current_price:
                return self._evaluate_condition(current_price, threshold, operator), current_price
        
        elif alert_type == "technical":
            # 技术指标预警
            if indicator_type:
                current_value = self._get_indicator_value(symbol, indicator_type, time_period)
                if current_value is not None:
                    return self._evaluate_condition(current_value, threshold, operator), current_value
        
        elif alert_type == "volume":
            # 成交量预警
            current_volume = self._get_volume_value(symbol, time_period)
            if current_volume is not None:
                return self._evaluate_condition(current_volume, threshold, operator), current_volume
        
        elif alert_type == "trend":
            # 趋势预警
            trend_strength = self._get_trend_strength(symbol, time_period)
            if trend_strength is not None:
                return self._evaluate_condition(trend_strength, threshold, operator), trend_strength
        
        return False, 0
    
    def _get_indicator_value(self, symbol: str, indicator_type: str, time_period: int) -> Optional[float]:
        """
        获取指标值
        """
        try:
            # 获取历史数据
            ohlc = self.data_provider.get_history(symbol, days=30)
            if not ohlc or len(ohlc) < time_period:
                return None
            
            # 转换为DataFrame
            import pandas as pd
            df = pd.DataFrame(ohlc)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 计算指标
            if indicator_type == "macd":
                macd, signal, histogram = self.tech_indicators.calculate_macd(df)
                return histogram.iloc[-1] if not histogram.empty else None
            
            elif indicator_type == "rsi":
                rsi = self.tech_indicators.calculate_rsi(df, period=time_period)
                return rsi.iloc[-1] if not rsi.empty else None
            
            elif indicator_type == "kdj":
                k, d, j = self.tech_indicators.calculate_kdj(df)
                return j.iloc[-1] if not j.empty else None
            
            elif indicator_type == "bollinger":
                upper, middle, lower = self.tech_indicators.calculate_bollinger_bands(df)
                current_price = df['close'].iloc[-1]
                return (current_price - middle.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            
            elif indicator_type == "ma":
                ma = self.tech_indicators.calculate_ma(df, period=time_period)
                current_price = df['close'].iloc[-1]
                return current_price / ma.iloc[-1] if not ma.empty else None
            
            elif indicator_type == "adx":
                adx = self.tech_indicators.calculate_adx(df)
                return adx.iloc[-1] if not adx.empty else None
            
            elif indicator_type == "williams_r":
                williams_r = self.tech_indicators.calculate_williams_r(df)
                return williams_r.iloc[-1] if not williams_r.empty else None
            
            elif indicator_type == "cci":
                cci = self.tech_indicators.calculate_cci(df)
                return cci.iloc[-1] if not cci.empty else None
            
            elif indicator_type == "obv":
                obv = self.tech_indicators.calculate_obv(df)
                return obv.iloc[-1] if not obv.empty else None
            
            elif indicator_type == "vwap":
                vwap = self.tech_indicators.calculate_vwap(df)
                current_price = df['close'].iloc[-1]
                return current_price / vwap.iloc[-1] if not vwap.empty else None
            
        except Exception as e:
            logger.error(f"Error getting indicator value: {e}")
        
        return None
    
    def _get_volume_value(self, symbol: str, time_period: int) -> Optional[float]:
        """
        获取成交量值
        """
        try:
            # 获取历史数据
            ohlc = self.data_provider.get_history(symbol, days=10)
            if not ohlc or len(ohlc) < 2:
                return None
            
            # 转换为DataFrame
            import pandas as pd
            df = pd.DataFrame(ohlc)
            
            # 计算成交量变化率
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-time_period:].mean()
            if avg_volume > 0:
                return current_volume / avg_volume
            
        except Exception as e:
            logger.error(f"Error getting volume value: {e}")
        
        return None
    
    def _get_trend_strength(self, symbol: str, time_period: int) -> Optional[float]:
        """
        获取趋势强度
        """
        try:
            # 获取历史数据
            ohlc = self.data_provider.get_history(symbol, days=30)
            if not ohlc or len(ohlc) < time_period:
                return None
            
            # 转换为DataFrame
            import pandas as pd
            df = pd.DataFrame(ohlc)
            
            # 计算趋势强度（基于ADX和MA斜率）
            adx = self.tech_indicators.calculate_adx(df)
            if adx.empty:
                return None
            
            adx_value = adx.iloc[-1]
            
            # 计算MA斜率
            ma_short = self.tech_indicators.calculate_ma(df, period=20)
            ma_long = self.tech_indicators.calculate_ma(df, period=60)
            
            if not ma_short.empty and not ma_long.empty:
                # 趋势方向
                if ma_short.iloc[-1] > ma_long.iloc[-1]:
                    trend_direction = 1  # 上升趋势
                else:
                    trend_direction = -1  # 下降趋势
                
                # 趋势强度 = 方向 * ADX值
                return trend_direction * adx_value
            
        except Exception as e:
            logger.error(f"Error getting trend strength: {e}")
        
        return None
    
    def _evaluate_condition(self, current_value: float, threshold: float, operator: str) -> bool:
        """
        评估条件是否满足
        """
        if operator == ">":
            return current_value > threshold
        elif operator == "<":
            return current_value < threshold
        elif operator == ">=":
            return current_value >= threshold
        elif operator == "<=":
            return current_value <= threshold
        elif operator == "==":
            return abs(current_value - threshold) < 0.001  # 允许小误差
        return False
    
    def _record_alert_trigger(self, conn: sqlite3.Connection, alert_id: int, user_id: int, 
                             symbol: str, alert_type: str, threshold: float, current_value: float):
        """
        记录预警触发事件
        """
        try:
            conn.execute(
                "INSERT INTO alert_triggers (alert_id, user_id, symbol, alert_type, trigger_value, current_value) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (alert_id, user_id, symbol, alert_type, threshold, current_value)
            )
        except Exception as e:
            logger.error(f"Error recording alert trigger: {e}")
    
    def get_alert_history(self, user_id: int, limit: int = 100) -> List[Dict]:
        """
        获取用户的预警触发历史
        
        Args:
            user_id: 用户ID
            limit: 返回记录数限制
            
        Returns:
            预警触发历史列表
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.execute(
                "SELECT at.id, at.alert_id, at.symbol, a.name, at.alert_type, a.indicator_type, "
                "at.trigger_value, at.current_value, at.timestamp "
                "FROM alert_triggers at "
                "JOIN alerts a ON at.alert_id = a.id "
                "WHERE at.user_id = ? "
                "ORDER BY at.timestamp DESC LIMIT ?",
                (user_id, limit)
            )
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "id": row[0],
                    "alert_id": row[1],
                    "symbol": row[2],
                    "alert_name": row[3],
                    "alert_type": row[4],
                    "indicator_type": row[5],
                    "trigger_value": row[6],
                    "current_value": row[7],
                    "timestamp": row[8]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting alert history: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_supported_indicators(self) -> List[Dict]:
        """
        获取支持的指标列表
        
        Returns:
            指标列表
        """
        return [
            {"type": "macd", "name": "MACD", "description": "MACD指标"},
            {"type": "rsi", "name": "RSI", "description": "相对强弱指标"},
            {"type": "kdj", "name": "KDJ", "description": "随机指标"},
            {"type": "bollinger", "name": "布林带", "description": "布林带指标"},
            {"type": "ma", "name": "移动平均线", "description": "移动平均线"},
            {"type": "adx", "name": "ADX", "description": "平均趋向指标"},
            {"type": "williams_r", "name": "威廉指标", "description": "威廉超买超卖指标"},
            {"type": "cci", "name": "CCI", "description": "顺势指标"},
            {"type": "obv", "name": "OBV", "description": "能量潮指标"},
            {"type": "vwap", "name": "VWAP", "description": "成交量加权平均价"}
        ]

# 全局预警系统实例
_alert_system_instance = None
_alert_system_lock = threading.Lock()

def get_alert_system() -> AlertSystem:
    """
    获取预警系统实例
    """
    global _alert_system_instance
    
    with _alert_system_lock:
        if _alert_system_instance is None:
            _alert_system_instance = AlertSystem()
        
    return _alert_system_instance
