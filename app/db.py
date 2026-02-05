import os
import sqlite3
import logging
from typing import Optional, Generator, List, Dict, Any
from contextlib import contextmanager
from queue import Queue, Empty
import pandas as pd
import akshare as ak
import json
import time
import threading
import asyncio
import atexit

from app.core.config import settings

logger = logging.getLogger(__name__)

BASE_DIR = settings.BASE_DIR
DATA_DIR = settings.DATA_DIR
DB_PATH = settings.DB_PATH


# ============ 数据库连接管理 ============
class DatabaseConnectionPool:
    """
    真正的 SQLite 连接池实现
    
    特性:
    - 连接复用：从池中获取连接，用完归还而非关闭
    - 懒初始化：首次使用时创建连接
    - 健康检查：归还前验证连接有效性
    - 优雅关闭：应用退出时关闭所有连接
    """
    
    def __init__(self, db_path: str, pool_size: int = 5, timeout: int = 30):
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        self._pool: Queue = Queue(maxsize=pool_size)
        self._created_count = 0
        self._lock = threading.Lock()
        self._closed = False
        
        # 确保数据目录存在
        os.makedirs(os.path.dirname(db_path) or DATA_DIR, exist_ok=True)
        logger.info(f"初始化数据库连接池: {db_path} (大小: {pool_size})")
        
        # 注册退出清理
        atexit.register(self.close_all)
    
    def _create_connection(self) -> sqlite3.Connection:
        """创建新的数据库连接"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,
            isolation_level='DEFERRED'
        )
        conn.row_factory = sqlite3.Row
        
        # 性能优化 PRAGMA
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        
        return conn
    
    def _is_connection_valid(self, conn: sqlite3.Connection) -> bool:
        """检查连接是否有效"""
        try:
            conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False
    
    def acquire(self) -> sqlite3.Connection:
        """从池中获取连接"""
        if self._closed:
            raise RuntimeError("连接池已关闭")
        
        # 尝试从池中获取
        try:
            conn = self._pool.get_nowait()
            # 验证连接有效性
            if self._is_connection_valid(conn):
                return conn
            else:
                # 连接失效，关闭并创建新的
                try:
                    conn.close()
                except:
                    pass
                with self._lock:
                    self._created_count -= 1
        except Empty:
            pass
        
        # 池为空或连接失效，尝试创建新连接
        with self._lock:
            if self._created_count < self.pool_size:
                conn = self._create_connection()
                self._created_count += 1
                return conn
        
        # 已达最大连接数，阻塞等待
        try:
            conn = self._pool.get(timeout=self.timeout)
            if self._is_connection_valid(conn):
                return conn
            else:
                try:
                    conn.close()
                except:
                    pass
                with self._lock:
                    self._created_count -= 1
                # 递归重试
                return self.acquire()
        except Empty:
            raise TimeoutError(f"获取数据库连接超时 ({self.timeout}s)")
    
    def release(self, conn: sqlite3.Connection):
        """归还连接到池"""
        if self._closed:
            try:
                conn.close()
            except:
                pass
            return
        
        try:
            conn.rollback()  # 清理未提交事务
            self._pool.put_nowait(conn)
        except:
            # 池已满，关闭连接
            try:
                conn.close()
            except:
                pass
            with self._lock:
                self._created_count -= 1
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """上下文管理器：获取连接并自动归还"""
        conn = self.acquire()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            self.release(conn)
    
    def close_all(self):
        """关闭池中所有连接"""
        self._closed = True
        closed_count = 0
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
                closed_count += 1
            except Empty:
                break
            except:
                pass
        
        if closed_count > 0:
            logger.info(f"已关闭 {closed_count} 个数据库连接")
    
    @property
    def stats(self) -> Dict[str, int]:
        """连接池统计"""
        return {
            "pool_size": self.pool_size,
            "created": self._created_count,
            "available": self._pool.qsize()
        }


# 全局连接池实例
_db_pool = DatabaseConnectionPool(DB_PATH, pool_size=5, timeout=30)


def get_conn() -> sqlite3.Connection:
    """
    获取数据库连接（直接方式）
    
    注意：调用者负责关闭连接，推荐使用 get_db_context() 或 get_db()
    """
    return _db_pool.acquire()


def release_conn(conn: sqlite3.Connection):
    """归还数据库连接"""
    _db_pool.release(conn)


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI依赖注入：为每个请求提供数据库连接"""
    with _db_pool.get_connection() as conn:
        yield conn


@contextmanager
def get_db_context() -> Generator[sqlite3.Connection, None, None]:
    """上下文管理器：自动处理连接和事务"""
    with _db_pool.get_connection() as conn:
        yield conn


def get_pool_stats() -> Dict[str, int]:
    """获取连接池统计信息"""
    return _db_pool.stats


def ensure_schema(conn: sqlite3.Connection):
    # Auth, Core, Trading, and other tables
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS visual_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            description TEXT,
            graph_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            type TEXT NOT NULL, -- 'sim' or 'real'
            initial_capital REAL DEFAULT 100000.0,
            balance REAL DEFAULT 100000.0,
            currency TEXT DEFAULT 'CNY',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            UNIQUE(user_id, type)
        )
        """
    )
    # 扩展 accounts 表
    try:
        cur = conn.execute("PRAGMA table_info(accounts)")
        cols = [r[1] for r in cur.fetchall()]
        if "total_assets" not in cols:
            conn.execute("ALTER TABLE accounts ADD COLUMN total_assets REAL DEFAULT 100000.0")
        if "day_pnl" not in cols:
            conn.execute("ALTER TABLE accounts ADD COLUMN day_pnl REAL DEFAULT 0.0")
    except Exception:
        pass

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER,
            symbol TEXT NOT NULL,
            name TEXT,
            quantity INTEGER NOT NULL,
            average_cost REAL NOT NULL,
            current_price REAL,
            market_value REAL,
            unrealized_pnl REAL,
            unrealized_pnl_pct REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(account_id) REFERENCES accounts(id),
            UNIQUE(account_id, symbol)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            account_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            side TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            open_price REAL,
            open_quantity INTEGER,
            open_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            close_price REAL,
            close_time TIMESTAMP,
            pnl REAL DEFAULT 0.0,
            pnl_pct REAL DEFAULT 0.0,
            strategy_note TEXT,
            stop_loss REAL,
            take_profit REAL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    # 兼容旧库：若无 stop_loss/take_profit 列则添加
    try:
        cur = conn.execute("PRAGMA table_info(trades)")
        cols = [r[1] for r in cur.fetchall()]
        if "stop_loss" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN stop_loss REAL")
        if "take_profit" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN take_profit REAL")
    except Exception:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            stock_name TEXT,
            stock_code TEXT,
            sector TEXT,
            analyze_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            win_rate REAL,
            rating TEXT,
            comment TEXT,
            json_data TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS psychology (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            trade_id INTEGER,
            account_type TEXT,
            log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            mood TEXT,
            decision_basis TEXT,
            confidence_level INTEGER,
            notes TEXT,
            FOREIGN KEY(trade_id) REFERENCES trades(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            stock_name TEXT NOT NULL,
            stock_code TEXT NOT NULL,
            sector TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            note TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            UNIQUE(user_id, stock_code)
        )
        """
    )
    # 条件单（价格/时间触发）占位表
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS condition_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            account_type TEXT NOT NULL DEFAULT 'sim',
            symbol TEXT NOT NULL,
            name TEXT,
            side TEXT NOT NULL,
            trigger_type TEXT NOT NULL,
            trigger_value REAL,
            trigger_time TEXT,
            price REAL,
            quantity INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            filled_at TIMESTAMP,
            note TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    # AI 提示词模板（可配置分析/选股模板，参考 go-stock）
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            key TEXT NOT NULL,
            content TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt_templates_key ON prompt_templates(key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt_templates_scope ON prompt_templates(scope)")
    # 预警规则（价格上穿/下穿，站内提醒）
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT NOT NULL,
            name TEXT,
            alert_type TEXT NOT NULL,
            value REAL NOT NULL,
            triggered_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_user_pending ON alerts(user_id, triggered_at)")
    conn.commit()

def init_db():
    with get_db_context() as conn:
        ensure_schema(conn)

# --- Auth Functions ---
def create_user(conn: sqlite3.Connection, username: str, password_hash: str, role: str = 'user'):
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role)
        )
        user_id = cur.lastrowid
        conn.execute("INSERT OR IGNORE INTO accounts (user_id, type) VALUES (?, 'sim')", (user_id,))
        conn.execute("INSERT OR IGNORE INTO accounts (user_id, type) VALUES (?, 'real')", (user_id,))
        conn.commit()
        return get_user_by_id(conn, user_id)
    except sqlite3.IntegrityError:
        conn.rollback()
        return None

def get_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    return dict(row) if row else None

def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> Optional[Dict[str, Any]]:
    cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    return dict(row) if row else None

# --- Trading Functions ---
def get_accounts(conn: sqlite3.Connection, user_id: int) -> List[Dict[str, Any]]:
    cur = conn.execute("SELECT * FROM accounts WHERE user_id = ?", (user_id,))
    return [dict(row) for row in cur.fetchall()]

def create_trade(conn: sqlite3.Connection, user_id: int, trade_data: Dict[str, Any]):
    conn.execute(
        """
        INSERT INTO trades (user_id, account_type, symbol, name, side, open_price, open_quantity, strategy_note, stop_loss, take_profit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id, trade_data['account_type'], trade_data['symbol'], trade_data['name'],
            trade_data['side'], trade_data['open_price'], trade_data['open_quantity'],
            trade_data.get('strategy_note', ''),
            trade_data.get('stop_loss'), trade_data.get('take_profit')
        )
    )
    conn.commit()


def get_open_trades_with_risk(conn: sqlite3.Connection, user_id: int) -> List[Dict[str, Any]]:
    """获取带止损/止盈的未平仓单，供风控检查"""
    cur = conn.execute(
        "SELECT * FROM trades WHERE user_id = ? AND status = 'open' AND (stop_loss IS NOT NULL OR take_profit IS NOT NULL) ORDER BY id",
        (user_id,)
    )
    return [dict(row) for row in cur.fetchall()]


def create_condition_order(conn: sqlite3.Connection, user_id: int, data: Dict[str, Any]) -> Optional[int]:
    try:
        cur = conn.execute(
            """INSERT INTO condition_orders (user_id, account_type, symbol, name, side, trigger_type, trigger_value, trigger_time, price, quantity, status, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (
                user_id, data.get('account_type', 'sim'), data['symbol'], data.get('name', ''),
                data['side'], data['trigger_type'], data.get('trigger_value'), data.get('trigger_time'),
                data.get('price', 0), data.get('quantity', 100), data.get('note', '')
            )
        )
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        logger.warning(f"create_condition_order: {e}")
        conn.rollback()
        return None


def cancel_condition_order(conn: sqlite3.Connection, user_id: int, order_id: int) -> bool:
    cur = conn.execute("UPDATE condition_orders SET status = 'cancelled' WHERE id = ? AND user_id = ? AND status = 'pending'", (order_id, user_id))
    conn.commit()
    return cur.rowcount > 0


def get_pending_condition_orders(conn: sqlite3.Connection, trigger_type: Optional[str] = None) -> List[Dict[str, Any]]:
    q = "SELECT * FROM condition_orders WHERE status = 'pending'"
    params = []
    if trigger_type:
        q += " AND trigger_type = ?"
        params.append(trigger_type)
    q += " ORDER BY id"
    cur = conn.execute(q, params)
    return [dict(row) for row in cur.fetchall()]


def fill_condition_order(conn: sqlite3.Connection, order_id: int):
    conn.execute("UPDATE condition_orders SET status = 'filled', filled_at = CURRENT_TIMESTAMP WHERE id = ?", (order_id,))
    conn.commit()


def create_alert(conn: sqlite3.Connection, user_id: int, symbol: str, name: str, alert_type: str, value: float) -> Optional[int]:
    try:
        cur = conn.execute(
            "INSERT INTO alerts (user_id, symbol, name, alert_type, value) VALUES (?, ?, ?, ?, ?)",
            (user_id, symbol, name or symbol, alert_type, value)
        )
        conn.commit()
        return cur.lastrowid
    except Exception:
        conn.rollback()
        return None


def list_alerts(conn: sqlite3.Connection, user_id: int, pending_only: bool = False) -> List[Dict[str, Any]]:
    if pending_only:
        cur = conn.execute("SELECT * FROM alerts WHERE user_id = ? AND triggered_at IS NULL ORDER BY id DESC", (user_id,))
    else:
        cur = conn.execute("SELECT * FROM alerts WHERE user_id = ? ORDER BY id DESC", (user_id,))
    return [dict(row) for row in cur.fetchall()]


def delete_alert(conn: sqlite3.Connection, user_id: int, alert_id: int) -> bool:
    cur = conn.execute("DELETE FROM alerts WHERE id = ? AND user_id = ?", (alert_id, user_id))
    conn.commit()
    return cur.rowcount > 0


def get_pending_alerts(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.execute("SELECT * FROM alerts WHERE triggered_at IS NULL ORDER BY id")
    return [dict(row) for row in cur.fetchall()]


def mark_alert_triggered(conn: sqlite3.Connection, alert_id: int):
    conn.execute("UPDATE alerts SET triggered_at = CURRENT_TIMESTAMP WHERE id = ?", (alert_id,))
    conn.commit()

def close_trade(conn: sqlite3.Connection, user_id: int, trade_id: int, close_price: float) -> bool:
    cur = conn.execute("SELECT * FROM trades WHERE id = ? AND user_id = ? AND status = 'open'", (trade_id, user_id))
    trade = cur.fetchone()
    if not trade:
        return False

    pnl = (close_price - trade['open_price']) * trade['open_quantity']
    pnl_pct = (pnl / (trade['open_price'] * trade['open_quantity'])) * 100

    conn.execute(
        """
        UPDATE trades
        SET status = 'closed', close_price = ?, close_time = CURRENT_TIMESTAMP, pnl = ?, pnl_pct = ?
        WHERE id = ?
        """,
        (close_price, pnl, pnl_pct, trade_id)
    )
    conn.execute(
        "UPDATE accounts SET balance = balance + ? WHERE user_id = ? AND type = ?",
        (pnl, user_id, trade['account_type'])
    )
    conn.commit()
    return True

def get_trades(conn: sqlite3.Connection, user_id: int, account_type: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    query = "SELECT * FROM trades WHERE user_id = ? AND account_type = ?"
    params = [user_id, account_type]
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY open_time DESC"
    cur = conn.execute(query, params)
    return [dict(row) for row in cur.fetchall()]

def get_history(conn: sqlite3.Connection, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    cur = conn.execute("SELECT * FROM history WHERE user_id = ? ORDER BY analyze_time DESC LIMIT ?", (user_id, limit))
    return [dict(row) for row in cur.fetchall()]

def add_history(conn: sqlite3.Connection, user_id: int, stock_name: str, stock_code: str, sector: str, win_rate: float, rating: str, comment: str, json_data: str):
    conn.execute(
        """
        INSERT INTO history (user_id, stock_name, stock_code, sector, win_rate, rating, comment, json_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, stock_name, stock_code, sector, win_rate, rating, comment, json_data)
    )
    conn.commit()

def clear_history(conn: sqlite3.Connection, user_id: int):
    conn.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    conn.commit()

def search(conn: sqlite3.Connection, q: str) -> List[Dict[str, Any]]:
    query = f"%{q}%"
    cur = conn.execute("SELECT id, name, size FROM docs WHERE name LIKE ? OR content LIKE ?", (query, query))
    return [dict(row) for row in cur.fetchall()]

def get_psychology_stats(conn: sqlite3.Connection, user_id: int) -> Dict[str, Any]:
    # This is a placeholder, you'd build a real query here
    return {"moods": [], "decisions": []}

# --- Watchlist Functions ---
def get_watchlist(conn: sqlite3.Connection, user_id: int) -> List[Dict[str, Any]]:
    cur = conn.execute("SELECT * FROM watchlist WHERE user_id = ? ORDER BY added_at DESC", (user_id,))
    return [dict(row) for row in cur.fetchall()]

def add_watchlist(conn: sqlite3.Connection, user_id: int, stock_name: str, stock_code: str, sector: Optional[str] = None) -> bool:
    try:
        conn.execute(
            "INSERT INTO watchlist (user_id, stock_name, stock_code, sector) VALUES (?, ?, ?, ?)",
            (user_id, stock_name, stock_code, sector)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def remove_watchlist(conn: sqlite3.Connection, user_id: int, stock_code: str):
    conn.execute("DELETE FROM watchlist WHERE user_id = ? AND stock_code = ?", (user_id, stock_code))
    conn.commit()

# --- Visual Strategy Functions ---
def create_visual_strategy(conn: sqlite3.Connection, user_id: int, name: str, description: Optional[str], graph_json: str) -> Optional[int]:
    try:
        cur = conn.execute(
            "INSERT INTO visual_strategies (user_id, name, description, graph_json, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (user_id, name, description, graph_json)
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Error creating visual strategy: {e}")
        conn.rollback()
        return None

def get_visual_strategy(conn: sqlite3.Connection, strategy_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    cur = conn.execute("SELECT * FROM visual_strategies WHERE id = ? AND user_id = ?", (strategy_id, user_id))
    row = cur.fetchone()
    return dict(row) if row else None

def get_all_visual_strategies(conn: sqlite3.Connection, user_id: int) -> List[Dict[str, Any]]:
    cur = conn.execute("SELECT id, name, description, updated_at FROM visual_strategies WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))
    return [dict(row) for row in cur.fetchall()]

def update_visual_strategy(conn: sqlite3.Connection, strategy_id: int, user_id: int, name: str, description: Optional[str], graph_json: str) -> bool:
    try:
        cur = conn.execute(
            """
            UPDATE visual_strategies
            SET name = ?, description = ?, graph_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (name, description, graph_json, strategy_id, user_id)
        )
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Error updating visual strategy: {e}")
        conn.rollback()
        return False

def delete_visual_strategy(conn: sqlite3.Connection, strategy_id: int, user_id: int) -> bool:
    try:
        cur = conn.execute("DELETE FROM visual_strategies WHERE id = ? AND user_id = ?", (strategy_id, user_id))
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Error deleting visual strategy: {e}")
        conn.rollback()
        return False


# --- AI 提示词模板（参考 go-stock） ---
def list_prompt_templates(conn: sqlite3.Connection, user_id: Optional[int] = None, scope: Optional[str] = None) -> List[Dict[str, Any]]:
    query = "SELECT id, name, key, scope, created_at, SUBSTR(content, 1, 200) as preview FROM prompt_templates WHERE 1=1"
    params = []
    if user_id is not None:
        query += " AND (user_id = ? OR scope = 'system')"
        params.append(user_id)
    if scope:
        query += " AND scope = ?"
        params.append(scope)
    query += " ORDER BY scope ASC, updated_at DESC"
    cur = conn.execute(query, params)
    return [dict(row) for row in cur.fetchall()]


def get_prompt_template(conn: sqlite3.Connection, key: str, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT id, name, key, content, scope FROM prompt_templates WHERE key = ? AND (user_id = ? OR scope = 'system') ORDER BY scope DESC LIMIT 1",
        (key, user_id)
    )
    row = cur.fetchone()
    return dict(row) if row else None


def save_prompt_template(conn: sqlite3.Connection, user_id: int, name: str, key: str, content: str, scope: str = "user") -> Optional[int]:
    try:
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        cur = conn.execute(
            "INSERT INTO prompt_templates (user_id, name, key, content, scope, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, name, key, content, scope, now)
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Error saving prompt template: {e}")
        conn.rollback()
        return None


# --- Market Data Helper Functions (Wrappers for AkShare) ---

# 使用统一的 TTLCache 替代手动缓存实现
from app.cache_utils import TTLCache

# 行情缓存配置
SPOT_CACHE_TTL = 45  # 秒
SPOT_CACHE_MAX_ROWS = 3000  # 最大行数限制

# 使用 TTLCache 实现行情缓存
_spot_ttl_cache = TTLCache(ttl=SPOT_CACHE_TTL, max_entries=1, name="spot_data")
_spot_lock = threading.Lock()


# ========== 兼容层 ==========
# 保留旧的 _spot_cache 字典接口，以兼容 market.py 和 system.py
class _SpotCacheCompat(dict):
    """
    兼容层：让旧代码可以通过 _spot_cache["data"] 和 _spot_cache["time"] 访问缓存
    实际数据存储在 _spot_ttl_cache 中
    """
    def __getitem__(self, key):
        if key == "data":
            cached = _spot_ttl_cache.get("all_stocks_spot")
            return cached if cached is not None else pd.DataFrame()
        elif key == "time":
            entry = _spot_ttl_cache.cache.get("all_stocks_spot")
            if entry:
                return entry.get("timestamp", 0)
            return 0
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        if key == "data":
            _spot_ttl_cache.set("all_stocks_spot", value)
        elif key == "time":
            # 时间戳由 TTLCache 自动管理，忽略外部设置
            pass
        else:
            super().__setitem__(key, value)
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# 向后兼容：旧代码可以继续使用 _spot_cache
_spot_cache = _SpotCacheCompat()


def clear_spot_cache():
    """释放行情缓存，供系统「清理缓存」使用"""
    _spot_ttl_cache.clear()
    logger.info("行情缓存已清理")


def _get_cached_spot():
    """从缓存获取行情数据"""
    return _spot_ttl_cache.get("all_stocks_spot")


def _set_cached_spot(df: pd.DataFrame):
    """设置行情缓存"""
    _spot_ttl_cache.set("all_stocks_spot", df)



def get_realtime_quotes_sina(stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    从新浪财经获取指定股票或所有A股的实时行情数据。
    返回包含股票代码、名称、最新价、涨跌幅等信息的DataFrame。
    """
    try:
        if stock_codes:
            # 格式化股票代码以适应新浪接口
            formatted_codes = []
            for code in stock_codes:
                if code.startswith(('60', '688')):  # 沪市
                    formatted_codes.append(f"sh{code}")
                elif code.startswith(('00', '30')):  # 深市
                    formatted_codes.append(f"sz{code}")
                else:
                    formatted_codes.append(code) # 保持原样，可能是指数或其他
            
            if not formatted_codes:
                logger.warning("No formatted stock codes to fetch from Sina.")
                return pd.DataFrame()

            # 新浪接口一次请求的股票数量有限，分批请求
            batch_size = 800  # 经验值，避免URL过长
            all_quotes_list = []

            for i in range(0, len(formatted_codes), batch_size):
                batch_codes = formatted_codes[i:i + batch_size]
                
                # 使用 AkShare 的 stock_zh_a_spot_sina 接口，传入 symbol 参数
                # 注意：ak.stock_zh_a_spot_sina 的 symbol 参数通常用于指定单个股票或指数
                # 对于批量获取，AkShare 内部可能没有直接支持传入列表的接口
                # 实际测试发现 ak.stock_zh_a_spot_sina 默认获取所有A股，且没有直接传入股票代码列表的参数
                # 因此，我们采取获取所有A股，然后筛选的方式，或者使用更底层的接口
                
                # 为了简化，我们直接调用 ak.stock_zh_a_spot_sina 获取所有A股，然后进行筛选
                # 这种方式在 stock_codes 列表很长时效率不高，但对于静态列表作为 fallback 足够
                sina_df_all = ak.stock_zh_a_spot_sina()
                if sina_df_all.empty:
                    logger.warning("AkShare stock_zh_a_spot_sina returned empty.")
                    continue
                
                # 筛选出需要的股票
                sina_df_filtered = sina_df_all[sina_df_all['code'].isin(formatted_codes)]
                all_quotes_list.append(sina_df_filtered)
            
            sina_df = pd.concat(all_quotes_list, ignore_index=True) if all_quotes_list else pd.DataFrame()

        else:
            # 如果没有提供股票代码列表，则获取所有A股
            sina_df = ak.stock_zh_a_spot_sina()
            if sina_df.empty:
                logger.warning("AkShare stock_zh_a_spot_sina returned empty.")
                return pd.DataFrame()
        
        # 统一字段名，使其与 stock_zh_a_spot_em 尽可能一致
        mapped_df = pd.DataFrame()
        if 'code' in sina_df.columns and 'name' in sina_df.columns and 'trade' in sina_df.columns and 'changepercent' in sina_df.columns:
            mapped_df['代码'] = sina_df['code'].apply(lambda x: x[2:] if len(x) > 2 and (x.startswith('sh') or x.startswith('sz')) else x) # 移除sh/sz前缀
            mapped_df['名称'] = sina_df['name']
            mapped_df['最新价'] = pd.to_numeric(sina_df['trade'], errors='coerce')
            mapped_df['涨跌幅'] = pd.to_numeric(sina_df['changepercent'], errors='coerce')
            
            if 'open' in sina_df.columns: mapped_df['开盘'] = pd.to_numeric(sina_df['open'], errors='coerce')
            if 'settlement' in sina_df.columns: mapped_df['昨收'] = pd.to_numeric(sina_df['settlement'], errors='coerce')
            if 'high' in sina_df.columns: mapped_df['最高'] = pd.to_numeric(sina_df['high'], errors='coerce')
            if 'low' in sina_df.columns: mapped_df['最低'] = pd.to_numeric(sina_df['low'], errors='coerce')
            if 'volume' in sina_df.columns: mapped_df['成交量'] = pd.to_numeric(sina_df['volume'], errors='coerce')
            if 'amount' in sina_df.columns: mapped_df['成交额'] = pd.to_numeric(sina_df['amount'], errors='coerce')
            
            # 确保关键列为数值类型，并处理NaN
            mapped_df['最新价'] = mapped_df['最新价'].fillna(0.0)
            mapped_df['涨跌幅'] = mapped_df['涨跌幅'].fillna(0.0)
            return mapped_df
        else:
            logger.warning(f"Sina data missing expected columns: {sina_df.columns.tolist()}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error fetching real-time quotes from Sina: {e}", exc_info=True)
        return pd.DataFrame()


async def get_all_stocks_spot():
    """
    获取所有 A 股实时行情数据
    
    数据源优先级:
    1. TTLCache 缓存 (45秒TTL)
    2. AkShare stock_zh_a_spot_em
    3. 新浪财经 (通过静态列表)
    4. 静态 CSV 文件
    """
    # 1. 尝试从缓存获取
    cached = _get_cached_spot()
    if cached is not None and not cached.empty:
        return cached
    
    # 防止并发请求
    if _spot_lock.locked():
        await asyncio.sleep(1)
        cached = _get_cached_spot()
        if cached is not None and not cached.empty:
            return cached

    loop = asyncio.get_event_loop()
    df = pd.DataFrame()
    
    try:
        # 2. 尝试从 AkShare 获取实时数据
        from app.utils.akshare_wrapper import async_ak_call
        df = await async_ak_call(
            lambda: ak.stock_zh_a_spot_em(),
            timeout=30.0,
            name="spot_em"
        )
        if df is None or df.empty:
            raise ValueError("AkShare stock_zh_a_spot_em returned empty.")
        logger.info("Successfully fetched spot data from AkShare.")
        
    except Exception as e:
        logger.warning(f"AkShare spot fetch error: {e}. Falling back to Sina.")
        try:
            # 3. AkShare 失败，尝试从新浪获取
            _base = os.path.dirname(os.path.abspath(__file__))
            _csv_path = os.path.join(_base, "static", "stock_list.csv")
            static_df = pd.read_csv(_csv_path, dtype=str)
            if static_df.empty or '代码' not in static_df.columns:
                raise ValueError("Static stock list is empty or missing '代码' column.")
            
            stock_codes = static_df['代码'].tolist()
            
            # 使用线程池并行获取 Sina 数据
            def fetch_sina_batch(codes_batch):
                return get_realtime_quotes_sina(codes_batch)

            batch_size = 500
            batches = [stock_codes[i:i + batch_size] for i in range(0, len(stock_codes), batch_size)]
            tasks = [loop.run_in_executor(None, fetch_sina_batch, batch) for batch in batches]
            results = await asyncio.gather(*tasks)
            
            df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

            if df is None or df.empty:
                raise ValueError("Sina real-time quotes returned empty.")
            logger.info("Successfully fetched spot data from Sina.")
            
        except Exception as e_sina:
            logger.error(f"Sina spot fetch error: {e_sina}. Falling back to cache/static.")
            
            # 尝试返回过期缓存
            cached = _get_cached_spot()
            if cached is not None and not cached.empty:
                logger.warning("Returning expired cached data due to fetch failures.")
                return cached
            
            # 4. 最后 fallback 到静态 CSV
            try:
                _base = os.path.dirname(os.path.abspath(__file__))
                _csv_path = os.path.join(_base, "static", "stock_list.csv")
                df = pd.read_csv(_csv_path, dtype=str)
                if '涨跌幅' not in df.columns:
                    df['涨跌幅'] = 0.0
                if '最新价' not in df.columns:
                    df['最新价'] = 0.0
                logger.warning("Loaded static stock list as a last resort.")
            except Exception as e_csv:
                logger.error(f"Static stock list read error: {e_csv}. Returning empty DataFrame.")
                return pd.DataFrame()

    # 更新缓存
    if not df.empty:
        # 瘦身：内存中只保留前 SPOT_CACHE_MAX_ROWS 条
        pct_col = next((c for c in df.columns if "涨跌幅" in c), None)
        if pct_col is not None and len(df) > SPOT_CACHE_MAX_ROWS:
            df = df.copy()
            df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce").fillna(0)
            df = df.sort_values(by=pct_col, ascending=False).head(SPOT_CACHE_MAX_ROWS)
        
        with _spot_lock:
            _set_cached_spot(df)
    
    return df


async def get_market_rankings():
    # Mock or Real
    # Return structure: {'sectors': [...], 'stocks': [...]}
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        # Top Sectors
        df_sector = await loop.run_in_executor(None, lambda: ak.stock_board_industry_name_em())
        sectors = []
        if df_sector is not None and not df_sector.empty:
            cols = df_sector.columns.tolist()
            pct_col = next((c for c in cols if '涨跌幅' in c), '涨跌幅')
            name_col = next((c for c in cols if ('板块名称' in c) or ('名称' in c)), '板块名称')
            df_sector[pct_col] = pd.to_numeric(df_sector[pct_col], errors='coerce').fillna(0)
            top5 = df_sector.sort_values(by=pct_col, ascending=False).head(5)
            sectors = [{"name": r[name_col], "pct": float(r[pct_col])} for _, r in top5.iterrows()]
            
        # Top Stocks
        df_stock = await get_all_stocks_spot()
        stocks = []
        if not df_stock.empty:
            df_stock['涨跌幅'] = pd.to_numeric(df_stock['涨跌幅'], errors='coerce').fillna(0)
            top5 = df_stock.sort_values(by='涨跌幅', ascending=False).head(5)
            stocks = [
                {
                    "name": r['名称'], 
                    "code": str(r['代码']),
                    "price": float(r['最新价']),
                    "pct": float(r['涨跌幅'])
                } for _, r in top5.iterrows()
            ]
        else:
            # Fallback: Try another AkShare endpoint or return empty list
            stocks = []
            
        return {"sectors": sectors, "stocks": stocks}
    except:
        return {"sectors": [], "stocks": []}

async def get_latest_news():
    news = []
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        df_list = []
        try:
            df_list.append(await loop.run_in_executor(None, lambda: ak.stock_info_global_cls(symbol="A股24小时电报")))
        except Exception: pass
        try:
            df_list.append(await loop.run_in_executor(None, lambda: ak.stock_info_global_cls(symbol="A股要闻")))
        except Exception: pass
        
        def collect(df, limit=20):
            if df is None or df.empty: 
                return []
            cols = df.columns.tolist()
            time_col = next((c for c in cols if ('时间' in c) or ('time' in c.lower()) or ('发布时间' in c)), cols[0])
            title_col = next((c for c in cols if ('标题' in c) or ('title' in c.lower())), cols[1] if len(cols)>1 else cols[0])
            url_col = next((c for c in cols if ('链接' in c) or ('url' in c.lower())), None)
            items = []
            for _, row in df.head(limit).iterrows():
                title = str(row.get(title_col, ''))
                url = str(row.get(url_col, '#')) if url_col else "#"
                
                if url == '#' or not url:
                    url = f"https://www.baidu.com/s?wd={title}"
                
                items.append({
                    "title": title,
                    "time": str(row.get(time_col, '')),
                    "url": url
                })
            return items
        
        agg = []
        for df in df_list:
            agg += collect(df, 20)
        seen = set()
        for item in agg:
            t = item['title']
            if t and t not in seen:
                seen.add(t)
                news.append(item)
        
    except Exception as e:
        logger.error(f"News fetch error: {e}")
    return news[:50]

async def get_north_fund():
    """
    获取最新的沪深港通资金流向（北向/南向）。

    兼容不同 AkShare 版本：优先使用 `stock_hsgt_fund_flow_summary_em()`，
    老版本/缺失时再尝试其它接口并转换为统一字段：
    - 资金方向
    - 成交净买额
    - 成交总额
    """
    import asyncio
    import pandas as pd
    import datetime as _dt

    loop = asyncio.get_event_loop()

    def _to_records(df: "pd.DataFrame") -> list[dict]:
        if df is None or df.empty:
            return []
        return df.to_dict("records")

    try:
        # 1) 首选：当日摘要（字段稳定，且你项目里 data_provider 已在使用）
        if hasattr(ak, "stock_hsgt_fund_flow_summary_em"):
            df = await loop.run_in_executor(None, ak.stock_hsgt_fund_flow_summary_em)
            rec = _to_records(df)
            if rec:
                return rec

        # 2) 兼容：历史接口（北向/南向），转成统一字段
        if hasattr(ak, "stock_hsgt_hist_em"):
            def _pick(df0: "pd.DataFrame") -> list[dict]:
                if df0 is None or df0.empty:
                    return []
                cols = df0.columns.tolist()
                date_col = next((c for c in cols if "日期" in c), None)
                net_col = next((c for c in cols if ("当日成交净买额" in c) or ("成交净买额" in c) or ("净买额" in c)), None)
                total_col = next((c for c in cols if ("成交总额" in c) or ("买入成交额" in c)), None)
                if date_col:
                    try:
                        df0 = df0.sort_values(by=date_col)
                    except Exception:
                        pass
                last = df0.iloc[-1]
                net = float(last.get(net_col, 0.0) or 0.0) if net_col else 0.0
                total = float(last.get(total_col, 0.0) or 0.0) if total_col else 0.0
                # 处理 NaN
                if net != net:
                    net = 0.0
                if total != total:
                    total = 0.0
                # 返回与 summary_em 对齐的字段名
                return [{
                    "日期": str(last.get(date_col, _dt.date.today().isoformat())) if date_col else _dt.date.today().isoformat(),
                    "资金方向": "",  # 由调用方/外层补齐
                    "成交净买额": net,
                    "成交总额": total,
                }]

            north_df = await loop.run_in_executor(None, lambda: ak.stock_hsgt_hist_em(symbol="北向资金"))
            south_df = await loop.run_in_executor(None, lambda: ak.stock_hsgt_hist_em(symbol="南向资金"))
            out: list[dict] = []
            for item in _pick(north_df):
                item["资金方向"] = "北向"
                out.append(item)
            for item in _pick(south_df):
                item["资金方向"] = "南向"
                out.append(item)
            if out:
                return out

        # 3) 旧接口（某些环境里可能存在，但你当前日志显示缺失）
        if hasattr(ak, "stock_hsgt_fund_flow_hsgt"):
            df = await loop.run_in_executor(None, ak.stock_hsgt_fund_flow_hsgt)
            rec = _to_records(df)
            if rec:
                return rec

        return []
    except Exception as e:
        logger.error(f"获取北向资金数据失败: {e}", exc_info=True)
        return []