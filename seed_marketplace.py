
import sqlite3
import os
from app.db import DB_PATH

def seed_marketplace():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Check if items exist
    cur.execute("SELECT count(*) FROM marketplace_items")
    count = cur.fetchone()[0]
    
    if count > 0:
        print(f"Marketplace already has {count} items.")
        return

    print("Seeding marketplace items...")
    items = [
        ("网格交易 Pro", "Official", "经典的网格交易策略，支持动态步长和自动止损。", 0.0, 2, 1.8, 15.5, 25.2, 120, "套利策略", "official/grid_master.py"),
        ("海龟交易法则", "Community", "著名的趋势跟随策略，适合大周期操作。", 0.0, 4, 1.2, 30.2, 45.8, 85, "趋势跟踪", "community/turtle_trend.py"),
        ("双均线突破", "System", "简单有效的均线交叉系统，新手必备。", 0.0, 3, 1.5, 22.8, 32.5, 95, "趋势跟踪", "system/ma_cross.py"),
        ("RSI 超买超卖", "TraderX", "利用 RSI 指标捕捉反转机会的高胜率策略。", 9.9, 3, 1.6, 18.5, 28.7, 150, "均值回归", "traderx/rsi_reversal.py"),
        ("多因子选股", "QuantLab", "基于市值、动量和价值因子的综合选股模型。", 29.9, 2, 2.1, 12.5, 38.2, 65, "机器学习", "quantlab/multi_factor.py")
    ]
    
    cur.executemany(
        "INSERT INTO marketplace_items (name, author, description, price, risk_level, sharpe_ratio, max_drawdown, avg_return, total_trades, category, file_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        items
    )
    conn.commit()
    print("Seeding complete.")
    conn.close()

if __name__ == "__main__":
    seed_marketplace()
