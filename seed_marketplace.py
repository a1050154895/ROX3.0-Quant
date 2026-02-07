
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
        ("网格交易 Pro", "Official", "经典的网格交易策略，支持动态步长和自动止损。", 0.0, "official/grid_master.py"),
        ("海龟交易法则", "Community", "著名的趋势跟随策略，适合大周期操作。", 0.0, "community/turtle_trend.py"),
        ("双均线突破", "System", "简单有效的均线交叉系统，新手必备。", 0.0, "system/ma_cross.py"),
        ("RSI 超买超卖", "TraderX", "利用 RSI 指标捕捉反转机会的高胜率策略。", 9.9, "traderx/rsi_reversal.py"),
        ("多因子选股", "QuantLab", "基于市值、动量和价值因子的综合选股模型。", 29.9, "quantlab/multi_factor.py")
    ]
    
    cur.executemany(
        "INSERT INTO marketplace_items (name, author, description, price, file_path) VALUES (?, ?, ?, ?, ?)",
        items
    )
    conn.commit()
    print("Seeding complete.")
    conn.close()

if __name__ == "__main__":
    seed_marketplace()
