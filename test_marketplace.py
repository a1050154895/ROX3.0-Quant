#!/usr/bin/env python3
# 测试策略超市功能

import sqlite3
import os
from app.db import DB_PATH

print(f"Testing marketplace functionality...")
print(f"Database path: {DB_PATH}")

# 检查数据库是否存在
if not os.path.exists(DB_PATH):
    print("Error: Database file not found!")
    exit(1)

# 连接数据库
try:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    print("Successfully connected to database")
    
    # 检查marketplace_items表结构
    cur = conn.execute("PRAGMA table_info(marketplace_items)")
    columns = [dict(row) for row in cur.fetchall()]
    print(f"\nMarketplace items table columns: {len(columns)}")
    for col in columns:
        print(f"- {col['name']} ({col['type']})")
    
    # 检查数据
    cur = conn.execute("SELECT * FROM marketplace_items LIMIT 5")
    items = [dict(row) for row in cur.fetchall()]
    print(f"\nMarketplace items count: {len(items)}")
    for item in items:
        print(f"\nItem: {item['name']}")
        print(f"  Author: {item['author']}")
        print(f"  Description: {item['description']}")
        print(f"  Price: {item['price']}")
        print(f"  Downloads: {item['downloads']}")
        print(f"  Rating: {item['rating']}")
        print(f"  Category: {item.get('category', 'N/A')}")
        print(f"  Risk level: {item.get('risk_level', 'N/A')}")
        print(f"  Sharpe ratio: {item.get('sharpe_ratio', 'N/A')}")
    
    conn.close()
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    exit(1)
