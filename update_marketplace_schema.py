#!/usr/bin/env python3
# 更新策略超市数据库表结构

import sqlite3
import os
from app.db import DB_PATH

print(f"Updating marketplace schema...")
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
    
    # 检查并添加缺失的字段
    cur = conn.execute("PRAGMA table_info(marketplace_items)")
    existing_columns = [row['name'] for row in cur.fetchall()]
    print(f"Existing columns: {existing_columns}")
    
    # 需要添加的字段
    columns_to_add = [
        ('downloads', 'INTEGER DEFAULT 0'),
        ('risk_level', 'INTEGER DEFAULT 3'),
        ('sharpe_ratio', 'REAL DEFAULT 0.0'),
        ('max_drawdown', 'REAL DEFAULT 0.0'),
        ('avg_return', 'REAL DEFAULT 0.0'),
        ('total_trades', 'INTEGER DEFAULT 0'),
        ('category', 'TEXT DEFAULT \'趋势跟踪\''),
        ('update_time', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
        ('win_rate', 'TEXT DEFAULT \'Unknown\''),
        ('return_rate', 'TEXT DEFAULT \'Unknown\''),
        ('rating_count', 'INTEGER DEFAULT 0')
    ]
    
    # 添加缺失的字段
    for col_name, col_def in columns_to_add:
        if col_name not in existing_columns:
            try:
                conn.execute(f"ALTER TABLE marketplace_items ADD COLUMN {col_name} {col_def}")
                print(f"Added column: {col_name}")
            except Exception as e:
                print(f"Error adding column {col_name}: {e}")
    
    # 提交更改
    conn.commit()
    
    # 验证更新
    cur = conn.execute("PRAGMA table_info(marketplace_items)")
    updated_columns = [row['name'] for row in cur.fetchall()]
    print(f"\nUpdated columns: {updated_columns}")
    print(f"Total columns: {len(updated_columns)}")
    
    conn.close()
    print("\nSchema updated successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    exit(1)
