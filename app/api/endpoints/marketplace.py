from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import sqlite3
from app.db import get_db
from app.auth import get_current_user, User
import os
import shutil

router = APIRouter(prefix="/marketplace", tags=["Marketplace"])

@router.get("/list")
async def list_marketplace_items(
    search: Optional[str] = Query(None, description="Search term for strategy name or description"),
    min_rating: Optional[float] = Query(None, ge=1, le=5, description="Minimum rating"),
    max_risk: Optional[int] = Query(None, ge=1, le=5, description="Maximum risk level"),
    category: Optional[str] = Query(None, description="Strategy category"),
    sort_by: Optional[str] = Query("install_count", description="Sort by field: install_count, rating, downloads"),
    conn: sqlite3.Connection = Depends(get_db)
):
    """
    List all available strategies in the marketplace with search and filter options.
    """
    # Build query
    query = "SELECT * FROM marketplace_items WHERE 1=1"
    params = []
    
    if search:
        query += " AND (name LIKE ? OR description LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
    
    if min_rating:
        query += " AND rating >= ?"
        params.append(min_rating)
    
    if max_risk:
        query += " AND risk_level <= ?"
        params.append(max_risk)
    
    if category:
        query += " AND category = ?"
        params.append(category)
    
    # Add sorting
    valid_sort_fields = ["install_count", "rating", "downloads", "created_at", "return_rate", "sharpe_ratio"]
    if sort_by in valid_sort_fields:
        query += f" ORDER BY {sort_by} DESC"
    else:
        query += " ORDER BY install_count DESC"
    
    cur = conn.execute(query, params)
    return [dict(row) for row in cur.fetchall()]

@router.post("/install/{item_id}")
async def install_strategy(
    item_id: int,
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    """
    Install a strategy from marketplace to local strategies folder.
    """
    # 1. Get Item info
    cur = conn.execute("SELECT * FROM marketplace_items WHERE id = ?", (item_id,))
    item = cur.fetchone()
    if not item:
        raise HTTPException(status_code=404, detail="Strategy not found")
        
    item = dict(item)
    
    # 2. Simulate Cloud Download
    # In a real app, this would download from S3/CDN.
    # Here we mock it by copying from a "seed" location or creating a dummy file if not exists.
    
    # Target path
    target_dir = "app/strategies"
    os.makedirs(target_dir, exist_ok=True)
    
    # Source path (mock)
    # If seeding via script didn't create real files, we'll create a dummy one on the fly.
    filename = os.path.basename(item['file_path'])
    target_path = os.path.join(target_dir, filename)
    
    if os.path.exists(target_path):
        return {"status": "already_installed", "message": f"{filename} already exists"}
        
    try:
        # 检查是否存在真实的策略文件
        # 首先检查app/strategies/目录中是否有对应文件
        existing_files = os.listdir(target_dir)
        
        # 尝试找到与策略名称相关的文件
        strategy_file = None
        for file in existing_files:
            if file.endswith('.py') and (item['name'].replace(' ', '').lower() in file.lower() or item['name'].lower() in file.lower()):
                strategy_file = os.path.join(target_dir, file)
                break
        
        # 如果没有找到相关文件，尝试使用文件名
        if not strategy_file:
            # 从file_path中提取文件名
            base_filename = os.path.basename(item['file_path'])
            for file in existing_files:
                if file == base_filename:
                    strategy_file = os.path.join(target_dir, file)
                    break
        
        # 如果找到策略文件，复制它
        if strategy_file and os.path.exists(strategy_file):
            shutil.copy2(strategy_file, target_path)
            content = f"""# {item['name']} Strategy
# Author: {item['author']}
# Desc: {item['description']}
# Installed from Marketplace
# Source: {strategy_file}
"""
            # 在文件开头添加注释
            with open(target_path, "r+", encoding="utf-8") as f:
                original_content = f.read()
                f.seek(0, 0)
                f.write(content + "\n" + original_content)
        else:
            # 如果没有找到真实文件，创建一个基于策略类型的模板
            strategy_type = item['name']
            class_name = item['name'].replace(' ', '')
            
            # 创建策略模板文件
            content = f'''
import random

class {class_name}Strategy:
    """
    {item['name']}
    Author: {item['author']}
    Desc: {item['description']}
    Installed from Marketplace
    """
    
    def initialize(self, context):
        """
        初始化策略
        """
        print(f"[{item['name']}] Initializing strategy...")
        context.universe = ["600519.SH", "000001.SZ", "300750.SZ"]
        context.max_position_pct = 0.2
    
    def on_tick(self, context, tick):
        """
        处理每 tick 数据
        """
        # {item['name']} 核心逻辑
        pass
    
    def handle_data(self, context, data):
        """
        处理数据
        """
        for code in context.universe:
            if code not in data:
                continue
            
            bar = data[code]
            close_price = bar['close']
            
            # 这里添加策略逻辑
            # 例如：趋势跟踪、均值回归、突破策略等
            pass
'''.strip()
            
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        # 3. Update Install Count and Downloads
        conn.execute("UPDATE marketplace_items SET install_count = install_count + 1, downloads = downloads + 1 WHERE id = ?", (item_id,))
        conn.commit()
        
        return {"status": "success", "message": f"Installed {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Install failed: {e}")

@router.get("/item/{item_id}")
async def get_marketplace_item(
    item_id: int,
    conn: sqlite3.Connection = Depends(get_db)
):
    """
    Get details of a specific marketplace item.
    """
    cur = conn.execute("SELECT * FROM marketplace_items WHERE id = ?", (item_id,))
    item = cur.fetchone()
    if not item:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # Get comments for this item
    cur = conn.execute("SELECT * FROM marketplace_comments WHERE item_id = ? ORDER BY created_at DESC", (item_id,))
    comments = [dict(row) for row in cur.fetchall()]
    
    item_dict = dict(item)
    item_dict['comments'] = comments
    
    return item_dict

@router.post("/rate/{item_id}")
async def rate_strategy(
    item_id: int,
    rating: int = Query(..., ge=1, le=5, description="Rating from 1 to 5"),
    comment: Optional[str] = Query(None, description="Optional comment"),
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    """
    Rate a strategy and optionally leave a comment.
    """
    # Check if item exists
    cur = conn.execute("SELECT * FROM marketplace_items WHERE id = ?", (item_id,))
    item = cur.fetchone()
    if not item:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # Add comment
    if comment:
        conn.execute(
            "INSERT INTO marketplace_comments (item_id, user_id, username, rating, comment) VALUES (?, ?, ?, ?, ?)",
            (item_id, current_user.id, current_user.username, rating, comment)
        )
    
    # Update rating
    conn.execute(
        "UPDATE marketplace_items SET rating = (SELECT AVG(rating) FROM marketplace_comments WHERE item_id = ?), rating_count = (SELECT COUNT(*) FROM marketplace_comments WHERE item_id = ?) WHERE id = ?",
        (item_id, item_id, item_id)
    )
    
    conn.commit()
    
    return {"status": "success", "message": "Rating submitted successfully"}

@router.get("/comments/{item_id}")
async def get_item_comments(
    item_id: int,
    conn: sqlite3.Connection = Depends(get_db)
):
    """
    Get comments for a specific marketplace item.
    """
    cur = conn.execute("SELECT * FROM marketplace_comments WHERE item_id = ? ORDER BY created_at DESC", (item_id,))
    comments = [dict(row) for row in cur.fetchall()]
    return comments
