#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场API - 自选股和预警
处理自选股和价格预警相关的API
"""

import logging
import asyncio
import re
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
import requests as req_lib

from app.auth import get_current_user, User
from app.db import (
    get_db, get_watchlist, add_watchlist, remove_watchlist,
    create_alert, list_alerts, delete_alert, get_pending_alerts
)
from app.api.endpoints.market.models import WatchlistItem, AlertCreate
from app.api.endpoints.market.services import symbol_prefix

logger = logging.getLogger(__name__)
router = APIRouter(tags=["watchlist-alerts"])


# ============ 自选股 API ============

@router.get("/watchlist")
async def api_get_watchlist(
    current_user: User = Depends(get_current_user),
    conn = Depends(get_db)
):
    """
    获取用户自选股列表
    
    Returns:
        自选股列表
    """
    items = get_watchlist(conn, current_user.id)
    return {"items": items}


@router.post("/watchlist")
async def api_add_watchlist(
    item: WatchlistItem,
    current_user: User = Depends(get_current_user),
    conn = Depends(get_db)
):
    """
    添加自选股
    
    Args:
        item: 自选股信息
    
    Returns:
        操作结果
    """
    success = add_watchlist(
        conn, current_user.id, 
        item.stock_name, item.stock_code, item.sector
    )
    
    if not success:
        return JSONResponse(
            {"error": "已在自选股中或添加失败"}, 
            status_code=400
        )
    
    return {"status": "ok"}


@router.delete("/watchlist")
async def api_remove_watchlist(
    stock_code: str,
    current_user: User = Depends(get_current_user),
    conn = Depends(get_db)
):
    """
    删除自选股
    
    Args:
        stock_code: 股票代码
    
    Returns:
        操作结果
    """
    remove_watchlist(conn, current_user.id, stock_code)
    return {"status": "ok"}


@router.get("/watchlist/export")
async def api_watchlist_export(
    format: str = "csv",
    current_user: User = Depends(get_current_user),
    conn = Depends(get_db)
):
    """
    导出自选股
    
    Args:
        format: 导出格式 (csv/json)
    
    Returns:
        导出文件
    """
    import io
    import csv
    
    items = get_watchlist(conn, current_user.id)
    
    if format == "json":
        return JSONResponse({"items": items})
    
    # CSV格式
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["股票名称", "股票代码", "板块", "添加时间"])
    
    for item in items:
        writer.writerow([
            item.get("stock_name", ""),
            item.get("stock_code", ""),
            item.get("sector", ""),
            item.get("created_at", ""),
        ])
    
    output.seek(0)
    return JSONResponse({
        "data": output.getvalue(),
        "filename": f"watchlist_{current_user.id}.csv"
    })


# ============ 预警 API ============

@router.get("/alerts")
async def api_list_alerts(
    pending_only: bool = False,
    current_user: User = Depends(get_current_user),
    conn = Depends(get_db)
):
    """
    获取预警列表
    
    Args:
        pending_only: 是否只显示待触发的预警
    
    Returns:
        预警列表
    """
    items = list_alerts(conn, current_user.id, pending_only=pending_only)
    return {"items": items}


@router.post("/alerts")
async def api_create_alert(
    req: AlertCreate,
    current_user: User = Depends(get_current_user),
    conn = Depends(get_db)
):
    """
    创建价格预警
    
    Args:
        req: 预警信息
    
    Returns:
        创建结果
    """
    if req.alert_type not in ("price_above", "price_below"):
        return JSONResponse(
            {"error": "alert_type 必须为 price_above 或 price_below"}, 
            status_code=400
        )
    
    aid = create_alert(
        conn, current_user.id, 
        req.symbol, req.name, req.alert_type, req.value
    )
    
    if aid is None:
        return JSONResponse({"error": "创建失败"}, status_code=500)
    
    return {"status": "ok", "id": aid}


@router.delete("/alerts/{alert_id}")
async def api_delete_alert(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    conn = Depends(get_db)
):
    """
    删除预警
    
    Args:
        alert_id: 预警ID
    
    Returns:
        操作结果
    """
    if delete_alert(conn, current_user.id, alert_id):
        return {"status": "ok"}
    return JSONResponse({"error": "未找到预警"}, status_code=404)


@router.post("/check-alerts")
async def api_check_alerts(
    current_user: User = Depends(get_current_user),
    conn = Depends(get_db)
):
    """
    检查价格预警并标记已触发
    
    Returns:
        本次触发的预警列表
    """
    triggered = []
    
    # 获取所有待触发预警
    alerts = get_pending_alerts(conn)
    if not alerts:
        return {"status": "ok", "triggered": []}
    
    loop = asyncio.get_event_loop()
    
    async def _check_single_alert(alert: Dict) -> Dict | None:
        """检查单个预警"""
        try:
            code = str(alert["symbol"]).strip().zfill(6)
            prefix_code = symbol_prefix(code)
            
            def _fetch_price():
                return req_lib.get(
                    f"http://hq.sinajs.cn/list={prefix_code}",
                    headers={"Referer": "http://finance.sina.com.cn/"},
                    timeout=2
                )
            
            r = await loop.run_in_executor(None, _fetch_price)
            m = re.search(r'"([^"]+)"', r.text)
            
            if not m or "," not in m.group(1):
                return None
            
            price = float(m.group(1).split(",")[3])
            
            # 检查是否触发
            hit = (
                (alert["alert_type"] == "price_above" and price >= float(alert["value"])) or
                (alert["alert_type"] == "price_below" and price <= float(alert["value"]))
            )
            
            if hit:
                return {"alert": alert, "price": price}
                
        except Exception as e:
            logger.error(f"检查预警失败 {alert['symbol']}: {e}")
        
        return None
    
    # 并发检查所有预警
    results = await asyncio.gather(*[_check_single_alert(a) for a in alerts])
    
    for res in results:
        if res:
            triggered.append(res)
            # 标记为已触发
            from app.db import mark_alert_triggered
            mark_alert_triggered(conn, res["alert"]["id"])
    
    return {"status": "ok", "triggered": triggered}
