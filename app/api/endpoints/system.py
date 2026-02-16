from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
import datetime
import time
import asyncio
import os
import akshare as ak
import pandas as pd
from app.db import _spot_cache, get_all_stocks_spot, clear_spot_cache

# Attempt to bypass proxy for EastMoney to avoid ProxyError in some environments
if "no_proxy" not in os.environ:
    os.environ["no_proxy"] = ".eastmoney.com,.dfcfw.com"
elif "eastmoney.com" not in os.environ["no_proxy"]:
    os.environ["no_proxy"] += ",.eastmoney.com,.dfcfw.com"

from app.rox_quant.data_provider import DataProvider
from app import cache_utils
from app.services.risk_monitor import get_risk_monitor

router = APIRouter()
_provider = DataProvider()

@router.get("/health")
async def api_health():
    """API 健康检查：供监控与前端判断服务是否可用"""
    return {"status": "ok", "service": "rox-quant", "docs": "/docs"}


@router.get("/ready")
async def api_ready():
    """就绪检查：DB 等依赖可用时返回 200，否则 503（供 K8s/负载均衡探活）"""
    try:
        from app.db import get_conn
        conn = get_conn()
        conn.execute("SELECT 1")
        conn.close()
        return {"status": "ok", "db": "ok"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "error": "数据库不可用", "code": "READINESS_FAIL", "detail": str(e)[:100]}
        )


@router.get("/status")
async def api_system_status():
    # Check AkShare connectivity (Ping test)
    ak_status = "Unknown"
    try:
        loop = asyncio.get_event_loop()
        
        def check_akshare():
            # List of checks to try in order
            # 1. Shanghai Composite (sh000001)
            # 2. Shanghai Composite (000001)
            # 3. Shenzhen Component (sz399001)
            # 4. General Spot Data (fallback)
            
            last_error = None
            
            # Try specific indices first (lightweight)
            for symbol in ["sh000001", "000001", "sz399001", "399001"]:
                try:
                    res = ak.stock_zh_index_spot_em(symbol=symbol)
                    if res is not None and not res.empty:
                        return True
                except Exception as e:
                    last_error = e
                    continue
            
            # Fallback to general spot data (heavier but reliable)
            try:
                res = ak.stock_zh_a_spot_em()
                if res is not None and not res.empty:
                    return True
            except Exception as e:
                last_error = e
                
            # If we get here, all checks failed
            raise last_error if last_error else Exception("AkShare connectivity check failed")

        await asyncio.wait_for(loop.run_in_executor(None, check_akshare), timeout=8.0)
        ak_status = "OK"
    except Exception as e:
        err_str = str(e)
        if "ProxyError" in err_str:
            ak_status = "Error: Proxy Issue (Check VPN/DNS)"
        elif "Connection" in err_str or "RemoteDisconnected" in err_str:
            ak_status = "Error: Network Blocked (EastMoney)"
        elif "403" in err_str:
            ak_status = "Error: Access Forbidden (403)"
        elif "KeyError" in err_str:
            ak_status = "Error: Data Format Mismatch"
        else:
            ak_status = f"Error: {err_str[:30]}..."

        cache_age = int(time.time() - _spot_cache["time"]) if _spot_cache["time"] > 0 else -1
    return JSONResponse({
        "server_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cache_age": cache_age,
        "akshare_status": ak_status,
        "spot_data_rows": len(_spot_cache["data"]) if not _spot_cache["data"].empty else 0,
        "scrape_progress": _provider.get_em_progress(),
        "tip": "需要释放内存时可点击下方「清理缓存」",
        "docs_url": "/docs"
    })


@router.post("/clear-cache")
async def api_clear_cache():
    """瘦身：清理行情缓存与通用 TTL 缓存，释放内存"""
    clear_spot_cache()
    try:
        cache_utils.clear_all_caches()
    except Exception:
        pass
    return JSONResponse({"status": "ok", "message": "缓存已清理"})

@router.post("/system/scrape/em_hsgt")
async def api_scrape_em_hsgt(background_tasks: BackgroundTasks):
    """
    Start the EastMoney HSGT scraping task in background.
    """
    if _provider.get_em_progress().get("step") in ["browser", "navigate", "extract"]:
        return JSONResponse({"status": "running", "message": "Task already running"})
    
    background_tasks.add_task(_provider.scrape_hsgt_eastmoney, headless=True)
    return JSONResponse({"status": "started", "message": "Scraping started"})

@router.get("/system/scrape/status")
async def api_scrape_status():
    """
    Get current scraping status.
    """
    return JSONResponse(_provider.get_em_progress())

@router.get("/system/scrape/result")
async def api_scrape_result():
    """
    Get last scraped result.
    """
    return JSONResponse(_provider.get_em_last())

@router.post("/system/refresh")
async def api_system_refresh():
    # Clear cache
    _spot_cache["data"] = pd.DataFrame()
    _spot_cache["time"] = 0
    
    # Trigger fetch immediately
    try:
        await get_all_stocks_spot()
        return JSONResponse({"status": "refreshed", "rows": len(_spot_cache["data"])})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("/system/monitor/status")
async def api_system_monitor_status():
    """获取系统监控状态"""
    try:
        risk_monitor = get_risk_monitor()
        system_status = risk_monitor.get_system_status()
        return JSONResponse({"status": "ok", "data": system_status})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("/system/monitor/alerts")
async def api_system_monitor_alerts(limit: int = 50):
    """获取系统监控告警"""
    try:
        risk_monitor = get_risk_monitor()
        system_alerts = risk_monitor.get_system_alerts(limit=limit)
        return JSONResponse({"status": "ok", "data": system_alerts})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("/system/monitor/summary")
async def api_system_monitor_summary():
    """获取系统监控摘要"""
    try:
        risk_monitor = get_risk_monitor()
        system_summary = risk_monitor.get_system_summary()
        return JSONResponse({"status": "ok", "data": system_summary})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
