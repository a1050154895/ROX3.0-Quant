"""
OpenClaw数据API端点 - 增强数据服务
"""
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import time

from app.auth import get_current_user
from app.db import (
    get_db_context,
    check_feature_permission,
    check_api_rate_limit,
    record_api_call,
    get_openclaw_config
)
from app.services.openclaw_service import get_openclaw_service


# ===========路由定义===========

router = APIRouter(prefix="/api/openclaw-data", tags=["openclaw-data"])


# ===========工具函数===========

def format_stock_code(code: Optional[str]) -> Optional[str]:
    """格式化股票代码：6位数字，前面补0"""
    if not code:
        return None
    digits = ''.join([c for c in code if c.isdigit()])
    if len(digits) < 6:
        digits = digits.zfill(6)
    return digits


# ===========数据模型===========

class FundFlowData(BaseModel):
    date: Optional[str]       # 日期
    stock_code: Optional[str] # 股票代码
    small_inflow: Optional[float]  # 小单流入
    medium_inflow: Optional[float] # 中单流入
    big_inflow: Optional[float]    # 大单流入


# ===========权限检查依赖===========

def require_feature(feature_key: str):
    """检查功能权限的依赖项工厂函数"""
    async def dependency(request: Request, current_user = Depends(get_current_user)):
        with get_db_context() as conn:
            has_perm = check_feature_permission(conn, current_user.id, feature_key)
            if not has_perm:
                raise HTTPException(status_code=403, detail=f"权限不足，需要开通更高等级会员")
            
            # 检查API调用限制
            ok, used, max_calls = check_api_rate_limit(conn, current_user.id)
            if not ok:
                raise HTTPException(status_code=429, detail=f"今日API调用次数已用完")
        return current_user
    return dependency


# ===========服务实例初始化===========

def _get_service_for_user(user_id: int):
    """为用户获取配置了OpenClaw服务"""
    with get_db_context() as conn:
        config = get_openclaw_config(conn, user_id)
        config_dict = dict(config) if config else None
        
    api_key = config_dict.get("api_key") if config_dict else None
    api_domain = config_dict.get("api_domain") if config_dict else None
    return get_openclaw_service(api_key, api_domain)


# ===========API端点===========

@router.get("/health")
async def check_health():
    """检查OpenClaw数据服务状态"""
    service = get_openclaw_service()
    return {
        "status": "ok" if service.available else "unavailable",
        "has_api_key": service.api_key != ""
    }

# ===========股票实时与历史数据===========

@router.get("/stocks/snapshot")
async def get_stock_snapshots(
    request: Request,
    current_user = Depends(require_feature("basic_market_data"))
):
    """获取股票实时快照"""
    start_time = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        data = service.get_stock_snapshot()
        elapsed = time.time() - start_time
        # 记录API调用
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/stocks/snapshot", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stocks/daily")
async def get_stock_daily(
    stock_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user = Depends(require_feature("basic_market_data"))
):
    """获取股票日K数据"""
    start_time = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        formatted_code = format_stock_code(stock_code)
        data = service.get_stock_daily(formatted_code, start_date, end_date)
        elapsed = time.time() - start_time
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/stocks/daily", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stocks/history")
async def get_stock_history(
    stock_code: str,
    level: str = "1min",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    current_user = Depends(require_feature("simple_chart"))
):
    """获取股票分钟级历史数据"""
    start_time_ts = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        formatted_code = format_stock_code(stock_code)
        data = service.get_stock_history(formatted_code, level, start_time, end_time)
        elapsed = time.time() - start_time_ts
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/stocks/history", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===========财务数据===========

@router.get("/finance/data")
async def get_finance_data(
    stock_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user = Depends(require_feature("advanced_chart"))
):
    """获取财务数据"""
    start_time = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        formatted_code = format_stock_code(stock_code)
        data = service.get_finance_data(formatted_code, start_date, end_date)
        elapsed = time.time() - start_time
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/finance/data", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===========主力资金流向===========

@router.get("/fund/main-flow")
async def get_main_fund_flow(
    stock_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user = Depends(require_feature("main_fund_flow"))
):
    """获取主力资金流向数据"""
    start_time = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        formatted_code = format_stock_code(stock_code)
        data = service.get_main_fund_flow(formatted_code, start_date, end_date)
        elapsed = time.time() - start_time
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/fund/main-flow", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===========债券数据===========

@router.get("/bonds/list")
async def get_bond_list(
    current_user = Depends(require_feature("bond_data"))
):
    """获取债券列表"""
    start_time = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        data = service.get_bond_list()
        elapsed = time.time() - start_time
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/bonds/list", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bonds/daily")
async def get_bond_daily(
    bond_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user = Depends(require_feature("bond_data"))
):
    """获取债券日K数据"""
    start_time = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        data = service.get_bond_daily(bond_code, start_date, end_date)
        elapsed = time.time() - start_time
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/bonds/daily", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===========ETF数据===========

@router.get("/etf/list")
async def get_etf_list(
    current_user = Depends(require_feature("etf_data"))
):
    """获取ETF列表"""
    start_time = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        data = service.get_etf_list()
        elapsed = time.time() - start_time
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/etf/list", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/etf/daily")
async def get_etf_daily(
    etf_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user = Depends(require_feature("etf_data"))
):
    """获取ETF日K数据"""
    start_time = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        data = service.get_etf_daily(etf_code, start_date, end_date)
        elapsed = time.time() - start_time
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/etf/daily", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===========基础行情数据===========

@router.get("/market/info")
async def get_market_info(
    current_user = Depends(require_feature("basic_market_data"))
):
    """获取基础市场信息"""
    start_time = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        data = service.get_market_info()
        elapsed = time.time() - start_time
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/market/info", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===========板块数据===========

@router.get("/sectors/list")
async def get_sector_list(
    current_user = Depends(require_feature("sector_data"))
):
    """获取板块列表"""
    start_time = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        data = service.get_sector_list()
        elapsed = time.time() - start_time
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/sectors/list", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sectors/stocks")
async def get_sector_stocks(
    sector_name: str,
    current_user = Depends(require_feature("sector_data"))
):
    """获取板块成分股"""
    start_time = time.time()
    try:
        service = _get_service_for_user(current_user.id)
        if not service.available:
            raise HTTPException(status_code=400, detail="需要配置OpenClaw API Key")
        data = service.get_sector_stocks(sector_name)
        elapsed = time.time() - start_time
        with get_db_context() as conn:
            record_api_call(conn, current_user.id, "/api/openclaw-data/sectors/stocks", "GET", 200, elapsed)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
