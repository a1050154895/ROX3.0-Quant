"""
会员管理API端点
"""
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
import sqlite3
import uuid

from app.db import (
    get_db, get_db_context,
    get_user_by_id,
    init_membership_plans, init_feature_permissions,
    get_all_membership_plans, get_membership_plan,
    create_membership, get_user_membership, has_active_membership,
    check_api_rate_limit, record_api_call,
    create_payment_record, update_payment_status,
    save_openclaw_config, get_openclaw_config,
    check_feature_permission
)
from app.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/membership", tags=["会员系统"])

# ===========Pydantic模型===========

class MembershipPlanResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float
    duration_days: int
    max_api_calls_per_day: int
    features: Optional[str] = None
    is_active: bool

class MembershipResponse(BaseModel):
    id: int
    user_id: int
    plan_id: int
    plan_name: Optional[str] = None
    start_date: str
    end_date: str
    status: str
    api_calls_used_today: int = 0
    max_api_calls_per_day: Optional[int] = None
    features: Optional[str] = None

class PurchaseRequest(BaseModel):
    plan_id: int
    payment_method: str = Field(default="wechat", description="支付方式 wechat/alipay")

class OpenClawConfigRequest(BaseModel):
    api_key: str
    api_domain: Optional[str] = "data.diemeng.chat"

class PaymentResponse(BaseModel):
    payment_id: int
    amount: float
    currency: str
    payment_status: str
    payment_url: Optional[str] = None  # 模拟用，实际项目会有真实的支付链接

# ===========API端点===========

@router.on_event("startup")
async def init_membership_data():
    """系统启动时初始化会员数据"""
    try:
        with get_db_context() as conn:
            init_membership_plans(conn)
            init_feature_permissions(conn)
        logger.info("会员系统初始化完成")
    except Exception as e:
        logger.warning(f"会员系统初始化失败: {e}")

@router.get("/plans", response_model=List[MembershipPlanResponse])
async def list_plans():
    """获取所有可用的会员套餐"""
    with get_db_context() as conn:
        plans = get_all_membership_plans(conn)
        return plans

@router.get("/plans/{plan_id}", response_model=MembershipPlanResponse)
async def get_plan(plan_id: int):
    """获取指定套餐详情"""
    with get_db_context() as conn:
        plan = get_membership_plan(conn, plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="套餐不存在")
        return plan

@router.get("/my-membership", response_model=Optional[MembershipResponse])
async def get_my_membership(current_user: dict = Depends(get_current_user)):
    """获取当前用户的会员信息"""
    with get_db_context() as conn:
        membership = get_user_membership(conn, current_user["id"])
        if membership:
            return {
                "id": membership["id"],
                "user_id": membership["user_id"],
                "plan_id": membership["plan_id"],
                "plan_name": membership["name"],
                "start_date": membership["start_date"],
                "end_date": membership["end_date"],
                "status": membership["status"],
                "api_calls_used_today": membership.get("api_calls_used_today", 0),
                "max_api_calls_per_day": membership.get("max_api_calls_per_day"),
                "features": membership.get("features")
            }
        return None

@router.post("/purchase", response_model=PaymentResponse)
async def purchase_membership(
    request: PurchaseRequest,
    current_user: dict = Depends(get_current_user)
):
    """购买会员"""
    with get_db_context() as conn:
        plan = get_membership_plan(conn, request.plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="套餐不存在")
        
        # 创建会员
        membership_id = create_membership(conn, current_user["id"], request.plan_id)
        if not membership_id:
            raise HTTPException(status_code=500, detail="创建会员失败")
        
        # 创建支付记录
        payment_id = create_payment_record(
            conn, current_user["id"], membership_id, plan["price"]
        )
        if not payment_id:
            raise HTTPException(status_code=500, detail="创建支付记录失败")
        
        # 模拟支付完成
        transaction_id = f"TXN{uuid.uuid4().hex[:8].upper()}"
        update_payment_status(conn, payment_id, "completed", transaction_id)
        
        return {
            "payment_id": payment_id,
            "amount": plan["price"],
            "currency": "CNY",
            "payment_status": "completed",
            "payment_url": None  # 演示环境直接完成
        }

@router.get("/check-permission/{feature_key}")
async def check_permission(
    feature_key: str,
    current_user: dict = Depends(get_current_user)
):
    """检查当前用户是否有某个功能的权限"""
    with get_db_context() as conn:
        has_perm = check_feature_permission(conn, current_user["id"], feature_key)
        return {"feature_key": feature_key, "has_permission": has_perm}

@router.get("/rate-limit")
async def get_rate_limit(
    current_user: dict = Depends(get_current_user)
):
    """获取当前用户API调用配额信息"""
    with get_db_context() as conn:
        ok, used, max_calls = check_api_rate_limit(conn, current_user["id"])
        return {
            "ok": ok,
            "calls_used": used,
            "calls_limit": max_calls,
            "calls_remaining": max_calls - used
        }

# ===========OpenClaw配置===========

@router.post("/openclaw/config")
async def set_openclaw_config(
    config: OpenClawConfigRequest,
    current_user: dict = Depends(get_current_user)
):
    """设置OpenClaw API配置"""
    with get_db_context() as conn:
        save_openclaw_config(conn, current_user["id"], config.api_key, config.api_domain)
        return {"status": "ok", "message": "配置已保存"}

@router.get("/openclaw/config")
async def get_user_openclaw_config(
    current_user: dict = Depends(get_current_user)
):
    """获取当前用户OpenClaw配置"""
    with get_db_context() as conn:
        config = get_openclaw_config(conn, current_user["id"])
        if not config:
            raise HTTPException(status_code=404, detail="未配置OpenClaw")
        return {
            "api_domain": config["api_domain"],
            "has_api_key": bool(config.get("api_key"))  # 不返回真实key
        }

