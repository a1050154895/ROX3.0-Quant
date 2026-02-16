from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
import sqlite3
from pydantic import BaseModel

from app.db import get_db
from app.auth import get_current_user, User
from app.services.account_manager import get_account_manager

router = APIRouter(prefix="/accounts", tags=["Accounts"])

class AccountCreate(BaseModel):
    type: str
    name: str
    initial_balance: float = 100000.0
    currency: str = "CNY"

class AccountUpdate(BaseModel):
    name: Optional[str] = None
    balance: Optional[float] = None

class TransferRequest(BaseModel):
    from_account_id: int
    to_account_id: int
    amount: float

@router.get("/", response_model=List[Dict[str, Any]])
async def get_accounts(
    current_user: User = Depends(get_current_user)
):
    """
    获取用户的所有账户
    """
    try:
        account_manager = get_account_manager()
        accounts = account_manager.get_user_accounts(current_user.id)
        return accounts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=Dict[str, Any])
async def create_account(
    account: AccountCreate,
    current_user: User = Depends(get_current_user)
):
    """
    创建新账户
    """
    try:
        account_manager = get_account_manager()
        account_id = account_manager.create_account(
            user_id=current_user.id,
            account_type=account.type,
            name=account.name,
            initial_balance=account.initial_balance,
            currency=account.currency
        )
        
        if not account_id:
            raise HTTPException(status_code=400, detail="Failed to create account")
        
        return {"id": account_id, "message": "Account created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{account_id}", response_model=Dict[str, Any])
async def get_account(
    account_id: int,
    current_user: User = Depends(get_current_user)
):
    """
    获取单个账户详情
    """
    try:
        account_manager = get_account_manager()
        account = account_manager.get_account(account_id, current_user.id)
        
        if not account:
            raise HTTPException(status_code=404, detail="Account not found")
        
        return account
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{account_id}", response_model=Dict[str, Any])
async def update_account(
    account_id: int,
    account_update: AccountUpdate,
    current_user: User = Depends(get_current_user)
):
    """
    更新账户信息
    """
    try:
        account_manager = get_account_manager()
        success = account_manager.update_account(
            account_id=account_id,
            user_id=current_user.id,
            name=account_update.name,
            balance=account_update.balance
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update account")
        
        # 获取更新后的账户信息
        account = account_manager.get_account(account_id, current_user.id)
        if not account:
            raise HTTPException(status_code=404, detail="Account not found")
        
        return account
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{account_id}", response_model=Dict[str, str])
async def delete_account(
    account_id: int,
    current_user: User = Depends(get_current_user)
):
    """
    删除账户
    """
    try:
        account_manager = get_account_manager()
        success = account_manager.delete_account(account_id, current_user.id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to delete account")
        
        return {"message": "Account deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transfer", response_model=Dict[str, str])
async def transfer_funds(
    transfer: TransferRequest,
    current_user: User = Depends(get_current_user)
):
    """
    在账户之间转移资金
    """
    try:
        account_manager = get_account_manager()
        success = account_manager.transfer_funds(
            from_account_id=transfer.from_account_id,
            to_account_id=transfer.to_account_id,
            user_id=current_user.id,
            amount=transfer.amount
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to transfer funds")
        
        return {"message": "Funds transferred successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{account_id}/performance", response_model=Dict[str, Any])
async def get_account_performance(
    account_id: int,
    current_user: User = Depends(get_current_user)
):
    """
    获取账户性能统计
    """
    try:
        account_manager = get_account_manager()
        performance = account_manager.get_account_performance(account_id, current_user.id)
        
        if "error" in performance:
            raise HTTPException(status_code=404, detail=performance["error"])
        
        return performance
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/combined", response_model=Dict[str, Any])
async def get_combined_performance(
    current_user: User = Depends(get_current_user)
):
    """
    获取用户所有账户的组合性能统计
    """
    try:
        account_manager = get_account_manager()
        performance = account_manager.get_combined_performance(current_user.id)
        
        if "error" in performance:
            raise HTTPException(status_code=404, detail=performance["error"])
        
        return performance
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
