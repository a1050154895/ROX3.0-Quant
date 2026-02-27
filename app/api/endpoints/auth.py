from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
import sqlite3
import datetime
import time
from collections import defaultdict
from jose import JWTError, jwt
from pydantic import BaseModel
from app.auth import (
    authenticate_user, create_access_token, get_password_hash, create_user,
    get_user_by_username, get_current_user, Token, User, UserCreate,
    ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY, ALGORITHM,
)
from app.db import get_db

router = APIRouter()
security = HTTPBearer(auto_error=False)

# 登录限流：每 IP 每分钟最多 10 次尝试，超限返回 429
_LOGIN_ATTEMPTS: dict = defaultdict(list)
_RATE_WINDOW = 60  # 秒
_RATE_MAX = 10

def _cleanup_login_attempts():
    """清理过期的登录尝试记录，防止内存泄漏"""
    now = time.time()
    # 仅当字典过大时清理
    if len(_LOGIN_ATTEMPTS) > 1000:
        for k in list(_LOGIN_ATTEMPTS.keys()):
            _LOGIN_ATTEMPTS[k] = [t for t in _LOGIN_ATTEMPTS[k] if now - t < _RATE_WINDOW]
            if not _LOGIN_ATTEMPTS[k]:
                del _LOGIN_ATTEMPTS[k]

def _check_login_rate(client_key: str) -> bool:
    _cleanup_login_attempts()
    now = time.time()
    _LOGIN_ATTEMPTS[client_key] = [t for t in _LOGIN_ATTEMPTS[client_key] if now - t < _RATE_WINDOW]
    if len(_LOGIN_ATTEMPTS[client_key]) >= _RATE_MAX:
        return False
    _LOGIN_ATTEMPTS[client_key].append(now)
    return True

@router.post("/token", response_model=Token)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    conn: sqlite3.Connection = Depends(get_db),
):
    client_key = request.client.host if request.client else request.headers.get("x-forwarded-for", "unknown").split(",")[0].strip()
    if not _check_login_rate(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="登录尝试过于频繁，请稍后再试",
        )
    user = authenticate_user(conn, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/token/refresh", response_model=Token)
async def refresh_access_token(
    creds: HTTPAuthorizationCredentials = Depends(security),
    conn: sqlite3.Connection = Depends(get_db),
):
    """统一 token 刷新：用当前 token 换新 token，避免多标签页即将过期时 401。"""
    if not creds or creds.scheme != "Bearer" or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = creds.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        username = payload.get("sub")
        exp = payload.get("exp")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        # 仅允许未过期或过期在 5 分钟内的 token 刷新
        now = int(time.time())
        if exp is not None and (now - exp) > 300:
            raise HTTPException(status_code=401, detail="Token expired too long ago, please login again")
        user_dict = get_user_by_username(conn, username)
        if not user_dict:
            raise HTTPException(status_code=401, detail="User not found")
        access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_token = create_access_token(data={"sub": username}, expires_delta=access_token_expires)
        return {"access_token": new_token, "token_type": "bearer"}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/register", response_model=User)
async def register(user: UserCreate, conn: sqlite3.Connection = Depends(get_db)):
    db_user = get_user_by_username(conn, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = create_user(conn, user.username, hashed_password, email=user.email, phone=user.phone)
    if not new_user:
        raise HTTPException(status_code=500, detail="Registration failed")
    return new_user

@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

class UserUpdate(BaseModel):
    bio: Optional[str] = None
    avatar: Optional[str] = None
    tags: Optional[str] = None

@router.patch("/users/me", response_model=User)
async def update_user_me(
    data: UserUpdate,
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    updates = []
    params = []
    
    if data.bio is not None:
        updates.append("bio = ?")
        params.append(data.bio)
    if data.avatar is not None:
        updates.append("avatar = ?")
        params.append(data.avatar)
    if data.tags is not None:
        updates.append("tags = ?")
        params.append(data.tags)
        
    if not updates:
        return current_user
        
    params.append(current_user.id)
    query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
    conn.execute(query, params)
    conn.commit()
    
    return get_user_by_username(conn, current_user.username)
