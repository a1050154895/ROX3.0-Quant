from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel, validator
import sqlite3
from app.db import get_db, get_user_by_username, create_user
from app.core.security_config import settings

# Configuration
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic Models

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    phone: Optional[str] = None
    role: str = "user"
    bio: Optional[str] = None
    avatar: Optional[str] = None
    tags: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    phone: Optional[str] = None

    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('用户名至少需要3个字符')
        return v

    @validator('email')
    def validate_email(cls, v):
        if v:
            import re
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
                raise ValueError('邮箱格式不正确')
        return v

    @validator('phone')
    def validate_phone(cls, v):
        if v:
            import re
            if not re.match(r'^1[3-9]\d{9}$', v):
                raise ValueError('手机号格式不正确')
        return v

    @validator('password')
    def validate_password(cls, v):
        from app.core.security_config import SecurityConfig
        valid, message = SecurityConfig.validate_password(v)
        if not valid:
            raise ValueError(message)
        return v

class UserInDB(User):
    password_hash: str

# Helper Functions
def verify_password(plain_password, hashed_password):
    if isinstance(plain_password, str):
        plain_password = plain_password.encode('utf-8')
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')
    return bcrypt.checkpw(plain_password, hashed_password)

def get_password_hash(password):
    if isinstance(password, str):
        password = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password, salt)
    return hashed.decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), conn: sqlite3.Connection = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user_dict = get_user_by_username(conn, token_data.username)
    if user_dict is None:
        raise credentials_exception
    return User(**user_dict)

def authenticate_user(conn, username, password):
    user_dict = get_user_by_username(conn, username)
    if not user_dict:
        return False
    if not verify_password(password, user_dict['password_hash']):
        return False
    return UserInDB(**user_dict)
