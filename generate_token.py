
import sys
import os
# Add current directory to sys.path
sys.path.append(os.getcwd())

from datetime import datetime, timedelta
from jose import jwt
from app.core.config import settings

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

# Use 'admin' as username
token = create_access_token({"sub": "admin"})
print(token)
