from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.auth import get_current_user, User
from app.db import get_db
import sqlite3
from typing import Optional
from pydantic import BaseModel
import os
import shutil

router = APIRouter(prefix="/users", tags=["Users"])

class UserUpdate(BaseModel):
    bio: Optional[str] = None
    tags: Optional[str] = None

@router.get("/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.patch("/me")
async def update_user_me(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    try:
        # Update user in DB
        if user_update.bio is not None:
            conn.execute("UPDATE users SET bio = ? WHERE id = ?", (user_update.bio, current_user.id))
        if user_update.tags is not None:
            conn.execute("UPDATE users SET tags = ? WHERE id = ?", (user_update.tags, current_user.id))
        conn.commit()
        
        # Fetch updated user
        cur = conn.execute("SELECT * FROM users WHERE id = ?", (current_user.id,))
        user = cur.fetchone()
        return dict(user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    try:
        # 1. Save file
        upload_dir = "app/static/avatars"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"user_{current_user.id}{file_ext}"
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Update DB
        avatar_url = f"/static/avatars/{filename}"
        conn.execute("UPDATE users SET avatar = ? WHERE id = ?", (avatar_url, current_user.id))
        conn.commit()
        
        return {"avatar": avatar_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
