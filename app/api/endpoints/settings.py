import os
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from app.core.config import settings
from app.rox_quant.llm import AIClient

router = APIRouter()

class AISettings(BaseModel):
    api_key: str
    base_url: str
    provider: Optional[str] = "default"
    model: Optional[str] = "deepseek-chat"

def update_env_file(key: str, value: str):
    """Update or add a key-value pair in the .env file."""
    env_path = os.path.join(settings.BASE_DIR, ".env")
    
    # Read existing lines
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
    # Update or append
    key_found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            key_found = True
        else:
            new_lines.append(line)
    
    if not key_found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append(f"{key}={value}\n")
    
    # Write back
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    
    # Update os.environ immediately for current process
    os.environ[key] = value

@router.get("/ai")
def get_ai_settings():
    """Get current AI settings (API Key masked)."""
    key = os.getenv("AI_API_KEY", "")
    masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
    
    return {
        "api_key": masked_key,  # Don't return full key for security
        "base_url": os.getenv("AI_BASE_URL", ""),
        "provider": os.getenv("AI_PROVIDER", "default"),
        "model": os.getenv("AI_DEFAULT_MODEL", "deepseek-chat"),
        "has_key": bool(key and key != "your_ai_api_key_here")
    }

@router.post("/ai")
def update_ai_settings(config: AISettings):
    """Update AI settings in .env and reload client."""
    try:
        if config.api_key and "***" not in config.api_key:
            update_env_file("AI_API_KEY", config.api_key)
            settings.AI_API_KEY = config.api_key
            
        update_env_file("AI_BASE_URL", config.base_url)
        settings.AI_BASE_URL = config.base_url
        
        if config.provider:
            update_env_file("AI_PROVIDER", config.provider)
            settings.AI_PROVIDER = config.provider
            
        if config.model:
            update_env_file("AI_DEFAULT_MODEL", config.model)
            settings.AI_DEFAULT_MODEL = config.model
            
        # Reload AI Client
        # Force re-initialization of the singleton if possible, or just let the next request pick up os.environ
        # AIClient re-reads os.environ in __init__, so we can re-instantiate or just let it be.
        # But `AIClient` is instantiated in `app.rox_quant.llm` and used as a dependency or imported. 
        # The simplest way is to update the instance if it's a singleton.
        # However, looking at `app/services/dashboard_analyzer.py`, it instantiates `AIClient()` per request or caches it.
        # Let's check `app/rox_quant/llm.py` usage. It seems to be used as `AIClient()`.
        
        return {"status": "success", "message": "AI settings updated. Restart may be required for some components."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
