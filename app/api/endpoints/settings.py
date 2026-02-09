import os
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from app.core.config import settings
from app.rox_quant.llm import AIClient

router = APIRouter()

import json

class AISettings(BaseModel):
    api_key: str
    base_url: str
    provider: Optional[str] = "default"
    model: Optional[str] = "deepseek-chat"
    # Secondary (Backup)
    secondary_api_key: Optional[str] = ""
    secondary_base_url: Optional[str] = ""
    secondary_model: Optional[str] = "gpt-3.5-turbo"

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
    
    # Parse Secondary from AI_PROVIDERS
    secondary = {}
    try:
        raw = os.getenv("AI_PROVIDERS", "{}")
        providers = json.loads(raw)
        secondary = providers.get("secondary", {})
        if not secondary and "backup" in providers: secondary = providers["backup"]
    except:
        pass

    sec_key = secondary.get("api_key", "")
    masked_sec_key = f"{sec_key[:4]}...{sec_key[-4:]}" if len(sec_key) > 8 else ""

    return {
        "api_key": masked_key,  # Don't return full key for security
        "base_url": os.getenv("AI_BASE_URL", ""),
        "provider": os.getenv("AI_PROVIDER", "default"),
        "model": os.getenv("AI_DEFAULT_MODEL", "deepseek-chat"),
        "has_key": bool(key and key != "your_ai_api_key_here"),
        
        "secondary_api_key": masked_sec_key,
        "secondary_base_url": secondary.get("base_url", ""),
        "secondary_model": secondary.get("default_model", ""),
        "has_secondary_key": bool(sec_key)
    }

@router.post("/ai")
def update_ai_settings(config: AISettings):
    """Update AI settings in .env and reload client."""
    try:
        # 1. Update Primary
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
            
        # 2. Update Secondary (stored in AI_PROVIDERS json)
        # Read existing providers first
        current_providers = {}
        try:
            raw = os.getenv("AI_PROVIDERS", "{}")
            current_providers = json.loads(raw)
        except:
            pass
            
        # Update 'secondary' entry
        if "secondary" not in current_providers:
            current_providers["secondary"] = {"name": "备用线路"}
            
        if config.secondary_api_key and "***" not in config.secondary_api_key:
            current_providers["secondary"]["api_key"] = config.secondary_api_key
        # If blank and not masked, maybe user wants to clear it? For now assume update if provided.
        
        if config.secondary_base_url:
            current_providers["secondary"]["base_url"] = config.secondary_base_url
            
        if config.secondary_model:
            current_providers["secondary"]["default_model"] = config.secondary_model
            
        # Save back to .env
        json_str = json.dumps(current_providers, ensure_ascii=False)
        update_env_file("AI_PROVIDERS", json_str)
        
        return {"status": "success", "message": "AI settings updated (Primary & Secondary)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
