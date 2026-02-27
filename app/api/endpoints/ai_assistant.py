from fastapi import APIRouter
from pydantic import BaseModel
from app.utils.external_services import OpenClawClient

router = APIRouter(tags=["ai_assistant"])
openclaw_client = OpenClawClient()

class ChatRequest(BaseModel):
    message: str

@router.get("/skills")
async def get_skills():
    return await openclaw_client.get_skills()

@router.post("/chat")
async def chat(req: ChatRequest):
    return await openclaw_client.chat(req.message)
