from fastapi import APIRouter
from app.utils.external_services import ClawFeedClient

router = APIRouter(tags=["news"])
clawfeed_client = ClawFeedClient()

@router.get("/digests")
async def get_news_digests(type: str = "daily", limit: int = 20):
    return await clawfeed_client.get_digests(type, limit)

@router.get("/digests/{digest_id}")
async def get_news_digest(digest_id: str):
    return await clawfeed_client.get_digest(digest_id)
