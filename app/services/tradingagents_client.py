"""TradingAgents-CN 适配客户端。"""

import asyncio
import logging
from typing import Any, Dict

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class TradingAgentsClient:
    """调用 TradingAgents-CN HTTP 服务的轻量客户端。"""

    def __init__(self):
        self.enabled = settings.TRADING_AGENTS_ENABLED
        self.base_url = settings.TRADING_AGENTS_BASE_URL.rstrip("/")
        self.timeout = settings.TRADING_AGENTS_TIMEOUT
        self.api_key = settings.TRADING_AGENTS_API_KEY
        endpoint = settings.TRADING_AGENTS_ENDPOINT.strip()
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.retry_count = max(settings.TRADING_AGENTS_RETRY_COUNT, 0)
        self.retry_backoff = max(settings.TRADING_AGENTS_RETRY_BACKOFF, 0.0)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _post_with_retry(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Exception | None = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.retry_count + 1):
                try:
                    response = await client.post(url, json=payload, headers=self._headers())
                    response.raise_for_status()
                    return response.json()
                except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    last_error = e
                    logger.warning(
                        "TradingAgents-CN 请求异常(可重试): attempt=%s/%s, err=%s",
                        attempt + 1,
                        self.retry_count + 1,
                        e,
                    )
                    if attempt < self.retry_count:
                        await asyncio.sleep(self.retry_backoff * (2**attempt))
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code
                    body = e.response.text
                    logger.error("TradingAgents-CN 请求失败: status=%s, body=%s", status, body)
                    raise RuntimeError(f"TradingAgents-CN 请求失败: HTTP {status}") from e
                except ValueError as e:
                    logger.error("TradingAgents-CN 返回非 JSON 响应")
                    raise RuntimeError("TradingAgents-CN 返回非 JSON 响应") from e

        raise RuntimeError(f"TradingAgents-CN 请求异常: {last_error}") from last_error

    async def analyze_stock(
        self,
        stock_code: str,
        stock_name: str = "",
        market: str = "cn",
        horizon: str = "swing",
    ) -> Dict[str, Any]:
        """转发单股分析请求给 TradingAgents-CN。"""
        if not self.enabled:
            raise RuntimeError("TradingAgents-CN 集成未启用，请设置 TRADING_AGENTS_ENABLED=true")
        if not self.base_url:
            raise RuntimeError("缺少 TRADING_AGENTS_BASE_URL 配置")

        payload = {
            "symbol": stock_code,
            "name": stock_name,
            "market": market,
            "horizon": horizon,
        }
        url = f"{self.base_url}{self.endpoint}"
        data = await self._post_with_retry(url, payload)

        return {
            "success": True,
            "provider": "tradingagents-cn",
            "stock_code": stock_code,
            "stock_name": stock_name,
            "result": data,
        }
