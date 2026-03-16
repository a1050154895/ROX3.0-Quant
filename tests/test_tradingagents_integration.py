import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import agents
from app.services.tradingagents_client import TradingAgentsClient


class DummyResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload or {}
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            req = httpx.Request("POST", "http://example.test/analyze")
            resp = httpx.Response(self.status_code, request=req, text=self.text)
            raise httpx.HTTPStatusError("error", request=req, response=resp)

    def json(self):
        return self._payload


class DummyAsyncClient:
    def __init__(self, *args, **kwargs):
        self.response = DummyResponse({"decision": "buy"}, 200)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return self.response


def test_tradingagents_client_success(monkeypatch):
    monkeypatch.setattr("app.services.tradingagents_client.httpx.AsyncClient", DummyAsyncClient)
    monkeypatch.setattr("app.services.tradingagents_client.settings.TRADING_AGENTS_ENABLED", True)
    monkeypatch.setattr("app.services.tradingagents_client.settings.TRADING_AGENTS_BASE_URL", "http://example.test")
    monkeypatch.setattr("app.services.tradingagents_client.settings.TRADING_AGENTS_ENDPOINT", "/analyze")
    monkeypatch.setattr("app.services.tradingagents_client.settings.TRADING_AGENTS_RETRY_COUNT", 0)
    monkeypatch.setattr("app.services.tradingagents_client.settings.TRADING_AGENTS_RETRY_BACKOFF", 0.0)

    client = TradingAgentsClient()
    result = asyncio.run(client.analyze_stock(stock_code="600519"))

    assert result["success"] is True
    assert result["provider"] == "tradingagents-cn"
    assert result["result"]["decision"] == "buy"


def test_tradingagents_endpoint_fallback(monkeypatch):
    app = FastAPI()
    app.include_router(agents.router, prefix="/api")
    client = TestClient(app)

    async def _raise_runtime_error(self, *args, **kwargs):
        raise RuntimeError("upstream failed")

    async def _local_ok(stock_code, stock_name=""):
        return {"success": True, "symbol": stock_code, "name": stock_name}

    monkeypatch.setattr(
        "app.services.tradingagents_client.TradingAgentsClient.analyze_stock",
        _raise_runtime_error,
    )
    monkeypatch.setattr("app.api.endpoints.agents._analyze_with_local_orchestrator", _local_ok)
    monkeypatch.setattr("app.core.config.settings.TRADING_AGENTS_FALLBACK_LOCAL", True)

    response = client.post(
        "/api/agents/tradingagents/analyze",
        json={"stock_code": "600519", "stock_name": "茅台", "market": "cn", "horizon": "swing"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "local-fallback"
    assert payload["result"]["symbol"] == "600519"


def test_tradingagents_health_endpoint(monkeypatch):
    app = FastAPI()
    app.include_router(agents.router, prefix="/api")
    client = TestClient(app)

    class HealthyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            class _Resp:
                status_code = 200
                headers = {"content-type": "application/json"}

                def raise_for_status(self):
                    return None

                def json(self):
                    return {"status": "ok"}

            return _Resp()

    monkeypatch.setattr("app.services.tradingagents_client.httpx.AsyncClient", HealthyAsyncClient)
    monkeypatch.setattr("app.core.config.settings.TRADING_AGENTS_ENABLED", True)
    monkeypatch.setattr("app.core.config.settings.TRADING_AGENTS_BASE_URL", "http://example.test")

    response = client.get("/api/agents/tradingagents/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["reachable"] is True
