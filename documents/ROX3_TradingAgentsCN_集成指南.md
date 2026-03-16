# ROX 3.0 × TradingAgents-CN 集成指南

可以结合，而且当前仓库已提供一条可直接接入的 API 通道。

## 1. 集成方式

ROX 新增端点：`POST /api/agents/tradingagents/analyze`

请求体：

```json
{
  "stock_code": "600519",
  "stock_name": "贵州茅台",
  "market": "cn",
  "horizon": "swing"
}
```

返回：
- `provider=tradingagents-cn`
- `result` 为 TradingAgents-CN 返回的原始结构

## 2. 环境变量

在 `.env` 里配置：

```env
TRADING_AGENTS_ENABLED=true
TRADING_AGENTS_BASE_URL=http://127.0.0.1:9000
TRADING_AGENTS_API_KEY=
TRADING_AGENTS_TIMEOUT=30
```

## 3. 对接建议

- **部署形态**：建议 TradingAgents-CN 独立进程/容器部署，ROX 通过 HTTP 调用。
- **失败降级**：当外部不可用时，前端回退到 ROX 本地多智能体分析 `/api/agents/analyze`。
- **统一输出**：可在后续把 TradingAgents-CN 输出映射到 ROX 的评分、信号、风控字段，便于同屏比较。

## 4. 最小验证

```bash
curl -X POST 'http://127.0.0.1:8008/api/agents/tradingagents/analyze' \
  -H 'Content-Type: application/json' \
  -d '{"stock_code":"600519","stock_name":"贵州茅台","market":"cn","horizon":"swing"}'
```

