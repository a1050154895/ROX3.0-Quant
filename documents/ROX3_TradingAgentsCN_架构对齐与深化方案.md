# ROX 3.0 × TradingAgents-CN 架构对齐与深化方案（Docker 视角）

## 1. 两者相似之处

基于 TradingAgents-CN Docker 部署文档常见形态与 ROX 3.0 当前工程结构，可明确看到以下同构点：

1. **服务化边界一致**：
   - ROX 3.0 以 FastAPI 作为统一网关；
   - TradingAgents-CN 以独立 HTTP 服务形态输出分析能力；
   - 两者天然适合通过 HTTP 进行“网关-分析引擎”解耦。

2. **容器编排模型一致**：
   - ROX 已有 `app + postgres + redis + nginx` 的 compose 拓扑；
   - TradingAgents-CN 也强调容器化部署与环境变量注入；
   - 适合直接合并为一个 compose 网络，减少跨机房通信复杂度。

3. **上游不稳定治理诉求一致**：
   - 两者都依赖外部模型/API（OpenAI/DeepSeek/其他）；
   - 都需要超时、重试、降级、健康检查等稳态机制。

---

## 2. 可结合点（已落地）

### A. 运行时弹性
- 新增上游健康检查接口：`GET /api/agents/tradingagents/health`。
- 增加 `TRADING_AGENTS_HEALTH_ENDPOINT` 和 `TRADING_AGENTS_HEALTH_STRICT`：
  - 非严格模式：返回可观测状态（reachable/detail）；
  - 严格模式：上游不可达即返回 503，便于接入监控告警。

### B. 编排层融合
- 在 `docker-compose.yml` 中新增可选 profile 服务 `tradingagents-cn`：
  - `docker compose up -d`：只起 ROX；
  - `docker compose --profile tradingagents up -d tradingagents-cn`：按需启用外部分析服务。
- ROX 侧默认支持通过环境变量把 `TRADING_AGENTS_BASE_URL` 指向容器名地址。

### C. 文档与可运维性
- README 补充：
  - TradingAgents 健康接口说明；
  - Docker profile 启动方式；
  - 环境变量完整配置。

---

## 3. 下一步建议（P1 / P2）

### P1（建议本周）
1. **指标化**：把 TradingAgents 请求耗时、失败率、fallback 触发次数输出到 metrics。
2. **熔断**：失败率超过阈值后短路 30-60 秒，避免雪崩。
3. **缓存**：对同一股票在短时间内重复分析做结果缓存，降低上游压力。

### P2（建议下个迭代）
1. **结果结构标准化**：将 TradingAgents 结果映射到 ROX 统一字段（signal/risk/score/reason）。
2. **异步队列化**：对耗时分析引入任务队列（RQ/Celery）并提供任务查询接口。
3. **多 provider 路由**：在 `provider router` 中支持 `local / tradingagents-cn / future providers` 策略分流。

---

## 4. 验收指标建议

- `tradingagents_health_reachable_rate` ≥ 99%
- `fallback_ratio` 在上游稳定时 < 5%
- `analyze_p95_latency` 相比当前下降 20%+
- `5xx_rate` 下降到 < 0.5%

