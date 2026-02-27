# ROX Quant API 文档

## 概述

ROX Quant Trading System 提供了一套完整的RESTful API，用于量化交易、市场数据获取和策略管理。

**基础URL**: `http://localhost:8002/api`

## 认证

大多数API端点需要JWT认证。在请求头中添加：

```
Authorization: Bearer <your_token>
```

### 认证 API

#### 登录获取token
```
POST /token
Content-Type: application/x-www-form-urlencoded

username=admin&password=password
```

**响应示例**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### 刷新token
```
POST /token/refresh
Authorization: Bearer <your_token>
```

**响应示例**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### 注册新用户
```
POST /register
Content-Type: application/json

{
  "username": "newuser",
  "password": "password123",
  "email": "user@example.com",
  "phone": "13800138000"
}
```

**响应示例**:
```json
{
  "id": 1,
  "username": "newuser",
  "email": "user@example.com",
  "phone": "13800138000",
  "bio": null,
  "avatar": null,
  "tags": null
}
```

#### 获取当前用户信息
```
GET /users/me
Authorization: Bearer <your_token>
```

**响应示例**:
```json
{
  "id": 1,
  "username": "admin",
  "email": "admin@example.com",
  "phone": "13800138000",
  "bio": "Quant trader",
  "avatar": null,
  "tags": "quant,trading"
}
```

#### 更新用户信息
```
PATCH /users/me
Authorization: Bearer <your_token>
Content-Type: application/json

{
  "bio": "Senior quant trader",
  "tags": "quant,trading,AI"
}
```

**响应示例**:
```json
{
  "id": 1,
  "username": "admin",
  "email": "admin@example.com",
  "phone": "13800138000",
  "bio": "Senior quant trader",
  "avatar": null,
  "tags": "quant,trading,AI"
}
```

## API 端点

### 市场数据 API

#### 获取实时行情
```
GET /market/spot?limit=500&offset=0
```

**响应示例**:
```json
{
  "items": [
    {
      "code": "000001",
      "name": "平安银行",
      "price": 12.34,
      "change_pct": 1.23,
      "volume": 12345678
    }
  ],
  "total": 5000
}
```

#### 获取K线数据
```
GET /market/kline?code=000001&period=daily&count=100
```

**参数**:
- `code`: 股票代码
- `period`: 周期
- `count`: 返回数量

#### 获取市场指数
```
GET /market/indices
```

#### 获取市场统计
```
GET /market/stats
```

#### 获取板块资金流向
```
GET /market/sector-fund-flow
```

#### 获取龙虎榜
```
GET /market/dragon-tiger?date=2024-01-01
```

### 自选股 API

#### 获取自选股列表
```
GET /market/watchlist
```

#### 添加自选股
```
POST /market/watchlist
Content-Type: application/json

{
  "stock_name": "平安银行",
  "stock_code": "000001",
  "sector": "银行"
}
```

#### 删除自选股
```
DELETE /market/watchlist?stock_code=000001
```

### 预警 API

#### 获取预警列表
```
GET /market/alerts?pending_only=false
```

#### 创建预警
```
POST /market/alerts
Content-Type: application/json

{
  "symbol": "000001",
  "name": "平安银行",
  "alert_type": "price_above",
  "value": 15.0
}
```

#### 删除预警
```
DELETE /market/alerts/{alert_id}
```

### 策略 API

#### 获取策略列表
```
GET /strategies/list
```

#### 获取策略统计
```
GET /strategies/stats
```

#### 执行策略
```
POST /strategies/execute
Content-Type: application/json

{
  "strategy_name": "策略名称",
  "params": {}
}
```

#### 回测策略
```
POST /strategies/backtest
Content-Type: application/json

{
  "strategy_name": "策略名称",
  "params": {}
}
```

### 宏观数据 API

#### 获取宏观经济指标
```
GET /macro/indicators
```

**响应示例**:
```json
{
  "money_supply": [
    {
      "date": "2024-01",
      "m2_yoy": 8.5,
      "m1_yoy": 3.2,
      "scissors": -5.3
    }
  ],
  "pmi": [
    {
      "date": "2024-01",
      "manufacturing": 49.2,
      "non_manufacturing": 52.5
    }
  ],
  "cpi": {
    "value": 2.1,
    "date": "2024-01"
  },
  "ppi": {
    "value": -1.2
  },
  "gdp": {
    "value": 5.2,
    "quarter": "2023Q4"
  }
}
```

### AI API

#### 获取AI提供商列表
```
GET /ai/providers
```

**响应示例**:
```json
{
  "current": "deepseek",
  "list": [
    {
      "name": "deepseek",
      "models": ["deepseek-chat", "deepseek-r1"]
    },
    {
      "name": "openai",
      "models": ["gpt-3.5-turbo", "gpt-4"]
    }
  ]
}
```

#### AI聊天
```
POST /ai/chat
Authorization: Bearer <your_token>
Content-Type: application/json

{
  "message": "什么是量化投资？",
  "context": "我是一名初学者",
  "model": "deepseek-chat",
  "provider": "deepseek"
}
```

**响应示例**:
```json
{
  "response": "量化投资是一种使用数学模型和计算机算法进行投资决策的方法..."
}
```

#### 股票分析
```
POST /ai/analyze
Authorization: Bearer <your_token>
Content-Type: application/json

{
  "stock_name": "贵州茅台",
  "stock_code": "600519",
  "price": 1700.0,
  "indicators": {
    "ma20": 1650.0,
    "ma60": 1600.0,
    "rsi": 65
  },
  "model": "deepseek-chat",
  "provider": "deepseek"
}
```

**响应示例**:
```json
{
  "analysis": "贵州茅台当前价格为1700元，处于上升趋势...",
  "recommendation": "持有",
  "confidence": 0.85
}
```

#### 获取AI提示词模板列表
```
GET /ai/templates
Authorization: Bearer <your_token>
```

**响应示例**:
```json
{
  "items": [
    {
      "key": "stock_analysis",
      "name": "股票分析模板",
      "content": "分析股票{stock_name}({stock_code})的技术面和基本面...",
      "scope": "user"
    }
  ]
}
```

#### 获取交易建议
```
POST /ai/decision/trade
Authorization: Bearer <your_token>
Content-Type: application/json

{
  "symbol": "600519",
  "signals": {
    "ma_crossover": "bullish",
    "rsi": 70,
    "volume": "high"
  }
}
```

**响应示例**:
```json
{
  "action": "buy",
  "price": 1700.0,
  "stop_loss": 1650.0,
  "take_profit": 1800.0,
  "reason": "技术指标显示看涨信号"
}
```

#### 获取投资组合建议
```
POST /ai/decision/portfolio
Authorization: Bearer <your_token>
Content-Type: application/json

{
  "positions": [
    {"symbol": "600519", "amount": 100, "price": 1700.0},
    {"symbol": "000001", "amount": 1000, "price": 12.0}
  ],
  "cash": 50000.0
}
```

**响应示例**:
```json
{
  "rebalance": [
    {"symbol": "600519", "target_weight": 0.6},
    {"symbol": "000001", "target_weight": 0.2},
    {"symbol": "300750", "target_weight": 0.2}
  ],
  "risk_level": "medium"
}
```

### 知识库 API

#### 搜索知识
```
GET /knowledge/search?q=量化投资&limit=10
```

**响应示例**:
```json
{
  "items": [
    {
      "title": "量化投资策略",
      "content": "量化投资是一种使用数学模型和计算机算法进行投资决策的方法...",
      "score": 0.95
    }
  ],
  "total": 10
}
```

#### 获取知识统计
```
GET /knowledge/stats
```

**响应示例**:
```json
{
  "total_documents": 100,
  "categories": {
    "策略": 30,
    "理论": 20,
    "案例": 50
  }
}
```

## 错误处理

API返回标准HTTP状态码：

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 401 | 未认证 |
| 403 | 权限不足 |
| 404 | 资源不存在 |
| 429 | 请求过于频繁 |
| 500 | 服务器内部错误 |

## 速率限制

API默认限制：每分钟100次请求

响应头包含速率限制信息：
- `X-RateLimit-Limit`: 限制总数
- `X-RateLimit-Remaining`: 剩余次数
- `X-RateLimit-Reset`: 重置时间（秒）

## WebSocket

实时数据推送端点：
```
ws://localhost:8002/ws
```

## 版本

当前API版本: v1.0.0
