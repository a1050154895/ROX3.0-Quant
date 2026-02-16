# ROX Quant API 文档

## 概述

ROX Quant Trading System 提供了一套完整的RESTful API，用于量化交易、市场数据获取和策略管理。

**基础URL**: `http://localhost:8002/api`

## 认证

大多数API端点需要JWT认证。在请求头中添加：

```
Authorization: Bearer <your_token>
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

### 知识库 API

#### 搜索知识
```
GET /knowledge/search?q=量化投资&limit=10
```

#### 获取知识统计
```
GET /knowledge/stats
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
