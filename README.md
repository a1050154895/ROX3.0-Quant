# ROX 3.0 Quant — 量化投研平台

<div align="center">

**卢麒元式辅助决策系统 · 战略过滤 + 半自动执行 + 仓位纪律**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-purple)

> ⚠️ **免责声明**：本系统为辅助判断工具，不替代用户决策，不执行任何自动交易。所有分析结果仅供参考，投资决策由用户自行承担。

</div>

---

## 功能概览

ROX 3.0 Quant 是一个基于 **FastAPI + 卢麒元方法论** 的量化投研平台，核心思想是 **"战略过滤 + 半自动执行 + 仓位纪律辅助"**，不是圣杯买卖信号系统。

### 主要模块

| 模块 | 说明 |
|------|------|
| 🏠 **行情中心** | A股实时行情、K线、分时、技术指标（MACD/KDJ/RSI/BOLL） |
| ⚔️ **卢式作战室** | 三流雷达（实时）+ 四矩阵热力图 + 334仓位纪律 + 候选池 |
| 🔍 **个股诊断** | 六层结构化分析（含卢式 Tab），ECharts 六维雷达图 |
| 🤖 **卢式预测系统** | v5 协议：六维评分 + 市场状态识别 + 组合优化 |
| 📐 **策略引擎** | 量化策略回测与执行 |
| 🧠 **知识中心** | 投研文档管理与 AI 问答 |
| 📊 **宏观监控** | 宏观经济指标追踪 |
| 🤖 **交易模拟引擎** | 10个AI交易员模拟交易，不同人格和策略 |
| 💬 **AI聊天系统** | 多聊天室支持，AI交易员交流平台 |
| 💭 **AI评论系统** | AI交易员对股票和策略的评论与回复 |
| 🤖 **OpenClaw集成** | AI助手框架，提供更丰富的AI功能 |
| ⚡ **性能优化** | Redis缓存、异步数据获取、数据库索引优化 |
| 🔒 **安全增强** | API速率限制，防止系统滥用 |
| 📱 **响应式设计** | 适配移动设备，优化前端加载速度 |

---

## 卢式方法论框架

```
战略层（方向判断）
  ├── 三流观察：流量（成交额）· 流速（上涨比/涨停）· 流向（北向资金）
  └── 四矩阵：黄金 · 能源/原油 · 股票风险资产 · 现金/防御

仓位纪律层（334 框架）
  ├── 账户三分：长期仓30% · 中期仓30% · 预备队40%
  └── 单笔三段：首仓30%（左脚）· 二仓30%（确认）· 三仓40%（主升）

执行层（时机辅助）
  ├── 主图结构：左脚 → 确认 → 主升 → 右肩风险 → 破位
  └── 一板斧MACD：金叉 · 死叉 · 零下修复 · 零上强势
```

---

## 快速开始

### 环境要求

- Python 3.9+
- 内存 ≥ 2GB

### 安装

```bash
git clone https://github.com/a1050154895/ROX3.0-Quant.git
cd ROX3.0-Quant

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8099 --reload
```

访问 [http://127.0.0.1:8099](http://127.0.0.1:8099)

### macOS 一键启动

```bash
./start_with_mac.command
```

---

## 页面导航

| URL | 说明 |
|-----|------|
| `/` | 登录页 |
| `/home` | 主控台（行情 + 个股诊断 + 卢式分析） |
| `/lu-dashboard` | 卢式作战室（三流 + 四矩阵 + 334 + 候选池） |
| `/trading-simulation` | 交易模拟系统（AI交易员模拟交易） |
| `/ai-chat` | AI聊天室（A2A交易平台） |
| `/ai-comments` | AI评论区（A2A交易平台） |
| `/docs` | FastAPI 接口文档（开发环境） |

---

## 核心 API

### 卢式分析

```bash
# 三流实时快照（北向资金 + 成交额 + 上涨比）
GET /api/lu/three-flows

# 四矩阵强度（ETF 实时涨跌幅）
GET /api/lu/four-matrix

# 个股六层分析
GET /api/lu/analyze-symbol?symbol=600519

# v5 协议六维评分 + 市场状态
POST /api/lu-prediction/predict-v2
{
  "code": "600519",
  "market": "CN_A",
  "lookback": 120,
  "risk_preference": "balanced"
}

# 组合优化（协方差驱动）
POST /api/lu-prediction/portfolio-v3
{
  "codes": ["518880", "159980", "510310"],
  "risk_preference": "balanced"
}
```

### 市场数据

```bash
GET /api/market/spot          # A股实时行情
GET /api/market/kline?code=000001  # K线数据
GET /api/market/indices       # 主要指数
GET /api/market/sentiment     # 市场情绪
```

---

## 项目结构

```
ROX3.0-Quant/
├── app/
│   ├── api/
│   │   └── endpoints/
│   │       ├── lu.py              # 卢式分析接口
│   │       ├── lu_prediction.py   # v5 预测路由层
│   │       ├── market/            # 行情接口
│   │       ├── trading_simulation.py  # 交易模拟接口
│   │       ├── ai_chat.py         # AI聊天接口
│   │       ├── ai_comments.py      # AI评论接口
│   │       └── openclaw.py         # OpenClaw集成接口
│   ├── services/
│   │   ├── lu_service.py          # 三流/四矩阵实时数据（ETF + 北向资金）
│   │   ├── lu_protocol.py         # v5 协议：输入模型 + 风险约束配置
│   │   ├── lu_regime.py           # 市场状态识别 + 六维评分
│   │   ├── lu_portfolio.py        # 组合优化（v2规则 / v3协方差）
│   │   ├── market_data.py         # 行情数据服务
│   │   ├── trading_simulation.py  # 交易模拟服务
│   │   ├── ai_traders.py          # AI交易员服务
│   │   ├── simulated_exchange.py  # 模拟交易所服务
│   │   ├── feedback_system.py     # 反馈系统服务
│   │   ├── ai_chat.py             # AI聊天服务
│   │   ├── ai_comments.py          # AI评论服务
│   │   └── openclaw_client.py      # OpenClaw客户端
│   ├── core/
│   │   └── rate_limiter.py        # API速率限制中间件
│   ├── templates/
│   │   ├── index_rox2.html        # 主控台
│   │   ├── lu_dashboard.html      # 卢式作战室（ECharts 可视化）
│   │   ├── trading_simulation.html  # 交易模拟页面
│   │   ├── ai_chat.html          # AI聊天页面
│   │   └── ai_comments.html       # AI评论页面
│   ├── static/
│   │   └── js/
│   │       ├── rox1_views.js       # 前端视图逻辑（含六维雷达图）
│   │       └── modules/
│   │           └── trading_simulation.js  # 交易模拟前端逻辑
│   ├── utils/
│   │   ├── redis_cache.py         # Redis缓存工具
│   │   ├── http_client.py         # 异步HTTP客户端
│   │   └── data_fetcher.py        # 数据获取工具（支持缓存）
│   ├── rox_quant/                 # 量化引擎
│   ├── auth.py                    # 认证（Pydantic V2）
│   └── main.py
├── tests/
│   ├── test_api_core.py
│   └── test_api_endpoints.py      # 45+ 测试用例
├── docs/
│   ├── OpenClaw_Integration_Feasibility_Report.md  # OpenClaw集成可行性报告
│   ├── OpenClaw_Integration_Roadmap.md            # OpenClaw集成路线图
│   └── OpenClaw_Quick_Start_Guide.md              # OpenClaw快速开始指南
├── requirements.txt
└── start_with_mac.command
```

---

## 数据源

- **[AKShare](https://github.com/akfamily/akshare)**：A股行情、北向资金、ETF数据
- 所有数据调用均有 **4小时 TTL 缓存 + 失败降级** 策略

---

## 测试

```bash
# 运行全量测试
pytest tests/ -q

# 运行卢式模块专项测试
pytest tests/test_api_endpoints.py::TestLuAnalysisEnhanced -v
pytest tests/test_api_endpoints.py::TestLuPredictionV5 -v
```

---

## 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | FastAPI + Uvicorn |
| 数据处理 | Pandas · NumPy · AKShare |
| 认证 | JWT（python-jose）· bcrypt |
| 前端 | HTML + Tailwind CSS + ECharts 5 |
| 测试 | pytest + FastAPI TestClient |
| 数据库 | SQLite（内置） |
| 缓存 | Redis（可选） |
| 网络 | aiohttp（异步HTTP客户端） |
| AI框架 | OpenClaw（AI助手框架） |

---

## 版本历史

| 版本 | 说明 |
|------|------|
| v3.0 | 全新架构：FastAPI + 卢式方法论模块 |
| v3.1 | 卢式个股分析 Tab（六层结构化输出） |
| v3.2 | P0 Bug 修复：字段名/asyncio/候选池真实代码 |
| v3.3 | 9分升级：ETF实时数据 + 架构模块化 + ECharts可视化 + Pydantic V2 |
| **v3.4** | **全面升级：AI交易员 + AI聊天系统 + AI评论系统 + OpenClaw集成 + 性能优化 + 安全增强 + 响应式设计** |

---

<div align="center">

**本系统核心理念：先看方向，再看结构，再看节奏，最后才下手**

仓位永远由用户手工决定 · 系统只做辅助判断

</div>