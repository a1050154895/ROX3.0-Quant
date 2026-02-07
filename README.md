
# 🚀 ROX 3.0 Quant Platform - 下一代全资产量化投研终端

> **"让量化投资像玩游戏一样简单"**

![ROX 3.0 Banner](https://img.shields.io/badge/ROX-3.0_Pro-blueviolet?style=for-the-badge&logo=python)
![Beginner Friendly](https://img.shields.io/badge/Beginner-One_Click_Start-success?style=for-the-badge&logo=apple)
![Pro Ready](https://img.shields.io/badge/Professional-Algo_Trading-blue?style=for-the-badge&logo=linux)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

**ROX 3.0** 不仅仅是一个量化软件，它是一个**双核**投研平台，完美平衡了专业深度与使用门槛。它整合了**A股、美股、加密货币**全球三大市场，将机构级的宏观数据、资金流向与 AI 投研能力免费带给每一位投资者。

---

## 📚 目录 (Table of Contents)

- [🚀 两大核心模式 (Dual Modes)](#-两大核心模式-dual-modes)
- [🍃 新手极速上手 (For Beginners)](#-新手极速上手-for-beginners)
- [⚡️ 开发者安装 (For Developers)](#-开发者安装-for-developers)
- [🔑 配置指南 (Configuration)](#-配置指南-configuration)
- [📖 功能详解 (Features Manual)](#-功能详解-features-manual)
    - [1. 市场看板 (Market Dashboard)](#1-市场看板-market-dashboard)
    - [2. 宏观宏观罗盘 (Macro Engine)](#2-宏观宏观罗盘-macro-engine)
    - [3. AI 投研顾问 (QBot)](#3-ai-投研顾问-qbot)
    - [4. 量化选股与策略 (Strategies)](#4-量化选股与策略-strategies)
- [🛠️ 系统架构 (Architecture)](#-系统架构-architecture)
- [📂 目录结构 (Structure)](#-目录结构-structure)
- [🛡️ 免责声明 (Disclaimer)](#-免责声明-disclaimer)

---

## 🚀 两大核心模式 (Dual Modes)

ROX 3.0 设计了两套完全不同的交互界面，以适应不同阶段的用户需求。

### 1. 🍃 小白模式 (Beginner Mode)
专为非金融背景、非编程背景的普通用户设计。
*   **极简界面**：隐藏复杂的 K 线、盘口和订单流。
*   **AI 驱动**：通过对话框与 "AI 投研顾问" 交互，获取投资建议。
*   **直观决策**：提供 "市场温度计"（情绪指标）和 "一键选股"（本周金股池）。
*   **适用人群**：价值投资者、长期持有者、量化初学者。

### 2. ⚡️ 专业极客模式 (Pro Mode)
专为宽客 (Quants)、全职交易员和开发者设计。
*   **全能终端**：类似 Bloomberg/Wind 的多屏工作站体验。
*   **深度数据**：Level-2 盘口、逐笔成交、资金流向、板块热力图。
*   **策略引擎**：支持 Python 策略编写、回测、仿真交易。
*   **适用人群**：日内交易员、算法工程师、全市场配置专家。

---

## 🍃 新手极速上手 (For Beginners)

**零代码、零配置，下载即用。**

我们为您准备了“一键启动脚本”，脚本会自动检测系统环境、安装 Python 依赖并启动浏览器。

### 🍎 macOS 用户
1. 点击本项目 GitHub 页面右上角的 **Code** -> **Download ZIP** 下载并解压。
2. 打开解压后的文件夹，找到 `start_with_mac.command`。
3. 双击运行（如遇安全提示，请右键点击文件 -> 打开 -> 确认打开）。
4. 脚本即刻启动，自动为您打开 ROX 3.0 系统界面。

### 🪟 Windows 用户
1. 下载并解压项目文件。
2. 找到文件夹内的 `start_with_win.bat`。
3. 双击运行，等待黑色终端窗口完成环境初始化。
4. 系统启动后会自动打开浏览器。

*(启动后，点击界面顶部导航栏右侧的 **“🍃 小白模式”** 按钮，即可进入极简界面)*

---

## ⚡️ 开发者安装 (For Developers)

如果您希望参与开发、修改代码或进行手动部署，请按照以下步骤操作。

### 环境要求
*   **Python**: 3.9 或更高版本
*   **Node.js**: (可选) 用于前端深度定制，默认已包含编译好的静态资源。

### 安装步骤

1. **克隆代码库**
   ```bash
   git clone https://github.com/a1050154895/ROX3.0-Quant.git
   cd ROX3.0-Quant
   ```

2. **创建虚拟环境 (强烈推荐)**
   ```bash
   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
   *注：国内用户可添加清华源加速：`-i https://pypi.tuna.tsinghua.edu.cn/simple`*

4. **启动服务**
   ```bash
   python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
   ```

5. **访问系统**
   打开浏览器访问：`http://localhost:8002`

---

## 🔑 配置指南 (Configuration)

ROX 3.0 开箱即用，但为了解锁 **AI 投研** 和 **更多数据源** 功能，建议配置环境变量。

在项目根目录下创建一个 `.env` 文件（参照 `.env.example`）：

```ini
# AI 配置 (支持 DeepSeek, OpenAI, Moonshot 等)
AI_API_KEY=your_api_key_here
AI_BASE_URL=https://api.deepseek.com/v1
AI_MODEL=deepseek-chat

# 数据库配置 (默认使用 SQLite，无需修改)
DB_URL=sqlite:///./data/db/rox.db

# 密钥配置 (用于 Session 加密)
SECRET_KEY=change_this_to_a_random_string
```

---

## 📖 功能详解 (Features Manual)

### 1. 市场看板 (Market Dashboard)
*   **多市场切换**：顶部导航栏支持 `[A股] [美股] [Crypto]` 一键切换。
*   **实时指数**：上证指数、纳斯达克、BTC/USDT 实时跳动。
*   **K线复盘**：集成了 TradingView 风格的图表，支持日/周/月线切换及技术指标叠加。
*   **五档盘口**：右侧边栏实时展示买一至买五、卖一至卖五的挂单详情（A股需开盘时间）。

### 2. 宏观宏观罗盘 (Macro Engine)
*   **数据源**：直连中国国家统计局 (NBS) 接口。
*   **M1-M2 剪刀差**：可视化展示货币供应量剪刀差。剪刀差走阔通常预示经济活力复苏，是牛市的重要先行指标。
*   **PMI 荣枯线**：追踪制造业采购经理指数，判断经济扩张与收缩周期。
*   **CSI/PPI**: 居民消费价格指数与工业生产者出厂价格指数，监控通胀水平。

### 3. AI 投研顾问 (QBot)
*(仅限小白模式或 Pro 模式多智能体面板)*
*   **对话式交互**：输入 "分析 600519" 或 "现在是牛市吗"，AI 将综合技术面、基本面与宏观数据给出建议。
*   **本地知识库 (RAG)**：ROX 会优先检索本地的策略文档与研报，确保回答符合您的投资体系。
*   **风险提示**：AI 会自动识别高风险标的并发出预警。

### 4. 量化选股与策略 (Strategies)
*   **策略工坊 (Lab)**：可视化的策略构建器，支持通过拖拽指标生成交易逻辑。
*   **每周金股**：系统通过算法每周一自动筛选出的高胜率潜力股池。
*   **个股诊断**：内置 "亢龙有悔"、"三色共振"、"游资暗盘" 等 7 大经典模型，为个股进行多维度打分。

---

## 🛠️ 系统架构 (Architecture)

ROX 3.0 采用前后端分离的现代化架构，确保高性能与可扩展性。

```mermaid
graph TD
    User[用户终端] --> |HTTP/WebSocket| Gateway[FastAPI 网关]
    
    subgraph "Backend Services"
        Gateway --> MarketService[行情服务]
        Gateway --> TradeEngine[交易/回测引擎]
        Gateway --> AIService[AI 投研服务]
        Gateway --> DataCenter[数据中心]
    end
    
    subgraph "Data Sources (Adapters)"
        DataCenter --> AkShare[AkShare (A股/宏观)]
        DataCenter --> YFinance[YFinance (美股)]
        DataCenter --> CCXT[CCXT (加密货币)]
        DataCenter --> LocalDB[(SQLite/CSV)]
    end
    
    subgraph "AI Core"
        AIService --> LLM[LLM (DeepSeek/GPT)]
        AIService --> RAG[RAG 向量检索]
    end
```

---

## 📂 目录结构 (Structure)

```
ROX3.0-Quant/
├── app/
│   ├── api/                 # API 接口路由 definition
│   ├── core/                # 核心配置与工具类
│   ├── rox_quant/           # 量化核心库 (Backtrader 扩展)
│   │   ├── datasources/     # 数据源适配器
│   │   ├── algos/           # 交易算法 (Grid, TWAP)
│   │   └── llm.py           # AI 大模型客户端
│   ├── strategies/          # 用户策略文件存放处
│   ├── static/              # 前端静态资源 (JS/CSS/Images)
│   └── templates/           # HTML 页面模板
├── data/                    # 数据存储 (DB, Logs)
├── start_with_mac.command   # Mac 一键启动脚本
├── start_with_win.bat       # Windows 一键启动脚本
├── requirements.txt         # Python 依赖清单
└── README.md                # 项目文档
```

---

## 🛡️ 免责声明 (Disclaimer)

1.  **风险提示**：量化投资涉及极其复杂的金融风险。本软件提供的所有数据、分析结果与 AI 建议仅供参考，**绝不构成任何投资建议**。
2.  **数据来源**：本项目数据来源于公开互联网接口（如 AkShare, Yahoo Finance），开发者不对数据的准确性与实时性做任何保证。
3.  **资金安全**：在使用模拟盘或实盘交易功能时，请务必保管好您的 API 密钥与账户信息。因软件 Bug 或网络问题导致的资金损失，开发者不承担法律责任。

---

MIT License © 2026 ROX Quant Team
