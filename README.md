
# 🚀 ROX 3.0 Quant Platform - 下一代全资产量化投研终端

> **"让量化投资像玩游戏一样简单"**

[![ROX 3.0 Banner](https://img.shields.io/badge/ROX-3.0_Pro-blueviolet?style=for-the-badge&logo=python)](https://github.com/a1050154895/ROX3.0-Quant)
[![Beginner Friendly](https://img.shields.io/badge/Beginner-One_Click_Start-success?style=for-the-badge&logo=apple)](https://github.com/a1050154895/ROX3.0-Quant)
[![Pro Ready](https://img.shields.io/badge/Professional-Algo_Trading-blue?style=for-the-badge&logo=linux)](https://github.com/a1050154895/ROX3.0-Quant)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

**ROX 3.0** 不仅仅是一个量化软件，它是一个**双核**投研平台，完美平衡了专业深度与使用门槛。它整合了**A股、美股、加密货币**全球三大市场，将机构级的宏观数据、资金流向与 AI 投研能力免费带给每一位投资者。

---

## 📢 重要提示：解锁 AI 黑科技 (AI Features)

> **"下载最新版后，点击右上角设置图标 ⚙️，填入自己的 DeepSeek 或 OpenAI Key 即可解锁 AI 功能。"**

本项目已深度集成由 **DeepSeek / OpenAI** 驱动的 AI 投研顾问。为保障服务稳定性，**请使用您自己的 API Key**。

1.  启动软件进入主界面。
2.  点击右上角的 **设置 (⚙️)** 按钮。
3.  在 **[AI 模型配置]** 中填入您的 API Key 和 Base URL (如 `https://api.deepseek.com`)。
4.  **[NEW] 双线路支持**：现在您可以配置 **备用线路 (Secondary API)**，主线路故障时自动切换，确保服务不中断。
5.  保存后即可立即使用 **智能个股诊断**、**市场每日简报** 和 **AI 自由问答**。

### 📅 最新更新 (v3.1)
*   **Dual AI API**: 支持主备双 API 自动故障切换。
*   **稳定性增强**: 修复了个股诊断与深度分析的已知崩溃问题。
*   **数据源扩容**: 集成 YFinance 与 CCXT，增强全球市场数据支持。

---

## 📚 目录 (Table of Contents)

- [🚀 两大核心模式 (Dual Modes)](#-两大核心模式-dual-modes)
- [🍃 新手极速上手 (Quick Start)](#-新手极速上手-quick-start)
- [⚡️ 开发者安装 (For Developers)](#-开发者安装-for-developers)
- [📖 功能详解 (Features)](#-功能详解-features)
    - [1. 市场看板 (Dashboard)](#1-市场看板-dashboard)
    - [2. 宏观罗盘 (Macro)](#2-宏观罗盘-macro)
    - [3. AI 投研顾问 (AI Agent)](#3-ai-投研顾问-ai-agent)
    - [4. 量化策略 (Strategies)](#4-量化策略-strategies)
- [🛠️ 系统架构 (Architecture)](#-系统架构-architecture)
- [🛡️ 免责声明 (Disclaimer)](#-免责声明-disclaimer)

---

## 🚀 两大核心模式 (Dual Modes)

ROX 3.0 设计了两套完全不同的交互界面，以适应不同阶段的用户需求。

### 1. 🍃 小白模式 (Beginner Mode)
专为非金融背景、非编程背景的普通用户设计。
*   **极简界面**：隐藏复杂的 K 线、盘口和订单流。
*   **AI 驱动**：通过对话框与 "AI 投研顾问" 交互，获取投资建议。
*   **直观决策**：提供 "市场温度计"（情绪指标）和 "一键选股"（本周金股池）。

### 2. ⚡️ 专业极客模式 (Pro Mode)
专为宽客 (Quants)、全职交易员和开发者设计。
*   **全能终端**：类似 Bloomberg/Wind 的多屏工作站体验。
*   **深度数据**：Level-2 盘口、逐笔成交、资金流向、板块热力图。
*   **策略引擎**：支持 Python 策略编写、回测、仿真交易。

---

## 🍃 新手极速上手 (Quick Start)

**零代码、零配置，下载即用。**

我们为您准备了“一键启动脚本”，脚本会自动检测系统环境、安装 Python 依赖并启动浏览器。

### 🍎 macOS 用户
1. 点击右上角 **Code** -> **Download ZIP** 下载并解压。
2. 打开解压后的文件夹，双击运行 `start_with_mac.command`。
3. 脚本即刻启动，自动为您打开 ROX 3.0 系统界面。

### 🪟 Windows 用户
1. 下载并解压项目文件。
2. 双击运行 `start_with_win.bat`。
3. 等待初始化完成，系统自动打开浏览器。

---

## ⚡️ 开发者安装 (For Developers)

如果您希望参与开发或手动部署：

### 环境要求
*   **Python**: 3.9+
*   **Git**: Version Control

### 安装步骤

1. **克隆代码库**
   ```bash
   git clone https://github.com/a1050154895/ROX3.0-Quant.git
   cd ROX3.0-Quant
   ```

2. **创建环境并安装依赖**
   ```bash
   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **启动服务**
   ```bash
   python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   访问: `http://localhost:8000`

---

## 📖 功能详解 (Features)

### 1. 市场看板 (Dashboard)
*   **多市场切换**：顶部导航栏支持 `[A股] [美股] [Crypto]` 一键切换。
*   **实时指数**：上证指数、纳斯达克、BTC/USDT 实时跳动。
*   **K线复盘**：集成了 TradingView 风格的图表，支持日/周/月线切换及技术指标叠加。

### 2. 宏观罗盘 (Macro)
*   **数据源**：直连中国国家统计局 (NBS) 接口。
*   **M1-M2 剪刀差**：可视化展示货币供应量剪刀差，预判牛熊周期。
*   **CSI/PPI**: 监控通胀水平与经济活力。

### 3. AI 投研顾问 (AI Agent)
*   **对话式交互**：输入 "分析 600519"，AI 综合技术面与基本面给出建议。
*   **本地知识库 (RAG)**：优先检索本地策略文档，确保回答专业。
*   **风险预警**：自动识别高风险标的。

### 4. 量化策略 (Strategies)
*   **策略工坊**：可视化拖拽生成交易逻辑。
*   **每周金股**：每周一自动更新的高胜率潜力股池。
*   **个股诊断**：内置 "亢龙有悔"、"三色共振" 等经典模型打分。

---

## 🛠️ 系统架构 (Architecture)

```mermaid
graph TD
    User[用户终端] --> |HTTP/WebSocket| Gateway[FastAPI 网关]
    
    subgraph "Backend Services"
        Gateway --> MarketService[行情服务]
        Gateway --> AIService[AI 投研服务]
        Gateway --> DataCenter[数据中心]
    end
    
    subgraph "Data Sources"
        DataCenter --> AkShare[AkShare (A股)]
        DataCenter --> YFinance[YFinance (美股)]
        DataCenter --> CCXT[CCXT (Crypto)]
    end
    
    subgraph "AI Core"
        AIService --> LLM[LLM (DeepSeek/GPT)]
    end
```

---

## 🛡️ 免责声明 (Disclaimer)

1.  **风险提示**：量化投资涉及风险。本软件所有数据与 AI 建议仅供参考，**绝不构成投资建议**。
2.  **数据来源**：数据来源于公开互联网接口，开发者不对数据的准确性与实时性做保证。
3.  **资金安全**：请保管好您的 API 密钥与账户信息。

---

MIT License © 2026 ROX Quant Team
