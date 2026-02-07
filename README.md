
# 🚀 ROX 3.0 量化投研平台

![ROX 3.0 Banner](https://img.shields.io/badge/ROX-3.0_Pro-blueviolet?style=for-the-badge&logo=python)
![Python](https://img.shields.io/badge/python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi&logoColor=white)
![Vue/Tailwind](https://img.shields.io/badge/Frontend-Tailwind_CSS-38B2AC?style=flat-square&logo=tailwindcss&logoColor=white)

**ROX 3.0** 是一个为专业个人投资者和小型基金设计的**多资产量化投研平台**。它集成了 **A股、美股、加密货币 (Crypto)** 的行情与交易能力，拥有高性能的 Python 回测引擎和现代化的 Web 数据看板。

> **当前状态**: 生产就绪 (Phase 1-6 已完成)

---

## ✨ 核心功能

### 🌍 1. 多资产全市场覆盖
- **统一接口**: 无缝切换 **A股 (CN)**、**美股 (US)** 和 **数字货币 (Crypto)** 市场视图。
- **数据集成**:
  - `yfinance`: 覆盖全球股票市场数据。
  - `ccxt`: 支持主流加密货币交易所。
  - `AkShare`: 接入 A 股实时行情与历史数据。

### 🧠 2. 高级策略引擎
- **Tick 级回测**: 内置高精度 `TickEngine`，支持订单流重放，还原真实交易场景。
- **算法交易**: 提供开箱即用的 **TWAP** (时间加权平均) 和 **网格交易 (Grid Trading)** 算法。
- **策略超市**: 社区化策略管理，支持一键安装精选策略（如 *Grid Master*, *Momentum Alpha*）。

### 📊 3. 宏观与数据智能 (Phase 6 新增)
- **宏观仪表盘**: 可视化展示 **GDP, CPI, PMI** 及关键的 **M1-M2 剪刀差** (流动性指标)。
- **资讯雷达**: 7x24 小时全球财经快讯流，实时推送个股公告。
- **主题挖掘**: 追踪一级市场与热门概念资金流向（如 *固态电池*、*低空经济* 等风口）。

### ☁️ 4. 云端与社交
- **云同步**: 支持系统配置与数据的完整备份与恢复 (ZIP 格式)，轻松跨设备迁移。
- **用户档案**: 包含头像、简介及标签的交易员身份系统。
- **社交交易**: (Beta) 模拟跟单信号引擎，支持 Signal Source 抽象。

---

## 🛠️ 技术栈

- **后端**: Python 3.9+, FastAPI, SQLite, Pandas, Numpy
- **数据源**: AkShare, YFinance, CCXT, 东方财富 (Proxy)
- **前端**: HTML5, Vanilla JS (ES6+), Tailwind CSS, ECharts
- **服务器**: Uvicorn (ASGI)

---

## 🚀 快速开始

### 环境要求
- Python 3.9 或更高版本
- Git

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/a1050154895/ROX3.0-Quant.git
   cd ROX3.0-Quant
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **启动服务**
   ```bash
   python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
   ```

4. **访问系统**
   打开浏览器访问: `http://localhost:8002`

---

## 📂 项目结构

```
ROX3.0-Quant/
├── app/
│   ├── api/             # API 接口 (宏观、行情、交易...)
│   ├── rox_quant/       # 核心量化引擎 (TickEngine, Algos)
│   ├── strategies/      # 用户策略文件目录
│   ├── static/          # JS, CSS, 静态资源
│   └── templates/       # HTML 模板 (index_rox2.html)
├── data/db/             # SQLite 数据库文件
├── requirements.txt     # Python 依赖列表
└── README.md            # 项目文档
```

---

## 🤝 参与贡献

欢迎提交 Pull Request 或 Issue 来改进 ROX 3.0！

## 📄 开源协议

MIT License © 2026 ROX Quant Team
