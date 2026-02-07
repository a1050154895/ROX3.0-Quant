
# 🚀 ROX 3.0 Quant Platform - 下一代开源量化终端

> **"让量化投资像玩游戏一样简单"**

![ROX 3.0 Banner](https://img.shields.io/badge/ROX-3.0_Pro-blueviolet?style=for-the-badge&logo=python)
![Beginner Friendly](https://img.shields.io/badge/Beginner-One_Click_Start-success?style=for-the-badge&logo=apple)
![Pro Ready](https://img.shields.io/badge/Professional-Algo_Trading-blue?style=for-the-badge&logo=linux)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

**ROX 3.0** 不仅仅是一个量化软件，它是一个**双核**投研平台，完美平衡了专业深度与使用门槛：

1.  🍃 **小白模式 (Beginner Mode)**：专为普通用户设计。内置 **AI 投研顾问**、**市场情绪温度计** 和 **一键选股**，无需看复杂的 K 线和代码。
2.  ⚡️ **极客模式 (Pro Mode)**：专为宽客设计。集成了 **A股/美股/Crypto** 全球行情、**宏观数据引擎** (GDP/CPI/PMI)、**Tick 级回测** 与 **算法交易**。

---

## 🍃 新手极速上手 (For Beginners)

**零代码、零配置，下载即用。**

无需懂 Python，无需会命令行。我们为您准备了“一键启动脚本”。

### 🍎 macOS 用户
1. 点击右上角 **Code** -> **Download ZIP** 下载本项目并解压。
2. 双击文件夹内的 `start_with_mac.command`。
3. 脚本会自动安装环境，并为您打开游览器进入系统。

### 🪟 Windows 用户
1. 点击右上角 **Code** -> **Download ZIP** 下载本项目并解压。
2. 双击文件夹内的 `start_with_win.bat`。
3. 等待黑色窗口跑完代码，系统即刻启动。

*(启动后，点击界面右上角的 “🍃 小白模式” 按钮，即可切换至极简界面)*

---

## ⚡️ 专业极客模式 (For Developers)

如果您是开发者或量化交易员，ROX 3.0 为您提供了无限的扩展能力。

### 🌟 核心特性 (Key Features)

#### 1. 🌍 上帝视角 (All-in-One Market)
*   **A股**: 深度集成 `AkShare`，支持沪深京全市场实时行情与历史数据。
*   **美股**: 连接 `YFinance`，纳斯达克/纽交所毫秒级延迟。
*   **Crypto**: 基于 `CCXT`，支持 Binance/OKX 等主流交易所。
*   *一键切换市场，无需打开多个软件。*

#### 2. 🧠 宏观罗盘 (Macro Engine)
*   从国家统计局直连数据，可视化展示 **M1-M2 剪刀差** (流动性指标) 与 **社融/PMI** (经济景气度)。
*   这是机构投资者做择时的大杀器，现在免费开放。

#### 3. 🔬 深度投研 (Deep Research)
*   **概念资金流**: 类似“一级市场”的热度扫描，捕捉独角兽与热门概念 (如固态电池、低空经济) 的资金流向。
*   **资讯雷达**: 7x24 小时全球财经快讯滚动，不错过任何黑天鹅。
*   **个股诊断**: 内置 7 大量化模型 (如亢龙有悔、三色共振) 自动评分。

---

## 🛠️ 安装与开发 (Manual Install)

如果您希望参与开发或手动部署：

```bash
# 1. 克隆代码
git clone https://github.com/a1050154895/ROX3.0-Quant.git
cd ROX3.0-Quant

# 2. 创建虚拟环境 (推荐)
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
```

---

## 📸 双模式展示

### 🍃 小白模式
*   **AI 投顾**: "这只股票能买吗？" —— 直接问 ROX QBot，基于 DeepSeek-V3 模型思考。
*   **市场温度计**: 实时计算市场多空情绪（Fear & Greed），一眼看懂现在是该贪婪还是该恐惧。
*   **本周金股**: AI 基于资金流与动量筛选的核心标的池，不做选择题。

### ⚡️ 专业模式
*   **K线复盘**: 多周期技术分析，叠加自研指标。
*   **资金流向**: 北向/南向资金实时监控，板块热力图。
*   **策略回测**: 在 `app/strategies/` 目录下编写您的 Python 策略，支持 Grid/TWAP/CTA。

---

## 🛡️ 免责声明 (Disclaimer)

本项目 (`ROX 3.0`) 仅供**量化投研学习与研究**使用，不构成任何投资建议。
*   金融市场风险巨大，自动化交易可能导致资金损失。
*   请务必在模拟盘充分测试后再考虑实盘。
*   开发者不对任何交易损失负责。

---

MIT License © 2026 ROX Quant Team
