
# 🚀 ROX 3.0 Quant Platform

![ROX 3.0 Banner](https://img.shields.io/badge/ROX-3.0_Pro-blueviolet?style=for-the-badge&logo=python)
![Python](https://img.shields.io/badge/python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi&logoColor=white)
![Vue/Tailwind](https://img.shields.io/badge/Frontend-Tailwind_CSS-38B2AC?style=flat-square&logo=tailwindcss&logoColor=white)

**ROX 3.0** is an advanced, multi-asset quantitative trading platform designed for serious retail investors and small funds. It integrates **A-Shares, US Stocks, and Crypto** into a unified research and trading environment, powered by a high-performance Python backend and a modern web interface.

> **Status**: Production Ready (Phase 1-6 Complete)

---

## ✨ Key Features

### 🌍 1. Multi-Asset Architecture
- **Unified Interface**: Seamlessly switch between **A-Shares (CN)**, **US Stocks (US)**, and **Crypto**.
- **Data Integration**:
  - `yfinance` for global equities.
  - `ccxt` for cryptocurrency exchanges.
  - `AskShare` for A-Share real-time data.

### 🧠 2. Advanced Strategy Engine
- **Tick-Level Backtesting**: High-precision `TickEngine` for accurate simulation.
- **Algo Trading**: Built-in **TWAP** (Time-Weighted Average Price) and **Grid Trading** algorithms.
- **Strategy Marketplace**: One-click install for community strategies (e.g., *Grid Master*, *Momentum Alpha*).

### 📊 3. Macro & Data Intelligence (New!)
- **Macro Dashboard**: Visualize **GDP, CPI, PMI**, and the critical **M1-M2 Liquidity Scissors**.
- **Info Radar**: 7x24 global financial news stream and real-time company announcements.
- **Theme Mining**: Track "Primary Market" hot money flows into concepts like *Lithium Batteries*, *Low-Altitude Economy*, etc.

### ☁️ 4. Cloud & Social
- **Cloud Sync**: Full system backup & restore (ZIP format) for cross-device migrations.
- **User Profile**: Identity system with avatars, bios, and trader tags.
- **Social Trading**: (Beta) Simulated copy-trading signal engine.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.9+, FastAPI, SQLite, Pandas, Numpy
- **Data Providers**: AkShare, YFinance, CCXT, Eastmoney (Proxy)
- **Frontend**: HTML5, Vanilla JS (ES6+), Tailwind CSS, ECharts
- **Server**: Uvicorn (ASGI)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/a1050154895/ROX3.0-Quant.git
   cd ROX3.0-Quant
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Server**
   ```bash
   python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
   ```

4. **Access the UI**
   Open your browser and navigate to: `http://localhost:8002`

---

## 📂 Project Structure

```
ROX3.0-Quant/
├── app/
│   ├── api/             # API Endpoints (Macro, Market, Trade...)
│   ├── rox_quant/       # Core Quant Engine (TickEngine, Algos)
│   ├── strategies/      # User Strategy Files
│   ├── static/          # JS, CSS, Assets
│   └── templates/       # HTML Templates (index_rox2.html)
├── data/db/             # SQLite Databases
├── requirements.txt     # Python Dependencies
└── README.md            # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📄 License

MIT License © 2026 ROX Quant Team
