// ROX 3.0 Trading Simulation Module
// 交易模拟系统的前端模块

class TradingSimulationModule {
    constructor() {
        this.isRunning = false;
        this.simulationInterval = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadTraders();
        this.loadMarketData();
    }

    bindEvents() {
        const startBtn = document.getElementById('start-simulation');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startSimulation());
        }

        const stopBtn = document.getElementById('stop-simulation');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopSimulation());
        }

        const refreshBtn = document.getElementById('refresh-data');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadTraders();
                this.loadMarketData();
                this.loadSimulationStatus();
            });
        }
    }

    async startSimulation() {
        const symbolsInput = document.getElementById('symbols-input');
        const durationInput = document.getElementById('duration-input');
        const intervalInput = document.getElementById('interval-input');

        const symbols = symbolsInput.value.split(',').map(s => s.trim()).filter(s => s);
        const duration = parseInt(durationInput.value) || 60;
        const interval = parseInt(intervalInput.value) || 5;

        if (symbols.length === 0) {
            alert('请输入交易品种');
            return;
        }

        try {
            const response = await fetch('/api/trading-simulation/simulation/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbols: symbols,
                    duration: duration,
                    interval: interval
                })
            });

            if (!response.ok) {
                throw new Error('启动模拟失败');
            }

            const result = await response.json();
            console.log('模拟启动成功:', result);
            this.isRunning = true;
            this.updateUI();
            this.startPolling();
        } catch (error) {
            console.error('启动模拟失败:', error);
            alert('启动模拟失败: ' + error.message);
        }
    }

    async stopSimulation() {
        try {
            const response = await fetch('/api/trading-simulation/simulation/stop', {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('停止模拟失败');
            }

            const result = await response.json();
            console.log('模拟停止成功:', result);
            this.isRunning = false;
            this.stopPolling();
            this.updateUI();
        } catch (error) {
            console.error('停止模拟失败:', error);
            alert('停止模拟失败: ' + error.message);
        }
    }

    async loadTraders() {
        try {
            const response = await fetch('/api/trading-simulation/traders');
            if (!response.ok) {
                throw new Error('获取交易员失败');
            }

            const traders = await response.json();
            this.renderTraders(traders);
        } catch (error) {
            console.error('获取交易员失败:', error);
            this.renderTradersError();
        }
    }

    async loadMarketData() {
        try {
            const response = await fetch('/api/trading-simulation/exchange/symbols');
            if (!response.ok) {
                throw new Error('获取交易品种失败');
            }

            const symbolsData = await response.json();
            const symbols = symbolsData.symbols;

            if (symbols && symbols.length > 0) {
                const symbol = symbols[0];
                const marketDataResponse = await fetch(`/api/trading-simulation/exchange/market-data/${symbol}`);
                if (marketDataResponse.ok) {
                    const marketData = await marketDataResponse.json();
                    this.renderMarketData(marketData);
                }
            } else {
                this.renderMarketDataEmpty();
            }
        } catch (error) {
            console.error('获取市场数据失败:', error);
            this.renderMarketDataError();
        }
    }

    async loadSimulationStatus() {
        try {
            const response = await fetch('/api/trading-simulation/simulation/status');
            if (!response.ok) {
                throw new Error('获取模拟状态失败');
            }

            const status = await response.json();
            this.isRunning = status.is_running;
            this.updateUI();
        } catch (error) {
            console.error('获取模拟状态失败:', error);
        }
    }

    async loadSimulationReport() {
        try {
            const response = await fetch('/api/trading-simulation/simulation/report');
            if (!response.ok) {
                throw new Error('获取模拟报告失败');
            }

            const report = await response.json();
            this.renderSimulationReport(report);
        } catch (error) {
            console.error('获取模拟报告失败:', error);
        }
    }

    renderTraders(traders) {
        const tradersContainer = document.getElementById('traders-container');
        if (!tradersContainer) return;

        if (!traders || traders.length === 0) {
            tradersContainer.innerHTML = `
                <div class="flex flex-col items-center justify-center py-12 text-slate-400">
                    <i class="fas fa-users text-3xl mb-2 opacity-50"></i>
                    <p>暂无交易员数据</p>
                </div>
            `;
            return;
        }

        let html = '';
        traders.forEach(trader => {
            const performanceClass = trader.performance >= 0 ? 'text-emerald-400' : 'text-red-400';
            const performanceIcon = trader.performance >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
            const emotionColor = trader.emotion > 0.7 ? 'bg-emerald-500' : trader.emotion > 0.3 ? 'bg-amber-500' : 'bg-red-500';
            const riskColor = trader.risk_appetite > 0.7 ? 'bg-red-500' : trader.risk_appetite > 0.3 ? 'bg-amber-500' : 'bg-emerald-500';
            
            html += `
                <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-4 hover:border-purple-500/50 transition-all">
                    <div class="flex justify-between items-start mb-3">
                        <div>
                            <h3 class="text-lg font-semibold text-slate-200">${trader.name}</h3>
                            <p class="text-sm text-slate-400">${trader.personality || '理性投资者'}</p>
                        </div>
                        <span class="px-2 py-1 rounded bg-purple-500/20 text-purple-400 text-xs">${trader.strategy}</span>
                    </div>
                    
                    <div class="grid grid-cols-2 gap-3 mb-3 text-sm">
                        <div>
                            <span class="text-slate-500">本金:</span>
                            <span class="text-slate-200 ml-1">¥${(trader.initial_balance || 10000).toLocaleString()}</span>
                        </div>
                        <div>
                            <span class="text-slate-500">余额:</span>
                            <span class="text-slate-200 ml-1">¥${trader.balance.toLocaleString()}</span>
                        </div>
                        <div>
                            <span class="text-slate-500">绩效:</span>
                            <span class="${performanceClass} ml-1">
                                <i class="fas ${performanceIcon} text-xs mr-1"></i>${trader.performance.toFixed(2)}%
                            </span>
                        </div>
                        <div>
                            <span class="text-slate-500">风险:</span>
                            <span class="text-slate-200 ml-1">${trader.risk_level}</span>
                        </div>
                    </div>
                    
                    <div class="space-y-2">
                        <div>
                            <div class="flex justify-between text-xs mb-1">
                                <span class="text-slate-500">情绪</span>
                                <span class="text-slate-400">${(trader.emotion * 100).toFixed(0)}%</span>
                            </div>
                            <div class="w-full bg-slate-700 rounded-full h-2">
                                <div class="${emotionColor} h-2 rounded-full transition-all" style="width: ${trader.emotion * 100}%"></div>
                            </div>
                        </div>
                        <div>
                            <div class="flex justify-between text-xs mb-1">
                                <span class="text-slate-500">风险偏好</span>
                                <span class="text-slate-400">${(trader.risk_appetite * 100).toFixed(0)}%</span>
                            </div>
                            <div class="w-full bg-slate-700 rounded-full h-2">
                                <div class="${riskColor} h-2 rounded-full transition-all" style="width: ${trader.risk_appetite * 100}%"></div>
                            </div>
                        </div>
                    </div>
                    
                    ${trader.selected_stocks && trader.selected_stocks.length > 0 ? `
                        <div class="mt-3 pt-3 border-t border-slate-700">
                            <span class="text-xs text-slate-500">持仓:</span>
                            <div class="flex flex-wrap gap-1 mt-1">
                                ${trader.selected_stocks.map(stock => `<span class="px-2 py-0.5 rounded bg-slate-700 text-slate-300 text-xs">${stock}</span>`).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            `;
        });

        tradersContainer.innerHTML = html;
    }

    renderTradersError() {
        const tradersContainer = document.getElementById('traders-container');
        if (!tradersContainer) return;

        tradersContainer.innerHTML = `
            <div class="flex flex-col items-center justify-center py-12 text-red-400">
                <i class="fas fa-exclamation-circle text-3xl mb-2 opacity-50"></i>
                <p>加载交易员数据失败</p>
            </div>
        `;
    }

    renderMarketData(marketData) {
        const marketDataContainer = document.getElementById('market-data-container');
        if (!marketDataContainer) return;

        if (!marketData) {
            this.renderMarketDataEmpty();
            return;
        }

        const html = `
            <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-4 mb-4">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold text-slate-200">${marketData.symbol}</h3>
                    <span class="text-2xl font-bold text-slate-200">¥${(marketData.last_price || 0).toFixed(2)}</span>
                </div>
                
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <h4 class="text-sm font-medium text-emerald-400 mb-2">
                            <i class="fas fa-arrow-up mr-1"></i>买单
                        </h4>
                        <div class="space-y-1">
                            ${(marketData.order_book?.buy || []).slice(0, 5).map(order => `
                                <div class="flex justify-between text-sm">
                                    <span class="text-emerald-400">¥${order.price.toFixed(2)}</span>
                                    <span class="text-slate-400">${order.quantity}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    <div>
                        <h4 class="text-sm font-medium text-red-400 mb-2">
                            <i class="fas fa-arrow-down mr-1"></i>卖单
                        </h4>
                        <div class="space-y-1">
                            ${(marketData.order_book?.sell || []).slice(0, 5).map(order => `
                                <div class="flex justify-between text-sm">
                                    <span class="text-red-400">¥${order.price.toFixed(2)}</span>
                                    <span class="text-slate-400">${order.quantity}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <h4 class="text-sm font-medium text-slate-300 mb-3">
                    <i class="fas fa-history mr-1"></i>最近交易
                </h4>
                <div class="space-y-2">
                    ${(marketData.recent_trades || []).slice(0, 5).map(trade => `
                        <div class="flex justify-between items-center text-sm py-1 border-b border-slate-700/50">
                            <span class="text-slate-200">¥${trade.price.toFixed(2)}</span>
                            <span class="text-slate-400">${trade.quantity}</span>
                            <span class="text-xs text-slate-500">${trade.buyer_id?.substring(0, 8) || '-'} → ${trade.seller_id?.substring(0, 8) || '-'}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        marketDataContainer.innerHTML = html;
    }

    renderMarketDataEmpty() {
        const marketDataContainer = document.getElementById('market-data-container');
        if (!marketDataContainer) return;

        marketDataContainer.innerHTML = `
            <div class="flex flex-col items-center justify-center py-12 text-slate-400">
                <i class="fas fa-chart-line text-3xl mb-2 opacity-50"></i>
                <p>暂无市场数据</p>
            </div>
        `;
    }

    renderMarketDataError() {
        const marketDataContainer = document.getElementById('market-data-container');
        if (!marketDataContainer) return;

        marketDataContainer.innerHTML = `
            <div class="flex flex-col items-center justify-center py-12 text-red-400">
                <i class="fas fa-exclamation-circle text-3xl mb-2 opacity-50"></i>
                <p>加载市场数据失败</p>
            </div>
        `;
    }

    renderSimulationReport(report) {
        const reportContainer = document.getElementById('simulation-report-container');
        if (!reportContainer) return;

        if (!report) {
            reportContainer.innerHTML = `
                <div class="flex flex-col items-center justify-center py-12 text-slate-400">
                    <i class="fas fa-inbox text-3xl mb-2 opacity-50"></i>
                    <p>尚未运行模拟</p>
                </div>
            `;
            return;
        }

        const html = `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                    <div class="text-2xl font-bold text-purple-400 mb-1">${report.total_trades || 0}</div>
                    <div class="text-sm text-slate-400">总交易次数</div>
                </div>
                <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                    <div class="text-2xl font-bold ${report.average_trader_performance >= 0 ? 'text-emerald-400' : 'text-red-400'} mb-1">
                        ${(report.average_trader_performance || 0).toFixed(2)}%
                    </div>
                    <div class="text-sm text-slate-400">平均绩效</div>
                </div>
                <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                    <div class="text-2xl font-bold text-sky-400 mb-1">${report.total_simulation_time || 0}s</div>
                    <div class="text-sm text-slate-400">模拟时长</div>
                </div>
            </div>
            
            ${report.improvement_suggestions && report.improvement_suggestions.length > 0 ? `
                <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-4 mb-4">
                    <h4 class="text-sm font-medium text-slate-300 mb-3">
                        <i class="fas fa-lightbulb text-amber-400 mr-2"></i>改进建议
                    </h4>
                    <ul class="space-y-2">
                        ${report.improvement_suggestions.map(suggestion => `
                            <li class="flex items-start gap-2 text-sm text-slate-400">
                                <i class="fas fa-check text-emerald-400 mt-0.5"></i>
                                <span>${suggestion}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${report.current_model_parameters ? `
                <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                    <h4 class="text-sm font-medium text-slate-300 mb-3">
                        <i class="fas fa-cog text-slate-400 mr-2"></i>当前模型参数
                    </h4>
                    <pre class="text-xs text-slate-400 bg-slate-900/50 rounded-lg p-3 overflow-x-auto">${JSON.stringify(report.current_model_parameters, null, 2)}</pre>
                </div>
            ` : ''}
        `;

        reportContainer.innerHTML = html;
    }

    updateUI() {
        const startBtn = document.getElementById('start-simulation');
        const stopBtn = document.getElementById('stop-simulation');

        if (startBtn) {
            startBtn.disabled = this.isRunning;
            if (this.isRunning) {
                startBtn.classList.add('opacity-50', 'cursor-not-allowed');
            } else {
                startBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        }

        if (stopBtn) {
            stopBtn.disabled = !this.isRunning;
            if (!this.isRunning) {
                stopBtn.classList.add('opacity-50', 'cursor-not-allowed');
            } else {
                stopBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        }

        const statusElement = document.getElementById('simulation-status');
        if (statusElement) {
            statusElement.textContent = this.isRunning ? '运行中' : '已停止';
            statusElement.className = this.isRunning 
                ? 'px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 text-sm font-medium'
                : 'px-3 py-1 rounded-full bg-slate-700 text-slate-300 text-sm font-medium';
        }
    }

    startPolling() {
        this.simulationInterval = setInterval(() => {
            this.loadTraders();
            this.loadMarketData();
            this.loadSimulationReport();
        }, 5000);
    }

    stopPolling() {
        if (this.simulationInterval) {
            clearInterval(this.simulationInterval);
            this.simulationInterval = null;
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.tradingSimulationModule = new TradingSimulationModule();
});
