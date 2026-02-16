/**
 * ROX 3.0 信号可视化增强模块
 * 提供7大核心交易信号的直观展示
 */

class SignalVisualization {
    constructor() {
        this.signals = {};
        this.signalHistory = {};
        this.charts = {};
        this.init();
    }

    init() {
        // 创建信号可视化容器
        this.createSignalContainer();
        
        // 初始化信号图表
        this.initSignalCharts();
        
        // 绑定事件
        this.bindEvents();
    }

    createSignalContainer() {
        const container = document.createElement('div');
        container.id = 'signal-visualization';
        container.className = 'fixed bottom-4 left-4 z-40 bg-slate-900/90 border border-slate-700 rounded-xl shadow-2xl p-4 w-80 backdrop-blur-sm';
        container.innerHTML = `
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-semibold text-white flex items-center">
                    <i class="fas fa-chart-line text-blue-400 mr-2"></i>
                    核心交易信号
                </h3>
                <button id="toggle-signals" class="text-slate-400 hover:text-white">
                    <i class="fas fa-chevron-down"></i>
                </button>
            </div>
            <div id="signal-content" class="space-y-3">
                <!-- 信号卡片将通过JavaScript动态添加 -->
            </div>
            <div id="signal-chart-container" class="mt-4 h-40">
                <!-- 信号趋势图表将通过JavaScript动态添加 -->
            </div>
        `;
        
        document.body.appendChild(container);
    }

    initSignalCharts() {
        // 检查是否已加载lightweight-charts库
        if (typeof LightweightCharts === 'undefined') {
            console.warn('LightweightCharts library not loaded');
            return;
        }

        const chartContainer = document.getElementById('signal-chart-container');
        if (!chartContainer) return;

        // 创建信号强度趋势图表
        this.charts.signalTrend = LightweightCharts.createChart(chartContainer, {
            width: chartContainer.clientWidth,
            height: 160,
            layout: {
                backgroundColor: 'transparent',
                textColor: '#94a3b8'
            },
            grid: {
                vertLines: {
                    color: 'rgba(148, 163, 184, 0.1)'
                },
                horzLines: {
                    color: 'rgba(148, 163, 184, 0.1)'
                }
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal
            },
            priceScale: {
                visible: true,
                min: 0,
                max: 100,
                ticks: {
                    font: '10px monospace',
                    color: '#94a3b8'
                }
            },
            timeScale: {
                visible: false
            }
        });

        // 添加信号强度线
        this.charts.signalLine = this.charts.signalTrend.addLineSeries({
            color: '#3b82f6',
            width: 2,
            lineType: LightweightCharts.LineType.Simple,
            crosshairMarkerVisible: false,
            priceMarkerVisible: false
        });
    }

    bindEvents() {
        // 切换信号面板显示/隐藏
        document.getElementById('toggle-signals').addEventListener('click', (e) => {
            const content = document.getElementById('signal-content');
            const chart = document.getElementById('signal-chart-container');
            const icon = e.target.closest('button').querySelector('i');
            
            content.classList.toggle('hidden');
            chart.classList.toggle('hidden');
            
            if (content.classList.contains('hidden')) {
                icon.classList.remove('fa-chevron-down');
                icon.classList.add('fa-chevron-up');
            } else {
                icon.classList.remove('fa-chevron-up');
                icon.classList.add('fa-chevron-down');
            }
        });
    }

    updateSignals(signals) {
        this.signals = signals;
        this.renderSignalCards();
        this.updateSignalHistory();
        this.updateSignalChart();
    }

    renderSignalCards() {
        const content = document.getElementById('signal-content');
        if (!content) return;

        const signalCards = Object.entries(this.signals).map(([name, signal]) => {
            const strengthClass = this.getStrengthClass(signal.strength);
            const signalIcon = this.getSignalIcon(signal.signal);
            const signalColor = this.getSignalColor(signal.signal);

            return `
                <div class="signal-card p-3 bg-slate-800/50 rounded-lg border border-slate-700 hover:bg-slate-800 transition-colors">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center">
                            <span class="text-white text-sm font-medium">${name}</span>
                            <span class="ml-2 text-xs px-2 py-0.5 rounded ${signalColor}">${signal.signal}</span>
                        </div>
                        <span class="text-slate-400 text-xs">${signal.confidence.toFixed(2)}</span>
                    </div>
                    <div class="space-y-2">
                        <div class="flex items-center justify-between">
                            <span class="text-slate-400 text-xs">信号强度</span>
                            <span class="text-${strengthClass} text-sm font-medium">${signal.strength.toFixed(0)}</span>
                        </div>
                        <div class="w-full bg-slate-700 rounded-full h-1.5">
                            <div class="h-1.5 rounded-full ${strengthClass === 'green-400' ? 'bg-green-400' : strengthClass === 'yellow-400' ? 'bg-yellow-400' : 'bg-red-400'}" style="width: ${signal.strength}%"></div>
                        </div>
                    </div>
                    ${signal.triggers && signal.triggers.length > 0 ? `
                        <div class="mt-2">
                            <span class="text-slate-500 text-xs">触发条件:</span>
                            <div class="text-slate-400 text-xs mt-1">${signal.triggers.slice(0, 2).join('; ')}</div>
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');

        content.innerHTML = signalCards;
    }

    updateSignalHistory() {
        const timestamp = Date.now();
        
        Object.entries(this.signals).forEach(([name, signal]) => {
            if (!this.signalHistory[name]) {
                this.signalHistory[name] = [];
            }
            
            this.signalHistory[name].push({
                timestamp,
                strength: signal.strength,
                signal: signal.signal
            });
            
            // 保留最近30个数据点
            if (this.signalHistory[name].length > 30) {
                this.signalHistory[name] = this.signalHistory[name].slice(-30);
            }
        });
    }

    updateSignalChart() {
        if (!this.charts.signalLine) return;
        
        // 计算综合信号强度
        const combinedStrength = Object.values(this.signals).reduce((sum, signal) => sum + signal.strength, 0) / Object.values(this.signals).length;
        
        // 更新图表数据
        if (!this.chartData) {
            this.chartData = [];
        }
        
        this.chartData.push({
            time: Date.now().toString(),
            value: combinedStrength
        });
        
        // 保留最近30个数据点
        if (this.chartData.length > 30) {
            this.chartData = this.chartData.slice(-30);
        }
        
        this.charts.signalLine.setData(this.chartData);
    }

    getStrengthClass(strength) {
        if (strength >= 70) return 'green-400';
        if (strength >= 40) return 'yellow-400';
        return 'red-400';
    }

    getSignalIcon(signal) {
        switch (signal) {
            case '强烈买入': return 'fas fa-arrow-up';
            case '买入': return 'fas fa-arrow-up';
            case '持有': return 'fas fa-minus';
            case '卖出': return 'fas fa-arrow-down';
            case '强烈卖出': return 'fas fa-arrow-down';
            default: return 'fas fa-question';
        }
    }

    getSignalColor(signal) {
        switch (signal) {
            case '强烈买入': return 'bg-green-500/20 text-green-400';
            case '买入': return 'bg-green-500/10 text-green-400';
            case '持有': return 'bg-yellow-500/20 text-yellow-400';
            case '卖出': return 'bg-red-500/10 text-red-400';
            case '强烈卖出': return 'bg-red-500/20 text-red-400';
            default: return 'bg-slate-500/20 text-slate-400';
        }
    }

    bindEvents() {
        // 监听信号更新事件
        window.addEventListener('signalsUpdated', (e) => {
            this.updateSignals(e.detail.signals);
        });
    }

    // 显示/隐藏信号面板
    toggleVisibility() {
        const container = document.getElementById('signal-visualization');
        if (container) {
            container.classList.toggle('hidden');
        }
    }

    // 从API加载信号数据
    async loadSignals(code) {
        try {
            const response = await fetch(`/api/professional-plus/signals/${code}`);
            const data = await response.json();
            
            if (data.core_signals) {
                const signals = {};
                data.core_signals.forEach(signal => {
                    signals[signal.name] = {
                        signal: signal.signal,
                        strength: signal.strength,
                        confidence: signal.confidence,
                        triggers: signal.triggers
                    };
                });
                
                this.updateSignals(signals);
                
                // 触发信号更新事件
                window.dispatchEvent(new CustomEvent('signalsUpdated', {
                    detail: { signals }
                }));
            }
        } catch (error) {
            console.error('加载信号数据失败:', error);
        }
    }
}

// 全局信号可视化实例
window.signalVisualization = new SignalVisualization();

// 全局函数
window.updateSignals = function(signals) {
    if (window.signalVisualization) {
        window.signalVisualization.updateSignals(signals);
    }
};

window.loadSignals = function(code) {
    if (window.signalVisualization) {
        window.signalVisualization.loadSignals(code);
    }
};

window.toggleSignalPanel = function() {
    if (window.signalVisualization) {
        window.signalVisualization.toggleVisibility();
    }
};
