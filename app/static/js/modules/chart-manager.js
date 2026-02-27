/**
 * 图表管理器 - 优化ECharts性能和内存使用
 * 
 * 功能特性：
 * - 图表实例复用和生命周期管理
 * - 自动resize处理和性能优化
 * - 内存泄漏防护
 * - 批量更新支持
 */
class ChartManager {
    constructor() {
        this.charts = new Map();
        this.resizeObserver = null;
        this.resizeTimers = new Map();
        this.initResizeObserver();
    }

    /**
     * 初始化图表
     * @param {string} domId - DOM元素ID
     * @param {Object} options - ECharts初始化选项
     * @returns {Object|null} ECharts实例
     */
    initChart(domId, options = {}) {
        const dom = document.getElementById(domId);
        if (!dom) {
            console.warn(`[ChartManager] Container ${domId} not found`);
            return null;
        }

        // 如果图表已存在，先销毁
        this.disposeChart(domId);

        const chart = echarts.init(dom, null, {
            renderer: 'canvas',
            useDirtyRect: true, // 启用脏矩形优化
            width: 'auto',
            height: 'auto',
            ...options
        });

        this.charts.set(domId, {
            instance: chart,
            dom: dom,
            lastUpdate: Date.now()
        });
        
        // 添加到resize观察器
        if (this.resizeObserver) {
            this.resizeObserver.observe(dom);
        }

        return chart;
    }

    /**
     * 获取图表实例
     * @param {string} domId - DOM元素ID
     * @returns {Object|null} ECharts实例
     */
    getChart(domId) {
        const chartData = this.charts.get(domId);
        return chartData ? chartData.instance : null;
    }

    /**
     * 销毁图表
     * @param {string} domId - DOM元素ID
     */
    disposeChart(domId) {
        const chartData = this.charts.get(domId);
        if (chartData) {
            // 清理resize定时器
            const timer = this.resizeTimers.get(domId);
            if (timer) {
                clearTimeout(timer);
                this.resizeTimers.delete(domId);
            }
            
            // 销毁图表实例
            chartData.instance.dispose();
            this.charts.delete(domId);
            
            console.log(`[ChartManager] Chart ${domId} disposed`);
        }
    }

    /**
     * 销毁所有图表
     */
    disposeAll() {
        console.log(`[ChartManager] Disposing ${this.charts.size} charts`);
        
        this.charts.forEach((chartData, domId) => {
            chartData.instance.dispose();
        });
        this.charts.clear();
        
        // 清理所有定时器
        this.resizeTimers.forEach(timer => clearTimeout(timer));
        this.resizeTimers.clear();
        
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
    }

    /**
     * 初始化resize观察器
     */
    initResizeObserver() {
        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(entries => {
                entries.forEach(entry => {
                    const domId = entry.target.id;
                    if (!domId) return;
                    
                    const chartData = this.charts.get(domId);
                    if (chartData) {
                        // 防抖resize - 避免频繁调用
                        const existingTimer = this.resizeTimers.get(domId);
                        if (existingTimer) {
                            clearTimeout(existingTimer);
                        }
                        
                        const timer = setTimeout(() => {
                            try {
                                chartData.instance.resize();
                                this.resizeTimers.delete(domId);
                            } catch (error) {
                                console.error(`[ChartManager] Resize error for ${domId}:`, error);
                            }
                        }, 100);
                        
                        this.resizeTimers.set(domId, timer);
                    }
                });
            });
        } else {
            // 降级到window resize事件
            let globalResizeTimer;
            window.addEventListener('resize', () => {
                clearTimeout(globalResizeTimer);
                globalResizeTimer = setTimeout(() => {
                    this.charts.forEach((chartData, domId) => {
                        try {
                            chartData.instance.resize();
                        } catch (error) {
                            console.error(`[ChartManager] Global resize error for ${domId}:`, error);
                        }
                    });
                }, 150);
            });
        }
    }

    /**
     * 批量更新图表数据
     * @param {Array} updates - 更新配置数组
     * @returns {Promise} 更新完成Promise
     */
    async batchUpdate(updates) {
        const promises = updates.map(({ domId, option, notMerge = false }) => {
            return new Promise(resolve => {
                const chartData = this.charts.get(domId);
                if (chartData) {
                    try {
                        chartData.instance.setOption(option, notMerge);
                        chartData.lastUpdate = Date.now();
                    } catch (error) {
                        console.error(`[ChartManager] Update error for ${domId}:`, error);
                    }
                }
                resolve();
            });
        });
        
        return Promise.all(promises);
    }

    /**
     * 创建K线图
     * @param {string} domId - DOM元素ID
     * @param {Object} data - K线数据
     * @returns {Object|null} ECharts实例
     */
    async addDarkPoolFund(domId, stockCode) {
        const chart = this.getChart(domId);
        if (!chart) {
            console.warn(`[ChartManager] Chart ${domId} not found.`);
            return;
        }

        try {
            const response = await fetch(`/api/analysis/dark-pool-fund/${stockCode}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch Dark Pool Fund data: ${response.statusText}`);
            }
            const data = await response.json();

            const darkPoolFundData = data.map(item => [item.日期, item.暗盘资金]);

            // Get current series, filter out existing '暗盘资金' series to prevent duplicates
            const existingSeries = chart.getOption().series;
            const newSeries = existingSeries.filter(s => s.name !== '暗盘资金');

            chart.setOption({
                legend: {
                    ...chart.getOption().legend,
                    data: [...chart.getOption().legend.data, '暗盘资金'],
                    selected: {
                        ...chart.getOption().legend.selected,
                        '暗盘资金': true
                    }
                },
                series: [
                    ...newSeries,
                    {
                        name: '暗盘资金',
                        type: 'line',
                        data: darkPoolFundData,
                        yAxisIndex: 1, // Use the volume y-axis
                        smooth: true,
                        lineStyle: {
                            width: 2,
                            color: '#d4af37'
                        },
                        itemStyle: {
                            color: '#d4af37'
                        },
                        showSymbol: false,
                    }
                ]
            });
        } catch (error) {
            console.error(`[ChartManager] Failed to add Dark Pool Fund indicator:`, error);
        }
    }

    async addHotMoney(domId, stockCode) {
        const chart = this.getChart(domId);
        if (!chart) return;

        try {
            const response = await fetch(`/api/analysis/hot-money/${stockCode}`);
            if (!response.ok) throw new Error('Failed to fetch Hot Money data');
            const data = await response.json();
            
            // Assuming data is [{date: '...', '游资净买': ...}, ...]
            const seriesData = data.map(item => [item.date, item['游资净买']]);

            const option = chart.getOption();
            const newSeries = option.series.filter(s => s.id !== 'hot_money_line');
            
            chart.setOption({
                legend: {
                    ...option.legend,
                    data: [...(option.legend.data || []), '游资暗盘']
                },
                series: [
                    ...newSeries,
                    {
                        id: 'hot_money_line',
                        name: '游资暗盘',
                        type: 'line',
                        data: seriesData,
                        yAxisIndex: 1,
                        smooth: true,
                        lineStyle: { width: 2, color: '#f43f5e' }, // Rose-500
                        itemStyle: { color: '#f43f5e' },
                        showSymbol: false
                    }
                ]
            });
        } catch (error) {
            console.error(`[ChartManager] Failed to add Hot Money indicator:`, error);
        }
    }

    async addThreeColor(domId, stockCode) {
        const chart = this.getChart(domId);
        if (!chart) return;

        try {
            const response = await fetch(`/api/analysis/three-color-resonance/${stockCode}`);
            if (!response.ok) throw new Error('Failed to fetch Three Color data');
            const data = await response.json();
            
            // Map to series
            const mainFlow = data.map(i => [i.date, i.main_force_money]);
            const hotFlow = data.map(i => [i.date, i.hot_money]);
            
            const option = chart.getOption();
            const newSeries = option.series.filter(s => !s.id || !s.id.startsWith('three_color'));

            chart.setOption({
                legend: {
                    ...option.legend,
                    data: [...(option.legend.data || []), '主力资金', '游资资金']
                },
                series: [
                    ...newSeries,
                    {
                        id: 'three_color_main',
                        name: '主力资金',
                        type: 'line',
                        data: mainFlow,
                        yAxisIndex: 1,
                        smooth: true,
                        lineStyle: { width: 2, color: '#ef4444' },
                        showSymbol: false
                    },
                    {
                        id: 'three_color_hot',
                        name: '游资资金',
                        type: 'line',
                        data: hotFlow,
                        yAxisIndex: 1,
                        smooth: true,
                        lineStyle: { width: 2, color: '#eab308' },
                        showSymbol: false
                    }
                ]
            });
        } catch (error) {
            console.error(`[ChartManager] Failed to add Three Color indicator:`, error);
        }
    }

    async addKangLong(domId, stockCode) {
        const chart = this.getChart(domId);
        if (!chart) return;

        try {
            const response = await fetch(`/api/analysis/kang-long-you-hui/${stockCode}`);
            if (!response.ok) throw new Error('Failed to fetch Kang Long data');
            const data = await response.json();
            
            // Using 'line_zhuli' (Main Force Line) as the primary line
            const lineData = data.map(i => [i.date, i.line_zhuli]);
            
            const option = chart.getOption();
            const newSeries = option.series.filter(s => s.id !== 'kang_long_line');

            chart.setOption({
                legend: {
                    ...option.legend,
                    data: [...(option.legend.data || []), '亢龙有悔']
                },
                series: [
                    ...newSeries,
                    {
                        id: 'kang_long_line',
                        name: '亢龙有悔',
                        type: 'line',
                        data: lineData,
                        yAxisIndex: 1,
                        smooth: true,
                        lineStyle: { width: 2, color: '#8b5cf6' }, // Violet
                        showSymbol: false
                    }
                ]
            });
        } catch (error) {
            console.error(`[ChartManager] Failed to add Kang Long indicator:`, error);
        }
    }

    removeIndicator(domId, indicatorName) {
        const chart = this.getChart(domId);
        if (!chart) return;

        const option = chart.getOption();
        
        // Map indicator names to series IDs or Names
        const prefixes = {
            'dark-pool': ['暗盘资金'],
            'precise-trading': ['precise_trading_buy', 'precise_trading_sell'],
            'hot-money': ['hot_money_line'],
            'three-color': ['three_color_main', 'three_color_hot', 'three_color_retail'],
            'kang-long': ['kang_long_line', 'kang_long_signal']
        };

        const targetIds = prefixes[indicatorName] || [indicatorName];
        
        // Filter out series
        const newSeries = option.series.filter(s => {
            // Check ID match
            if (s.id && targetIds.some(tid => s.id.startsWith(tid))) return false;
            // Check Name match (legacy)
            if (s.name === '暗盘资金' && indicatorName === 'dark-pool') return false;
            if ((s.name === '买入信号' || s.name === '卖出信号') && indicatorName === 'precise-trading') return false;
            return true;
        });

        // Update Legend
        // We can just set the series, ECharts handles legend removal if the series is gone? 
        // No, we should clean up legend data too to be safe.
        // Actually, let's just update series. ECharts might keep legend items if we don't remove them.
        
        // Simplification: just reset series. 
        chart.setOption({ series: newSeries }, { replaceMerge: ['series'] });
        console.log(`[ChartManager] Indicator '${indicatorName}' removed`);
    }

    async addPreciseTrading(domId, stockCode) {
        const chart = this.getChart(domId);
        if (!chart) {
            console.warn(`[ChartManager] Chart ${domId} not found.`);
            return;
        }

        try {
            const response = await fetch(`/api/analysis/precise-trading/${stockCode}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch Precise Trading data: ${response.statusText}`);
            }
            const data = await response.json();

            const buySignals = data.map(item => {
                if (item.buy_signal !== null) {
                    return {
                        name: '买入点',
                        coord: [item.date, item.low * 0.98], // Place below the low
                        value: 'B',
                        symbol: 'arrow',
                        symbolSize: 15,
                        itemStyle: {
                            color: '#ef4444' // red
                        }
                    };
                }
                return null;
            }).filter(p => p);

            const sellSignals = data.map(item => {
                if (item.sell_signal !== null) {
                    return {
                        name: '卖出点',
                        coord: [item.date, item.high * 1.02], // Place above the high
                        value: 'S',
                        symbol: 'arrow',
                        symbolRotate: 180,
                        symbolSize: 15,
                        itemStyle: {
                            color: '#22c55e' // green
                        }
                    };
                }
                return null;
            }).filter(p => p);

            const option = chart.getOption();
            const existingSeries = option.series;
            // Filter out old precise trading series before adding new ones
            const newSeries = existingSeries.filter(s => !s.id || !s.id.startsWith('precise_trading'));

            const currentLegend = (Array.isArray(option.legend) ? option.legend[0] : option.legend) || {};
            const currentLegendData = currentLegend.data || [];
            const newLegendData = [...new Set([...currentLegendData, '买入信号', '卖出信号'])];

            chart.setOption({
                legend: {
                    ...currentLegend,
                    data: newLegendData,
                },
                series: [
                    ...newSeries,
                    {
                        id: 'precise_trading_buy',
                        name: '买入信号',
                        type: 'scatter', // Using scatter to host markPoints
                        yAxisIndex: 0,
                        data: [], // No data points, just markers
                        markPoint: {
                            data: buySignals
                        }
                    },
                    {
                        id: 'precise_trading_sell',
                        name: '卖出信号',
                        type: 'scatter',
                        yAxisIndex: 0,
                        data: [],
                        markPoint: {
                            data: sellSignals
                        }
                    }
                ]
            });
            console.log(`[ChartManager] Precise Trading signals added to chart ${domId}.`);

        } catch (error) {
            console.error(`[ChartManager] Failed to add Precise Trading indicator:`, error);
        }
    }

    createKLineChart(domId, data) {
        const chart = this.initChart(domId);
        if (!chart) return null;

        const { dates, values, volumes, signals = [] } = data;
        
        const option = {
            backgroundColor: 'transparent',
            animation: false, // 禁用动画提升性能
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross' },
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                borderColor: '#334155',
                textStyle: { color: '#e2e8f0' },
                formatter: function(params) {
                    if (!params || params.length === 0) return '';
                    
                    const data = params[0];
                    const klineData = values[data.dataIndex];
                    if (!klineData) return '';
                    
                    return `
                        <div style="font-size: 12px;">
                            <div>日期: ${dates[data.dataIndex]}</div>
                            <div>开盘: <span style="font-family:monospace;font-weight:bold;">${FormatUtils.formatPrice(klineData[0])}</span></div>
                            <div>收盘: <span style="font-family:monospace;font-weight:bold;">${FormatUtils.formatPrice(klineData[1])}</span></div>
                            <div>最低: <span style="font-family:monospace;font-weight:bold;">${FormatUtils.formatPrice(klineData[2])}</span></div>
                            <div>最高: <span style="font-family:monospace;font-weight:bold;">${FormatUtils.formatPrice(klineData[3])}</span></div>
                            <div>成交量: ${FormatUtils.formatBigNumber(volumes[data.dataIndex][1])}</div>
                        </div>
                    `;
                }
            },
            grid: [
                { left: '3%', right: '3%', height: '60%' },
                { left: '3%', right: '3%', top: '75%', height: '15%' }
            ],
            xAxis: [
                { 
                    type: 'category', 
                    data: dates, 
                    scale: true, 
                    boundaryGap: false,
                    axisLine: { onZero: false },
                    splitLine: { show: false },
                    axisLabel: { 
                        color: '#94a3b8',
                        fontSize: 10
                    }
                },
                { 
                    type: 'category', 
                    gridIndex: 1, 
                    data: dates, 
                    axisLabel: { show: false } 
                }
            ],
            yAxis: [
                { 
                    scale: true, 
                    splitLine: { lineStyle: { color: '#334155' } },
                    axisLabel: { 
                        color: '#94a3b8',
                        fontSize: 10
                    }
                },
                { 
                    scale: true, 
                    gridIndex: 1, 
                    splitNumber: 2, 
                    axisLabel: { show: false }, 
                    axisLine: { show: false }, 
                    splitLine: { show: false } 
                }
            ],
            dataZoom: [
                { 
                    type: 'inside', 
                    xAxisIndex: [0, 1], 
                    start: 50, 
                    end: 100,
                    filterMode: 'filter'
                },
                { 
                    show: true, 
                    xAxisIndex: [0, 1], 
                    type: 'slider', 
                    bottom: 0, 
                    start: 50, 
                    end: 100, 
                    borderColor: '#334155',
                    textStyle: { color: '#94a3b8' }
                }
            ],
            series: [
                {
                    name: 'KLine',
                    type: 'candlestick',
                    data: values,
                    itemStyle: {
                        color: '#ef4444',
                        color0: '#22c55e',
                        borderColor: '#ef4444',
                        borderColor0: '#22c55e'
                    },
                    markPoint: {
                        data: this.formatSignals(signals, values, dates)
                    }
                },
                {
                    name: 'MA5',
                    type: 'line',
                    data: this.calculateMA(5, values),
                    smooth: true,
                    lineStyle: { 
                        opacity: 0.7,
                        width: 1,
                        color: '#f59e0b'
                    },
                    symbol: 'none',
                    showSymbol: false
                },
                {
                    name: 'MA20',
                    type: 'line',
                    data: this.calculateMA(20, values),
                    smooth: true,
                    lineStyle: { 
                        opacity: 0.7,
                        width: 1,
                        color: '#8b5cf6'
                    },
                    symbol: 'none',
                    showSymbol: false
                },
                {
                    name: 'Volume',
                    type: 'bar',
                    xAxisIndex: 1,
                    yAxisIndex: 1,
                    data: volumes.map(v => ({
                        value: v[1],
                        itemStyle: { 
                            color: v[2] > 0 ? '#ef444480' : '#22c55e80' 
                        }
                    }))
                }
            ]
        };

        chart.setOption(option);
        return chart;
    }

    /**
     * 创建简单折线图
     * @param {string} domId - DOM元素ID
     * @param {Array} data - 数据数组
     * @param {string} name - 系列名称
     * @param {string} color - 线条颜色
     * @returns {Object|null} ECharts实例
     */
    createLineChart(domId, data, name = '数据', color = '#38bdf8') {
        const chart = this.initChart(domId);
        if (!chart) return null;

        const xData = data.map(item => item.time || item.date);
        const yData = data.map(item => item.value);

        const option = {
            backgroundColor: 'transparent',
            animation: false,
            tooltip: { 
                trigger: 'axis',
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                borderColor: '#334155',
                textStyle: { color: '#e2e8f0' }
            },
            grid: { 
                top: 20, 
                bottom: 30, 
                left: 50, 
                right: 20,
                containLabel: true
            },
            xAxis: { 
                type: 'category', 
                data: xData, 
                axisLabel: { 
                    fontSize: 10,
                    color: '#94a3b8'
                },
                axisLine: {
                    lineStyle: { color: '#334155' }
                }
            },
            yAxis: { 
                type: 'value', 
                splitLine: { 
                    lineStyle: { color: '#334155' } 
                },
                axisLabel: { 
                    fontSize: 10,
                    color: '#94a3b8'
                }
            },
            series: [{
                name: name,
                type: 'line',
                data: yData,
                showSymbol: false,
                itemStyle: { color: color },
                lineStyle: { 
                    width: 2,
                    color: color
                },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: color + '40' },
                        { offset: 1, color: color + '10' }
                    ])
                }
            }]
        };

        chart.setOption(option);
        return chart;
    }

    /**
     * 计算移动平均线
     * @param {number} dayCount - 天数
     * @param {Array} data - OHLC数据
     * @returns {Array} MA数据
     */
    calculateMA(dayCount, data) {
        const result = [];
        for (let i = 0; i < data.length; i++) {
            if (i < dayCount) {
                result.push('-');
                continue;
            }
            let sum = 0;
            for (let j = 0; j < dayCount; j++) {
                sum += +data[i - j][1]; // 使用收盘价
            }
            result.push(+(sum / dayCount).toFixed(2));
        }
        return result;
    }

    /**
     * 格式化交易信号
     * @param {Array} signals - 信号数组
     * @param {Array} values - OHLC数据
     * @param {Array} dates - 日期数组
     * @returns {Array} 格式化的信号数据
     */
    formatSignals(signals, values, dates) {
        return signals.map(s => {
            const isBuy = s.type === 'buy';
            const isStop = s.type === 'stop';
            
            let color = '#333';
            let label = '';
            let priceVal = values[s.index] ? values[s.index][1] : 0;
            
            if (isBuy) {
                color = '#ef4444';
                label = '↑';
                priceVal = values[s.index] ? values[s.index][2] : 0; // 最低价
            } else if (isStop) {
                color = '#22c55e';
                label = '↓';
                priceVal = values[s.index] ? values[s.index][3] : 0; // 最高价
            }
            
            return {
                name: s.label || (isBuy ? '买入' : '卖出'),
                coord: [dates[s.index], priceVal],
                value: label,
                symbol: 'circle',
                symbolSize: 20,
                symbolOffset: isBuy ? [0, '50%'] : [0, '-50%'],
                label: {
                    show: true,
                    color: color,
                    fontSize: 14,
                    fontWeight: 'bold',
                    formatter: label
                },
                itemStyle: {
                    color: '#ffffff',
                    borderColor: color,
                    borderWidth: 2,
                    shadowBlur: 3,
                    shadowColor: 'rgba(0,0,0,0.5)'
                },
                z: 100
            };
        });
    }

    /**
     * 创建宏观数据图表
     * @param {string} domId - DOM元素ID
     * @param {Object} data - 宏观数据
     * @returns {Object|null} ECharts实例
     */
    createMacroChart(domId, data) {
        const chart = this.initChart(domId);
        if (!chart) return null;

        const moneySupply = data.money_supply || [];
        const pmiData = data.pmi || [];
        const cpi = data.cpi || { value: 0, date: '' };
        const ppi = data.ppi || { value: 0 };
        const gdp = data.gdp || { value: 0, quarter: '' };

        // 准备M1-M2剪刀差数据
        const dates = moneySupply.map(item => item.date);
        const m1Data = moneySupply.map(item => item.m1_yoy);
        const m2Data = moneySupply.map(item => item.m2_yoy);
        const scissorsData = moneySupply.map(item => item.scissors);

        // 准备PMI数据
        const pmiDates = pmiData.map(item => item.date);
        const manufacturingData = pmiData.map(item => item.manufacturing);
        const nonManufacturingData = pmiData.map(item => item.non_manufacturing);

        const option = {
            backgroundColor: 'transparent',
            animation: false,
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                borderColor: '#334155',
                textStyle: { color: '#e2e8f0' }
            },
            legend: {
                data: ['M1同比', 'M2同比', '剪刀差', '制造业PMI', '非制造业PMI'],
                textStyle: { color: '#94a3b8' },
                top: 10
            },
            grid: [
                { left: '3%', right: '3%', height: '40%', top: 50 },
                { left: '3%', right: '3%', height: '30%', top: '60%' }
            ],
            xAxis: [
                {
                    type: 'category',
                    data: dates,
                    axisLabel: {
                        color: '#94a3b8',
                        fontSize: 10,
                        rotate: 45
                    },
                    axisLine: { lineStyle: { color: '#334155' } }
                },
                {
                    type: 'category',
                    data: pmiDates,
                    gridIndex: 1,
                    axisLabel: {
                        color: '#94a3b8',
                        fontSize: 10,
                        rotate: 45
                    },
                    axisLine: { lineStyle: { color: '#334155' } }
                }
            ],
            yAxis: [
                {
                    type: 'value',
                    name: '同比增长(%)',
                    nameTextStyle: { color: '#94a3b8' },
                    splitLine: { lineStyle: { color: '#334155' } },
                    axisLabel: { color: '#94a3b8' }
                },
                {
                    type: 'value',
                    name: 'PMI指数',
                    nameTextStyle: { color: '#94a3b8' },
                    gridIndex: 1,
                    splitLine: { lineStyle: { color: '#334155' } },
                    axisLabel: { color: '#94a3b8' },
                    min: 40,
                    max: 60
                }
            ],
            series: [
                // M1-M2剪刀差相关
                {
                    name: 'M1同比',
                    type: 'line',
                    data: m1Data,
                    smooth: true,
                    lineStyle: { width: 2, color: '#38bdf8' },
                    itemStyle: { color: '#38bdf8' },
                    showSymbol: false
                },
                {
                    name: 'M2同比',
                    type: 'line',
                    data: m2Data,
                    smooth: true,
                    lineStyle: { width: 2, color: '#8b5cf6' },
                    itemStyle: { color: '#8b5cf6' },
                    showSymbol: false
                },
                {
                    name: '剪刀差',
                    type: 'line',
                    data: scissorsData,
                    smooth: true,
                    lineStyle: { width: 2, color: '#f43f5e' },
                    itemStyle: { color: '#f43f5e' },
                    showSymbol: false,
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: '#f43f5e40' },
                            { offset: 1, color: '#f43f5e10' }
                        ])
                    }
                },
                // PMI相关
                {
                    name: '制造业PMI',
                    type: 'line',
                    xAxisIndex: 1,
                    yAxisIndex: 1,
                    data: manufacturingData,
                    smooth: true,
                    lineStyle: { width: 2, color: '#22c55e' },
                    itemStyle: { color: '#22c55e' },
                    showSymbol: false
                },
                {
                    name: '非制造业PMI',
                    type: 'line',
                    xAxisIndex: 1,
                    yAxisIndex: 1,
                    data: nonManufacturingData,
                    smooth: true,
                    lineStyle: { width: 2, color: '#f59e0b' },
                    itemStyle: { color: '#f59e0b' },
                    showSymbol: false
                }
            ]
        };

        chart.setOption(option);
        return chart;
    }

    /**
     * 创建CPI/PPI对比图表
     * @param {string} domId - DOM元素ID
     * @param {Object} data - CPI/PPI数据
     * @returns {Object|null} ECharts实例
     */
    createCpiPpiChart(domId, data) {
        const chart = this.initChart(domId);
        if (!chart) return null;

        const cpi = data.cpi || { value: 0, date: '' };
        const ppi = data.ppi || { value: 0 };

        const option = {
            backgroundColor: 'transparent',
            animation: false,
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                borderColor: '#334155',
                textStyle: { color: '#e2e8f0' }
            },
            grid: {
                top: 30,
                bottom: 30,
                left: 50,
                right: 20,
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: ['CPI', 'PPI'],
                axisLabel: {
                    color: '#94a3b8',
                    fontSize: 12
                },
                axisLine: { lineStyle: { color: '#334155' } }
            },
            yAxis: {
                type: 'value',
                name: '同比增长(%)',
                nameTextStyle: { color: '#94a3b8' },
                splitLine: { lineStyle: { color: '#334155' } },
                axisLabel: { color: '#94a3b8' }
            },
            series: [{
                name: '通胀指标',
                type: 'bar',
                data: [
                    {
                        value: cpi.value,
                        itemStyle: {
                            color: cpi.value >= 0 ? '#f59e0b' : '#38bdf8'
                        }
                    },
                    {
                        value: ppi.value,
                        itemStyle: {
                            color: ppi.value >= 0 ? '#f59e0b' : '#38bdf8'
                        }
                    }
                ],
                label: {
                    show: true,
                    position: 'top',
                    color: '#e2e8f0',
                    formatter: '{c}%'
                },
                barWidth: '40%'
            }]
        };

        chart.setOption(option);
        return chart;
    }

    /**
     * 创建GDP图表
     * @param {string} domId - DOM元素ID
     * @param {Object} data - GDP数据
     * @returns {Object|null} ECharts实例
     */
    createGdpChart(domId, data) {
        const chart = this.initChart(domId);
        if (!chart) return null;

        const gdp = data.gdp || { value: 0, quarter: '' };

        const option = {
            backgroundColor: 'transparent',
            animation: false,
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                borderColor: '#334155',
                textStyle: { color: '#e2e8f0' }
            },
            series: [{
                name: 'GDP增长',
                type: 'gauge',
                startAngle: 180,
                endAngle: 0,
                min: 0,
                max: 10,
                splitNumber: 10,
                axisLine: {
                    lineStyle: {
                        width: 20,
                        color: [
                            [0.3, '#38bdf8'],
                            [0.7, '#f59e0b'],
                            [1, '#ef4444']
                        ]
                    }
                },
                pointer: {
                    icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                    length: '12%',
                    width: 20,
                    offsetCenter: [0, '-60%'],
                    itemStyle: {
                        color: 'auto'
                    }
                },
                axisTick: {
                    length: 12,
                    lineStyle: {
                        color: 'auto',
                        width: 2
                    }
                },
                splitLine: {
                    length: 20,
                    lineStyle: {
                        color: 'auto',
                        width: 5
                    }
                },
                axisLabel: {
                    color: '#94a3b8',
                    fontSize: 12,
                    distance: -60,
                    formatter: function (value) {
                        return value + '%';
                    }
                },
                title: {
                    offsetCenter: [0, '-10%'],
                    fontSize: 14,
                    color: '#e2e8f0'
                },
                detail: {
                    fontSize: 30,
                    offsetCenter: [0, '-35%'],
                    valueAnimation: true,
                    formatter: function (value) {
                        return value.toFixed(1) + '%';
                    },
                    color: 'auto'
                },
                data: [{
                    value: gdp.value,
                    name: 'GDP同比增长'
                }]
            }]
        };

        chart.setOption(option);
        return chart;
    }

    /**
     * 获取图表统计信息
     * @returns {Object} 统计信息
     */
    getStats() {
        return {
            totalCharts: this.charts.size,
            activeTimers: this.resizeTimers.size,
            charts: Array.from(this.charts.keys())
        };
    }

    /**
     * 清理过期图表（超过指定时间未更新）
     * @param {number} maxAge - 最大存活时间（毫秒）
     */
    cleanupStaleCharts(maxAge = 5 * 60 * 1000) { // 默认5分钟
        const now = Date.now();
        const staleCharts = [];
        
        this.charts.forEach((chartData, domId) => {
            if (now - chartData.lastUpdate > maxAge) {
                staleCharts.push(domId);
            }
        });
        
        staleCharts.forEach(domId => {
            console.log(`[ChartManager] Cleaning up stale chart: ${domId}`);
            this.disposeChart(domId);
        });
        
        return staleCharts.length;
    }
}

// 导出单例实例
window.chartManager = new ChartManager();

// 页面卸载时清理资源
window.addEventListener('beforeunload', () => {
    if (window.chartManager) {
        window.chartManager.disposeAll();
    }
});

// 定期清理过期图表
setInterval(() => {
    if (window.chartManager) {
        window.chartManager.cleanupStaleCharts();
    }
}, 5 * 60 * 1000); // 每5分钟清理一次

console.log('[ChartManager] Initialized successfully');
