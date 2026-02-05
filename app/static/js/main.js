
import { AIChatWidget } from './modules/chat.js';
import { initProfessionalSystem, switchProfessionalTab, fetchProfessionalSignal, calculateRisk } from './modules/professional-system.js';
import { AIAgentController } from './modules/ai_agent.js';
import { loadWatchlist, removeStockFromWatchlist } from './watchlist.js';

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("ROX 3.0 Core Initialized");
    
    try {
        if (typeof AIChatWidget === 'function') {
            window.roxChat = new AIChatWidget();
        }
    } catch (e) { console.error("Chat Widget Init Failed", e); }

    try {
        if (typeof AIAgentController === 'function') {
            window.aiAgent = new AIAgentController();
        }
    } catch (e) { console.error("AI Agent Init Failed", e); }
    
    window.switchProfessionalTab = switchProfessionalTab;
    window.fetchProfessionalSignal = fetchProfessionalSignal;
    window.calculateRisk = calculateRisk;
    window.initProfessionalSystem = initProfessionalSystem;
    window.loadWatchlist = loadWatchlist;
    window.removeStockFromWatchlist = removeStockFromWatchlist;
    
    initMainLogic();
});

// --- Window Management ---
window.currentStockCode = '600519';

function selectStock(code, name) {
    window.currentStockCode = code;
    const n = document.getElementById('stock-name-header');
    const c = document.getElementById('stock-code-header');
    if (n) n.textContent = name || code;
    if (c) c.textContent = code;
    
    // Sync price immediately
    updateStockHeader(code);
    
    fetchKLineData('daily');
    updateIndicatorChart(code);
}
window.selectStock = selectStock;

async function updateStockHeader(code) {
    try {
        const resp = await fetch('/api/market/fetch-realtime', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stock_name: code })
        });
        const data = await resp.json();
        if (data.error) return;
        
        const priceEl = document.querySelector('.price-font.text-2xl');
        if (priceEl) {
            priceEl.textContent = data.p_now.toFixed(2);
            // Update color based on change
            const change = data.p_change || 0;
            priceEl.className = `text-2xl font-mono font-bold price-font ${change >= 0 ? 'text-up' : 'text-down'}`;
        }
        
        // Update change and percent
        const changeEls = document.querySelectorAll('#stock-name-header ~ div .text-up, #stock-name-header ~ div .text-down');
        if (changeEls.length >= 2) {
             const changeVal = data.p_change || 0;
             const pctVal = data.p_pct || 0;
             const cls = changeVal >= 0 ? 'text-up' : 'text-down';
             
             changeEls[0].textContent = (changeVal > 0 ? '+' : '') + changeVal.toFixed(2);
             changeEls[0].className = cls;
             
             changeEls[1].textContent = (pctVal > 0 ? '+' : '') + pctVal.toFixed(2) + '%';
             changeEls[1].className = cls;
        }
        
    } catch (e) {
        console.error("Failed to sync header price", e);
    }
}

let chartMode = 'kline';
let fenshiChart = null;

async function fetchAndRenderFenshi() {
    const wrap = document.getElementById('fenshi-placeholder');
    const chartEl = document.getElementById('fenshi-chart');
    const loadingEl = document.getElementById('fenshi-loading');
    const code = window.currentStockCode || '600519';
    if (!chartEl || !wrap) return;
        if (loadingEl) { loadingEl.classList.remove('hidden'); loadingEl.textContent = '分时图加载中…'; }
    try {
        const r = await fetch(`/api/market/fenshi?code=${encodeURIComponent(code)}`);
        const d = await r.json().catch(() => ({}));
        if (!r.ok || d.error) {
            if (loadingEl) loadingEl.textContent = d.error || '分时图暂无数据';
            return;
        }
        const times = d.times || [];
        const prices = d.prices || [];
        const volumes = d.volumes || [];
        const ma5 = d.ma5 || [];
        const ma10 = d.ma10 || [];
        if (loadingEl) loadingEl.classList.add('hidden');
        if (!fenshiChart && typeof echarts !== 'undefined') fenshiChart = echarts.init(chartEl);
        if (!fenshiChart) return;
        const series = [
            { type: 'line', data: prices, smooth: true, symbol: 'none', lineStyle: { color: '#38bdf8', width: 2 }, xAxisIndex: 0, yAxisIndex: 0 }
        ];
        if (ma5.length) series.push({ type: 'line', data: ma5, smooth: true, symbol: 'none', lineStyle: { color: '#eab308', width: 1 }, xAxisIndex: 0, yAxisIndex: 0 });
        if (ma10.length) series.push({ type: 'line', data: ma10, smooth: true, symbol: 'none', lineStyle: { color: '#a855f7', width: 1 }, xAxisIndex: 0, yAxisIndex: 0 });
        series.push({ type: 'bar', data: volumes, itemStyle: { color: (p) => (prices[p.dataIndex] >= (prices[p.dataIndex - 1] || prices[p.dataIndex]) ? '#ff333a' : '#00aa3b') }, xAxisIndex: 1, yAxisIndex: 1 });
        fenshiChart.setOption({
            backgroundColor: 'transparent',
            grid: [{ left: 50, right: 30, top: 20, bottom: 60 }, { left: 50, right: 30, top: '75%', height: '18%' }],
            xAxis: [
                { type: 'category', data: times, gridIndex: 0, axisLabel: { color: '#94a3b8', fontSize: 10 } },
                { type: 'category', data: times, gridIndex: 1, axisLabel: { show: false } }
            ],
            yAxis: [
                { type: 'value', gridIndex: 0, scale: true, splitLine: { lineStyle: { color: '#334155' } }, axisLabel: { color: '#94a3b8', fontSize: 10 } },
                { type: 'value', gridIndex: 1, scale: true, axisLabel: { show: false }, splitLine: { show: false } }
            ],
            series
        });
        fenshiChart.resize();
    } catch (e) {
        if (loadingEl) { loadingEl.classList.remove('hidden'); loadingEl.textContent = '分时图加载失败'; }
    }
}

function handleFKey(key) {
    const klineEl = document.getElementById('kline-chart-container');
    const fenshiEl = document.getElementById('fenshi-placeholder');
    const f10Modal = document.getElementById('f10-modal');
    if (key === 'F1') {
        chartMode = 'fenshi';
        if (klineEl) klineEl.classList.add('hidden');
        if (fenshiEl) { fenshiEl.classList.remove('hidden'); fenshiEl.classList.add('flex'); }
        fetchAndRenderFenshi();
    } else if (key === 'F2') {
        chartMode = 'kline';
        if (klineEl) klineEl.classList.remove('hidden');
        if (fenshiEl) { fenshiEl.classList.add('hidden'); fenshiEl.classList.remove('flex'); }
        fetchKLineData('daily');
        if (typeof klineChart !== 'undefined' && klineChart) klineChart.resize();
    } else if (key === 'F10') {
        if (f10Modal) {
            f10Modal.classList.remove('hidden');
            const codeEl = document.getElementById('f10-code');
            const nameEl = document.getElementById('f10-name');
            if (codeEl) codeEl.textContent = window.currentStockCode || '—';
            if (nameEl) nameEl.textContent = document.getElementById('stock-name-header')?.textContent || '—';
            loadF10ValueLaw();
            return;
        }
        const modal = document.createElement('div');
        modal.id = 'f10-modal';
        modal.className = 'fixed top-[12%] left-[20%] w-[700px] max-h-[75vh] bg-[#0c0c0c] border border-gray-700 shadow-2xl z-50 flex flex-col rounded-lg overflow-hidden';
        modal.innerHTML = `
            <div class="h-9 bg-gray-800 flex items-center justify-between px-3 cursor-move" id="f10-header">
                <span class="text-yellow-500 font-bold text-sm">个股资料 / AI F10</span>
                <button type="button" onclick="document.getElementById('f10-modal').classList.add('hidden')" class="text-gray-400 hover:text-white"><i class="fas fa-times"></i></button>
            </div>
            <div class="p-4 flex-1 overflow-y-auto">
                <div class="flex gap-4 mb-4">
                    <span id="f10-name" class="font-bold text-white">—</span>
                    <span id="f10-code" class="text-gray-400 font-mono">—</span>
                </div>
                <div id="f10-content" class="text-gray-400 text-sm min-h-[200px]">加载中…</div>
                <button type="button" id="f10-ai-btn" class="mt-4 px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white text-xs rounded">AI 解读</button>
            </div>
        `;
        document.body.appendChild(modal);
        const codeEl = document.getElementById('f10-code');
        const nameEl = document.getElementById('f10-name');
        if (codeEl) codeEl.textContent = window.currentStockCode || '—';
        if (nameEl) nameEl.textContent = document.getElementById('stock-name-header')?.textContent || '—';
        const contentEl = document.getElementById('f10-content');
        const aiBtn = document.getElementById('f10-ai-btn');
        if (contentEl) contentEl.textContent = '加载中…';
        if (aiBtn) aiBtn.onclick = () => {
            if (typeof window.toggleAIAgent === 'function') window.toggleAIAgent();
            if (typeof window.aiAgent !== 'undefined' && window.aiAgent?.analyzeStock) window.aiAgent.analyzeStock(window.currentStockCode);
            document.getElementById('f10-modal').classList.add('hidden');
        };
        if (typeof setupDraggable === 'function') setupDraggable(modal, 'f10-header');
        loadF10ValueLaw();
    }
}
window.handleFKey = handleFKey;

async function loadF10ValueLaw() {
    const code = window.currentStockCode || '600519';
    const contentEl = document.getElementById('f10-content');
    if (!contentEl) return;
    contentEl.textContent = '加载中…';
    try {
        const resp = await fetch(`/api/stock/value-law/${encodeURIComponent(code)}`);
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || data.error || data.detail) {
            contentEl.textContent = data.detail || data.error || '暂无法获取价值规律数据。';
            return;
        }
        const price = data.market_price != null ? data.market_price.toFixed(2) : '—';
        const iv = data.intrinsic_value != null ? data.intrinsic_value.toFixed(2) : '—';
        const devPct = data.deviation != null ? (data.deviation * 100).toFixed(1) + '%' : '—';
        const surplus = data.surplus_value_score != null ? data.surplus_value_score : '—';
        const industry = data.industry || (data.fundamentals && data.fundamentals.industry) || '—';
        const signal = data.signal || 'unknown';
        const comment = data.comment || '';
        contentEl.innerHTML = `
            <div class="space-y-2 text-xs leading-relaxed">
                <div class="flex flex-wrap gap-4">
                    <div>所属行业：<span class="text-yellow-500">${industry}</span></div>
                    <div>现价：<span class="text-sky-400 font-mono">${price}</span></div>
                    <div>内在价值估算：<span class="text-sky-400 font-mono">${iv}</span></div>
                    <div>价格偏离度：<span class="${data.deviation != null && data.deviation < 0 ? 'text-up' : 'text-down'} font-mono">${devPct}</span></div>
                </div>
                <div>剩余价值创造能力评分：<span class="text-emerald-400 font-mono">${surplus}</span> / 100</div>
                <div>信号：<span class="font-mono ${
                    signal === 'strong_buy' ? 'text-up' :
                    signal === 'buy' ? 'text-up' :
                    signal === 'sell' || signal === 'strong_sell' ? 'text-down' : 'text-gray-300'
                }">${signal}</span></div>
                <div class="mt-2 text-gray-300">${comment}</div>
            </div>
        `;
    } catch (e) {
        contentEl.textContent = '加载价值规律数据时出错。';
    }
}

// Open Professional Window (Deep Analysis)
window.openProfessionalWindow = function() {
    // Check if the professional module is loaded
    if (typeof import('./modules/professional-system.js') !== 'undefined') {
        // If it's a module, we might need to access it differently or it attaches to window
        // Assuming professional-system.js exports initProfessionalSystem but also maybe we can just load the UI
    }
    
    // For now, we'll check if the modal exists, if not create it or show it
    let modal = document.getElementById('professional-modal');
    if (!modal) {
        // Create modal structure if missing (it might be in a separate template, but let's ensure it exists)
        // Actually, professional-system.js likely handles this. 
        // Let's try to import and init it dynamically if needed.
        import('/static/js/modules/professional-system.js').then(module => {
             if (module && module.initProfessionalSystem) {
                 module.initProfessionalSystem();
                 // Show the modal
                 const m = document.getElementById('professional-modal');
                 if(m) m.classList.remove('hidden');
                 else alert("深度分析模块初始化失败：界面未找到");
             }
        }).catch(e => {
            console.error("Failed to load professional system:", e);
            alert("无法加载深度分析模块");
        });
    } else {
        modal.classList.remove('hidden');
        // Trigger analysis for current stock
        if (window.currentStockCode) {
             const input = document.getElementById('prof-signal-symbol');
             if(input) {
                 input.value = window.currentStockCode;
                 // Optionally auto-click analyze
                 // if(typeof fetchProfessionalSignal === 'function') fetchProfessionalSignal(); 
                 // But fetchProfessionalSignal is likely not global.
                 // We need to rely on the module's event listeners.
             }
        }
    }
};

window.runAIBacktest = function() {
    // Check if window exists
    if(document.getElementById('ai-backtest-modal')) {
        document.getElementById('ai-backtest-modal').classList.remove('hidden');
        return;
    }
    
    // Create Modal (Placeholder for Qbot UI)
    const modal = document.createElement('div');
    modal.id = 'ai-backtest-modal';
    modal.className = 'fixed top-[15%] left-[15%] w-[800px] h-[600px] bg-[#0c0c0c] border border-gray-700 shadow-2xl z-50 flex flex-col rounded-lg overflow-hidden';
    const defaultEnd = new Date();
    const defaultStart = new Date(defaultEnd);
    defaultStart.setFullYear(defaultStart.getFullYear() - 1);
    modal.innerHTML = `
        <div class="h-8 bg-gray-800 flex items-center justify-between px-2 cursor-move" id="ai-backtest-header">
            <span class="text-yellow-500 font-bold text-xs">AI 策略回测 (Qbot)</span>
            <button onclick="document.getElementById('ai-backtest-modal').classList.add('hidden')" class="text-gray-400 hover:text-white"><i class="fas fa-times"></i></button>
        </div>
        <div class="flex-1 bg-black p-4 text-gray-300 font-mono text-sm overflow-y-auto">
            <div class="grid grid-cols-3 gap-4 mb-4">
                <div>
                    <label class="block text-xs text-gray-500 mb-1">开始日期</label>
                    <input id="ai-backtest-start" type="date" class="w-full bg-[#111] border border-gray-700 rounded px-2 py-1 text-xs text-white" value="${defaultStart.toISOString().slice(0,10)}">
                </div>
                <div>
                    <label class="block text-xs text-gray-500 mb-1">结束日期</label>
                    <input id="ai-backtest-end" type="date" class="w-full bg-[#111] border border-gray-700 rounded px-2 py-1 text-xs text-white" value="${defaultEnd.toISOString().slice(0,10)}">
                </div>
                <div>
                    <label class="block text-xs text-gray-500 mb-1">初始资金</label>
                    <input id="ai-backtest-capital" type="number" class="w-full bg-[#111] border border-gray-700 rounded px-2 py-1 text-xs text-white" value="100000" min="1000" step="1000">
                </div>
            </div>
            <div id="ai-backtest-chart-wrap" class="relative h-[300px] border border-gray-800 rounded bg-[#080808] min-h-[200px]">
                <span id="ai-backtest-placeholder" class="absolute inset-0 flex items-center justify-center text-gray-600">设置参数后点击开始回测</span>
                <div id="ai-backtest-chart" class="absolute inset-0 w-full h-full hidden"></div>
            </div>
            <div id="ai-backtest-error" class="mt-2 text-red-400 text-xs hidden"></div>
            <div class="mt-4 flex justify-end gap-2">
                <button id="ai-backtest-run" class="bg-yellow-600 hover:bg-yellow-500 text-black font-bold py-1 px-4 rounded text-xs">开始回测</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    setupDraggable(modal, 'ai-backtest-header');
    document.getElementById('ai-backtest-run').addEventListener('click', runAIBacktestSubmit);
};

let _aiBacktestChart = null;

async function runAIBacktestSubmit() {
    const startEl = document.getElementById('ai-backtest-start');
    const endEl = document.getElementById('ai-backtest-end');
    const capitalEl = document.getElementById('ai-backtest-capital');
    const runBtn = document.getElementById('ai-backtest-run');
    const placeholder = document.getElementById('ai-backtest-placeholder');
    const chartEl = document.getElementById('ai-backtest-chart');
    const errEl = document.getElementById('ai-backtest-error');
    if (!startEl || !endEl || !capitalEl || !runBtn || !placeholder || !chartEl || !errEl) return;

    const start_date = startEl.value;
    const end_date = endEl.value;
    const capital = parseFloat(capitalEl.value) || 100000;

    runBtn.disabled = true;
    errEl.classList.add('hidden');
    errEl.textContent = '';
    placeholder.textContent = '回测计算中...';
    chartEl.classList.add('hidden');
    placeholder.classList.remove('hidden');

    try {
        const resp = await fetch('/api/strategy/backtest/ai_qbot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_date, end_date, capital })
        });
        const data = await resp.json();

        if (data.status === 'error' || !resp.ok) {
            errEl.textContent = data.message || data.error || '回测失败';
            errEl.classList.remove('hidden');
            placeholder.textContent = '设置参数后点击开始回测';
            return;
        }

        const history = data.history || [];
        if (history.length === 0) {
            placeholder.textContent = '无回测数据';
            return;
        }

        placeholder.classList.add('hidden');
        chartEl.classList.remove('hidden');

        if (!_aiBacktestChart) _aiBacktestChart = (typeof echarts !== 'undefined' && echarts.init(chartEl)) || null;
        if (_aiBacktestChart) {
            const dates = history.map(h => h.date);
            const values = history.map(h => h.portfolio_value);
            _aiBacktestChart.setOption({
                backgroundColor: 'transparent',
                grid: { left: 50, right: 20, top: 20, bottom: 30 },
                xAxis: { type: 'category', data: dates, axisLabel: { color: '#94a3b8', fontSize: 10 } },
                yAxis: { type: 'value', scale: true, axisLabel: { color: '#94a3b8', fontSize: 10 }, splitLine: { lineStyle: { color: '#334155' } } },
                series: [{ type: 'line', data: values, smooth: true, lineStyle: { color: '#38bdf8' }, areaStyle: { color: 'rgba(56,189,248,0.2)' } }]
            });
            _aiBacktestChart.resize();
        } else {
            placeholder.classList.remove('hidden');
            placeholder.textContent = `回测完成，最终权益: ${FormatUtils.formatBigNumber(history[history.length - 1].portfolio_value || 0)}`;
        }
    } catch (e) {
        errEl.textContent = e.message || '请求异常';
        errEl.classList.remove('hidden');
        placeholder.textContent = '设置参数后点击开始回测';
    } finally {
        runBtn.disabled = false;
    }
}
window.runAIBacktestSubmit = runAIBacktestSubmit;

// --- Professional System Window ---

window.openProfessionalWindow = function() {
    let modalId = 'prof-system-modal';
    let modal = document.getElementById(modalId);
    
    if (modal) {
        modal.classList.remove('hidden');
        return;
    }

    modal = document.createElement('div');
    modal.id = modalId;
    modal.className = 'fixed top-[10%] left-[10%] w-[900px] h-[700px] bg-[#0c0c0c] border border-gray-700 shadow-2xl z-40 flex flex-col rounded-lg overflow-hidden';
    modal.style.boxShadow = '0 0 50px rgba(0,0,0,0.8)';
    
    modal.innerHTML = `
        <!-- Header -->
        <div class="h-10 bg-gray-800 border-b border-gray-700 flex items-center justify-between px-4 select-none cursor-move" id="${modalId}-header">
            <div class="flex items-center space-x-2">
                <i class="fas fa-layer-group text-cyan-500"></i>
                <span class="text-gray-200 font-bold text-sm">ROX 专业量化分析系统</span>
            </div>
            <button class="text-gray-400 hover:text-white" onclick="document.getElementById('${modalId}').remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
        
        <!-- Toolbar -->
        <div class="h-10 bg-[#151515] border-b border-gray-800 flex items-center px-4 space-x-4">
            <button id="prof-tab-signal" class="px-3 py-1 text-sm font-bold text-[#06b6d4] border-b-2 border-[#06b6d4]" onclick="switchProfessionalTab('signal')">核心信号</button>
            <button id="prof-tab-risk" class="px-3 py-1 text-sm text-gray-400 hover:text-white" onclick="switchProfessionalTab('risk')">风控系统</button>
        </div>
        
        <!-- Content -->
        <div class="flex-1 bg-black p-4 overflow-y-auto">
            
            <!-- SIGNAL VIEW -->
            <div id="prof-view-signal" class="h-full flex flex-col">
                <!-- Controls -->
                <div class="flex items-center space-x-4 mb-4 bg-[#111] p-3 rounded border border-gray-800">
                    <div class="flex flex-col">
                        <label class="text-[10px] text-gray-500">标的</label>
                        <input id="prof-signal-symbol" type="text" class="bg-black border border-gray-700 text-yellow-500 px-2 py-1 text-xs w-24 rounded" value="600519">
                    </div>
                    <div class="flex flex-col">
                        <label class="text-[10px] text-gray-500">模型模版</label>
                        <select id="prof-signal-template" class="bg-black border border-gray-700 text-gray-300 px-2 py-1 text-xs rounded">
                            <option value="trend_following">趋势跟踪 (Trend Following)</option>
                            <option value="mean_reversion">均值回归 (Mean Reversion)</option>
                            <option value="dark_pool">主力潜伏 (Dark Pool)</option>
                            <option value="limit_up">涨停战法 (Limit Up)</option>
                        </select>
                    </div>
                    <div class="flex flex-col">
                        <label class="text-[10px] text-gray-500">周期</label>
                        <select id="prof-signal-period" class="bg-black border border-gray-700 text-gray-300 px-2 py-1 text-xs rounded">
                            <option value="daily">日线</option>
                            <option value="60min">60分钟</option>
                        </select>
                    </div>
                    <button class="mt-3 px-4 py-1 bg-cyan-600 hover:bg-cyan-500 text-white text-xs rounded shadow-lg shadow-cyan-500/20" onclick="fetchProfessionalSignal()">
                        <i class="fas fa-play mr-1"></i> 开始分析
                    </button>
                </div>
                
                <!-- Loading -->
                <div id="prof-signal-loading" class="hidden flex-1 flex flex-col items-center justify-center text-cyan-500">
                    <i class="fas fa-circle-notch fa-spin text-3xl mb-2"></i>
                    <span class="text-xs">量化引擎计算中...</span>
                </div>
                
                <!-- Empty State -->
                <div id="prof-signal-empty" class="flex-1 flex flex-col items-center justify-center text-gray-600">
                    <i class="fas fa-wave-square text-4xl mb-2"></i>
                    <span class="text-xs">请选择模版并开始分析</span>
                </div>

                <!-- Result Card -->
                <div id="prof-signal-result-card" class="hidden flex-1 flex flex-col space-y-4">
                     <div class="grid grid-cols-3 gap-4">
                        <div class="bg-[#111] p-3 rounded border border-gray-800">
                            <div class="text-gray-500 text-xs">信号强度</div>
                            <div class="text-2xl font-bold text-up mt-1" id="prof-res-strength">---</div>
                        </div>
                        <div class="bg-[#111] p-3 rounded border border-gray-800">
                            <div class="text-gray-500 text-xs">建议操作</div>
                            <div class="text-xl font-bold text-white mt-1" id="prof-res-action">---</div>
                        </div>
                        <div class="bg-[#111] p-3 rounded border border-gray-800">
                            <div class="text-gray-500 text-xs">置信度</div>
                            <div class="text-xl font-bold text-yellow-500 mt-1" id="prof-res-confidence">---</div>
                        </div>
                     </div>
                     
                     <div class="flex-1 bg-[#111] rounded border border-gray-800 p-3 relative">
                        <div class="text-xs text-gray-500 mb-2">信号逻辑详解</div>
                        <div id="prof-signal-content" class="text-sm text-gray-300 font-mono leading-relaxed h-full overflow-y-auto">
                            <!-- Content injected here -->
                        </div>
                     </div>
                </div>
            </div>

            <!-- RISK VIEW -->
            <div id="prof-view-risk" class="hidden h-full flex flex-col">
                <div class="flex-1 flex items-center justify-center text-gray-500">
                    <div class="text-center">
                        <i class="fas fa-shield-alt text-4xl mb-2"></i>
                        <p>风控模块正在连接风控服务器...</p>
                    </div>
                </div>
            </div>
            
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Drag Logic (Reuse)
    setupDraggable(modal, `${modalId}-header`);
    
    // Initialize Logic
    if(typeof initProfessionalSystem === 'function') {
        initProfessionalSystem();
    }
};

function setupDraggable(modal, headerId) {
    const header = document.getElementById(headerId);
    let isDragging = false;
    let startX, startY, initialLeft, initialTop;
    
    header.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        const rect = modal.getBoundingClientRect();
        initialLeft = rect.left;
        initialTop = rect.top;
        modal.style.transform = 'none';
        modal.style.left = initialLeft + 'px';
        modal.style.top = initialTop + 'px';
    });
    
    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        modal.style.left = (initialLeft + dx) + 'px';
        modal.style.top = (initialTop + dy) + 'px';
    });
    
    window.addEventListener('mouseup', () => { isDragging = false; });
}

// --- Left Pane Tabs: 自选股 | 涨跌排行 | 板块 | 沪深A股 ---
const ROX_SKILLS = [
    { id: 'youzi_anpan', name: '游资暗盘', desc: '游资净买、JJ/XJ 线、建仓/卖出图标、暗盘买入标记等，用于跟踪短线资金动向。' },
    { id: 'kanlong_youhui', name: '看龙有悔', desc: '前高、倍量、突破前高、启明/揽月线、亢龙有悔卖点，用于趋势与卖点提示。' },
    { id: 'xianren_zhilu', name: '仙人指路2.0', desc: '金钻趋势、金牛、回调买、金钻起涨，用于趋势与买点。' },
    { id: 'jigou_caopan', name: '机构操盘3.0', desc: '红色持股/青色观望、短买/白色离场、今日/明日阻力支撑，用于机构风格买卖点与价位。' },
    { id: 'xunlongjue', name: '寻龙诀选股器', desc: '倍量+突破前高+涨停等条件选股，用于筛选强势股。' },
    { id: 'anpan_zijin', name: '暗盘资金 hf1.0', desc: '特大单/大单/中单/小单买卖、暗盘资金，需 Level2 或分档数据，用于资金结构分析。' }
];

function getSkillsVisible() {
    try {
        const raw = localStorage.getItem('rox_skills_visible');
        return raw ? JSON.parse(raw) : {};
    } catch (e) { return {}; }
}
function setSkillVisible(id, visible) {
    const o = getSkillsVisible();
    o[id] = !!visible;
    localStorage.setItem('rox_skills_visible', JSON.stringify(o));
}

function renderSkillsList() {
    const container = document.getElementById('skills-list');
    if (!container || container.dataset.rendered === '1') return;
    const visible = getSkillsVisible();
    container.innerHTML = ROX_SKILLS.map(s => {
        const isVisible = visible[s.id] !== false;
        return `
            <div class="rounded border border-[#333] bg-[#1a1a1a] p-2 text-xxs" data-skill-id="${s.id}">
                <div class="font-bold text-yellow-500 mb-1">${s.name}</div>
                <p class="text-gray-400 mb-2 leading-relaxed">${s.desc}</p>
                <button type="button" class="skill-toggle px-2 py-0.5 rounded text-[10px] ${isVisible ? 'bg-green-800 text-green-200' : 'bg-gray-700 text-gray-400'}" data-skill-id="${s.id}">${isVisible ? '显示' : '隐藏'}</button>
            </div>`;
    }).join('');
    container.dataset.rendered = '1';
    container.querySelectorAll('.skill-toggle').forEach(btn => {
        btn.addEventListener('click', () => {
            const id = btn.dataset.skillId;
            const visible = getSkillsVisible();
            const next = !visible[id];
            setSkillVisible(id, next);
            btn.textContent = next ? '显示' : '隐藏';
            btn.classList.toggle('bg-green-800', next);
            btn.classList.toggle('text-green-200', next);
            btn.classList.toggle('bg-gray-700', !next);
            btn.classList.toggle('text-gray-400', !next);
            
            // Trigger chart update to reflect visibility changes
            if (typeof renderKLineChart === 'function' && window._cachedKLineData) {
                renderKLineChart(window._cachedKLineData, document.querySelector('[data-period].active')?.dataset.period || 'daily');
            } else if (typeof fetchKLineData === 'function') {
                fetchKLineData(document.querySelector('[data-period].active')?.dataset.period || 'daily');
            }
        });
    });
}

function switchLeftTab(tab) {
    const header = document.getElementById('left-rankings-header');
    const rankingsList = document.getElementById('stock-list-container');
    const watchlistEl = document.getElementById('watchlist-container');
    const sectorEl = document.getElementById('sector-list-container');
    const spotEl = document.getElementById('spot-list-container');
    const skillsEl = document.getElementById('skills-container');
    const btns = document.querySelectorAll('.tab-btn[data-tab]');
    btns.forEach(b => {
        b.classList.toggle('active', b.dataset.tab === tab);
    });
    [rankingsList, watchlistEl, sectorEl, spotEl, skillsEl].forEach(el => { if (el) el.classList.add('hidden'); });
    if (tab === 'zixuan') {
        if (header) header.classList.add('hidden');
        if (watchlistEl) { watchlistEl.classList.remove('hidden'); loadWatchlist(); }
    } else if (tab === 'sector') {
        if (header) header.classList.remove('hidden');
        if (sectorEl) { sectorEl.classList.remove('hidden'); loadSectorList(); }
    } else if (tab === 'spot') {
        if (header) header.classList.remove('hidden');
        if (spotEl) { spotEl.classList.remove('hidden'); loadSpotList(); }
    } else if (tab === 'skills') {
        if (header) header.classList.add('hidden');
        if (skillsEl) { skillsEl.classList.remove('hidden'); renderSkillsList(); }
    } else {
        if (header) header.classList.remove('hidden');
        if (rankingsList) rankingsList.classList.remove('hidden');
        loadStockList();
    }
}
window.switchLeftTab = switchLeftTab;
window.getSkillsVisible = getSkillsVisible;
window.ROX_SKILLS = ROX_SKILLS;

async function loadSectorList() {
    const container = document.getElementById('sector-list-container');
    if (!container) return;
    try {
        const r = await fetch('/api/market/rankings');
        const d = await r.json().catch(() => ({}));
        const sectors = d.sectors || [];
        container.innerHTML = '';
        sectors.forEach(s => {
            const div = document.createElement('div');
            div.className = 'grid grid-cols-[1fr_80px_60px] px-2 py-1 border-b border-[#1a1a1a] hover:bg-[#222] cursor-default';
            const pct = s.pct != null ? s.pct : 0;
            const colorClass = pct > 0 ? 'text-up' : (pct < 0 ? 'text-down' : 'text-gray-400');
            div.innerHTML = `<div class="text-yellow-500 font-bold">${s.name || '—'}</div><div class="text-right text-gray-500">—</div><div class="text-right ${colorClass} font-mono">${pct > 0 ? '+' : ''}${pct}%</div>`;
            container.appendChild(div);
        });
    } catch (e) {
        console.error('loadSectorList', e);
    }
}

const SPOT_PAGE_SIZE = 500;
let spotListOffset = 0;
let spotListTotal = 0;

function renderSpotRow(s, listEl) {
    const div = document.createElement('div');
    div.className = 'grid grid-cols-[1fr_80px_60px] px-2 py-1 border-b border-[#1a1a1a] hover:bg-[#222] cursor-pointer stock-row';
    div.onclick = () => selectStock(s.code, s.name);
    const colorClass = (s.pct || 0) > 0 ? 'text-up' : ((s.pct || 0) < 0 ? 'text-down' : 'text-gray-400');
    div.innerHTML = `
        <div><div class="text-yellow-500 font-bold">${s.name}</div><div class="text-xxs text-gray-500">${s.code}</div></div>
        <div class="text-right ${colorClass} self-center font-mono">${(s.price || 0).toFixed(2)}</div>
        <div class="text-right ${colorClass} self-center font-mono">${(s.pct || 0) > 0 ? '+' : ''}${s.pct || 0}%</div>
    `;
    listEl.appendChild(div);
}

async function loadSpotList(append = false) {
    const container = document.getElementById('spot-list-container');
    if (!container) return;
    const offset = append ? spotListOffset : 0;
    try {
        const r = await fetch(`/api/market/spot?limit=${SPOT_PAGE_SIZE}&offset=${offset}`);
        const d = await r.json().catch(() => ({}));
        const stocks = d.stocks || [];
        const total = d.total != null ? d.total : 0;
        if (!append) {
            spotListOffset = stocks.length;
            spotListTotal = total;
            container.innerHTML = ''
                + '<p class="text-xxs text-gray-500 px-2 py-1 border-b border-[#1a1a1a]">默认 500 条，按涨跌幅排序；完整列表可用下方「加载更多」或搜索。</p>'
                + '<div id="spot-list-body"></div>'
                + '<div id="spot-list-footer" class="p-2"></div>';
        } else {
            spotListOffset += stocks.length;
        }
        const listEl = document.getElementById('spot-list-body');
        if (d.error && !append && listEl) {
            listEl.innerHTML = '<p class="text-xxs text-amber-400 px-2 py-2">' + (d.error || '行情数据暂时不可用，请稍后重试') + '</p>';
        } else if (listEl) {
            stocks.forEach(s => renderSpotRow(s, listEl));
        }
        const footer = document.getElementById('spot-list-footer');
        if (footer) {
            if (spotListOffset < spotListTotal && spotListTotal > SPOT_PAGE_SIZE) {
                footer.innerHTML = '<button type="button" id="spot-load-more-btn" class="w-full py-1.5 text-xxs rounded bg-[#333] hover:bg-[#444] text-gray-300">加载更多（已显示 ' + spotListOffset + ' / ' + spotListTotal + '）</button>';
                const btn = document.getElementById('spot-load-more-btn');
                if (btn) btn.addEventListener('click', () => loadSpotList(true));
            } else {
                footer.innerHTML = spotListTotal > 0 ? '<p class="text-xxs text-gray-500 text-center">已显示全部 ' + spotListTotal + ' 条</p>' : '';
            }
        }
    } catch (e) {
        console.error('loadSpotList', e);
    }
}

// --- Main Logic (Charts, Data) ---
function initMainLogic() {
    window.searchStock = searchStock;

    document.querySelectorAll('.tab-btn[data-tab]').forEach(btn => {
        btn.addEventListener('click', () => switchLeftTab(btn.dataset.tab));
    });

    const compactBtn = document.getElementById('compact-toggle');
    if (compactBtn) {
        compactBtn.addEventListener('click', () => {
            document.body.classList.toggle('compact');
            compactBtn.classList.toggle('bg-gray-600', document.body.classList.contains('compact'));
            compactBtn.classList.toggle('text-gray-300', document.body.classList.contains('compact'));
        });
    }
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        const savedTheme = localStorage.getItem('rox_theme');
        if (savedTheme === 'light') {
            document.body.classList.add('theme-light');
            themeBtn.textContent = '深色';
        }
        themeBtn.addEventListener('click', () => {
            document.body.classList.toggle('theme-light');
            const isLight = document.body.classList.contains('theme-light');
            localStorage.setItem('rox_theme', isLight ? 'light' : 'dark');
            themeBtn.textContent = isLight ? '深色' : '主题';
        });
    }
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const modals = ['auth-modal', 'settings-modal', 'news-center-modal', 'alerts-modal', 'python-sandbox-modal', 'contradictions-modal', 'value-scatter-modal', 'pred-chart-modal'];
            for (const id of modals) {
                const el = document.getElementById(id);
                if (el && !el.classList.contains('hidden')) {
                    el.classList.add('hidden');
                    e.preventDefault();
                    break;
                }
            }
        }
    });
    const exportWatchlistBtn = document.getElementById('export-watchlist-btn');
    if (exportWatchlistBtn) {
        exportWatchlistBtn.addEventListener('click', async () => {
            try {
                const r = await fetch('/api/market/watchlist', { credentials: 'include' });
                if (!r.ok) {
                    if (r.status === 401 && typeof showAuthModal === 'function') showAuthModal();
                    else (typeof showToast === 'function' ? showToast : alert)('请先登录');
                    return;
                }
                const d = await r.json();
                const blob = new Blob([JSON.stringify(d.items || [], null, 2)], { type: 'application/json' });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = `rox-watchlist-${new Date().toISOString().slice(0,10)}.json`;
                a.click();
                URL.revokeObjectURL(a.href);
                if (typeof showToast === 'function') showToast('已导出');
            } catch (e) {
                (typeof showToast === 'function' ? showToast : alert)('导出失败');
            }
        });
    }
    const exportCsvBtn = document.getElementById('export-watchlist-csv-btn');
    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', async () => {
            try {
                const token = localStorage.getItem('access_token');
                const r = await fetch('/api/market/watchlist/export?format=csv', {
                    credentials: 'include',
                    headers: token ? { 'Authorization': 'Bearer ' + token } : {}
                });
                if (!r.ok) {
                    if (r.status === 401 && typeof showAuthModal === 'function') showAuthModal();
                    else (typeof showToast === 'function' ? showToast : alert)('请先登录');
                    return;
                }
                const blob = await r.blob();
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'rox-watchlist.csv';
                a.click();
                URL.revokeObjectURL(a.href);
                if (typeof showToast === 'function') showToast('已导出 CSV');
            } catch (e) {
                (typeof showToast === 'function' ? showToast : alert)('导出失败');
            }
        });
    }
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    if (settingsBtn && settingsModal) {
        settingsBtn.addEventListener('click', () => {
            settingsModal.classList.remove('hidden');
            settingsModal.classList.add('flex');
            const iv = localStorage.getItem('rox-refresh-interval');
            const input = document.getElementById('settings-refresh-interval');
            if (input && iv) input.value = iv;
        });
        const input = document.getElementById('settings-refresh-interval');
        if (input) {
            const v = localStorage.getItem('rox-refresh-interval');
            if (v) input.value = v;
            input.addEventListener('change', () => localStorage.setItem('rox-refresh-interval', String(input.value)));
        }
    }

    initKLineChart();
    initFundsFlowChart();
    // 1.0→3.0 带上下文：从 URL 读取 code/name，直接打开该只 K 线/分时
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    const name = params.get('name') || '';
    if (code && code.length >= 5) {
        selectStock(code.trim(), name.trim() || code);
        history.replaceState({}, '', window.location.pathname); // 去掉 URL 参数，便于刷新
    }
    switchLeftTab('rankings');

    document.querySelectorAll('.indicator-tab').forEach(el => {
        el.addEventListener('click', () => setIndicatorMode(el.dataset.indicator || 'hot_money'));
    });

    setInterval(() => {
        const active = document.querySelector('.tab-btn[data-tab].active');
        const t = active ? active.dataset.tab : '';
        if (t === 'rankings') loadStockList();
        else if (t === 'sector') loadSectorList();
        else if (t === 'spot') loadSpotList();
    }, 60000);

    // Load header indices dynamically
    loadHeaderIndices();
    setInterval(loadHeaderIndices, 60000);
}

// --- Header Indices Dynamic Loading ---
async function loadHeaderIndices() {
    const container = document.getElementById('header-indices');
    if (!container) return;
    
    try {
        const r = await fetch('/api/market/indices');
        const data = await r.json().catch(() => ({}));
        const indices = data.indices || [];
        
        // Map index names to element data-index attributes
        const mapping = {
            '上证指数': 'sh',
            '深证成指': 'sz', 
            '创业板指': 'cyb'
        };
        
        for (const idx of indices) {
            const key = mapping[idx.name];
            if (!key) continue;
            
            const item = container.querySelector(`[data-index="${key}"]`);
            if (!item) continue;
            
            const priceEl = item.querySelector('.index-price');
            const pctEl = item.querySelector('.index-pct');
            
            if (priceEl) {
                priceEl.textContent = idx.price?.toFixed(2) || '----';
                priceEl.classList.remove('skeleton-text', 'text-slate-400');
                priceEl.classList.add(idx.pct >= 0 ? 'text-up' : 'text-down');
            }
            
            if (pctEl) {
                const pctVal = idx.pct || 0;
                pctEl.textContent = `${pctVal >= 0 ? '+' : ''}${pctVal.toFixed(2)}%`;
                pctEl.classList.remove('text-slate-500', 'bg-slate-800');
                if (pctVal >= 0) {
                    pctEl.classList.add('text-up', 'bg-up-dim');
                } else {
                    pctEl.classList.add('text-down', 'bg-down-dim');
                }
            }
        }
    } catch (e) {
        console.warn('Failed to load header indices:', e);
    }
}

// --- Phase 3: 消息中心、提醒系统 ---
function openNewsCenter() {
    const modal = document.getElementById('news-center-modal');
    if (!modal) return;
    modal.classList.remove('hidden');
    const content = document.getElementById('news-center-content');
    if (content) {
        content.innerHTML = '<div class="text-gray-500 mb-2">📰 最新消息</div><div class="space-y-2">' +
            '<div class="p-2 bg-[#111] rounded border border-gray-800"><div class="text-yellow-500 text-xs font-bold">600519 贵州茅台</div><div class="text-gray-400 text-xxs mt-1">2025-01-30 10:30</div><div class="text-gray-300 text-xs mt-1">公司发布2024年度业绩预告，净利润同比增长15%</div></div>' +
            '<div class="p-2 bg-[#111] rounded border border-gray-800"><div class="text-yellow-500 text-xs font-bold">300750 宁德时代</div><div class="text-gray-400 text-xxs mt-1">2025-01-30 09:15</div><div class="text-gray-300 text-xs mt-1">与某车企签署长期合作协议</div></div>' +
            '</div>';
    }
    if (typeof setupDraggable === 'function') setupDraggable(modal, 'news-center-header');
}
window.openNewsCenter = openNewsCenter;

function openAlerts() {
    const modal = document.getElementById('alerts-modal');
    if (!modal) return;
    modal.classList.remove('hidden');
    loadAlerts();
    const addBtn = document.getElementById('alert-add-btn');
    if (addBtn) {
        addBtn.onclick = () => {
            const code = document.getElementById('alert-stock-code')?.value;
            const above = document.getElementById('alert-price-above')?.value;
            const below = document.getElementById('alert-price-below')?.value;
            if (!code) { (typeof showToast === 'function' ? showToast : alert)('请输入股票代码'); return; }
            const alerts = JSON.parse(localStorage.getItem('rox-alerts') || '[]');
            alerts.push({ code, above: above ? parseFloat(above) : null, below: below ? parseFloat(below) : null, id: Date.now() });
            localStorage.setItem('rox-alerts', JSON.stringify(alerts));
            loadAlerts();
            if (typeof showToast === 'function') showToast('提醒已添加');
        };
    }
    if (typeof setupDraggable === 'function') setupDraggable(modal, 'alerts-header');
}
window.openAlerts = openAlerts;

function loadAlerts() {
    const list = document.getElementById('alerts-list');
    if (!list) return;
    const alerts = JSON.parse(localStorage.getItem('rox-alerts') || '[]');
    list.innerHTML = alerts.length ? alerts.map(a => `
        <div class="p-2 bg-[#111] rounded border border-gray-800 flex justify-between items-center">
            <div><span class="text-yellow-500 font-bold">${a.code}</span> ${a.above ? `≥${a.above}` : ''} ${a.below ? `≤${a.below}` : ''}</div>
            <button onclick="removeAlert(${a.id})" class="text-red-400 hover:text-red-300 text-xs">删除</button>
        </div>
    `).join('') : '<div class="text-gray-500 text-xs">暂无提醒</div>';
}
window.removeAlert = (id) => {
    const alerts = JSON.parse(localStorage.getItem('rox-alerts') || '[]');
    localStorage.setItem('rox-alerts', JSON.stringify(alerts.filter(a => a.id !== id)));
    loadAlerts();
};

// --- Phase 5: Python沙箱 ---
function openPythonSandbox() {
    const modal = document.getElementById('python-sandbox-modal');
    if (!modal) return;
    modal.classList.remove('hidden');
    const runBtn = document.getElementById('python-run-btn');
    const saveBtn = document.getElementById('python-save-btn');
    const editor = document.getElementById('python-code-editor');
    const output = document.getElementById('python-output');
    if (runBtn) {
        runBtn.onclick = async () => {
            if (!editor || !output) return;
            const code = editor.value;
            output.textContent = '执行中…';
            try {
                const r = await fetch('/api/strategy/python-exec', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code, stock_code: window.currentStockCode || '600519' })
                });
                const d = await r.json().catch(() => ({}));
                if (r.ok && d.result) {
                    output.textContent = typeof d.result === 'string' ? d.result : JSON.stringify(d.result, null, 2);
                } else {
                    output.textContent = d.error || '执行失败';
                }
            } catch (e) {
                output.textContent = '错误: ' + e.message;
            }
        };
    }
    if (saveBtn) {
        saveBtn.onclick = () => {
            const code = editor?.value;
            if (!code) return;
            const saved = JSON.parse(localStorage.getItem('rox-python-strategies') || '[]');
            saved.push({ name: `策略_${Date.now()}`, code, created: new Date().toISOString() });
            localStorage.setItem('rox-python-strategies', JSON.stringify(saved));
            if (typeof showToast === 'function') showToast('策略已保存');
        };
    }
    if (typeof setupDraggable === 'function') setupDraggable(modal, 'python-sandbox-header');
}
window.openPythonSandbox = openPythonSandbox;

// --- Philosophy: 矛盾分析 & 价值散点图 ---
function openContradictions() {
    const modal = document.getElementById('contradictions-modal');
    const content = document.getElementById('contradictions-content');
    if (!modal || !content) return;
    modal.classList.remove('hidden');
    content.textContent = '加载中…';
    (async () => {
        try {
            const r = await fetch('/api/philosophy/contradictions');
            const d = await r.json().catch(() => ({}));
            if (!r.ok || d.error) {
                content.textContent = d.detail || d.error || '加载失败';
                return;
            }
            window.roxMarketRegime = { data: d, ts: Date.now() };
            const main = d.main;
            const items = d.items || [];
            const snap = d.snapshot || {};
            content.innerHTML = `
                <div class="space-y-3">
                    <div class="text-xs text-gray-500">市场快照：上涨 ${snap.up ?? '--'} / 下跌 ${snap.down ?? '--'} ｜ 成交额 ${snap.volume_yi != null ? snap.volume_yi.toFixed(0)+'亿' : '--'} ｜ 北向 ${snap.north_yi != null ? snap.north_yi.toFixed(2)+'亿' : '--'} ｜ 主力 ${snap.main_yi != null ? snap.main_yi.toFixed(2)+'亿' : '--'}</div>
                    <div class="flex gap-2">
                        <button type="button" onclick="openValueScatter()" class="text-xxs px-2 py-1 rounded bg-sky-700 hover:bg-sky-600 text-white">打开价值散点</button>
                        <button type="button" onclick="applyRegimeToValueScatter()" class="text-xxs px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-gray-200 border border-slate-700">按主矛盾筛选</button>
                    </div>
                    <div class="p-3 bg-[#111] border border-gray-800 rounded">
                        <div class="text-yellow-500 font-bold text-sm">主矛盾：${main?.name || '—'}</div>
                        <div class="mt-1 text-gray-300 text-xs">${main?.summary || ''}</div>
                        <div class="mt-2 text-gray-400 text-xs">建议：${main?.suggestion || ''}</div>
                        <div class="mt-2 text-gray-500 text-xxs">强度：${main?.strength ?? '--'} / 100 ｜ 方向：${main?.direction ?? '--'}</div>
                    </div>
                    <div class="grid grid-cols-1 gap-2">
                        ${(items || []).map(it => `
                            <div class="p-3 bg-[#0f0f0f] border border-gray-800 rounded">
                                <div class="flex justify-between items-center">
                                    <div class="text-gray-200 text-xs font-bold">${it.name}</div>
                                    <div class="text-xxs text-gray-500">强度 ${it.strength}/100</div>
                                </div>
                                <div class="mt-1 text-xxs text-gray-400">${it.summary}</div>
                                <div class="mt-1 text-xxs text-gray-500">建议：${it.suggestion}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        } catch (e) {
            content.textContent = '加载失败';
        }
    })();
    if (typeof setupDraggable === 'function') setupDraggable(modal, 'contradictions-header');
}
window.openContradictions = openContradictions;

let valueScatterChart = null;
let valueScatterCache = { ts: 0, items: [] };
let valueScatterFollowRegime = true;
let valueScatterManualFilter = null;

async function ensureMarketRegime(force = false) {
    const cached = window.roxMarketRegime;
    if (!force && cached?.data && cached?.ts && (Date.now() - cached.ts) < 5 * 60 * 1000) {
        return cached.data;
    }
    try {
        const r = await fetch('/api/philosophy/contradictions');
        const d = await r.json().catch(() => ({}));
        if (r.ok && !d.error) {
            window.roxMarketRegime = { data: d, ts: Date.now() };
            return d;
        }
    } catch (_) {}
    return cached?.data || null;
}

function deriveFocusPredicateFromRegime(regime) {
    const main = regime?.main;
    const id = main?.id;
    const strength = Number(main?.strength ?? 0);
    const direction = Number(main?.direction ?? 0);

    // 兜底：只看“低估 + 质量不差”
    const base = (it) => (it.surplus_score ?? 0) >= 60 && (it.deviation ?? 0) <= -0.10;

    if (!id || strength < 10) return base;

    // 量能 vs 赚钱效应
    if (id === 'liquidity_vs_breadth') {
        // direction<0：偏分歧/风险 → 防守：大市值 + 高质量 + 轻度低估
        if (direction < -0.05) {
            return (it) =>
                (it.mv_yi ?? 0) >= 200 &&
                (it.surplus_score ?? 0) >= 65 &&
                (it.deviation ?? 0) <= -0.08 &&
                (it.deviation ?? 0) >= -0.35;
        }
        // direction>0：偏顺风 → 进攻：允许更深低估/更广范围
        if (direction > 0.05) {
            return (it) =>
                (it.surplus_score ?? 0) >= 55 &&
                (it.deviation ?? 0) <= -0.12 &&
                (it.deviation ?? 0) >= -0.45;
        }
        return base;
    }

    // 外资 vs 内资分歧：高分歧时偏防守
    if (id === 'foreign_vs_domestic') {
        return (it) =>
            (it.mv_yi ?? 0) >= 300 &&
            (it.surplus_score ?? 0) >= 65 &&
            (it.deviation ?? 0) <= -0.08 &&
            (it.deviation ?? 0) >= -0.30;
    }

    // 行业分化强：不追求“便宜”，更看“强者更强/高质量”
    if (id === 'sector_rotation') {
        return (it) =>
            (it.surplus_score ?? 0) >= 70 &&
            (it.deviation ?? 0) <= 0.15 && // 允许略贵
            (it.deviation ?? 0) >= -0.25;
    }

    return base;
}

function buildScatterSeries(items, focusPredicate, follow) {
    const ptsFocus = [];
    const ptsOther = [];
    for (const it of items) {
        const x = (it.deviation ?? 0) * 100;
        const y = it.surplus_score ?? 50;
        const mv = it.mv_yi ?? 0;
        const row = [x, y, mv, it.code, it.name, it.price, it.intrinsic, it.signal];
        const isFocus = follow ? !!focusPredicate(it) : true;
        (isFocus ? ptsFocus : ptsOther).push(row);
    }
    return { ptsFocus, ptsOther };
}

function colorForSignal(sig) {
    if (sig === 'strong_buy') return '#ff333a';
    if (sig === 'buy') return '#fb7185';
    if (sig === 'strong_sell') return '#00aa3b';
    if (sig === 'sell') return '#22c55e';
    return '#94a3b8';
}

function openValueScatter() {
    const modal = document.getElementById('value-scatter-modal');
    const chartEl = document.getElementById('value-scatter-chart');
    if (!modal || !chartEl) return;
    modal.classList.remove('hidden');

    if (!valueScatterChart && typeof echarts !== 'undefined') {
        valueScatterChart = echarts.init(chartEl);
        valueScatterChart.on('click', (params) => {
            const p = params?.data;
            if (!p) return;
            const code = p[3];
            const name = p[4];
            if (code && typeof window.selectStock === 'function') window.selectStock(code, name);
        });
        window.addEventListener('resize', () => valueScatterChart && valueScatterChart.resize());
    }

    const followEl = document.getElementById('value-scatter-follow');
    const refreshEl = document.getElementById('value-scatter-refresh');
    const resetEl = document.getElementById('value-scatter-reset');
    if (followEl && followEl.dataset.bound !== '1') {
        followEl.dataset.bound = '1';
        followEl.checked = true;
        followEl.addEventListener('change', () => {
            valueScatterFollowRegime = !!followEl.checked;
            render();
        });
    }
    if (refreshEl && refreshEl.dataset.bound !== '1') {
        refreshEl.dataset.bound = '1';
        refreshEl.addEventListener('click', async () => {
            await ensureMarketRegime(true);
            valueScatterCache.ts = 0;
            await render();
        });
    }
    if (resetEl && resetEl.dataset.bound !== '1') {
        resetEl.dataset.bound = '1';
        resetEl.addEventListener('click', () => {
            valueScatterFollowRegime = true;
            valueScatterManualFilter = null;
            if (followEl) followEl.checked = true;
            render();
        });
    }

    const render = async () => {
        if (!valueScatterChart) return;
        valueScatterChart.showLoading({ color: '#38bdf8', maskColor: 'rgba(0,0,0,0.2)' });
        try {
            const regime = await ensureMarketRegime(false);
            const main = regime?.main;
            const regimeLabelEl = document.getElementById('value-scatter-regime');
            const regimeNoteEl = document.getElementById('value-scatter-regime-note');
            if (regimeLabelEl) regimeLabelEl.textContent = main?.name || '—';
            if (regimeNoteEl) regimeNoteEl.textContent = main ? `强度 ${main.strength}/100` : '';

            if (!valueScatterCache.items.length || (Date.now() - valueScatterCache.ts) > 60 * 1000) {
                const r = await fetch('/api/philosophy/value-scatter?limit=900&sort=mv');
                const d = await r.json().catch(() => ({}));
                valueScatterCache.items = d.items || [];
                valueScatterCache.ts = Date.now();
            }
            const items = valueScatterCache.items || [];

            const focusPred = valueScatterManualFilter || deriveFocusPredicateFromRegime(regime);
            const { ptsFocus, ptsOther } = buildScatterSeries(items, focusPred, valueScatterFollowRegime);

            valueScatterChart.setOption({
                backgroundColor: 'transparent',
                grid: { left: 60, right: 20, top: 30, bottom: 50 },
                tooltip: {
                    trigger: 'item',
                    formatter: (p) => {
                        const d = p.data;
                        const code = d[3], name = d[4];
                        const price = d[5], iv = d[6];
                        const dev = d[0];
                        const score = d[1];
                        const mv = d[2];
                        const sig = d[7];
                        return `
                            <div style="font-size:12px;line-height:1.4">
                                <div><b>${name}</b> <span style="color:#999">${code}</span></div>
                                <div>偏离度：<b>${dev.toFixed(1)}%</b> ｜ 剩余价值：<b>${score}</b></div>
                                <div>现价：${Number(price).toFixed(2)} ｜ 内在：${Number(iv).toFixed(2)}</div>
                                <div>市值：${mv ? mv.toFixed(0) : '--'} 亿 ｜ 信号：${sig}</div>
                                <div style="color:#888">点击点：切换到该股票</div>
                            </div>
                        `;
                    }
                },
                xAxis: {
                    name: '价格偏离度(%)',
                    nameTextStyle: { color: '#94a3b8' },
                    axisLabel: { color: '#94a3b8' },
                    splitLine: { lineStyle: { color: '#1e293b' } },
                },
                yAxis: {
                    name: '剩余价值能力(0-100)',
                    nameTextStyle: { color: '#94a3b8' },
                    axisLabel: { color: '#94a3b8' },
                    splitLine: { lineStyle: { color: '#1e293b' } },
                    min: 0,
                    max: 100,
                },
                series: [{
                    type: 'scatter',
                    data: ptsOther,
                    symbolSize: (d) => {
                        const mv = d[2] || 0;
                        return Math.max(4, Math.min(18, Math.sqrt(mv || 1)));
                    },
                    itemStyle: {
                        color: (p) => colorForSignal(p.data[7]),
                        opacity: valueScatterFollowRegime ? 0.10 : 0.65
                    },
                    silent: true
                },{
                    type: 'scatter',
                    data: ptsFocus,
                    symbolSize: (d) => {
                        const mv = d[2] || 0;
                        return Math.max(4, Math.min(18, Math.sqrt(mv || 1)));
                    },
                    itemStyle: {
                        color: (p) => colorForSignal(p.data[7]),
                        opacity: 0.90
                    },
                    markLine: {
                        silent: true,
                        symbol: ['none', 'none'],
                        lineStyle: { color: '#64748b', type: 'dashed' },
                        data: [
                            { xAxis: 0 },
                            { xAxis: -15 }, { xAxis: -30 },
                            { xAxis: 15 }, { xAxis: 30 },
                            { yAxis: 50 }
                        ]
                    }
                }]
            });
        } catch (e) {
            // ignore
        } finally {
            if (valueScatterChart) valueScatterChart.hideLoading();
        }
    };

    render();
    if (typeof setupDraggable === 'function') setupDraggable(modal, 'value-scatter-header');
}
window.openValueScatter = openValueScatter;

function applyRegimeToValueScatter() {
    valueScatterFollowRegime = true;
    const followEl = document.getElementById('value-scatter-follow');
    if (followEl) followEl.checked = true;
    openValueScatter();
}
window.applyRegimeToValueScatter = applyRegimeToValueScatter;

let fundsFlowChart = null;

function initFundsFlowChart() {
    const container = document.getElementById('indicator-chart-container');
    if (!container) return;
    fundsFlowChart = typeof echarts !== 'undefined' ? echarts.init(container) : null;
    const placeholder = Array.from({ length: 60 }, (_, i) => 0);
    const option = {
        backgroundColor: 'transparent',
        grid: { left: 40, right: 10, top: 20, bottom: 20 },
        xAxis: { type: 'category', data: placeholder.map((_, i) => i), show: false },
        yAxis: { scale: true, splitLine: { show: false }, axisLabel: { color: '#666', fontSize: 10 } },
        series: [{ type: 'bar', data: placeholder, itemStyle: { color: (p) => (p.value > 0 ? '#ff333a' : '#00aa3b') }, barWidth: '60%' }]
    };
    if (fundsFlowChart) fundsFlowChart.setOption(option);
    window.addEventListener('resize', () => { if (fundsFlowChart) fundsFlowChart.resize(); });
}

async function updateIndicatorChart(code) {
    if (!fundsFlowChart) return;
    const period = document.querySelector('[data-period].active')?.dataset.period || 'daily';
    const setBarOption = (dates, vals) => {
        if (!fundsFlowChart) return;
        fundsFlowChart.setOption({
            xAxis: { type: 'category', data: dates, show: false },
            yAxis: { scale: true, splitLine: { show: false }, axisLabel: { color: '#666', fontSize: 10 } },
            series: [{ type: 'bar', data: vals, barWidth: '60%', itemStyle: { color: (p) => (p.value > 0 ? '#ff333a' : '#00aa3b') } }]
        });
        fundsFlowChart.resize();
    };
    const setLineOption = (dates, series) => {
        if (!fundsFlowChart) return;
        fundsFlowChart.setOption({
            xAxis: { type: 'category', data: dates, show: false },
            yAxis: { scale: true, splitLine: { show: false }, axisLabel: { color: '#666', fontSize: 10 } },
            series
        });
        fundsFlowChart.resize();
    };
    if (indicatorMode === 'vol' && _lastKlineData && _lastKlineData.dates && _lastKlineData.volumes) {
        setBarOption(_lastKlineData.dates, _lastKlineData.volumes);
        return;
    }
    if (['macd', 'kdj', 'rsi'].includes(indicatorMode)) {
        try {
            const r = await fetch(`/api/market/indicators?code=${encodeURIComponent(code)}&period=${period}`);
            const d = await r.json().catch(() => ({}));
            if (!r.ok || d.error) return;
            const dates = d.dates || [];
            if (indicatorMode === 'macd' && d.macd) {
                setLineOption(dates, [
                    { type: 'line', data: d.macd.dif, name: 'DIF', lineStyle: { color: '#38bdf8' }, symbol: 'none' },
                    { type: 'line', data: d.macd.dea, name: 'DEA', lineStyle: { color: '#eab308' }, symbol: 'none' },
                    { type: 'bar', data: d.macd.histogram, name: 'MACD', itemStyle: { color: (p) => (p.value > 0 ? '#ff333a' : '#00aa3b') }, barWidth: '40%' }
                ]);
            } else if (indicatorMode === 'kdj' && d.kdj) {
                setLineOption(dates, [
                    { type: 'line', data: d.kdj.k, name: 'K', lineStyle: { color: '#38bdf8' }, symbol: 'none' },
                    { type: 'line', data: d.kdj.d, name: 'D', lineStyle: { color: '#eab308' }, symbol: 'none' },
                    { type: 'line', data: d.kdj.j, name: 'J', lineStyle: { color: '#a855f7' }, symbol: 'none' }
                ]);
            } else if (indicatorMode === 'rsi' && d.rsi) {
                setLineOption(dates, [
                    { type: 'line', data: d.rsi, name: 'RSI', lineStyle: { color: '#38bdf8' }, symbol: 'none', markLine: { data: [{ yAxis: 70 }, { yAxis: 30 }], lineStyle: { color: '#666', type: 'dashed' } } }
                ]);
            }
            return;
        } catch (e) {
            console.debug('Indicators fetch failed', e);
        }
    }
    if (!code) return;
    try {
        const r = await fetch(`/api/analysis/hot-money/${encodeURIComponent(code)}`);
        const data = await r.json().catch(() => []);
        if (!Array.isArray(data) || data.length === 0) return;
        const key = data[0] && '游资净买' in data[0] ? '游资净买' : (data[0] && 'hot_money' in data[0] ? 'hot_money' : null);
        if (!key) return;
        const vals = data.map((d) => (d[key] != null ? Number(d[key]) : 0));
        const dates = data.map((d) => (d.date != null ? String(d.date).slice(0, 10) : ''));
        setBarOption(dates, vals);
    } catch (e) {
        console.debug('Indicator chart update failed', e);
    }
}

function setIndicatorMode(mode) {
    indicatorMode = mode;
    document.querySelectorAll('.indicator-tab').forEach(el => {
        el.classList.toggle('text-trade-highlight', el.dataset.indicator === mode);
        el.classList.toggle('font-bold', el.dataset.indicator === mode);
    });
    updateIndicatorChart(window.currentStockCode || '600519');
}

async function searchStock() {
    const input = document.getElementById('stock-search-input');
    const query = input.value.trim();
    if (!query) return;
    try {
        const resp = await fetch('/api/market/fetch-realtime', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stock_name: query })
        });
        const data = await resp.json();
        if (data.error) {
            if (typeof showToast === 'function') showToast(data.error);
            else alert(data.error);
            return;
        }
        selectStock(data.code, data.name || data.code);
    } catch (e) {
        console.error(e);
    }
}

let _lastRankings = {};
let _lastKlineData = null;
let indicatorMode = 'hot_money';

async function loadStockList() {
    const container = document.getElementById('stock-list-container');
    if (!container) return;
    try {
        const resp = await fetch('/api/market/rankings');
        const data = await resp.json();
        const stocks = data.stocks || [];
        container.innerHTML = '';
        stocks.forEach(s => {
            const prev = _lastRankings[s.code];
            const price = typeof s.price === 'number' ? s.price : parseFloat(s.price) || 0;
            const flash = prev != null && prev !== price ? (price > prev ? 'flash-up' : 'flash-down') : '';
            _lastRankings[s.code] = price;

            const div = document.createElement('div');
            div.className = `grid grid-cols-[1fr_80px_60px] px-2 py-1 border-b border-[#1a1a1a] hover:bg-[#222] cursor-pointer stock-row ${flash}`.trim();
            div.dataset.code = s.code;
            div.onclick = () => selectStock(s.code, s.name);

            const colorClass = s.pct > 0 ? 'text-up' : (s.pct < 0 ? 'text-down' : 'text-gray-400');
            div.innerHTML = `
                <div>
                    <div class="text-yellow-500 font-bold">${s.name}</div>
                    <div class="text-xxs text-gray-500">${s.code}</div>
                </div>
                <div class="text-right ${colorClass} self-center font-mono">${(s.price || 0).toFixed(2)}</div>
                <div class="text-right ${colorClass} self-center font-mono">${(s.pct > 0 ? '+' : '')}${(s.pct || 0)}%</div>
            `;
            container.appendChild(div);
            if (flash) {
                setTimeout(() => div.classList.remove('flash-up', 'flash-down'), 300);
            }
        });
    } catch (e) {
        console.error("Failed to load stock list", e);
    }
}

// --- K-Line Logic ---
let klineChart = null;

let zoomStart = 50;
let zoomEnd = 100;

function initKLineChart() {
    const container = document.getElementById('kline-chart-container');
    if(!container) return;
    
    // ECharts Init
    klineChart = echarts.init(container);
    
    // Bind Period Buttons
    const periods = ['daily', 'weekly', 'monthly', '1min', '5min', '15min', '30min', '60min'];
    periods.forEach(p => {
        const btn = document.querySelector(`[data-period="${p}"]`);
        if(btn) {
            btn.addEventListener('click', () => fetchKLineData(p));
        }
    });

    // Capture Zoom Events to persist zoom level
    klineChart.on('datazoom', function (params) {
        if (params.batch && params.batch.length > 0) {
            zoomStart = params.batch[0].start;
            zoomEnd = params.batch[0].end;
        } else {
            zoomStart = params.start;
            zoomEnd = params.end;
        }
    });

    // Load Default Data (Daily)
    fetchKLineData('daily');
    
    window.addEventListener('resize', () => klineChart.resize());
}

function calculateMA(dayCount, data) {
    var result = [];
    for (var i = 0, len = data.length; i < len; i++) {
        if (i < dayCount) {
            result.push('-');
            continue;
        }
        var sum = 0;
        for (var j = 0; j < dayCount; j++) {
            sum += data[i - j][1]; // Close price is index 1 in OHLC [open, close, low, high]
        }
        result.push((sum / dayCount).toFixed(2));
    }
    return result;
}

function renderKLineChart(data, period = 'daily') {
    if(!klineChart) return;
    
    try {
        // Prepare MarkPoints for Buy/Sell Signals
        const markPoints = [];
        const skillsVisible = typeof getSkillsVisible === 'function' ? getSkillsVisible() : {};
        
        // Helper to check if a skill is visible (default true)
        const isSkillVisible = (id) => skillsVisible[id] !== false;

        if (data.indicators) {
            // Dark Pool Signals (对应: anpan_zijin)
            if (data.indicators.buy_signals && isSkillVisible('anpan_zijin')) {
                data.indicators.buy_signals.forEach((signal, index) => {
                    if (signal) {
                        markPoints.push({
                            name: '暗盘买入',
                            coord: [data.dates[index], data.ohlc[index][2]], // Low price
                            value: '暗盘',
                            itemStyle: { color: '#eab308' } // Yellow-500
                        });
                    }
                });
            }
            
            // Precision Buy Signals (对应: xianren_zhilu - 假设)
            if (data.indicators.precision_buy && isSkillVisible('xianren_zhilu')) {
                data.indicators.precision_buy.forEach((signal, index) => {
                    if (signal) {
                        markPoints.push({
                            name: '精准买入',
                            coord: [data.dates[index], data.ohlc[index][2] * 0.98], // Below Low
                            value: '买进',
                            symbol: 'arrow',
                            symbolRotate: 0, // Point Up
                            itemStyle: { color: '#ef4444' } // Red
                        });
                    }
                });
            }
            
            // Precision Sell Signals (对应: xianren_zhilu - 假设)
            if (data.indicators.precision_sell && isSkillVisible('xianren_zhilu')) {
                data.indicators.precision_sell.forEach((signal, index) => {
                    if (signal) {
                        markPoints.push({
                            name: '精准卖出',
                            coord: [data.dates[index], data.ohlc[index][3] * 1.02], // Above High
                            value: '卖出',
                            symbol: 'arrow',
                            symbolRotate: 180, // Point Down
                            itemStyle: { color: '#22c55e' } // Green
                        });
                    }
                });
            }
            // KangLongYouHui Buy Signals (强庄) (对应: kanlong_youhui)
            if (data.indicators.kanglong_xg && isSkillVisible('kanlong_youhui')) {
                data.indicators.kanglong_xg.forEach((signal, index) => {
                    if (signal) {
                        markPoints.push({
                            name: '强庄突破',
                            coord: [data.dates[index], data.ohlc[index][2] * 0.96], // Below Low
                            value: '强庄',
                            symbol: 'diamond',
                            itemStyle: { color: '#ec4899' } // Pink-500
                        });
                    }
                });
            }

            // KangLongYouHui Sell Signals (亢龙有悔) (对应: kanlong_youhui)
            if (data.indicators.kanglong_sell && isSkillVisible('kanlong_youhui')) {
                data.indicators.kanglong_sell.forEach((signal, index) => {
                    if (signal) {
                        markPoints.push({
                            name: '亢龙有悔',
                            coord: [data.dates[index], data.ohlc[index][3] * 1.04], // Above High
                            value: '悔',
                            symbol: 'pin',
                            itemStyle: { color: '#a855f7' } // Purple-500
                        });
                    }
                });
            }
            
            // XunLongJue Signals (寻龙诀) (对应: xunlongjue)
            if (data.indicators.xunlong_signal && isSkillVisible('xunlongjue')) {
                data.indicators.xunlong_signal.forEach((signal, index) => {
                    if (signal) {
                        markPoints.push({
                            name: '寻龙诀',
                            coord: [data.dates[index], data.ohlc[index][2] * 0.94], // Below Low
                            value: '寻龙',
                            symbol: 'rect', // Rectangle to stand out
                            symbolSize: [12, 12],
                            itemStyle: { color: '#3b82f6' } // Blue-500
                        });
                    }
                });
            }
        }

        const series = [
            {
                name: 'KLine',
                type: 'candlestick',
                data: data.ohlc,
                itemStyle: {
                    color: '#ef4444',
                    color0: '#22c55e',
                    borderColor: '#ef4444',
                    borderColor0: '#22c55e'
                },
                markPoint: {
                    data: markPoints,
                    symbol: 'arrow',
                    symbolSize: 10,
                    label: { offset: [0, 10] }
                }
            },
            {
                name: 'MA5',
                type: 'line',
                data: calculateMA(5, data.ohlc),
                smooth: true,
                lineStyle: { opacity: 0.5, width: 1 }
            }
        ];

        // AMA (对应: jigou_caopan)
        if (isSkillVisible('jigou_caopan')) {
             series.push({
                name: 'AMA',
                type: 'line',
                data: data.indicators ? data.indicators.ama : [],
                smooth: true,
                lineStyle: { 
                    width: 2,
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 1, y2: 0,
                        colorStops: data.indicators && data.indicators.ama_color ? 
                            data.indicators.ama_color.map((c, i, arr) => ({
                                offset: i / (arr.length - 1), 
                                color: c === 1 ? '#ef4444' : '#22c55e'
                            })) 
                            : [{offset: 0, color: '#ef4444'}] 
                    }
                }
            });
        }

        // 启明线/揽月线 (对应: kanlong_youhui)
        if (isSkillVisible('kanlong_youhui')) {
            series.push({
                name: '启明线',
                type: 'line',
                data: data.indicators ? data.indicators.qiming : [],
                lineStyle: { type: 'dashed', color: '#ffffff', width: 1, opacity: 0.5 },
                symbol: 'none'
            });
            series.push({
                name: '揽月线',
                type: 'line',
                data: data.indicators ? data.indicators.lanyue : [],
                lineStyle: { type: 'dashed', color: '#facc15', width: 1, opacity: 0.5 }, // Yellow-400
                symbol: 'none'
            });
        }

        // 游资净买 (对应: youzi_anpan)
        if (isSkillVisible('youzi_anpan')) {
            series.push({
                name: '游资净买',
                type: 'line',
                xAxisIndex: 1,
                yAxisIndex: 1,
                data: data.indicators ? data.indicators.hot_money : [],
                itemStyle: { color: '#f43f5e' }, // Rose-500
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(244, 63, 94, 0.5)' },
                        { offset: 1, color: 'rgba(244, 63, 94, 0.0)' }
                    ])
                }
            });
        }

        const option = {
            backgroundColor: '#0f172a', // Slate-900
            animation: false,
            graphic: data.fallback ? [{
                type: 'group',
                left: 'center',
                top: '15%',
                children: [
                    {
                        type: 'rect',
                        z: 100,
                        left: 'center',
                        top: 'middle',
                        shape: { width: 200, height: 30 },
                        style: { fill: 'rgba(245, 158, 11, 0.2)' }
                    },
                    {
                        type: 'text',
                        z: 100,
                        left: 'center',
                        top: 'middle',
                        style: { text: '⚠️ 模拟数据演示模式', fill: '#fbbf24', fontSize: 14, fontWeight: 'bold' }
                    }
                ]
            }] : [],
            grid: [
                { left: 50, right: 10, top: 20, height: '60%' }, // Main K-Line
                { left: 50, right: 10, top: '82%', height: '15%' }  // Indicator (Hot Money)
            ],
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross' },
                backgroundColor: 'rgba(30, 41, 59, 0.9)',
                borderColor: '#334155',
                textStyle: { color: '#cbd5e1' }
            },
            axisPointer: { link: { xAxisIndex: 'all' } },
            xAxis: [
                { 
                    type: 'category', 
                    data: data.dates,
                    axisLine: { lineStyle: { color: '#475569' } },
                    axisLabel: { show: false } // Hide label on top chart
                },
                { 
                    type: 'category', 
                    gridIndex: 1,
                    data: data.dates,
                    axisLine: { lineStyle: { color: '#475569' } },
                    axisLabel: { color: '#94a3b8' }
                }
            ],
            yAxis: [
                { 
                    scale: true, 
                    splitLine: { lineStyle: { color: '#1e293b' } },
                    axisLabel: { color: '#94a3b8' }
                },
                { 
                    gridIndex: 1, 
                    scale: true, 
                    splitLine: { show: false },
                    axisLabel: { show: false }
                }
            ],
            dataZoom: [
                { type: 'inside', xAxisIndex: [0, 1], start: zoomStart, end: zoomEnd },
                { type: 'slider', xAxisIndex: [0, 1], bottom: 0, height: 20, borderColor: '#334155', start: zoomStart, end: zoomEnd }
            ],
            series: series
        };
        
        // Use notMerge: true to ensure hidden series are removed
        klineChart.setOption(option, true);
        klineChart.hideLoading();
    } catch(e) {
        console.error("K-Line Render Failed", e);
    }
}

window.renderKLineChart = renderKLineChart;

async function fetchKLineData(period = 'daily') {
    if(!klineChart) return;

    // Show Loading
    klineChart.showLoading({ color: '#38bdf8', maskColor: 'rgba(15, 23, 42, 0.2)' });

    try {
        const stockCode = window.currentStockCode || '600519'; // Default to Moutai
        
        const resp = await fetch(`/api/market/kline?code=${stockCode}&period=${period}`);
        const data = await resp.json().catch(() => ({}));
        if (resp.status === 401 && typeof showAuthModal === 'function') { showAuthModal(); klineChart.hideLoading(); return; }
        if (resp.status === 503) { if (typeof showToast === 'function') showToast(data.error || '行情数据源暂时不可用'); klineChart.hideLoading(); return; }
        const hasData = data && Array.isArray(data.dates) && data.dates.length > 0 && Array.isArray(data.ohlc) && data.ohlc.length > 0;
        if (resp.status === 404 || data.error || !hasData) {
            if (typeof showToast === 'function') showToast(data.error || 'K线数据加载失败，请检查网络或稍后重试');
            klineChart.setOption({
                backgroundColor: '#0f172a',
                graphic: {
                    type: 'text',
                    left: 'center',
                    top: 'middle',
                    style: { text: 'K线数据加载失败\n请检查网络或稍后重试', fill: '#94a3b8', fontSize: 14 }
                }
            });
            klineChart.hideLoading();
            return;
        }

        // Cache full data for toggling skills without re-fetching
        window._cachedKLineData = data;
        _lastKlineData = { dates: data.dates || [], ohlc: data.ohlc || [], volumes: (data.volumes || []).map(Number) };
        
        renderKLineChart(data, period);

        // Highlight active button
        document.querySelectorAll('[data-period]').forEach(b => b.classList.remove('active'));
        const activeBtn = document.querySelector(`[data-period="${period}"]`);
        if(activeBtn) activeBtn.classList.add('active');

        updateIndicatorChart(window.currentStockCode || '600519');

    } catch(e) {
        console.error("K-Line Fetch Failed", e);
        if (typeof showToast === 'function') showToast('K线加载失败，请稍后重试');
        if (klineChart) {
            klineChart.setOption({
                backgroundColor: '#0f172a',
                graphic: {
                    type: 'text',
                    left: 'center',
                    top: 'middle',
                    style: { text: 'K线数据加载失败\n请检查网络或稍后重试', fill: '#94a3b8', fontSize: 14 }
                }
            });
            klineChart.hideLoading();
        }
    }
}
