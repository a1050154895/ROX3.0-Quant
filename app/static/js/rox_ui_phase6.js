/**
 * ROX 3.0 Phase 6 UI Logic
 * Handles Macro Dashboard, News Feed, Concept/Theme Visualization, and Beginner Mode.
 */

// ================= BEGINNER MODE =================
window.isBeginnerMode = false;

window.toggleBeginnerMode = function () {
    window.isBeginnerMode = !window.isBeginnerMode;
    const btn = document.getElementById('btn-beginner-mode');
    const proWorkspace = document.getElementById('pro-workspace');
    const beginnerPanel = document.getElementById('beginner-panel');

    // Fallback for proWorkspace if id not found (in case HTML edit failed or cached)
    const targetPro = proWorkspace || document.querySelector('.grid-cols-\\[260px_1fr_280px\\]');

    if (window.isBeginnerMode) {
        // Switch to Beginner
        if (targetPro) targetPro.classList.add('hidden');
        if (beginnerPanel) beginnerPanel.classList.remove('hidden');
        beginnerPanel.style.display = 'flex'; // Force flex layout

        btn.classList.add('bg-emerald-500', 'text-white');
        btn.classList.remove('bg-gradient-to-r', 'text-emerald-400');
        btn.innerHTML = '<i class="fas fa-check"></i> 小白模式 (ON)';

        showToast('已切换至小白模式：简单、直观、智能');
    } else {
        // Switch back to Pro
        if (targetPro) targetPro.classList.remove('hidden');
        if (beginnerPanel) beginnerPanel.classList.add('hidden');
        beginnerPanel.style.display = 'none';

        btn.classList.remove('bg-emerald-500', 'text-white');
        btn.classList.add('bg-gradient-to-r', 'text-emerald-400');
        btn.innerHTML = '<i class="fas fa-leaf"></i> 小白模式';

        showToast('已切回专业模式');
    }
};

// ================= MACRO DASHBOARD =================
let macroChart = null;

async function loadMacroDashboard() {
    const container = document.querySelector('.rox1-macro-inner');
    if (!container) return;

    container.innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div class="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <div class="text-xs text-slate-500 mb-1">GDP (季度)</div>
                <div id="macro-gdp" class="text-xl font-bold text-white">--</div>
            </div>
             <div class="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <div class="text-xs text-slate-500 mb-1">CPI (通胀)</div>
                <div id="macro-cpi" class="text-xl font-bold text-slate-300">--</div>
            </div>
             <div class="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <div class="text-xs text-slate-500 mb-1">PPI (工业)</div>
                <div id="macro-ppi" class="text-xl font-bold text-slate-300">--</div>
            </div>
             <div class="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <div class="text-xs text-slate-500 mb-1">M1-M2 剪刀差</div>
                <div id="macro-scissors" class="text-xl font-bold text-slate-300">--</div>
            </div>
        </div>
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="bg-slate-800/30 p-4 rounded-xl border border-slate-700/50">
                <h3 class="font-bold text-slate-300 mb-4">货币供应量剪刀差 (Liquidity)</h3>
                <div id="macro-chart-money" class="h-64 w-full"></div>
                <p class="text-xs text-slate-500 mt-2">* 剪刀差走阔通常对应股市牛市，收窄则流动性紧缩。</p>
            </div>
             <div class="bg-slate-800/30 p-4 rounded-xl border border-slate-700/50">
                <h3 class="font-bold text-slate-300 mb-4">制造业 PMI 趋势</h3>
                <div id="macro-chart-pmi" class="h-64 w-full"></div>
                <p class="text-xs text-slate-500 mt-2">* 50 为荣枯线，高于 50 代表经济扩张。</p>
            </div>
        </div>
    `;

    try {
        const res = await fetch('/api/macro/indicators');
        const data = await res.json();

        // Render Cards
        document.getElementById('macro-gdp').innerText = `${data.gdp.value}% (${data.gdp.quarter})`;
        document.getElementById('macro-cpi').innerText = `${data.cpi.value}%`;
        document.getElementById('macro-ppi').innerText = `${data.ppi.value}%`;

        if (data.money_supply && data.money_supply.length > 0) {
            document.getElementById('macro-scissors').innerText = `${data.money_supply[0].scissors}%`;
            document.getElementById('macro-scissors').className = data.money_supply[0].scissors > 0 ? "text-xl font-bold text-up" : "text-xl font-bold text-down";

            initMoneyChart(data.money_supply);
        }

        if (data.pmi && data.pmi.length > 0) {
            initPMIChart(data.pmi);
        }
    } catch (e) {
        console.error("Macro Load Error", e);
    }
}

function initMoneyChart(data) {
    const chartDom = document.getElementById('macro-chart-money');
    const myChart = echarts.init(chartDom);
    const dates = data.map(i => i.date).reverse();
    const m1 = data.map(i => i.m1_yoy).reverse();
    const m2 = data.map(i => i.m2_yoy).reverse();
    const scissors = data.map(i => i.scissors).reverse();

    const option = {
        tooltip: { trigger: 'axis' },
        legend: { data: ['M1同比', 'M2同比', '剪刀差'], textStyle: { color: '#94a3b8' } },
        grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
        xAxis: { type: 'category', data: dates, axisLine: { lineStyle: { color: '#334155' } } },
        yAxis: { type: 'value', splitLine: { lineStyle: { color: '#1e293b' } } },
        series: [
            { name: 'M1同比', type: 'line', data: m1, smooth: true },
            { name: 'M2同比', type: 'line', data: m2, smooth: true },
            {
                name: '剪刀差', type: 'bar', data: scissors, itemStyle: {
                    color: (p) => p.value > 0 ? '#ef4444' : '#22c55e'
                }
            }
        ]
    };
    myChart.setOption(option);
}

function initPMIChart(data) {
    const chartDom = document.getElementById('macro-chart-pmi');
    const myChart = echarts.init(chartDom);
    const dates = data.map(i => i.date).reverse();
    const manu = data.map(i => i.manufacturing).reverse();

    const option = {
        tooltip: { trigger: 'axis' },
        legend: { data: ['制造业PMI'], textStyle: { color: '#94a3b8' } },
        grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
        xAxis: { type: 'category', data: dates, axisLine: { lineStyle: { color: '#334155' } } },
        yAxis: { type: 'value', min: 40, max: 60, splitLine: { lineStyle: { color: '#1e293b' } } },
        visualMap: {
            show: false,
            pieces: [{ gt: 50, color: '#ef4444' }, { lte: 50, color: '#22c55e' }],
            outOfRange: { color: '#999' }
        },
        series: [
            {
                name: '制造业PMI', type: 'line', data: manu,
                markLine: {
                    data: [{ yAxis: 50, name: '荣枯线' }],
                    lineStyle: { color: '#fbbf24', type: 'dashed' }
                }
            }
        ]
    };
    myChart.setOption(option);
}


// ================= NEWS & NOTICES =================
async function loadNewsFeed() {
    const container = document.getElementById('market-news-list');
    if (!container) return;

    try {
        const res = await fetch('/api/info/news?limit=10');
        const news = await res.json();

        container.innerHTML = news.map(item => `
            <div class="flex gap-2 items-start group cursor-pointer hover:bg-slate-800/30 p-2 rounded transition-colors" onclick="window.open('${item.url}', '_blank')">
                <span class="text-xs text-slate-500 whitespace-nowrap mt-0.5">${item.time.split(' ')[1]}</span>
                <div>
                     <div class="text-slate-300 text-sm group-hover:text-sky-400 transition-colors">${item.title}</div>
                </div>
            </div>
        `).join('');
    } catch (e) {
        console.error("News Load Error", e);
    }
}

// ================= CONCEPT THEMES (IT Juzi Proxy) =================
async function loadConceptThemes() {
    const container = document.getElementById('sector-heatmap'); // We'll append here or replace
    if (!container) return;

    // We append a "Concept" section if not exists
    let conceptContainer = document.getElementById('concept-themes-list');
    if (!conceptContainer) {
        const wrap = document.createElement('div');
        wrap.className = "w-full mt-4 border-t border-slate-800 pt-4";
        wrap.innerHTML = `
            <h4 class="text-xs font-bold text-slate-400 mb-2">一级概念/独角兽资金流 (Proxy)</h4>
            <div id="concept-themes-list" class="flex flex-wrap gap-2"></div>
        `;
        container.parentElement.appendChild(wrap);
        conceptContainer = document.getElementById('concept-themes-list');
    }

    try {
        const res = await fetch('/api/market/concepts?limit=8');
        const concepts = await res.json();

        conceptContainer.innerHTML = concepts.map(c => `
             <div class="px-3 py-1.5 bg-slate-800 rounded border border-slate-700 flex flex-col items-center min-w-[80px]">
                <span class="text-xs text-slate-300">${c.name}</span>
                <span class="text-xs font-bold ${c.net_inflow > 0 ? 'text-up' : 'text-down'}">
                    ${(c.net_inflow / 10000).toFixed(1)}亿
                </span>
            </div>
        `).join('');
    } catch (e) {
        console.error("Concept Load Error", e);
    }
}

// Hook into View Switch
const oldSwitchModeP6 = window.switchMode;
window.switchMode = function (mode) {
    if (oldSwitchModeP6) oldSwitchModeP6(mode);

    if (mode === 'macro') {
        loadMacroDashboard();
    }
    if (mode === 'market') {
        loadNewsFeed();
        loadConceptThemes();
    }
}
