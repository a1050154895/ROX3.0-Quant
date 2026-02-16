/**
 * ROX QUANT 增强功能模块
 * 包含: 龙虎榜、板块轮动、价格预警、数据导出、K线形态、快捷键
 */

// ============ 初始化 ============
document.addEventListener('DOMContentLoaded', () => {
    initEnhancedFeatures();
});

function initEnhancedFeatures() {
    initKeyboardShortcuts();
    initAlertChecker();
    console.log('[ROX] 增强功能模块已加载');
}

// ============ 龙虎榜 ============
async function loadDragonTiger() {
    try {
        const res = await fetch('/api/market/dragon-tiger');
        const data = await res.json();
        return data.data || [];
    } catch (e) {
        console.error('龙虎榜加载失败:', e);
        return [];
    }
}

function renderDragonTigerCard(container, data) {
    if (!container) return;

    const html = `
        <div class="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-semibold text-white flex items-center">
                    <i class="fas fa-dragon text-orange-400 mr-2"></i>
                    龙虎榜
                </h3>
                <button onclick="exportDragonTiger()" class="text-xs text-slate-400 hover:text-white">
                    <i class="fas fa-download mr-1"></i>导出
                </button>
            </div>
            <div class="space-y-2 max-h-64 overflow-y-auto">
                ${data.slice(0, 10).map(item => `
                    <div class="flex items-center justify-between py-2 border-b border-slate-700/50 hover:bg-slate-700/30 cursor-pointer"
                         onclick="searchStock('${item.code}')">
                        <div class="flex-1">
                            <span class="text-white text-sm">${item.name}</span>
                            <span class="text-slate-400 text-xs ml-2">${item.code}</span>
                        </div>
                        <div class="text-right">
                            <span class="${item.change_pct >= 0 ? 'text-red-400' : 'text-green-400'} text-sm">
                                ${item.change_pct >= 0 ? '+' : ''}${item.change_pct.toFixed(2)}%
                            </span>
                            <span class="text-slate-400 text-xs block">${item.reason}</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    container.innerHTML = html;
}

// ============ 板块轮动 ============
async function loadSectorRotation() {
    try {
        const res = await fetch('/api/market/rotation');
        const data = await res.json();
        return data.data || [];
    } catch (e) {
        console.error('板块轮动加载失败:', e);
        return [];
    }
}

function renderRotationCard(container, data) {
    if (!container) return;

    const html = `
        <div class="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-semibold text-white flex items-center">
                    <i class="fas fa-sync-alt text-purple-400 mr-2"></i>
                    板块轮动
                </h3>
            </div>
            <div class="space-y-2 max-h-64 overflow-y-auto">
                ${data.slice(0, 10).map((item, idx) => `
                    <div class="flex items-center justify-between py-2 border-b border-slate-700/50">
                        <div class="flex items-center">
                            <span class="w-6 h-6 rounded-full ${idx < 3 ? 'bg-red-500/20 text-red-400' : 'bg-slate-600/20 text-slate-400'} 
                                  flex items-center justify-center text-xs mr-2">${idx + 1}</span>
                            <span class="text-white text-sm">${item.name}</span>
                        </div>
                        <div class="text-right">
                            <span class="${item.pct_1d >= 0 ? 'text-red-400' : 'text-green-400'} text-sm">
                                ${item.pct_1d >= 0 ? '+' : ''}${item.pct_1d.toFixed(2)}%
                            </span>
                            <span class="text-slate-400 text-xs block">5日: ${item.pct_5d >= 0 ? '+' : ''}${item.pct_5d}%</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    container.innerHTML = html;
}

// ============ 价格预警 ============
let alertCheckInterval = null;

function initAlertChecker() {
    // 每60秒检查一次预警
    alertCheckInterval = setInterval(checkAlerts, 60000);
}

async function checkAlerts() {
    try {
        const res = await fetch('/api/alerts/check', { method: 'POST' });
        const data = await res.json();

        if (data.triggered && data.triggered.length > 0) {
            data.triggered.forEach(alert => {
                showAlertNotification(alert);
            });
        }
    } catch (e) {
        console.error('预警检查失败:', e);
    }
}

function showAlertNotification(alert) {
    // 浏览器通知
    if (Notification.permission === 'granted') {
        const typeText = {
            'price_above': '价格突破上限',
            'price_below': '价格跌破下限',
            'change_pct_above': '涨幅超过',
            'change_pct_below': '跌幅超过'
        };

        new Notification(`🔔 ${alert.name || alert.symbol} 预警触发`, {
            body: `${typeText[alert.alert_type]}: ${alert.value}\n当前价格: ${alert.current_price}`,
            icon: '/static/icons/icon-192x192.png'
        });
    }

    // 页面内通知
    showToast(`预警触发: ${alert.name || alert.symbol}`, 'warning');
}

async function createAlert(symbol, name, alertType, value) {
    try {
        const res = await fetch('/api/alerts/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol, name, alert_type: alertType, value })
        });
        const data = await res.json();
        if (data.success) {
            showToast('预警创建成功', 'success');
        }
        return data;
    } catch (e) {
        console.error('创建预警失败:', e);
        showToast('创建预警失败', 'error');
    }
}

function openAlertModal(symbol = '', name = '') {
    const modal = document.getElementById('alert-modal');
    if (!modal) {
        createAlertModal();
    }

    document.getElementById('alert-symbol').value = symbol;
    document.getElementById('alert-name').value = name;
    document.getElementById('alert-modal').style.display = 'flex';
}

function closeAlertModal() {
    document.getElementById('alert-modal').style.display = 'none';
}

function createAlertModal() {
    const html = `
        <div id="alert-modal" class="fixed inset-0 z-[100] hidden items-center justify-center bg-black/50 backdrop-blur-sm" style="display: none;">
            <div class="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-96 p-6">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold text-white">
                        <i class="fas fa-bell text-yellow-400 mr-2"></i>设置价格预警
                    </h3>
                    <button onclick="closeAlertModal()" class="text-slate-400 hover:text-white">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="space-y-4">
                    <div>
                        <label class="text-sm text-slate-400 block mb-1">股票代码</label>
                        <input type="text" id="alert-symbol" class="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-white" placeholder="如: 600519">
                    </div>
                    <div>
                        <label class="text-sm text-slate-400 block mb-1">股票名称</label>
                        <input type="text" id="alert-name" class="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-white" placeholder="如: 贵州茅台">
                    </div>
                    <div>
                        <label class="text-sm text-slate-400 block mb-1">预警类型</label>
                        <select id="alert-type" class="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-white">
                            <option value="price_above">价格突破上限</option>
                            <option value="price_below">价格跌破下限</option>
                            <option value="change_pct_above">涨幅超过</option>
                            <option value="change_pct_below">跌幅超过</option>
                        </select>
                    </div>
                    <div>
                        <label class="text-sm text-slate-400 block mb-1">预警值</label>
                        <input type="number" id="alert-value" class="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-white" placeholder="如: 1800 或 5">
                    </div>
                    <button onclick="submitAlert()" class="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg py-2 font-medium hover:opacity-90 transition">
                        创建预警
                    </button>
                </div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', html);
}

async function submitAlert() {
    const symbol = document.getElementById('alert-symbol').value;
    const name = document.getElementById('alert-name').value;
    const type = document.getElementById('alert-type').value;
    const value = parseFloat(document.getElementById('alert-value').value);

    if (!symbol || !value) {
        showToast('请填写完整信息', 'error');
        return;
    }

    await createAlert(symbol, name, type, value);
    closeAlertModal();
}

// ============ 数据导出 ============
function exportWatchlist(format = 'csv') {
    window.open(`/api/export/watchlist?format=${format}`, '_blank');
}

function exportMarketData(code, days = 60, format = 'csv') {
    window.open(`/api/export/market-data/${code}?days=${days}&format=${format}`, '_blank');
}

function exportDragonTiger(format = 'csv') {
    window.open(`/api/export/dragon-tiger?format=${format}`, '_blank');
}

// ============ K线形态 ============
async function loadPatterns(code) {
    try {
        const res = await fetch(`/api/market/patterns/${code}`);
        const data = await res.json();
        return data.patterns || [];
    } catch (e) {
        console.error('形态识别失败:', e);
        return [];
    }
}

function renderPatternsCard(container, patterns) {
    if (!container || !patterns.length) return;

    const typeColors = {
        '看涨': 'text-red-400 bg-red-500/10',
        '看跌': 'text-green-400 bg-green-500/10',
        '反转信号': 'text-yellow-400 bg-yellow-500/10'
    };

    const html = `
        <div class="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
            <h3 class="text-lg font-semibold text-white flex items-center mb-4">
                <i class="fas fa-chart-bar text-cyan-400 mr-2"></i>
                K线形态识别
            </h3>
            <div class="space-y-2">
                ${patterns.map(p => `
                    <div class="flex items-center justify-between py-2 border-b border-slate-700/50">
                        <div>
                            <span class="text-white text-sm font-medium">${p.pattern}</span>
                            <span class="text-slate-400 text-xs ml-2">${p.date}</span>
                        </div>
                        <div class="flex items-center space-x-2">
                            <span class="px-2 py-1 rounded text-xs ${typeColors[p.type] || 'text-slate-400 bg-slate-600/20'}">${p.type}</span>
                            <span class="text-slate-500 text-xs">${p.reliability}</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    container.innerHTML = html;
}

// ============ 键盘快捷键 ============
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // 忽略输入框
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        switch (e.key) {
            case '?':
                showShortcutHelp();
                break;
            case 'Escape':
                closeAllModals();
                break;
            case 's':
            case 'S':
                if (!e.ctrlKey && !e.metaKey) {
                    e.preventDefault();
                    focusSearch();
                }
                break;
            case '1':
                if (e.altKey) switchToView('market');
                break;
            case '2':
                if (e.altKey) switchToView('stock');
                break;
            case '3':
                if (e.altKey) switchToView('strategy');
                break;
            case 'a':
            case 'A':
                if (e.altKey) openAlertModal();
                break;
        }
    });
}

function showShortcutHelp() {
    const existing = document.getElementById('shortcut-help');
    if (existing) {
        existing.remove();
        return;
    }

    const html = `
        <div id="shortcut-help" class="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm" onclick="this.remove()">
            <div class="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-96 p-6" onclick="event.stopPropagation()">
                <h3 class="text-lg font-semibold text-white mb-4">
                    <i class="fas fa-keyboard text-blue-400 mr-2"></i>键盘快捷键
                </h3>
                <div class="space-y-3 text-sm">
                    <div class="flex justify-between"><span class="text-slate-400">搜索股票</span><kbd class="bg-slate-700 px-2 py-1 rounded text-white">S</kbd></div>
                    <div class="flex justify-between"><span class="text-slate-400">市场看板</span><kbd class="bg-slate-700 px-2 py-1 rounded text-white">Alt + 1</kbd></div>
                    <div class="flex justify-between"><span class="text-slate-400">个股诊断</span><kbd class="bg-slate-700 px-2 py-1 rounded text-white">Alt + 2</kbd></div>
                    <div class="flex justify-between"><span class="text-slate-400">策略工坊</span><kbd class="bg-slate-700 px-2 py-1 rounded text-white">Alt + 3</kbd></div>
                    <div class="flex justify-between"><span class="text-slate-400">添加预警</span><kbd class="bg-slate-700 px-2 py-1 rounded text-white">Alt + A</kbd></div>
                    <div class="flex justify-between"><span class="text-slate-400">关闭弹窗</span><kbd class="bg-slate-700 px-2 py-1 rounded text-white">Esc</kbd></div>
                    <div class="flex justify-between"><span class="text-slate-400">显示帮助</span><kbd class="bg-slate-700 px-2 py-1 rounded text-white">?</kbd></div>
                </div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', html);
}

function closeAllModals() {
    document.querySelectorAll('[id$="-modal"]').forEach(modal => {
        modal.style.display = 'none';
    });
    document.getElementById('shortcut-help')?.remove();
}

function focusSearch() {
    const searchInput = document.querySelector('input[type="text"][placeholder*="搜索"]') ||
        document.querySelector('input[type="text"][placeholder*="代码"]') ||
        document.querySelector('#search-input');
    if (searchInput) {
        searchInput.focus();
        searchInput.select();
    }
}

function switchToView(view) {
    // 调用已有的视图切换函数
    if (typeof window.showView === 'function') {
        window.showView(view);
    }
}

// ============ 通用工具 ============
// 通知队列管理
let toastQueue = [];
const MAX_TOASTS = 5;

function showToast(message, type = 'info', title = '', duration = 3000, position = 'top-right') {
    const colors = {
        success: {
            bg: 'bg-emerald-900/90 border-emerald-500/30',
            text: 'text-emerald-400',
            icon: 'fa-check-circle',
            border: 'border-l-4 border-emerald-500'
        },
        error: {
            bg: 'bg-red-900/90 border-red-500/30',
            text: 'text-red-400',
            icon: 'fa-exclamation-circle',
            border: 'border-l-4 border-red-500'
        },
        warning: {
            bg: 'bg-amber-900/90 border-amber-500/30',
            text: 'text-amber-400',
            icon: 'fa-exclamation-triangle',
            border: 'border-l-4 border-amber-500'
        },
        info: {
            bg: 'bg-sky-900/90 border-sky-500/30',
            text: 'text-sky-400',
            icon: 'fa-info-circle',
            border: 'border-l-4 border-sky-500'
        }
    };

    const config = colors[type] || colors.info;
    
    // 创建通知元素
    const toast = document.createElement('div');
    toast.className = `fixed z-[200] ${config.bg} ${config.text} ${config.border} px-4 py-3 rounded-lg shadow-lg flex items-start space-x-3 transition-all duration-300 scale-in max-w-sm w-full`;
    
    // 设置位置
    const positionClasses = {
        'top-right': 'top-4 right-4',
        'top-left': 'top-4 left-4',
        'bottom-right': 'bottom-4 right-4',
        'bottom-left': 'bottom-4 left-4',
        'top-center': 'top-4 left-1/2 transform -translate-x-1/2',
        'bottom-center': 'bottom-4 left-1/2 transform -translate-x-1/2'
    };
    
    Object.assign(toast.style, {
        ...positionClasses[position] ? {} : positionClasses['top-right'],
        animation: 'slideInRight 0.3s ease forwards'
    });
    
    // 添加位置类
    if (positionClasses[position]) {
        toast.classList.add(...positionClasses[position].split(' '));
    }
    
    // 构建通知内容
    const content = `
        <div class="text-lg ${config.text}">
            <i class="fas ${config.icon}"></i>
        </div>
        <div class="flex-1 min-w-0">
            ${title ? `<div class="font-bold text-sm mb-1">${title}</div>` : ''}
            <div class="text-sm">${message}</div>
        </div>
        <button type="button" class="text-slate-400 hover:text-white transition-colors ml-2 flex-shrink-0" onclick="this.closest('div').remove();">
            <i class="fas fa-times text-xs"></i>
        </button>
    `;
    
    toast.innerHTML = content;
    document.body.appendChild(toast);
    
    // 添加到队列
    toastQueue.push(toast);
    
    // 限制通知数量
    if (toastQueue.length > MAX_TOASTS) {
        const oldestToast = toastQueue.shift();
        if (oldestToast) {
            oldestToast.style.animation = 'fadeOut 0.3s ease forwards';
            setTimeout(() => oldestToast.remove(), 300);
        }
    }
    
    // 自动关闭
    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s ease forwards';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
                // 从队列中移除
                toastQueue = toastQueue.filter(t => t !== toast);
            }
        }, 300);
    }, duration);
    
    // 返回通知元素，以便外部可以手动控制
    return toast;
}

// 添加动画样式
if (!document.getElementById('toast-animations')) {
    const style = document.createElement('style');
    style.id = 'toast-animations';
    style.innerHTML = `
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(100%);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-100%);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeOut {
            from {
                opacity: 1;
                transform: translateX(0);
            }
            to {
                opacity: 0;
                transform: translateX(100%);
            }
        }
        
        .scale-in {
            animation: scaleIn 0.3s ease forwards;
        }
        
        @keyframes scaleIn {
            from {
                opacity: 0;
                transform: scale(0.9);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
    `;
    document.head.appendChild(style);
}

// 请求通知权限
if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
}

// 导出到全局
window.loadDragonTiger = loadDragonTiger;
window.renderDragonTigerCard = renderDragonTigerCard;
window.loadSectorRotation = loadSectorRotation;
window.renderRotationCard = renderRotationCard;
window.openAlertModal = openAlertModal;
window.closeAlertModal = closeAlertModal;
window.exportWatchlist = exportWatchlist;
window.exportMarketData = exportMarketData;
window.exportDragonTiger = exportDragonTiger;
window.loadPatterns = loadPatterns;
window.renderPatternsCard = renderPatternsCard;
window.showShortcutHelp = showShortcutHelp;
window.showToast = showToast;
