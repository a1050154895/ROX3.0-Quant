/**
 * ROX 3.0 UI Logic (Phase 5)
 * Handles Market Switching, Strategy Store, Profile Management, and Sync.
 */

// ================= MARKET SWITCHER =================
window.currentMarket = 'CN'; // CN, US, CRYPTO

window.switchMarket = function (market) {
    window.currentMarket = market;

    // 1. Update Buttons
    const buttons = ['CN', 'US', 'CRYPTO'];
    buttons.forEach(m => {
        const btn = document.getElementById(`market-btn-${m}`);
        if (m === market) {
            btn.className = "px-3 py-1 rounded text-xs font-bold text-white bg-sky-600 transition-colors";
        } else {
            btn.className = "px-3 py-1 rounded text-xs font-bold text-slate-400 hover:text-white hover:bg-slate-700 transition-colors";
        }
    });

    // 2. Update Index Bar (Mock Data for now, ideally fetch from backend)
    const bar = document.getElementById('market-index-bar');
    if (market === 'CN') {
        bar.innerHTML = `
            <div><span class="text-slate-400">上证</span> <span class="text-up font-bold">3050.23</span> <span class="text-up text-xs">+0.5%</span></div>
            <div><span class="text-slate-400">创业</span> <span class="text-down font-bold">2100.12</span> <span class="text-down text-xs">-0.2%</span></div>
        `;
    } else if (market === 'US') {
        bar.innerHTML = `
            <div><span class="text-slate-400">DJIA</span> <span class="text-up font-bold">38500.10</span> <span class="text-up text-xs">+0.8%</span></div>
            <div><span class="text-slate-400">NDX</span> <span class="text-up font-bold">17800.50</span> <span class="text-up text-xs">+1.2%</span></div>
        `;
    } else if (market === 'CRYPTO') {
        bar.innerHTML = `
            <div><span class="text-slate-400">BTC</span> <span class="text-up font-bold">95,400</span> <span class="text-up text-xs">+3.5%</span></div>
            <div><span class="text-slate-400">ETH</span> <span class="text-up font-bold">3,650</span> <span class="text-up text-xs">+2.1%</span></div>
        `;
    }

    // 3. Trigger view update if on market board
    if (document.getElementById('view-market').classList.contains('view-active')) {
        // Refresh charts logic could go here
        showToast(`已切换至 ${market} 市场数据`);
    }
};

// Initialize with CN
document.addEventListener('DOMContentLoaded', () => {
    switchMarket('CN');
});


// ================= STRATEGY STORE =================
let storeLoaded = false;

// Hook into the existing switchMode to load data when Store is opened
const originalSwitchMode = window.switchMode;
window.switchMode = function (mode) {
    originalSwitchMode(mode);
    if (mode === 'store') {
        if (!storeLoaded) {
            initMarketplaceFilters();
            loadStrategyMarketplace();
        }
    }
};

// 初始化策略超市搜索和筛选功能
function initMarketplaceFilters() {
    const storeSection = document.getElementById('store-section');
    if (!storeSection) return;
    
    // 找到store-list容器
    const storeList = document.getElementById('store-list');
    if (!storeList) return;
    
    // 创建搜索和筛选组件
    const filtersHTML = `
        <div class="grid grid-cols-1 gap-4 mb-6">
            <div class="flex flex-col gap-3">
                <!-- 搜索框 -->
                <div class="w-full">
                    <div class="relative">
                        <input 
                            type="text" 
                            id="strategy-search" 
                            placeholder="搜索策略名称或描述..." 
                            class="w-full px-4 py-2 pl-10 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent text-slate-300"
                        >
                        <i class="fas fa-search absolute left-3 top-3 text-slate-500"></i>
                    </div>
                </div>
                
                <!-- 筛选选项行 1 -->
                <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    <!-- 分类筛选 -->
                    <div>
                        <select id="strategy-category" class="w-full px-3 py-2 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent text-slate-300 text-sm">
                            <option value="">所有分类</option>
                            <option value="趋势跟踪">趋势跟踪</option>
                            <option value="均值回归">均值回归</option>
                            <option value="突破策略">突破策略</option>
                            <option value="机器学习">机器学习</option>
                            <option value="高股息">高股息</option>
                            <option value="小市值">小市值</option>
                            <option value="指数增强">指数增强</option>
                        </select>
                    </div>
                    
                    <!-- 评分筛选 -->
                    <div>
                        <select id="strategy-rating" class="w-full px-3 py-2 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent text-slate-300 text-sm">
                            <option value="">所有评分</option>
                            <option value="4">4星及以上</option>
                            <option value="3">3星及以上</option>
                            <option value="2">2星及以上</option>
                        </select>
                    </div>
                    
                    <!-- 风险等级筛选 -->
                    <div class="sm:col-span-1 col-span-2">
                        <select id="strategy-risk" class="w-full px-3 py-2 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent text-slate-300 text-sm">
                            <option value="">所有风险</option>
                            <option value="1">低风险</option>
                            <option value="2">中低风险</option>
                            <option value="3">中风险</option>
                            <option value="4">中高风险</option>
                            <option value="5">高风险</option>
                        </select>
                    </div>
                </div>
                
                <!-- 筛选选项行 2 -->
                <div class="flex flex-col sm:flex-row gap-3">
                    <!-- 排序选项 -->
                    <div class="flex-1">
                        <select id="strategy-sort" class="w-full px-3 py-2 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent text-slate-300 text-sm">
                            <option value="install_count">热门程度</option>
                            <option value="rating">评分最高</option>
                            <option value="downloads">下载最多</option>
                            <option value="created_at">最新发布</option>
                            <option value="return_rate">收益率最高</option>
                            <option value="sharpe_ratio">夏普比率最高</option>
                        </select>
                    </div>
                    
                    <!-- 重置按钮 -->
                    <div>
                        <button id="reset-filters" class="w-full sm:w-auto px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded-lg transition-colors text-sm">
                            <i class="fas fa-redo mr-1"></i>重置
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // 在store-list前插入搜索筛选组件
    storeList.insertAdjacentHTML('beforebegin', filtersHTML);
    
    // 添加事件监听器
    document.getElementById('strategy-search').addEventListener('input', debounce(handleSearch, 300));
    document.getElementById('strategy-category').addEventListener('change', handleSearch);
    document.getElementById('strategy-rating').addEventListener('change', handleSearch);
    document.getElementById('strategy-risk').addEventListener('change', handleSearch);
    document.getElementById('strategy-sort').addEventListener('change', handleSearch);
    document.getElementById('reset-filters').addEventListener('click', resetFilters);
}

// 防抖函数
function debounce(func, wait) {
    let timeout;
    return function() {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, arguments), wait);
    };
}

// 处理搜索和筛选
function handleSearch() {
    const search = document.getElementById('strategy-search').value;
    const category = document.getElementById('strategy-category').value;
    const minRating = document.getElementById('strategy-rating').value;
    const maxRisk = document.getElementById('strategy-risk').value;
    const sortBy = document.getElementById('strategy-sort').value;
    
    loadStrategyMarketplace(search, {
        minRating: minRating || undefined,
        maxRisk: maxRisk || undefined,
        category: category || undefined,
        sortBy: sortBy
    });
}

// 重置筛选条件
function resetFilters() {
    document.getElementById('strategy-search').value = '';
    document.getElementById('strategy-category').value = '';
    document.getElementById('strategy-rating').value = '';
    document.getElementById('strategy-risk').value = '';
    document.getElementById('strategy-sort').value = 'install_count';
    loadStrategyMarketplace();
}

async function loadStrategyMarketplace(search = '', filters = {}) {
    const container = document.getElementById('store-list');
    
    // 显示加载状态
    container.innerHTML = `
        <div class="col-span-full py-16 text-center">
            <div class="loading-spinner-lg mx-auto mb-4"></div>
            <p class="text-slate-400">正在加载策略超市...</p>
            <p class="text-xs text-slate-500 mt-2">正在为您筛选最适合的策略</p>
        </div>
    `;
    
    try {
        // 构建查询参数
        const params = new URLSearchParams();
        if (search) params.append('search', search);
        if (filters.minRating) params.append('min_rating', filters.minRating);
        if (filters.maxRisk) params.append('max_risk', filters.maxRisk);
        if (filters.category) params.append('category', filters.category);
        if (filters.sortBy) params.append('sort_by', filters.sortBy);
        
        const queryString = params.toString();
        const url = `/api/marketplace/list${queryString ? '?' + queryString : ''}`;
        
        const res = await fetch(url);
        
        if (!res.ok) {
            throw new Error(`服务器响应错误: ${res.status} ${res.statusText}`);
        }
        
        const items = await res.json();

        container.innerHTML = '';
        if (items.length === 0) {
            container.innerHTML = `
                <div class="col-span-full py-16 text-center">
                    <div class="w-16 h-16 mx-auto mb-4 flex items-center justify-center bg-slate-800 rounded-full">
                        <i class="fas fa-search text-2xl text-slate-500"></i>
                    </div>
                    <p class="text-slate-400 mb-2">暂无策略上架</p>
                    <p class="text-xs text-slate-500">尝试调整筛选条件或稍后再来查看</p>
                </div>
            `;
            return;
        }

        items.forEach(item => {
                const card = document.createElement('div');
                card.className = "glass-card p-4 sm:p-5 border border-slate-700/50 hover:border-pink-500/30 transition-all group hover:shadow-lg hover:shadow-pink-900/10 transform hover:-translate-y-1 duration-300 scale-in";
                card.innerHTML = `
                    <div class="flex justify-between items-start mb-3 sm:mb-4">
                        <div class="h-10 sm:h-12 w-10 sm:w-12 rounded-lg bg-gradient-to-br from-pink-600 to-rose-600 flex items-center justify-center text-white text-sm sm:text-xl shadow-lg shadow-pink-900/20 group-hover:scale-110 transition-transform duration-300">
                            <i class="fas fa-chess-knight"></i>
                        </div>
                        <span class="text-xs bg-slate-800 text-slate-400 px-2 py-1 rounded group-hover:bg-pink-900/50 group-hover:text-pink-400 transition-colors">${item.author}</span>
                    </div>
                    <h3 class="text-base sm:text-lg font-bold text-white mb-2 group-hover:text-pink-400 transition-colors">${item.name}</h3>
                    <p class="text-sm text-slate-400 mb-3 sm:mb-4 h-10 line-clamp-2 group-hover:text-slate-300 transition-colors">${item.description}</p>
                    
                    <div class="grid grid-cols-2 gap-2 text-xs mb-3 sm:mb-4 p-2 sm:p-3 bg-slate-900/30 rounded group-hover:bg-slate-800/50 transition-colors">
                        <div>
                            <div class="text-slate-500">胜率</div>
                            <div class="text-up font-bold">${item.win_rate || 'Unknown'}</div>
                        </div>
                        <div>
                            <div class="text-slate-500">下载量</div>
                            <div class="text-slate-300">${item.downloads || 0}</div>
                        </div>
                        <div>
                            <div class="text-slate-500">收益率</div>
                            <div class="text-up font-bold">${item.return_rate || 'Unknown'}</div>
                        </div>
                        <div>
                            <div class="text-slate-500">评分</div>
                            <div class="text-yellow-500">${item.rating || 0} ★</div>
                        </div>
                    </div>
                    
                    <button onclick="installStrategy('${item.id}')" class="w-full py-2 bg-slate-800 hover:bg-pink-600 text-slate-300 hover:text-white border border-slate-700 hover:border-pink-500 rounded transition-all font-medium text-sm flex items-center justify-center gap-2 group-hover:scale-[1.02] active:scale-[0.98]">
                        <i class="fas fa-download"></i> 安装策略
                    </button>
                    
                    <div class="mt-2 sm:mt-3 flex justify-between items-center">
                        <span class="text-xs text-slate-500">${item.update_time || '未知更新时间'}</span>
                        <button onclick="viewStrategyDetails('${item.id}')" class="text-xs text-sky-400 hover:text-sky-300 transition-colors flex items-center gap-1">
                            <span>查看详情</span>
                            <i class="fas fa-angle-right text-xs"></i>
                        </button>
                    </div>
                `;
                container.appendChild(card);
                
                // 添加动画延迟，使卡片依次出现
                setTimeout(() => {
                    card.classList.add('scale-in');
                }, Math.random() * 200);
            });
        storeLoaded = true;
    } catch (e) {
        console.error('加载策略市场失败:', e);
        container.innerHTML = `
            <div class="col-span-full py-16 text-center">
                <div class="w-16 h-16 mx-auto mb-4 flex items-center justify-center bg-red-900/30 rounded-full border border-red-500/30">
                    <i class="fas fa-exclamation-triangle text-2xl text-red-400"></i>
                </div>
                <p class="text-red-400 mb-2">加载失败</p>
                <p class="text-xs text-slate-500 mb-4">${e.message || '网络连接错误，请检查网络设置'}</p>
                <button onclick="loadStrategyMarketplace('${search}', ${JSON.stringify(filters)})" class="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-all text-sm">
                    <i class="fas fa-redo mr-1"></i> 重试
                </button>
            </div>
        `;
    }
}

// 查看策略详情
async function viewStrategyDetails(strategyId) {
    // 创建策略详情模态框
    const modal = document.createElement('div');
    modal.id = 'strategy-detail-modal';
    modal.className = 'fixed inset-0 bg-slate-900/90 z-[100] flex items-center justify-center p-4 scale-in';
    modal.innerHTML = `
        <div class="bg-slate-800 w-full max-w-4xl rounded-2xl border border-slate-700 flex flex-col shadow-2xl overflow-hidden max-h-[90vh] scale-in">
            <div class="p-6 overflow-y-auto">
                <div class="flex justify-between items-start mb-6">
                    <h2 class="text-xl font-bold bg-gradient-to-r from-pink-400 to-rose-400 bg-clip-text text-transparent flex items-center gap-2">
                        <i class="fas fa-chess-knight text-pink-500"></i> 策略详情
                    </h2>
                    <button type="button" onclick="document.getElementById('strategy-detail-modal').remove()" class="text-slate-500 hover:text-white p-1 transition-colors">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                
                <div id="strategy-detail-loading" class="py-16 text-center">
                    <div class="loading-spinner-lg mx-auto mb-4"></div>
                    <p class="text-slate-400">正在加载策略详情...</p>
                    <p class="text-xs text-slate-500 mt-2">正在分析策略性能数据</p>
                </div>
                
                <div id="strategy-detail-content" class="hidden space-y-6">
                    <!-- 标签页导航 -->
                    <div class="flex border-b border-slate-700 mb-6">
                        <button class="tab-btn active px-4 py-2 text-sm font-medium transition-all" data-tab="info">基本信息</button>
                        <button class="tab-btn px-4 py-2 text-sm font-medium transition-all" data-tab="performance">性能分析</button>
                        <button class="tab-btn px-4 py-2 text-sm font-medium transition-all" data-tab="parameters">参数配置</button>
                        <button class="tab-btn px-4 py-2 text-sm font-medium transition-all" data-tab="comments">用户评论</button>
                    </div>
                    
                    <!-- 基本信息标签页 -->
                    <div class="tab-content active" data-tab="info">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div class="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                                <h3 class="text-lg font-bold text-white mb-3">基本信息</h3>
                                <div class="space-y-2">
                                    <div class="flex justify-between">
                                        <span class="text-slate-400">策略名称</span>
                                        <span id="detail-name" class="text-white font-medium"></span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-slate-400">作者</span>
                                        <span id="detail-author" class="text-white font-medium"></span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-slate-400">价格</span>
                                        <span id="detail-price" class="text-white font-medium"></span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-slate-400">下载量</span>
                                        <span id="detail-downloads" class="text-white font-medium"></span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-slate-400">安装次数</span>
                                        <span id="detail-install-count" class="text-white font-medium"></span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-slate-400">评分</span>
                                        <span id="detail-rating" class="text-white font-medium"></span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-slate-400">更新时间</span>
                                        <span id="detail-update-time" class="text-white font-medium"></span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                                <h3 class="text-lg font-bold text-white mb-3">性能指标</h3>
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div class="space-y-2">
                                        <div class="flex justify-between">
                                            <span class="text-slate-400">胜率</span>
                                            <span id="detail-win-rate" class="text-up font-medium"></span>
                                        </div>
                                        <div class="flex justify-between">
                                            <span class="text-slate-400">收益率</span>
                                            <span id="detail-return-rate" class="text-up font-medium"></span>
                                        </div>
                                        <div class="flex justify-between">
                                            <span class="text-slate-400">夏普比率</span>
                                            <span id="detail-sharpe-ratio" class="text-white font-medium"></span>
                                        </div>
                                        <div class="flex justify-between">
                                            <span class="text-slate-400">最大回撤</span>
                                            <span id="detail-max-drawdown" class="text-white font-medium"></span>
                                        </div>
                                    </div>
                                    <div class="space-y-2">
                                        <div class="flex justify-between">
                                            <span class="text-slate-400">平均收益率</span>
                                            <span id="detail-avg-return" class="text-up font-medium"></span>
                                        </div>
                                        <div class="flex justify-between">
                                            <span class="text-slate-400">总交易次数</span>
                                            <span id="detail-total-trades" class="text-white font-medium"></span>
                                        </div>
                                        <div class="flex justify-between">
                                            <span class="text-slate-400">风险等级</span>
                                            <span id="detail-risk-level" class="text-white font-medium"></span>
                                        </div>
                                        <div class="flex justify-between">
                                            <span class="text-slate-400">策略分类</span>
                                            <span id="detail-category" class="text-white font-medium"></span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                            <h3 class="text-lg font-bold text-white mb-3">策略描述</h3>
                            <p id="detail-description" class="text-slate-300"></p>
                        </div>
                    </div>
                    
                    <!-- 性能分析标签页 -->
                    <div class="tab-content" data-tab="performance">
                        <div class="bg-slate-900/50 p-4 rounded-lg border border-slate-700 mb-4">
                            <h3 class="text-lg font-bold text-white mb-3">性能图表</h3>
                            <div id="performance-chart" class="h-80 w-full"></div>
                        </div>
                        
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div class="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                                <h4 class="text-sm font-bold text-slate-400 mb-2">年化收益率</h4>
                                <div class="text-2xl font-bold text-up" id="detail-annual-return">--</div>
                            </div>
                            <div class="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                                <h4 class="text-sm font-bold text-slate-400 mb-2">最大回撤</h4>
                                <div class="text-2xl font-bold text-down" id="detail-max-drawdown-chart">--</div>
                            </div>
                            <div class="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                                <h4 class="text-sm font-bold text-slate-400 mb-2">夏普比率</h4>
                                <div class="text-2xl font-bold text-white" id="detail-sharpe-ratio-chart">--</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- 参数配置标签页 -->
                    <div class="tab-content" data-tab="parameters">
                        <div class="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                            <h3 class="text-lg font-bold text-white mb-3">策略参数</h3>
                            <div id="detail-parameters" class="space-y-3">
                                <!-- 参数将在这里动态添加 -->
                                <div class="text-slate-500 text-center py-4">暂无参数信息</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- 用户评论标签页 -->
                    <div class="tab-content" data-tab="comments">
                        <div class="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                            <div class="flex justify-between items-center mb-3">
                                <h3 class="text-lg font-bold text-white">用户评论</h3>
                                <button onclick="openRatingModal('${strategyId}')" class="px-3 py-1 bg-pink-600 hover:bg-pink-500 text-white rounded text-sm transition-colors">
                                    <i class="fas fa-star mr-1"></i> 评分
                                </button>
                            </div>
                            <div id="detail-comments" class="space-y-3">
                                <!-- 评论将在这里动态添加 -->
                            </div>
                        </div>
                    </div>
                    
                    <div class="flex gap-3 justify-end">
                        <button onclick="installStrategy('${strategyId}')" class="px-6 py-2 bg-gradient-to-r from-pink-600 to-rose-600 hover:from-pink-500 hover:to-rose-500 text-white font-bold rounded-lg transition-all flex items-center gap-2">
                            <i class="fas fa-download"></i> 安装策略
                        </button>
                        <button onclick="document.getElementById('strategy-detail-modal').remove()" class="px-6 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-all">
                            关闭
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    
    // 加载策略详情
    try {
        const res = await fetch(`/api/marketplace/item/${strategyId}`);
        
        if (!res.ok) {
            throw new Error(`服务器响应错误: ${res.status} ${res.statusText}`);
        }
        
        const data = await res.json();
        
        // 隐藏加载状态，显示内容
        document.getElementById('strategy-detail-loading').classList.add('hidden');
        document.getElementById('strategy-detail-content').classList.remove('hidden');
        
        // 填充策略信息
        document.getElementById('detail-name').textContent = data.name;
        document.getElementById('detail-author').textContent = data.author;
        document.getElementById('detail-price').textContent = data.price > 0 ? `¥${data.price}` : '免费';
        document.getElementById('detail-downloads').textContent = data.downloads || 0;
        document.getElementById('detail-install-count').textContent = data.install_count || 0;
        document.getElementById('detail-rating').textContent = `${data.rating || 5.0} ★ (${data.rating_count || 0}人评价)`;
        document.getElementById('detail-win-rate').textContent = data.win_rate || 'Unknown';
        document.getElementById('detail-return-rate').textContent = data.return_rate || 'Unknown';
        document.getElementById('detail-sharpe-ratio').textContent = data.sharpe_ratio || '0.0';
        document.getElementById('detail-max-drawdown').textContent = `${data.max_drawdown || 0.0}%`;
        document.getElementById('detail-avg-return').textContent = `${data.avg_return || 0.0}%`;
        document.getElementById('detail-total-trades').textContent = data.total_trades || 0;
        document.getElementById('detail-risk-level').textContent = `${data.risk_level || 3}/5`;
        document.getElementById('detail-category').textContent = data.category || '未分类';
        document.getElementById('detail-update-time').textContent = data.update_time || 'Unknown';
        document.getElementById('detail-description').textContent = data.description || '暂无描述';
        
        // 填充图表数据
        document.getElementById('detail-annual-return').textContent = `${(data.avg_return || 0) * 12 || 0}%`;
        document.getElementById('detail-max-drawdown-chart').textContent = `${data.max_drawdown || 0}%`;
        document.getElementById('detail-sharpe-ratio-chart').textContent = data.sharpe_ratio || 0;
        
        // 初始化性能图表
        initPerformanceChart(data);
        
        // 填充评论
        const commentsContainer = document.getElementById('detail-comments');
        if (data.comments && data.comments.length > 0) {
            commentsContainer.innerHTML = data.comments.map(comment => `
                <div class="p-3 bg-slate-800/50 rounded-lg border border-slate-700 scale-in">
                    <div class="flex justify-between items-start mb-2">
                        <div>
                            <span class="text-white font-medium">${comment.username}</span>
                            <span class="ml-2 text-yellow-500">${'★'.repeat(comment.rating)}${'☆'.repeat(5 - comment.rating)}</span>
                        </div>
                        <span class="text-xs text-slate-500">${comment.created_at}</span>
                    </div>
                    <p class="text-slate-300 text-sm">${comment.comment}</p>
                </div>
            `).join('');
        } else {
            commentsContainer.innerHTML = `
                <div class="text-center py-8">
                    <div class="w-12 h-12 mx-auto mb-3 flex items-center justify-center bg-slate-800 rounded-full">
                        <i class="fas fa-comment-dots text-xl text-slate-500"></i>
                    </div>
                    <p class="text-slate-500 mb-2">暂无评论</p>
                    <p class="text-xs text-slate-600">快来成为第一个评价的人吧！</p>
                </div>
            `;
        }
        
        // 初始化标签页切换
        initTabs();
    } catch (e) {
        console.error('加载策略详情失败:', e);
        document.getElementById('strategy-detail-loading').innerHTML = `
            <div class="py-16 text-center">
                <div class="w-16 h-16 mx-auto mb-4 flex items-center justify-center bg-red-900/30 rounded-full border border-red-500/30">
                    <i class="fas fa-exclamation-triangle text-2xl text-red-400"></i>
                </div>
                <p class="text-red-400 mb-2">加载失败</p>
                <p class="text-xs text-slate-500 mb-4">${e.message || '网络连接错误，请检查网络设置'}</p>
                <button onclick="viewStrategyDetails('${strategyId}')" class="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-all text-sm">
                    <i class="fas fa-redo mr-1"></i> 重试
                </button>
            </div>
        `;
    }
}

// 初始化标签页切换
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            
            // 更新标签按钮状态
            tabBtns.forEach(b => {
                b.classList.remove('active', 'text-pink-400', 'border-b-2', 'border-pink-400');
                b.classList.add('text-slate-400');
            });
            btn.classList.add('active', 'text-pink-400', 'border-b-2', 'border-pink-400');
            btn.classList.remove('text-slate-400');
            
            // 更新内容显示
            tabContents.forEach(content => {
                content.classList.remove('active', 'block');
                content.classList.add('hidden');
            });
            const activeContent = document.querySelector(`.tab-content[data-tab="${tab}"]`);
            if (activeContent) {
                activeContent.classList.add('active', 'block');
                activeContent.classList.remove('hidden');
            }
        });
    });
}

// 初始化性能图表
function initPerformanceChart(data) {
    const chartDom = document.getElementById('performance-chart');
    if (!chartDom || typeof echarts === 'undefined') return;
    
    const myChart = echarts.init(chartDom);
    
    // 模拟性能数据
    const dates = [];
    const values = [];
    const baseValue = 10000;
    let currentValue = baseValue;
    
    // 生成30天的模拟数据
    for (let i = 30; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        dates.push(date.toLocaleDateString());
        
        // 模拟每日波动
        const change = (Math.random() - 0.48) * 0.02 * currentValue;
        currentValue += change;
        values.push(currentValue);
    }
    
    const option = {
        backgroundColor: 'transparent',
        // 动画效果
        animation: true,
        animationDuration: 1500,
        animationEasing: 'cubicOut',
        animationDelay: function(idx) {
            return idx * 20;
        },
        // 交互配置
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross',
                label: {
                    backgroundColor: '#6a7985'
                }
            },
            formatter: function(params) {
                const value = params[0].value;
                const change = ((value - baseValue) / baseValue * 100).toFixed(2);
                const changeClass = change >= 0 ? 'text-up' : 'text-down';
                return `
                    <div style="padding: 8px 12px; background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; font-size: 12px;">
                        <div style="font-weight: bold; margin-bottom: 4px; color: #f8fafc;">${params[0].name}</div>
                        <div style="margin-bottom: 2px;">资产: <span style="font-weight: bold; color: #f8fafc;">¥${value.toFixed(2)}</span></div>
                        <div>收益: <span class="${changeClass}" style="font-weight: bold;">${change}%</span></div>
                    </div>
                `;
            },
            backgroundColor: 'transparent',
            borderColor: 'transparent'
        },
        // 图例
        legend: {
            data: ['资产价值'],
            textStyle: {
                color: '#94a3b8'
            },
            top: 0
        },
        // 网格
        grid: {
            left: '3%',
            right: '4%',
            bottom: '10%',
            top: '15%',
            containLabel: true
        },
        // 工具箱
        toolbox: {
            feature: {
                saveAsImage: {
                    iconStyle: {
                        normal: {
                            color: '#94a3b8'
                        }
                    }
                },
                dataZoom: {
                    iconStyle: {
                        normal: {
                            color: '#94a3b8'
                        }
                    }
                },
                restore: {
                    iconStyle: {
                        normal: {
                            color: '#94a3b8'
                        }
                    }
                }
            },
            iconStyle: {
                normal: {
                    borderColor: '#475569'
                }
            }
        },
        // 数据缩放
        dataZoom: [
            {
                type: 'inside',
                start: 0,
                end: 100
            },
            {
                start: 0,
                end: 100,
                height: 20,
                bottom: 0,
                fillerColor: 'rgba(56, 189, 248, 0.1)',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                handleIcon: 'M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
                handleSize: '80%',
                handleStyle: {
                    color: '#38bdf8',
                    shadowBlur: 3,
                    shadowColor: 'rgba(0, 0, 0, 0.6)',
                    shadowOffsetX: 2,
                    shadowOffsetY: 2
                },
                textStyle: {
                    color: '#94a3b8'
                }
            }
        ],
        // X轴
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: dates,
            axisLabel: {
                color: '#94a3b8',
                fontSize: 10,
                rotate: 45,
                interval: 4 // 每5天显示一个标签
            },
            axisLine: {
                lineStyle: {
                    color: '#334155'
                }
            },
            axisTick: {
                show: false
            }
        },
        // Y轴
        yAxis: {
            type: 'value',
            axisLabel: {
                color: '#94a3b8',
                fontSize: 10,
                formatter: function(value) {
                    return '¥' + value.toFixed(0);
                }
            },
            axisLine: {
                lineStyle: {
                    color: '#334155'
                }
            },
            splitLine: {
                lineStyle: {
                    color: 'rgba(51, 65, 85, 0.6)',
                    type: 'dashed'
                }
            },
            axisTick: {
                show: false
            }
        },
        // 系列
        series: [{
            name: '资产价值',
            data: values,
            type: 'line',
            smooth: true,
            symbol: 'circle',
            symbolSize: 6,
            sampling: 'average',
            itemStyle: {
                color: '#38bdf8',
                borderColor: '#1e293b',
                borderWidth: 2
            },
            lineStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                    { offset: 0, color: '#38bdf8' },
                    { offset: 1, color: '#818cf8' }
                ]),
                width: 3,
                shadowColor: 'rgba(56, 189, 248, 0.3)',
                shadowBlur: 10,
                shadowOffsetY: 5
            },
            areaStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    {
                        offset: 0,
                        color: 'rgba(56, 189, 248, 0.3)'
                    },
                    {
                        offset: 1,
                        color: 'rgba(56, 189, 248, 0.05)'
                    }
                ])
            },
            // 标记点
            markPoint: {
                data: [
                    { type: 'max', name: '最大值' },
                    { type: 'min', name: '最小值' }
                ],
                itemStyle: {
                    color: '#ec4899'
                },
                label: {
                    color: '#f8fafc'
                }
            },
            // 标记线
            markLine: {
                data: [
                    { type: 'average', name: '平均值' }
                ],
                lineStyle: {
                    color: '#f59e0b',
                    type: 'dashed'
                },
                label: {
                    color: '#94a3b8'
                }
            }
        }]
    };
    
    myChart.setOption(option);
    
    // 响应式调整
    window.addEventListener('resize', function() {
        myChart.resize();
    });
    
    // 添加鼠标移动效果
    chartDom.addEventListener('mousemove', function() {
        myChart.dispatchAction({
            type: 'showTip',
            seriesIndex: 0
        });
    });
}

// 打开评分模态框
function openRatingModal(strategyId) {
    // 创建评分模态框
    const modal = document.createElement('div');
    modal.id = 'rating-modal';
    modal.className = 'fixed inset-0 bg-slate-900/90 z-[101] flex items-center justify-center p-4';
    modal.innerHTML = `
        <div class="bg-slate-800 w-full max-w-md rounded-2xl border border-slate-700 flex flex-col shadow-2xl overflow-hidden">
            <div class="p-6">
                <div class="flex justify-between items-start mb-6">
                    <h2 class="text-xl font-bold bg-gradient-to-r from-yellow-400 to-amber-400 bg-clip-text text-transparent flex items-center gap-2">
                        <i class="fas fa-star text-yellow-500"></i> 评分与评论
                    </h2>
                    <button type="button" onclick="document.getElementById('rating-modal').remove()" class="text-slate-500 hover:text-white p-1">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                
                <div class="space-y-4">
                    <div>
                        <label class="block text-slate-400 mb-2">评分</label>
                        <div class="flex gap-2">
                            ${[1, 2, 3, 4, 5].map(star => `
                                <button type="button" class="star-btn text-2xl ${star <= 5 ? 'text-yellow-500' : 'text-slate-600'}" data-rating="${star}">
                                    ★
                                </button>
                            `).join('')}
                        </div>
                    </div>
                    
                    <div>
                        <label class="block text-slate-400 mb-2">评论</label>
                        <textarea id="rating-comment" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-slate-300 focus:border-yellow-500 focus:outline-none" rows="4" placeholder="分享你的使用体验..."></textarea>
                    </div>
                    
                    <div class="flex gap-3 justify-end">
                        <button onclick="document.getElementById('rating-modal').remove()" class="px-6 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-all">
                            取消
                        </button>
                        <button onclick="submitRating('${strategyId}')" class="px-6 py-2 bg-gradient-to-r from-yellow-600 to-amber-600 hover:from-yellow-500 hover:to-amber-500 text-white font-bold rounded-lg transition-all">
                            提交评分
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    
    // 绑定星级评分按钮
    modal.querySelectorAll('.star-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const rating = parseInt(this.dataset.rating);
            modal.querySelectorAll('.star-btn').forEach((b, index) => {
                b.className = `star-btn text-2xl ${(index + 1) <= rating ? 'text-yellow-500' : 'text-slate-600'}`;
            });
        });
    });
}

// 提交评分
async function submitRating(strategyId) {
    const modal = document.getElementById('rating-modal');
    const selectedRating = modal.querySelector('.star-btn.text-yellow-500:last-child')?.dataset.rating;
    const comment = document.getElementById('rating-comment').value;
    
    if (!selectedRating) {
        alert('请选择评分');
        return;
    }
    
    try {
        const res = await fetch(`/api/marketplace/rate/${strategyId}?rating=${selectedRating}&comment=${encodeURIComponent(comment)}`, {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + localStorage.getItem('access_token')
            }
        });
        
        if (res.ok) {
            alert('评分提交成功！');
            document.getElementById('rating-modal').remove();
            // 刷新策略详情
            viewStrategyDetails(strategyId);
        } else {
            alert('评分提交失败，请登录后重试');
        }
    } catch (e) {
        console.error(e);
        alert('评分提交失败，请稍后重试');
    }
}

window.installStrategy = async function (strategyId) {
    // 获取当前按钮并保存原始状态
    const button = event.target.closest('button');
    const originalText = button.innerHTML;
    const originalClasses = button.className;
    
    // 显示加载状态
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 准备安装...';
    button.classList.add('opacity-70', 'cursor-not-allowed', 'bg-slate-800/80');
    
    if (!confirm("确定要下载并安装此策略吗？")) {
        // 恢复按钮状态
        button.innerHTML = originalText;
        button.className = originalClasses;
        button.disabled = false;
        return;
    }
    
    try {
        // 显示安装进度
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 连接服务器...';
        showToast("🔄 正在连接服务器...", "info");
        
        const res = await fetch(`/api/marketplace/install/${strategyId}`, { method: 'POST' });
        
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 处理安装请求...';
        showToast("🔄 正在处理安装请求...", "info");
        
        if (res.ok) {
            const data = await res.json();
            
            if (data.status === "success") {
                // 显示成功状态
                button.innerHTML = '<i class="fas fa-check"></i> 安装成功';
                button.className = 'px-6 py-2 bg-green-900/50 text-green-400 border border-green-700 rounded-lg transition-all flex items-center gap-2';
                
                showToast("✅ 策略安装成功！请在 strategies 目录查看");
                
                // 延迟跳转到策略中心
                setTimeout(() => {
                    window.location.href = "/strategies";
                }, 1500);
            } else if (data.status === "already_installed") {
                // 恢复按钮状态
                button.innerHTML = originalText;
                button.className = originalClasses;
                button.disabled = false;
                
                showToast("ℹ️ 策略已经安装过了！", "info");
            } else {
                // 恢复按钮状态
                button.innerHTML = originalText;
                button.className = originalClasses;
                button.disabled = false;
                
                showToast("❌ 安装失败：" + (data.message || "未知错误"), "error");
            }
        } else {
            // 恢复按钮状态
            button.innerHTML = originalText;
            button.className = originalClasses;
            button.disabled = false;
            
            // 尝试获取错误详情
            try {
                const errorData = await res.json();
                showToast(`❌ 安装失败：${errorData.detail || errorData.message || "服务器错误"}`, "error");
            } catch {
                showToast("❌ 安装失败，服务器错误！", "error");
            }
        }
    } catch (e) {
        console.error("安装失败:", e);
        
        // 恢复按钮状态
        button.innerHTML = originalText;
        button.className = originalClasses;
        button.disabled = false;
        
        if (e.name === 'AbortError') {
            showToast("⏹️ 安装已取消", "info");
        } else if (e.message.includes('Network')) {
            showToast("❌ 安装失败，网络错误！请检查网络连接", "error");
        } else {
            showToast(`❌ 安装失败：${e.message}`, "error");
        }
    }
};


// ================= PROFILE MANAGMENT =================
window.openProfileModal = async function () {
    const modal = document.getElementById('profile-modal');
    modal.classList.remove('hidden');
    modal.style.display = 'flex';

    // Load data
    try {
        const res = await fetch('/api/users/me', {
            headers: { 'Authorization': 'Bearer ' + localStorage.getItem('access_token') }
        });
        if (res.ok) {
            const user = await res.json();
            document.getElementById('profile-username').innerText = user.username;
            document.getElementById('profile-bio-input').value = user.bio || '';
            document.getElementById('profile-tags-input').value = user.tags || '';
            if (user.avatar) {
                document.getElementById('profile-avatar-img').src = user.avatar;
                document.getElementById('header-avatar').src = user.avatar; // Update header too
            }
            // Parse tags for display
            renderTags(user.tags);
        }
    } catch (e) {
        console.error(e);
    }
};

window.closeProfileModal = function () {
    document.getElementById('profile-modal').style.display = 'none';
};

window.saveProfile = async function () {
    const bio = document.getElementById('profile-bio-input').value;
    const tags = document.getElementById('profile-tags-input').value;

    try {
        const res = await fetch('/api/users/me', {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + localStorage.getItem('access_token')
            },
            body: JSON.stringify({ bio, tags })
        });

        if (res.ok) {
            showToast("✅ 个人名片已更新");
            closeProfileModal();
            // Update UI reflectively
            const user = await res.json();
            document.getElementById('header-avatar').src = user.avatar || '/static/avatars/default.png';
        } else {
            showToast("更新失败");
        }
    } catch (e) {
        showToast("网络错误");
    }
};

window.uploadAvatar = async function (input) {
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/users/avatar', {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + localStorage.getItem('access_token')
            },
            body: formData
        });

        if (res.ok) {
            const data = await res.json();
            document.getElementById('profile-avatar-img').src = data.avatar;
            document.getElementById('header-avatar').src = data.avatar;
            showToast("✅ 头像更新成功");
        } else {
            showToast("❌ 上传失败");
        }
    } catch (e) {
        showToast("网络错误: " + e.message);
    }
};

function renderTags(tagsStr) {
    const container = document.getElementById('profile-tags-display');
    if (!tagsStr) {
        container.innerHTML = '';
        return;
    }
    const tags = tagsStr.split(/[,，]/).filter(t => t.trim());
    container.innerHTML = tags.map(t => `<span class="px-2 py-0.5 bg-slate-800 rounded text-xxs text-slate-400 border border-slate-700">${t}</span>`).join('');
}

// ================= CLOUD SYNC =================
window.downloadBackup = function () {
    window.open('/api/sync/backup', '_blank');
};

window.uploadRestore = async function (input) {
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    if (!confirm("⚠️ 警告：恢复备份将覆盖当前的数据库和配置，确定继续吗？")) {
        input.value = '';
        return;
    }

    showToast("⏳ 正在恢复数据...");
    try {
        const res = await fetch('/api/sync/restore', {
            method: 'POST',
            body: formData
        });
        if (res.ok) {
            alert("✅ 数据恢复成功！页面即将刷新");
            location.reload();
        } else {
            showToast("❌ 恢复失败，文件可能已损坏");
        }
    } catch (e) {
        showToast("网络错误");
    }
    input.value = '';
};


// ================= SYSTEM SETTINGS =================
async function loadSystemSettings() {
    try {
        const res = await fetch('/api/settings/ai');
        const data = await res.json();

        document.getElementById('settings-ai-key').value = data.api_key || '';
        document.getElementById('settings-ai-url').value = data.base_url || '';
        document.getElementById('settings-ai-model').value = data.model || '';
        document.getElementById('settings-ai-provider').value = data.provider || 'default';

        if (data.has_key) {
            document.getElementById('settings-ai-key').placeholder = "********** (已配置)";
        }

        // Secondary
        if (document.getElementById('settings-ai-sec-key')) {
            document.getElementById('settings-ai-sec-key').value = data.secondary_api_key || '';
            document.getElementById('settings-ai-sec-url').value = data.secondary_base_url || '';
            document.getElementById('settings-ai-sec-model').value = data.secondary_model || '';

            if (data.has_secondary_key) {
                document.getElementById('settings-ai-sec-key').placeholder = "********** (已配置)";
            }
        }
    } catch (e) {
        console.error("Failed to load settings", e);
    }
}

window.saveSystemSettings = async function () {
    const key = document.getElementById('settings-ai-key').value;
    const url = document.getElementById('settings-ai-url').value;
    const model = document.getElementById('settings-ai-model').value;
    const provider = document.getElementById('settings-ai-provider').value;

    // Secondary
    const secKey = document.getElementById('settings-ai-sec-key') ? document.getElementById('settings-ai-sec-key').value : "";
    const secUrl = document.getElementById('settings-ai-sec-url') ? document.getElementById('settings-ai-sec-url').value : "";
    const secModel = document.getElementById('settings-ai-sec-model') ? document.getElementById('settings-ai-sec-model').value : "";

    if (!url) {
        showToast("❌ Base URL 不能为空");
        return;
    }

    try {
        const res = await fetch('/api/settings/ai', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                api_key: key,
                base_url: url,
                model: model,
                provider: provider,
                secondary_api_key: secKey,
                secondary_base_url: secUrl,
                secondary_model: secModel
            })
        });

        if (res.ok) {
            showToast("✅ AI 配置已保存 (主线路 + 备用线路)");
            // Reload to show masked key
            loadSystemSettings();
        } else {
            const err = await res.json();
            showToast("❌ 保存失败: " + (err.detail || "未知错误"));
        }
    } catch (e) {
        showToast("网络错误");
    }
};


// Hook into switchMode for System view
const originalSwitchModeP5 = window.switchMode;
window.switchMode = function (mode) {
    if (originalSwitchModeP5) originalSwitchModeP5(mode);
    if (mode === 'system') {
        loadSystemSettings();
    }
};

// Initialize Settings Button
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('settings-btn');
    if (btn) {
        btn.onclick = function () {
            loadSystemSettings();
            document.getElementById('settings-modal').classList.remove('hidden');
        }
    }
});
