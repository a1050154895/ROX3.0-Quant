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
    if (mode === 'store' && !storeLoaded) {
        loadStrategyMarketplace();
    }
};

async function loadStrategyMarketplace() {
    const container = document.getElementById('store-list');
    try {
        const res = await fetch('/api/marketplace/list');
        const items = await res.json();

        container.innerHTML = '';
        if (items.length === 0) {
            container.innerHTML = '<div class="col-span-full text-slate-500">暂无策略上架</div>';
            return;
        }

        items.forEach(item => {
            const card = document.createElement('div');
            card.className = "glass-card p-5 border border-slate-700/50 hover:border-pink-500/30 transition-all group";
            card.innerHTML = `
                <div class="flex justify-between items-start mb-4">
                    <div class="h-12 w-12 rounded-lg bg-gradient-to-br from-pink-600 to-rose-600 flex items-center justify-center text-white text-xl shadow-lg shadow-pink-900/20">
                        <i class="fas fa-chess-knight"></i>
                    </div>
                    <span class="text-xs bg-slate-800 text-slate-400 px-2 py-1 rounded">${item.author}</span>
                </div>
                <h3 class="text-lg font-bold text-white mb-2">${item.name}</h3>
                <p class="text-sm text-slate-400 mb-4 h-10 line-clamp-2">${item.description}</p>
                
                <div class="grid grid-cols-2 gap-2 text-xs mb-4 p-3 bg-slate-900/30 rounded">
                    <div>
                        <div class="text-slate-500">胜率</div>
                        <div class="text-up font-bold">Unknown</div>
                    </div>
                     <div>
                        <div class="text-slate-500">下载量</div>
                        <div class="text-slate-300">${item.downloads}</div>
                    </div>
                </div>
                
                <button onclick="installStrategy('${item.id}')" class="w-full py-2 bg-slate-800 hover:bg-pink-600/20 text-slate-300 hover:text-pink-400 border border-slate-700 hover:border-pink-500/50 rounded transition-all font-medium text-sm flex items-center justify-center gap-2">
                    <i class="fas fa-download"></i> 安装策略
                </button>
            `;
            container.appendChild(card);
        });
        storeLoaded = true;
    } catch (e) {
        console.error(e);
        container.innerHTML = '<div class="text-red-500">加载失败</div>';
    }
}

window.installStrategy = async function (strategyId) {
    if (!confirm("确定要下载并安装此策略吗？")) return;
    try {
        const res = await fetch(`/api/marketplace/install/${strategyId}`, { method: 'POST' });
        if (res.ok) {
            showToast("✅ 策略安装成功！请在 strategies 目录查看");
        } else {
            showToast("❌ 安装失败");
        }
    } catch (e) {
        showToast("❌ 请求错误");
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
    } catch (e) {
        console.error("Failed to load settings", e);
    }
}

window.saveSystemSettings = async function () {
    const key = document.getElementById('settings-ai-key').value;
    const url = document.getElementById('settings-ai-url').value;
    const model = document.getElementById('settings-ai-model').value;
    const provider = document.getElementById('settings-ai-provider').value;

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
                provider: provider
            })
        });

        if (res.ok) {
            showToast("✅ AI 配置已保存");
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
