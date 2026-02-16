/**
 * ROX 3.0 键盘精灵（对标通达信）
 * 全局输入数字/拼音 → 弹出搜索 → Enter 切股，Up/Down 选择
 */

class KeyboardCommander {
    constructor() {
        this.isVisible = false;
        this.selectedIndex = 0;
        this.matches = [];
        this.debounceTimer = null;
        this.el = document.getElementById('keyboard-commander');
        this.inputEl = document.getElementById('commander-input');
        this.resultsEl = document.getElementById('commander-results');
        this.stockDb = [
            { code: '600519', name: '贵州茅台' },
            { code: '600030', name: '中信证券' },
            { code: '600036', name: '招商银行' },
            { code: '000001', name: '平安银行' },
            { code: '300059', name: '东方财富' },
            { code: '601899', name: '紫金矿业' },
            { code: '300750', name: '宁德时代' },
            { code: '601138', name: '工业富联' },
        ];
        this.init();
    }

    init() {
        if (!this.el || !this.inputEl || !this.resultsEl) return;
        document.addEventListener('keydown', (e) => this.handleGlobalKey(e));
        this.inputEl.addEventListener('input', () => this.onInput());
        this.inputEl.addEventListener('keydown', (e) => this.handleInputKey(e));
        this.inputEl.addEventListener('focus', () => this.onInput());
    }

    handleGlobalKey(e) {
        const active = document.activeElement;
        const inInput = active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA');
        if (inInput && active !== this.inputEl) return;

        // 快捷键处理
        if (this.handleShortcuts(e)) {
            e.preventDefault();
            return;
        }

        if (e.key === 'Escape' && this.isVisible) {
            this.hide();
            e.preventDefault();
            return;
        }

        if (/^[a-zA-Z0-9]$/.test(e.key) && !this.isVisible) {
            this.show();
            this.inputEl.value = e.key;
            this.inputEl.focus();
            this.onInput();
            e.preventDefault();
            return;
        }

        if (['F1', 'F2', 'F10'].includes(e.key)) {
            e.preventDefault();
            if (typeof window.handleFKey === 'function') window.handleFKey(e.key);
        }
    }

    handleShortcuts(e) {
        // Ctrl/Command + 快捷键
        if (e.ctrlKey || e.metaKey) {
            switch (e.key.toLowerCase()) {
                case '1':
                    this.switchLayout('compact');
                    return true;
                case '2':
                    this.switchLayout('normal');
                    return true;
                case 'd':
                    this.toggleDarkMode();
                    return true;
                case 'k':
                    this.show();
                    return true;
                case 'r':
                    this.refreshData();
                    return true;
                case 's':
                    this.saveLayout();
                    return true;
                default:
                    return false;
            }
        }

        // 单键快捷键
        switch (e.key) {
            case 'F5':
                this.refreshData();
                return true;
            case 'F6':
                this.toggleWatchlist();
                return true;
            case 'F7':
                this.toggleIndicators();
                return true;
            case 'F8':
                this.toggleStrategyBuilder();
                return true;
            case 'F9':
                this.toggleTradePanel();
                return true;
            default:
                return false;
        }
    }

    switchLayout(layout) {
        if (typeof window.switchLayout === 'function') {
            window.switchLayout(layout);
        } else {
            document.body.dataset.layout = layout;
            localStorage.setItem('rox-layout', layout);
            location.reload();
        }
    }

    toggleDarkMode() {
        const html = document.documentElement;
        html.classList.toggle('dark');
        localStorage.setItem('rox-dark-mode', html.classList.contains('dark'));
    }

    refreshData() {
        if (typeof window.refreshData === 'function') {
            window.refreshData();
        } else {
            location.reload();
        }
    }

    saveLayout() {
        if (typeof window.saveLayout === 'function') {
            window.saveLayout();
        } else {
            alert('布局已保存');
        }
    }

    toggleWatchlist() {
        const watchlist = document.getElementById('watchlist');
        if (watchlist) {
            watchlist.classList.toggle('hidden');
        }
    }

    toggleIndicators() {
        const indicators = document.getElementById('indicators-panel');
        if (indicators) {
            indicators.classList.toggle('hidden');
        }
    }

    toggleStrategyBuilder() {
        const builder = document.getElementById('strategy-builder');
        if (builder) {
            builder.classList.toggle('hidden');
        }
    }

    toggleTradePanel() {
        const tradePanel = document.getElementById('trade-panel');
        if (tradePanel) {
            tradePanel.classList.toggle('hidden');
        }
    }

    onInput() {
        const q = (this.inputEl?.value || '').trim();
        if (this.debounceTimer) clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(() => {
            this.search(q);
            this.debounceTimer = null;
        }, 150);
    }

    handleInputKey(e) {
        if (e.key === 'Enter') {
            this.selectCurrent();
            e.preventDefault();
            return;
        }
        if (e.key === 'ArrowDown') {
            this.moveSelection(1);
            e.preventDefault();
            return;
        }
        if (e.key === 'ArrowUp') {
            this.moveSelection(-1);
            e.preventDefault();
        }
    }

    moveSelection(delta) {
        const n = this.matches.length;
        if (n === 0) return;
        this.selectedIndex = (this.selectedIndex + delta + n) % n;
        this.renderResults();
        const sel = this.resultsEl?.querySelector('.result-item.selected');
        if (sel) sel.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    selectCurrent() {
        const m = this.matches[this.selectedIndex];
        if (m) this.selectStock(m.code, m.name);
    }

    async search(query) {
        this.matches = [];
        this.selectedIndex = 0;
        if (!this.resultsEl) return;

        if (!query) {
            this.resultsEl.innerHTML = '<div class="px-3 py-2 text-xxs text-gray-500">输入代码或名称</div>';
            return;
        }

        try {
            const r = await fetch(`/api/market/stock-suggest?q=${encodeURIComponent(query)}&limit=12`);
            const data = await r.json().catch(() => ({}));
            const items = data.items || [];
            if (items.length) {
                this.matches = items;
                this.renderResults();
                return;
            }
        } catch (_) {}

        const q = query.toUpperCase();
        this.matches = this.stockDb.filter(s =>
            String(s.code).includes(q) || String(s.name).includes(q)
        ).slice(0, 12);
        this.renderResults();
    }

    renderResults() {
        if (!this.resultsEl) return;
        if (this.matches.length === 0) {
            this.resultsEl.innerHTML = '<div class="px-3 py-2 text-xxs text-gray-500">无匹配</div>';
            return;
        }
        this.resultsEl.innerHTML = this.matches.map((m, i) => `
            <div class="result-item px-3 py-1.5 flex justify-between text-sm cursor-pointer ${i === this.selectedIndex ? 'selected bg-blue-900/80 text-white' : 'text-gray-300 hover:bg-slate-800'}"
                 data-code="${m.code}" data-name="${m.name}"
                 data-index="${i}">
                <span class="font-mono text-yellow-500">${m.code}</span>
                <span>${m.name}</span>
            </div>
        `).join('');

        this.resultsEl.querySelectorAll('.result-item').forEach((el, i) => {
            el.addEventListener('click', () => {
                this.selectedIndex = i;
                this.selectStock(el.dataset.code, el.dataset.name);
            });
        });
    }

    selectStock(code, name) {
        if (typeof window.selectStock === 'function') {
            window.selectStock(code, name);
        } else {
            const nh = document.getElementById('stock-name-header');
            const ch = document.getElementById('stock-code-header');
            if (nh) nh.textContent = name || code;
            if (ch) ch.textContent = code;
            if (window.currentStockCode !== undefined) window.currentStockCode = code;
        }
        this.hide();
    }

    show() {
        this.isVisible = true;
        if (this.el) this.el.classList.remove('hidden');
        this.inputEl.value = '';
        this.matches = [];
        this.selectedIndex = 0;
        this.resultsEl.innerHTML = '<div class="px-3 py-2 text-xxs text-gray-500">输入代码或名称</div>';
        this.inputEl.focus();
    }

    hide() {
        this.isVisible = false;
        if (this.el) this.el.classList.add('hidden');
        this.inputEl.blur();
    }
}

window.commander = new KeyboardCommander();
