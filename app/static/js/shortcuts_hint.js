/**
 * ROX 3.0 快捷键提示组件
 * 显示所有可用的快捷键
 */

class ShortcutsHint {
    constructor() {
        this.isVisible = false;
        this.init();
    }

    init() {
        // 创建快捷键提示元素
        this.createHintElement();
        
        // 添加显示/隐藏快捷键
        document.addEventListener('keydown', (e) => {
            // Shift + ? 显示/隐藏快捷键提示
            if (e.shiftKey && e.key === '?') {
                e.preventDefault();
                this.toggleVisibility();
            }
        });
        
        // 点击其他地方隐藏
        document.addEventListener('click', (e) => {
            const hintEl = document.getElementById('shortcuts-hint');
            if (this.isVisible && hintEl && !hintEl.contains(e.target)) {
                this.hide();
            }
        });
    }

    createHintElement() {
        const hintEl = document.createElement('div');
        hintEl.id = 'shortcuts-hint';
        hintEl.className = 'keyboard-shortcuts';
        hintEl.innerHTML = `
            <div class="font-bold mb-2">快捷键提示</div>
            <div class="space-y-1">
                <div><kbd>Shift</kbd> + <kbd>?</kbd> - 显示/隐藏快捷键</div>
                <div><kbd>Ctrl</kbd> + <kbd>1</kbd> - 切换到紧凑模式</div>
                <div><kbd>Ctrl</kbd> + <kbd>2</kbd> - 切换到正常模式</div>
                <div><kbd>Ctrl</kbd> + <kbd>D</kbd> - 切换深色/浅色模式</div>
                <div><kbd>Ctrl</kbd> + <kbd>K</kbd> - 打开键盘精灵</div>
                <div><kbd>Ctrl</kbd> + <kbd>R</kbd> - 刷新数据</div>
                <div><kbd>Ctrl</kbd> + <kbd>S</kbd> - 保存布局</div>
                <div><kbd>F5</kbd> - 刷新数据</div>
                <div><kbd>F6</kbd> - 切换自选股面板</div>
                <div><kbd>F7</kbd> - 切换指标面板</div>
                <div><kbd>F8</kbd> - 切换策略构建器</div>
                <div><kbd>F9</kbd> - 切换交易面板</div>
                <div><kbd>Esc</kbd> - 关闭键盘精灵</div>
            </div>
        `;
        
        document.body.appendChild(hintEl);
    }

    toggleVisibility() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }

    show() {
        const hintEl = document.getElementById('shortcuts-hint');
        if (hintEl) {
            hintEl.classList.add('show');
            this.isVisible = true;
        }
    }

    hide() {
        const hintEl = document.getElementById('shortcuts-hint');
        if (hintEl) {
            hintEl.classList.remove('show');
            this.isVisible = false;
        }
    }
}

// 全局快捷键提示实例
window.shortcutsHint = new ShortcutsHint();
