/**
 * ROX 3.0 布局管理器
 * 支持紧凑模式和正常模式切换
 */

class LayoutManager {
    constructor() {
        this.currentLayout = localStorage.getItem('rox-layout') || 'normal';
        this.init();
    }

    init() {
        // 应用保存的布局
        this.applyLayout(this.currentLayout);
        
        // 初始化布局切换按钮
        this.initLayoutButtons();
        
        // 响应式布局
        this.initResponsiveLayout();
    }

    applyLayout(layout) {
        const body = document.body;
        
        // 移除所有布局类
        body.classList.remove('layout-compact', 'layout-normal');
        
        // 添加当前布局类
        body.classList.add(`layout-${layout}`);
        body.dataset.layout = layout;
        
        this.currentLayout = layout;
        localStorage.setItem('rox-layout', layout);
        
        // 通知其他组件布局变更
        this.notifyLayoutChange(layout);
    }

    initLayoutButtons() {
        // 创建布局切换按钮
        const layoutToggle = document.createElement('div');
        layoutToggle.id = 'layout-toggle';
        layoutToggle.className = 'fixed top-4 right-4 z-50 flex gap-2';
        layoutToggle.innerHTML = `
            <button id="btn-compact" class="px-3 py-1 bg-blue-700 text-white rounded-md text-sm hover:bg-blue-600 transition-colors">
                紧凑模式
            </button>
            <button id="btn-normal" class="px-3 py-1 bg-gray-700 text-white rounded-md text-sm hover:bg-gray-600 transition-colors">
                正常模式
            </button>
        `;
        
        document.body.appendChild(layoutToggle);
        
        // 添加事件监听器
        document.getElementById('btn-compact').addEventListener('click', () => {
            this.applyLayout('compact');
        });
        
        document.getElementById('btn-normal').addEventListener('click', () => {
            this.applyLayout('normal');
        });
        
        // 更新按钮状态
        this.updateButtonStates();
    }

    updateButtonStates() {
        const btnCompact = document.getElementById('btn-compact');
        const btnNormal = document.getElementById('btn-normal');
        
        if (btnCompact && btnNormal) {
            if (this.currentLayout === 'compact') {
                btnCompact.classList.add('bg-blue-600');
                btnCompact.classList.remove('bg-blue-700');
                btnNormal.classList.add('bg-gray-700');
                btnNormal.classList.remove('bg-gray-600');
            } else {
                btnCompact.classList.add('bg-blue-700');
                btnCompact.classList.remove('bg-blue-600');
                btnNormal.classList.add('bg-gray-600');
                btnNormal.classList.remove('bg-gray-700');
            }
        }
    }

    initResponsiveLayout() {
        const handleResize = () => {
            const width = window.innerWidth;
            
            // 在小屏幕上自动切换到紧凑模式
            if (width < 1024 && this.currentLayout !== 'compact') {
                this.applyLayout('compact');
            }
        };
        
        window.addEventListener('resize', handleResize);
        handleResize(); // 初始检查
    }

    notifyLayoutChange(layout) {
        // 触发自定义事件
        const event = new CustomEvent('layoutChange', {
            detail: { layout }
        });
        window.dispatchEvent(event);
        
        // 通知图表组件
        if (window.chart && typeof window.chart.resize === 'function') {
            setTimeout(() => {
                window.chart.resize();
            }, 100);
        }
    }

    getCurrentLayout() {
        return this.currentLayout;
    }
}

// 全局布局管理器实例
window.layoutManager = new LayoutManager();

// 全局布局切换函数
window.switchLayout = function(layout) {
    if (window.layoutManager) {
        window.layoutManager.applyLayout(layout);
    }
};

// 全局保存布局函数
window.saveLayout = function() {
    // 保存当前布局设置
    localStorage.setItem('rox-layout-saved', JSON.stringify({
        layout: window.layoutManager.getCurrentLayout(),
        timestamp: new Date().toISOString()
    }));
    
    // 显示保存成功提示
    const toast = document.createElement('div');
    toast.className = 'fixed bottom-4 right-4 bg-green-600 text-white px-4 py-2 rounded-md z-50';
    toast.textContent = '布局已保存';
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 2000);
};
