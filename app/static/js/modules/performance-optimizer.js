/**
 * 前端性能优化模块
 * 负责提升ROX 3.0的响应速度和用户体验
 */
class PerformanceOptimizer {
    constructor() {
        this.initialized = false;
        this.observers = [];
        this.debounceTimers = {};
        this.throttleTimers = {};
        this.cache = new Map();
    }

    /**
     * 初始化性能优化模块
     */
    init() {
        if (this.initialized) return;
        
        this._optimizeDOM();
        this._optimizeEvents();
        this._optimizeAnimations();
        this._setupLazyLoading();
        this._setupVirtualScrolling();
        this._optimizeNetwork();
        this._setupMemoryManagement();
        
        this.initialized = true;
        console.log('PerformanceOptimizer initialized');
    }

    /**
     * 优化DOM操作
     */
    _optimizeDOM() {
        // 替换直接的DOM操作为文档片段
        if (!Element.prototype.appendChildren) {
            Element.prototype.appendChildren = function(children) {
                const fragment = document.createDocumentFragment();
                children.forEach(child => fragment.appendChild(child));
                this.appendChild(fragment);
                return this;
            };
        }

        // 批量DOM操作
        this.batchDOMOperations = (operations) => {
            const fragment = document.createDocumentFragment();
            operations.forEach(op => op(fragment));
            document.body.appendChild(fragment);
        };

        // 优化DOM查询
        this.querySelector = (selector, context = document) => {
            const key = `${selector}_${context === document ? 'document' : context.className || context.id}`;
            if (this.cache.has(key)) {
                return this.cache.get(key);
            }
            const element = context.querySelector(selector);
            if (element) {
                this.cache.set(key, element);
            }
            return element;
        };

        this.querySelectorAll = (selector, context = document) => {
            const key = `${selector}_all_${context === document ? 'document' : context.className || context.id}`;
            if (this.cache.has(key)) {
                return this.cache.get(key);
            }
            const elements = Array.from(context.querySelectorAll(selector));
            if (elements.length > 0) {
                this.cache.set(key, elements);
            }
            return elements;
        };
    }

    /**
     * 优化事件处理
     */
    _optimizeEvents() {
        // 事件委托
        this.delegateEvent = (parent, selector, eventType, handler) => {
            parent.addEventListener(eventType, (e) => {
                const target = e.target.closest(selector);
                if (target) {
                    handler.call(target, e);
                }
            }, {
                passive: true,
                capture: false
            });
        };

        // 防抖
        this.debounce = (func, wait, immediate = false) => {
            return (...args) => {
                const key = func.toString();
                clearTimeout(this.debounceTimers[key]);
                
                if (immediate && !this.debounceTimers[key]) {
                    func.apply(this, args);
                }
                
                this.debounceTimers[key] = setTimeout(() => {
                    if (!immediate) {
                        func.apply(this, args);
                    }
                    delete this.debounceTimers[key];
                }, wait);
            };
        };

        // 节流
        this.throttle = (func, limit) => {
            return (...args) => {
                const key = func.toString();
                if (!this.throttleTimers[key]) {
                    func.apply(this, args);
                    this.throttleTimers[key] = setTimeout(() => {
                        delete this.throttleTimers[key];
                    }, limit);
                }
            };
        };

        // 优化滚动事件
        this.optimizedScroll = (handler) => {
            const throttled = this.throttle(handler, 16); // ~60fps
            window.addEventListener('scroll', throttled, { passive: true });
            return throttled;
        };

        // 优化 resize 事件
        this.optimizedResize = (handler) => {
            const debounced = this.debounce(handler, 250);
            window.addEventListener('resize', debounced);
            return debounced;
        };
    }

    /**
     * 优化动画性能
     */
    _optimizeAnimations() {
        // 使用 transform 和 opacity 进行动画
        this.animate = (element, properties, duration = 300, easing = 'ease') => {
            const start = performance.now();
            const startProperties = {};
            
            // 获取初始值
            Object.keys(properties).forEach(prop => {
                if (prop === 'opacity') {
                    startProperties[prop] = parseFloat(getComputedStyle(element)[prop]);
                } else if (prop.startsWith('transform')) {
                    // 处理 transform 属性
                    startProperties[prop] = 0;
                }
            });

            const animateFrame = (timestamp) => {
                const elapsed = timestamp - start;
                const progress = Math.min(elapsed / duration, 1);
                
                // 应用缓动函数
                const easedProgress = this._ease(progress, easing);
                
                // 计算并应用中间值
                Object.keys(properties).forEach(prop => {
                    if (prop === 'opacity') {
                        const startValue = startProperties[prop];
                        const endValue = properties[prop];
                        element.style[prop] = startValue + (endValue - startValue) * easedProgress;
                    } else if (prop.startsWith('transform')) {
                        element.style[prop] = properties[prop];
                    }
                });

                if (progress < 1) {
                    requestAnimationFrame(animateFrame);
                }
            };

            requestAnimationFrame(animateFrame);
        };

        // 停止动画
        this.stopAnimation = (element) => {
            element.style.animation = 'none';
            element.style.transition = 'none';
        };

        // 使用 will-change 优化
        this.optimizeForAnimation = (element) => {
            element.style.willChange = 'transform, opacity';
        };

        // 恢复 will-change
        this.resetWillChange = (element) => {
            element.style.willChange = 'auto';
        };
    }

    /**
     * 设置懒加载
     */
    _setupLazyLoading() {
        // 图片懒加载
        this.setupImageLazyLoading = () => {
            const imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.classList.remove('lazy');
                        observer.unobserve(img);
                    }
                });
            }, {
                rootMargin: '200px 0px'
            });

            document.querySelectorAll('img[data-src]').forEach(img => {
                imageObserver.observe(img);
            });

            this.observers.push(imageObserver);
        };

        // 组件懒加载
        this.setupComponentLazyLoading = (selector, loader) => {
            const componentObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const component = entry.target;
                        loader(component);
                        observer.unobserve(component);
                    }
                });
            }, {
                rootMargin: '100px 0px'
            });

            document.querySelectorAll(selector).forEach(component => {
                componentObserver.observe(component);
            });

            this.observers.push(componentObserver);
        };
    }

    /**
     * 设置虚拟滚动
     */
    _setupVirtualScrolling() {
        /**
         * 创建虚拟滚动列表
         * @param {HTMLElement} container - 容器元素
         * @param {Array} items - 数据项
         * @param {Function} renderItem - 渲染函数
         * @param {number} itemHeight - 每项高度
         */
        this.createVirtualList = (container, items, renderItem, itemHeight = 50) => {
            const containerHeight = container.clientHeight;
            const visibleCount = Math.ceil(containerHeight / itemHeight);
            const bufferCount = 2;
            const totalHeight = items.length * itemHeight;

            // 创建滚动容器
            const scrollContainer = document.createElement('div');
            scrollContainer.style.height = `${totalHeight}px`;
            scrollContainer.style.position = 'relative';

            // 创建可见项目容器
            const itemsContainer = document.createElement('div');
            itemsContainer.style.position = 'absolute';
            itemsContainer.style.top = '0';
            itemsContainer.style.left = '0';
            itemsContainer.style.width = '100%';

            scrollContainer.appendChild(itemsContainer);
            container.appendChild(scrollContainer);

            // 渲染可见项目
            const render = () => {
                const scrollTop = container.scrollTop;
                const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - bufferCount);
                const endIndex = Math.min(
                    items.length,
                    startIndex + visibleCount + bufferCount * 2
                );

                // 清空容器
                itemsContainer.innerHTML = '';
                itemsContainer.style.transform = `translateY(${startIndex * itemHeight}px)`;

                // 渲染可见项目
                for (let i = startIndex; i < endIndex; i++) {
                    const itemElement = renderItem(items[i], i);
                    itemElement.style.height = `${itemHeight}px`;
                    itemsContainer.appendChild(itemElement);
                }
            };

            // 监听滚动事件
            container.addEventListener('scroll', this.throttle(render, 16), { passive: true });

            // 初始渲染
            render();

            return {
                update: (newItems) => {
                    items = newItems;
                    scrollContainer.style.height = `${items.length * itemHeight}px`;
                    render();
                },
                destroy: () => {
                    container.removeEventListener('scroll', render);
                }
            };
        };
    }

    /**
     * 优化网络请求
     */
    _optimizeNetwork() {
        // 缓存网络请求
        this.cachedFetch = async (url, options = {}) => {
            const cacheKey = `${url}_${JSON.stringify(options)}`;
            
            if (this.cache.has(cacheKey)) {
                return this.cache.get(cacheKey);
            }
            
            try {
                const response = await fetch(url, options);
                const data = await response.json();
                
                // 缓存结果，有效期5分钟
                this.cache.set(cacheKey, data);
                setTimeout(() => {
                    this.cache.delete(cacheKey);
                }, 5 * 60 * 1000);
                
                return data;
            } catch (error) {
                console.error('Cached fetch error:', error);
                throw error;
            }
        };

        // 批量请求
        this.batchRequests = async (requests) => {
            try {
                const results = await Promise.all(requests);
                return results;
            } catch (error) {
                console.error('Batch requests error:', error);
                throw error;
            }
        };

        // 取消重复请求
        this.cancelableFetch = (url, options = {}) => {
            const controller = new AbortController();
            const signal = controller.signal;
            
            const fetchPromise = fetch(url, {
                ...options,
                signal
            });
            
            return {
                promise: fetchPromise,
                cancel: () => controller.abort()
            };
        };
    }

    /**
     * 设置内存管理
     */
    _setupMemoryManagement() {
        // 清理缓存
        this.clearCache = () => {
            this.cache.clear();
            console.log('Cache cleared');
        };

        // 清理定时器
        this.clearTimers = () => {
            Object.values(this.debounceTimers).forEach(timer => clearTimeout(timer));
            Object.values(this.throttleTimers).forEach(timer => clearTimeout(timer));
            this.debounceTimers = {};
            this.throttleTimers = {};
            console.log('Timers cleared');
        };

        // 清理观察者
        this.clearObservers = () => {
            this.observers.forEach(observer => observer.disconnect());
            this.observers = [];
            console.log('Observers cleared');
        };

        // 完全清理
        this.destroy = () => {
            this.clearCache();
            this.clearTimers();
            this.clearObservers();
            this.initialized = false;
            console.log('PerformanceOptimizer destroyed');
        };
    }

    /**
     * 缓动函数
     */
    _ease(t, type) {
        switch (type) {
            case 'linear':
                return t;
            case 'ease':
                return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
            case 'ease-in':
                return t * t * t;
            case 'ease-out':
                return 1 - Math.pow(1 - t, 3);
            case 'ease-in-out':
                return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
            default:
                return t;
        }
    }

    /**
     * 测量代码执行时间
     */
    measureTime(name, fn) {
        const start = performance.now();
        const result = fn();
        const end = performance.now();
        console.log(`${name} took ${end - start}ms`);
        return result;
    }

    /**
     * 检测性能瓶颈
     */
    detectPerformanceIssues() {
        if ('performance' in window && 'measure' in window.performance) {
            // 检测长任务
            const observer = new PerformanceObserver((list) => {
                list.getEntries().forEach((entry) => {
                    if (entry.duration > 50) {
                        console.warn(`Long task detected: ${entry.name} took ${entry.duration}ms`);
                    }
                });
            });

            observer.observe({ entryTypes: ['longtask'] });
            this.observers.push(observer);
        }

        // 检测内存使用
        if ('memory' in performance) {
            setInterval(() => {
                const memory = performance.memory;
                console.log(`Memory usage: ${Math.round(memory.usedJSHeapSize / 1024 / 1024)}MB`);
                
                if (memory.usedJSHeapSize > memory.jsHeapSizeLimit * 0.8) {
                    console.warn('Memory usage approaching limit');
                    this.clearCache();
                }
            }, 30000); // 每30秒检查一次
        }
    }
}

// 导出单例
const performanceOptimizer = new PerformanceOptimizer();
if (typeof module !== 'undefined' && module.exports) {
    module.exports = performanceOptimizer;
} else {
    window.PerformanceOptimizer = performanceOptimizer;
}
