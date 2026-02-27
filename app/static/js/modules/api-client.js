/**
 * API客户端 - 统一管理所有API请求
 * 
 * 功能特性：
 * - 请求拦截和响应处理
 * - 自动重试机制
 * - 请求缓存
 * - 错误处理
 * - 加载状态管理
 */
class APIClient {
    constructor() {
        this.baseURL = '';
        this.defaultTimeout = 10000;
        this.retryAttempts = 3;
        this.retryDelay = 1000;
        this.cache = new Map();
        this.pendingRequests = new Map();
        this.loadingStates = new Map();
        
        // 绑定原始fetch以避免被覆盖
        this.originalFetch = window.fetch;
        
        // 初始化请求拦截器
        this.initInterceptors();
    }

    /**
     * 统一 token 刷新：即将过期（5 分钟内）时自动刷新，避免多标签 401
     */
    async ensureTokenRefreshed() {
        const token = localStorage.getItem('access_token');
        if (!token || !token.includes('.')) return;
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            const exp = payload.exp ? payload.exp * 1000 : 0;
            const needRefresh = exp && (exp - Date.now() < 5 * 60 * 1000); // 5 分钟内过期
            if (!needRefresh) return;
            const r = await this.originalFetch('/token/refresh', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (r.ok) {
                const data = await r.json();
                if (data.access_token) localStorage.setItem('access_token', data.access_token);
            }
        } catch (e) { /* ignore */ }
    }

    /**
     * 初始化请求拦截器：自动附加认证、超时、GET 请求 5xx/网络错误重试
     */
    initInterceptors() {
        const originalFetch = this.originalFetch;
        const self = this;

        const doOne = async (url, options, signal) => {
            const opts = { ...options };
            if (signal) opts.signal = signal;
            return originalFetch(url, opts);
        };

        window.fetch = async (url, options = {}, attempt = 1) => {
            await self.ensureTokenRefreshed();
            const token = localStorage.getItem('access_token');
            if (token && (url.startsWith('/') || url.startsWith(window.location.origin)) && !url.includes('/token')) {
                options.headers = options.headers || {};
                if (!options.headers['Authorization']) {
                    options.headers['Authorization'] = `Bearer ${token}`;
                }
            }

            const method = (options.method || 'GET').toUpperCase();
            const timeout = options.timeout != null ? options.timeout : self.defaultTimeout;
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);
            const mergedOpts = { ...options, signal: controller.signal };

            let response;
            try {
                response = await doOne(url, mergedOpts, controller.signal);
            } catch (err) {
                clearTimeout(timeoutId);
                const isRetryable = method === 'GET' && attempt < self.retryAttempts && (err.name === 'AbortError' || err.message === 'Failed to fetch');
                if (isRetryable) {
                    console.warn("[APIClient] Request failed (attempt " + attempt + "/" + self.retryAttempts + "), retrying...", err.message);
                    await self.delay(self.retryDelay);
                    return window.fetch(url, options, attempt + 1);
                }
                throw err;
            }
            clearTimeout(timeoutId);

            const serverError = response.status >= 500 && response.status < 600;
            if (method === 'GET' && serverError && attempt < self.retryAttempts) {
                console.warn("[APIClient] Server error " + response.status + " (attempt " + attempt + "/" + self.retryAttempts + "), retrying...");
                await self.delay(self.retryDelay);
                return window.fetch(url, options, attempt + 1);
            }

            if (response.status === 401 && !url.includes('/token')) {
                console.warn("[APIClient] Unauthorized, redirecting to login...");
                localStorage.removeItem('access_token');
                self.handleAuthError();
            }
            return response;
        };
    }

    /**
     * 处理认证错误
     */
    handleAuthError() {
        // 触发登录模态框
        if (typeof showAuthModal === 'function') {
            showAuthModal();
        } else {
            const authModal = document.getElementById('auth-modal');
            if (authModal) {
                authModal.classList.remove('hidden');
            }
        }
    }

    /**
     * 生成缓存键
     * @param {string} url - 请求URL
     * @param {Object} options - 请求选项
     * @returns {string} 缓存键
     */
    getCacheKey(url, options = {}) {
        const method = options.method || 'GET';
        const body = options.body || '';
        return `${method}:${url}:${typeof body === 'string' ? body : JSON.stringify(body)}`;
    }

    /**
     * 设置加载状态
     * @param {string} key - 状态键
     * @param {boolean} loading - 是否加载中
     */
    setLoading(key, loading) {
        this.loadingStates.set(key, loading);
        
        // 触发加载状态变化事件
        window.dispatchEvent(new CustomEvent('loadingStateChange', {
            detail: { key, loading }
        }));
    }

    /**
     * 获取加载状态
     * @param {string} key - 状态键
     * @returns {boolean} 是否加载中
     */
    isLoading(key) {
        return this.loadingStates.get(key) || false;
    }

    /**
     * 延迟函数
     * @param {number} ms - 延迟毫秒数
     * @returns {Promise} 延迟Promise
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * 带重试的请求
     * @param {string} url - 请求URL
     * @param {Object} options - 请求选项
     * @param {number} attempt - 当前尝试次数
     * @returns {Promise} 响应Promise
     */
    async fetchWithRetry(url, options = {}, attempt = 1) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.defaultTimeout);
            
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return response;
        } catch (error) {
            if (attempt < this.retryAttempts && !error.name === 'AbortError') {
                console.warn(`[APIClient] Request failed (attempt ${attempt}/${this.retryAttempts}):`, error.message);
                await this.delay(this.retryDelay * attempt);
                return this.fetchWithRetry(url, options, attempt + 1);
            }
            throw error;
        }
    }

    /**
     * 显示错误提示
     * @param {string} message - 错误消息
     * @param {string} type - 错误类型
     */
    showError(message, type = 'error') {
        // 触发错误事件，让其他组件可以处理
        window.dispatchEvent(new CustomEvent('apiError', {
            detail: { message, type }
        }));
        
        // 显示错误提示
        if (typeof showToast === 'function') {
            showToast(message, type);
        } else if (typeof toastr !== 'undefined') {
            toastr[type](message);
        } else {
            console.error(`[APIClient] ${message}`);
        }
    }

    /**
     * 处理API错误响应
     * @param {Response} response - 响应对象
     * @returns {Promise} 处理后的错误
     */
    async handleErrorResponse(response) {
        try {
            const errorData = await response.json();
            const errorMessage = errorData.error || `HTTP ${response.status}: ${response.statusText}`;
            const errorCode = errorData.code;
            
            // 显示友好的错误提示
            this.showError(errorMessage, 'error');
            
            // 特殊处理认证错误
            if (response.status === 401) {
                this.handleAuthError();
            }
            
            throw new Error(errorMessage);
        } catch (error) {
            const errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            this.showError(errorMessage, 'error');
            throw new Error(errorMessage);
        }
    }

    /**
     * GET请求
     * @param {string} url - 请求URL
     * @param {Object} options - 请求选项
     * @returns {Promise} 响应数据
     */
    async get(url, options = {}) {
        const { cache = false, cacheTime = 60000, ...fetchOptions } = options;
        const cacheKey = this.getCacheKey(url, fetchOptions);
        
        // 检查缓存
        if (cache) {
            const cached = this.cache.get(cacheKey);
            if (cached && Date.now() - cached.timestamp < cacheTime) {
                return cached.data;
            }
        }
        
        // 检查是否有相同的请求正在进行
        if (this.pendingRequests.has(cacheKey)) {
            return this.pendingRequests.get(cacheKey);
        }
        
        this.setLoading(url, true);
        
        const requestPromise = (async () => {
            try {
                const response = await this.fetchWithRetry(url, {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                        ...fetchOptions.headers
                    },
                    ...fetchOptions
                });
                
                if (!response.ok) {
                    await this.handleErrorResponse(response);
                }
                
                const data = await response.json();
                
                // 缓存响应
                if (cache) {
                    this.cache.set(cacheKey, {
                        data,
                        timestamp: Date.now()
                    });
                }
                
                return data;
            } catch (error) {
                console.error(`[APIClient] GET ${url} failed:`, error);
                // 显示网络错误提示
                if (error.name === 'AbortError') {
                    this.showError('请求超时，请稍后重试', 'warning');
                } else if (error.message === 'Failed to fetch') {
                    this.showError('网络连接失败，请检查网络设置', 'error');
                } else {
                    this.showError(error.message, 'error');
                }
                throw error;
            } finally {
                this.setLoading(url, false);
                this.pendingRequests.delete(cacheKey);
            }
        })();
        
        this.pendingRequests.set(cacheKey, requestPromise);
        return requestPromise;
    }

    /**
     * POST请求
     * @param {string} url - 请求URL
     * @param {Object} data - 请求数据
     * @param {Object} options - 请求选项
     * @returns {Promise} 响应数据
     */
    async post(url, data = null, options = {}) {
        this.setLoading(url, true);
        
        try {
            const response = await this.fetchWithRetry(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                body: data ? JSON.stringify(data) : null,
                ...options
            });
            
            if (!response.ok) {
                await this.handleErrorResponse(response);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`[APIClient] POST ${url} failed:`, error);
            // 显示网络错误提示
            if (error.name === 'AbortError') {
                this.showError('请求超时，请稍后重试', 'warning');
            } else if (error.message === 'Failed to fetch') {
                this.showError('网络连接失败，请检查网络设置', 'error');
            } else {
                this.showError(error.message, 'error');
            }
            throw error;
        } finally {
            this.setLoading(url, false);
        }
    }

    /**
     * PUT请求
     * @param {string} url - 请求URL
     * @param {Object} data - 请求数据
     * @param {Object} options - 请求选项
     * @returns {Promise} 响应数据
     */
    async put(url, data = null, options = {}) {
        this.setLoading(url, true);
        
        try {
            const response = await this.fetchWithRetry(url, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                body: data ? JSON.stringify(data) : null,
                ...options
            });
            
            if (!response.ok) {
                await this.handleErrorResponse(response);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`[APIClient] PUT ${url} failed:`, error);
            // 显示网络错误提示
            if (error.name === 'AbortError') {
                this.showError('请求超时，请稍后重试', 'warning');
            } else if (error.message === 'Failed to fetch') {
                this.showError('网络连接失败，请检查网络设置', 'error');
            } else {
                this.showError(error.message, 'error');
            }
            throw error;
        } finally {
            this.setLoading(url, false);
        }
    }

    /**
     * DELETE请求
     * @param {string} url - 请求URL
     * @param {Object} options - 请求选项
     * @returns {Promise} 响应数据
     */
    async delete(url, options = {}) {
        this.setLoading(url, true);
        
        try {
            const response = await this.fetchWithRetry(url, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                await this.handleErrorResponse(response);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`[APIClient] DELETE ${url} failed:`, error);
            // 显示网络错误提示
            if (error.name === 'AbortError') {
                this.showError('请求超时，请稍后重试', 'warning');
            } else if (error.message === 'Failed to fetch') {
                this.showError('网络连接失败，请检查网络设置', 'error');
            } else {
                this.showError(error.message, 'error');
            }
            throw error;
        } finally {
            this.setLoading(url, false);
        }
    }

    /**
     * 批量请求
     * @param {Array} requests - 请求配置数组
     * @returns {Promise} 所有响应的Promise
     */
    async batch(requests) {
        const promises = requests.map(async (config) => {
            const { method = 'GET', url, data, options = {} } = config;
            
            try {
                switch (method.toLowerCase()) {
                    case 'get':
                        return await this.get(url, options);
                    case 'post':
                        return await this.post(url, data, options);
                    case 'put':
                        return await this.put(url, data, options);
                    case 'delete':
                        return await this.delete(url, options);
                    default:
                        throw new Error(`Unsupported method: ${method}`);
                }
            } catch (error) {
                return { error: error.message, url };
            }
        });
        
        return Promise.all(promises);
    }

    /**
     * 清除缓存
     * @param {string} pattern - 缓存键模式（可选）
     */
    clearCache(pattern = null) {
        if (pattern) {
            const regex = new RegExp(pattern);
            for (const [key] of this.cache) {
                if (regex.test(key)) {
                    this.cache.delete(key);
                }
            }
        } else {
            this.cache.clear();
        }
    }

    /**
     * 获取缓存统计
     * @returns {Object} 缓存统计信息
     */
    getCacheStats() {
        return {
            size: this.cache.size,
            keys: Array.from(this.cache.keys())
        };
    }

    /**
     * 取消所有待处理的请求
     */
    cancelAllRequests() {
        this.pendingRequests.clear();
        this.loadingStates.clear();
    }
}

// 创建全局API客户端实例
window.apiClient = new APIClient();

// 为了向后兼容，保留原有的fetchJSON函数
window.fetchJSON = async function(url, options = {}) {
    try {
        return await window.apiClient.get(url, { 
            cache: true, 
            cacheTime: 30000, // 30秒缓存
            ...options 
        });
    } catch (error) {
        console.error(`[fetchJSON] ${url} failed:`, error);
        throw error;
    }
};

console.log('[APIClient] Initialized successfully');