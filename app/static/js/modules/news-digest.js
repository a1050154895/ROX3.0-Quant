class NewsDigest {
    constructor() {
        this.container = document.getElementById('news-container');
        if (this.container) {
            this.init();
        }
    }

    async init() {
        await this.loadDigests();
    }

    async loadDigests() {
        try {
            const response = await fetch('/api/news/digests?type=daily&limit=10');
            const data = await response.json();
            this.renderDigests(data);
        } catch (error) {
            console.error('Failed to load news digests:', error);
        }
    }

    renderDigests(digests) {
        if (!this.container) return;
        this.container.innerHTML = digests.map(digest => `
            <div class="news-card bg-[#1a1a1a] border border-[#333] rounded-lg p-4 mb-4">
                <h3 class="text-yellow-500 font-bold mb-2">${digest.title}</h3>
                <p class="text-gray-400 text-sm mb-3">${digest.summary}</p>
                                          etween text-xs text-gray-500">
                    <span>${digest.date}</span>
                    <button class="text-sky-400 hover:text-sky-300" onclick="loadDigestDetail('${digest.id}')">查看详情</button>
                </div>
            </div>
        `).join('');
    }
}
