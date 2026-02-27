class AIAssistant {
    constructor() {
        this.container = document.getElementById('ai-assistant-container');
        this.input = document.getElementById('ai-chat-input');
        this.sendBtn = document.getElementById('ai-chat-send');
        this.chatHistory = document.getElementById('ai-chat-history');
        if (this.container) {
            this.init();
        }
    }

    init() {
        if (this.sendBtn) {
            this.sendBtn.addEventListener('click', () => this.sendMessage());
        }
        if (this.input) {
            this.input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                                                                                                                                                                                                                                                                                                        '';

                          const response = await fetch('/api/ai_assistant/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });
            const data = await response.json();
            // 添加 AI 回复
            this.addMessage('ai', data.response);
        } catch (error) {
            console.error('Failed to send message:', error);
            this.addMessage('ai', '抱歉，AI 服务暂时不可用，请稍后再试。');
        }
    }

    addMessage(sender, content) {
        if (!this.chatHistory) return;
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender === 'user' ? 'user-message' : 'ai-message'} mb-4`;
        messageDiv.innerHTML = `
            <div class="flex items-start">
                <div class="w-8 h-8 rounded-full flex items-center justify-center ${sender === 'user' ? 'bg-sky-600' : 'bg-purple-600'} text-white">
                    ${sender === 'user' ? 'U' : 'AI'}
                </div>
                <div class="ml-3 flex-1">
                    <p class="text-sm ${sender === 'user' ? 'text-white' : 'text-gray-300'}">${content}</p>
                </div>
            </div>
        `;
        this.chatHistory.appendChild(messageDiv);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }
}
