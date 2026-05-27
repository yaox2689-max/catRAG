import { marked } from "marked";
import hljs from "highlight.js";

export default {
    data() {
        return {
            messages: [],
            userInput: '',
            isLoading: false,
            activeNav: 'newChat',
            abortController: null,
            sessionId: 'session_' + Date.now(),
            sessions: [],
            showHistorySidebar: false,
            isComposing: false,
            documents: [],
            documentsLoading: false,
            resourcePage: 1,
            resourcePageSize: 6,
            previewImageUrl: '',
            previewImageName: '',
            previewOcrText: '',
            previewOcrMessage: '',
            previewExpanded: true,
            showImagePreview: false,
            imageContextText: '',
            chatAttachments: [],
            selectedFile: null,
            isUploading: false,
            uploadProgress: '',
            uploadSteps: [],
            uploadProgressCollapsed: false,
            activeUploadJobId: '',
            uploadPollTimer: null,
            showAttachMenu: false,
            ocrSelectedFile: null,
            isOcrUploading: false,
            ocrResult: '',
            ocrMessage: '',
            deleteJobs: {},
            deletePollTimers: {},
            deleteRemoveTimers: {},
            token: localStorage.getItem('accessToken') || '',
            currentUser: null,
            authMode: 'login',
            authForm: {
                username: '',
                password: '',
                role: 'user',
                admin_code: ''
            },
            authLoading: false
        };
    },
    computed: {
        isAuthenticated() {
            return !!this.token && !!this.currentUser;
        },
        isAdmin() {
            return this.currentUser?.role === 'admin';
        },
        resourceGroups() {
            const imageExt = /\.(png|jpe?g|gif|bmp|webp|tiff?)$/i;
            const images = (this.documents || []).filter(item => {
                const type = (item.file_type || '').toLowerCase();
                const name = (item.filename || '').toLowerCase();
                return type === 'image' || imageExt.test(name);
            });
            const docs = (this.documents || []).filter(item => !images.some(img => img.filename === item.filename));
            return { docs, images };
        },
        pagedResources() {
            const all = this.documents || [];
            const start = (this.resourcePage - 1) * this.resourcePageSize;
            return all.slice(start, start + this.resourcePageSize);
        },
        resourceTotalPages() {
            return Math.max(1, Math.ceil((this.documents.length || 0) / this.resourcePageSize));
        }
    },
    async mounted() {
        this.configureMarked();
        if (this.token) {
            try {
                await this.fetchMe();
            } catch (_) {
                this.handleLogout();
            }
        }
    },
    beforeUnmount() {
        this.stopUploadJobPolling();
        this.stopAllDeleteJobPolling();
        Object.values(this.deleteRemoveTimers).forEach(timer => clearTimeout(timer));
    },
    methods: {
        configureMarked() {
            marked.setOptions({
                highlight: function(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                },
                langPrefix: 'hljs language-',
                breaks: true,
                gfm: true
            });
        },

        parseMarkdown(text) {
            return marked.parse(text);
        },

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },

        authHeaders(extra = {}) {
            const headers = { ...extra };
            if (this.token) {
                headers.Authorization = `Bearer ${this.token}`;
            }
            return headers;
        },

        async authFetch(url, options = {}) {
            const opts = { ...options };
            opts.headers = this.authHeaders(opts.headers || {});
            const response = await fetch(url, opts);
            if (response.status === 401) {
                this.handleLogout();
                throw new Error('登录已过期，请重新登录');
            }
            return response;
        },

        async fetchMe() {
            const response = await this.authFetch('/auth/me');
            if (!response.ok) {
                throw new Error('认证失败');
            }
            this.currentUser = await response.json();
        },

        async handleAuthSubmit() {
            if (this.authLoading) return;
            const username = this.authForm.username.trim();
            const password = this.authForm.password.trim();
            if (!username || !password) {
                alert('用户名和密码不能为空');
                return;
            }

            this.authLoading = true;
            try {
                const endpoint = this.authMode === 'login' ? '/auth/login' : '/auth/register';
                const payload = {
                    username,
                    password
                };
                if (this.authMode === 'register') {
                    payload.role = this.authForm.role;
                    payload.admin_code = this.authForm.admin_code || null;
                }

                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(data.detail || '认证失败');
                }

                this.token = data.access_token;
                this.currentUser = { username: data.username, role: data.role };
                localStorage.setItem('accessToken', this.token);
                this.authForm.password = '';
                this.authForm.admin_code = '';
                this.messages = [];
                this.sessionId = 'session_' + Date.now();
                this.activeNav = 'newChat';
            } catch (error) {
                alert(error.message);
            } finally {
                this.authLoading = false;
            }
        },

        handleLogout() {
            this.token = '';
            this.currentUser = null;
            this.messages = [];
            this.sessions = [];
            this.documents = [];
            this.resourcePage = 1;
            this.previewImageUrl = '';
            this.previewImageName = '';
            this.previewOcrText = '';
            this.previewOcrMessage = '';
            this.imageContextText = '';
            this.chatAttachments = [];
            this.selectedFile = null;
            this.activeNav = 'newChat';
            this.showHistorySidebar = false;
            this.showAttachMenu = false;
            localStorage.removeItem('accessToken');
        },

        handleCompositionStart() {
            this.isComposing = true;
        },

        handleCompositionEnd() {
            this.isComposing = false;
        },

        handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey && !this.isComposing) {
                event.preventDefault();
                this.handleSend();
            }
        },

        handleStop() {
            if (this.abortController) {
                this.abortController.abort();
            }
        },

        toggleAttachMenu() {
            this.showAttachMenu = !this.showAttachMenu;
        },

        closeAttachMenu() {
            this.showAttachMenu = false;
        },

        handleChatDocumentSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                const file = files[0];
                const item = {
                    id: `file_${Date.now()}_${file.name}`,
                    kind: 'file',
                    name: file.name,
                    file,
                    previewUrl: '',
                    text: '',
                    status: 'ready',
                    contentB64: '',
                    size: file.size,
                    extension: this.getFileExtension(file.name),
                };
                this.chatAttachments = [item, ...this.chatAttachments.filter(v => v.kind !== 'file')];
                const reader = new FileReader();
                reader.onload = () => {
                    const result = String(reader.result || '');
                    const commaIdx = result.indexOf(',');
                    item.contentB64 = commaIdx >= 0 ? result.slice(commaIdx + 1) : '';
                };
                reader.readAsDataURL(file);
            }
            if (this.$refs.chatFileInput) {
                this.$refs.chatFileInput.value = '';
            }
            this.closeAttachMenu();
        },

        handleChatImageSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                const file = files[0];
                const previewUrl = URL.createObjectURL(file);
                const item = {
                    id: `image_${Date.now()}_${file.name}`,
                    kind: 'image',
                    name: file.name,
                    file,
                    previewUrl,
                    text: '',
                    status: 'pending',
                    size: file.size,
                    extension: this.getFileExtension(file.name),
                };
                this.chatAttachments = [item, ...this.chatAttachments.filter(v => v.kind !== 'image')];
                this.ocrSelectedFile = file;
                this.previewImageUrl = previewUrl;
                this.previewImageName = file.name;
                this.previewOcrText = '';
                this.previewOcrMessage = '准备识别...';
                this.ocrResult = '';
                this.ocrMessage = '';
                this.imageContextText = '';
                this.uploadImageOcr(item);
            }
            if (this.$refs.chatImageInput) {
                this.$refs.chatImageInput.value = '';
            }
            this.closeAttachMenu();
        },

        handleChatAttachClick(kind) {
            this.closeAttachMenu();
            if (kind === 'image' && this.$refs.chatImageInput) {
                this.$refs.chatImageInput.click();
            }
            if (kind === 'document' && this.$refs.chatFileInput) {
                this.$refs.chatFileInput.click();
            }
        },

        openAttachment(item) {
            if (!item) return;
            if (item.kind === 'image' && item.previewUrl) {
                this.previewImageUrl = item.previewUrl;
                this.previewImageName = item.name;
                this.previewOcrText = item.text || '';
                this.previewOcrMessage = item.status === 'done' ? '已完成识别' : (item.status === 'pending' ? '识别中...' : '');
                this.showImagePreview = true;
                return;
            }
        },

        closeImagePreview() {
            this.showImagePreview = false;
        },

        async handleSend() {
            if (!this.isAuthenticated) {
                alert('请先登录');
                return;
            }

            const text = this.userInput.trim();
            const imageContext = (this.imageContextText || '').trim();
            const fileAttachment = this.chatAttachments.find(v => v.kind === 'file');
            const hasFileAttachment = Boolean(fileAttachment?.contentB64);
            if ((!text && !imageContext && !hasFileAttachment) || this.isLoading || this.isComposing) return;

            const userContent = text
                || (imageContext ? `【图片内容】\n${imageContext}` : '')
                || (hasFileAttachment ? `【已上传文件：${fileAttachment.name}】` : '');
            this.messages.push({
                text: userContent,
                isUser: true,
                imageContext: imageContext || ''
            });

            this.userInput = '';
            this.imageContextText = '';
            this.previewImageUrl = '';
            this.previewImageName = '';
            this.previewOcrText = '';
            this.previewOcrMessage = '';
            this.chatAttachments = [];
            this.$nextTick(() => {
                this.resetTextareaHeight();
                this.scrollToBottom();
            });

            this.isLoading = true;
            this.messages.push({
                text: '',
                isUser: false,
                isThinking: true,
                ragTrace: null,
                ragSteps: []
            });
            const botMsgIdx = this.messages.length - 1;

            this.abortController = new AbortController();

            try {
                const response = await this.authFetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: userContent,
                        session_id: this.sessionId,
                        image_context: imageContext || '',
                        file_name: fileAttachment?.name || '',
                        file_content_b64: fileAttachment?.contentB64 || ''
                    }),
                    signal: this.abortController.signal,
                });

                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);

                        if (eventStr.startsWith('data: ')) {
                            const dataStr = eventStr.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.type === 'content') {
                                    if (this.messages[botMsgIdx].isThinking) {
                                        this.messages[botMsgIdx].isThinking = false;
                                    }
                                    this.messages[botMsgIdx].text += data.content;
                                } else if (data.type === 'trace') {
                                    this.messages[botMsgIdx].ragTrace = data.rag_trace;
                                } else if (data.type === 'rag_step') {
                                    if (!this.messages[botMsgIdx].ragSteps) {
                                        this.messages[botMsgIdx].ragSteps = [];
                                    }
                                    this.messages[botMsgIdx].ragSteps.push(data.step);
                                } else if (data.type === 'error') {
                                    this.messages[botMsgIdx].isThinking = false;
                                    this.messages[botMsgIdx].text += `\n[Error: ${data.content}]`;
                                }
                            } catch (e) {
                                console.warn('SSE parse error:', e);
                            }
                        }
                    }
                    this.$nextTick(() => this.scrollToBottom());
                }

            } catch (error) {
                if (error.name === 'AbortError') {
                    this.messages[botMsgIdx].isThinking = false;
                    if (!this.messages[botMsgIdx].text) {
                        this.messages[botMsgIdx].text = '(已终止回答)';
                    } else {
                        this.messages[botMsgIdx].text += '\n\n_(回答已被终止)_';
                    }
                } else {
                    this.messages[botMsgIdx].isThinking = false;
                    this.messages[botMsgIdx].text = `喵呜... 出了点问题：${error.message}`;
                }
            } finally {
                this.isLoading = false;
                this.abortController = null;
                this.$nextTick(() => this.scrollToBottom());
            }
        },

        autoResize(event) {
            const textarea = event.target;
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        },

        resetTextareaHeight() {
            if (this.$refs.textarea) {
                this.$refs.textarea.style.height = 'auto';
            }
        },

        scrollToBottom() {
            if (this.$refs.chatContainer) {
                this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
            }
        },

        handleNewChat() {
            if (!this.isAuthenticated) return;
            this.messages = [];
            this.sessionId = 'session_' + Date.now();
            this.activeNav = 'newChat';
            this.showHistorySidebar = false;
        },

        handleClearChat() {
            if (confirm('确定要清空当前对话吗？喵？')) {
                this.messages = [];
            }
        },

        async handleHistory() {
            if (!this.isAuthenticated) return;
            this.activeNav = 'history';
            this.showHistorySidebar = true;
            try {
                const response = await this.authFetch('/sessions');
                if (!response.ok) {
                    throw new Error('Failed to load sessions');
                }
                const data = await response.json();
                this.sessions = data.sessions;
            } catch (error) {
                alert('加载历史记录失败：' + error.message);
            }
        },

        async loadSession(sessionId) {
            this.sessionId = sessionId;
            this.showHistorySidebar = false;
            this.activeNav = 'newChat';

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`);
                if (!response.ok) {
                    throw new Error('Failed to load session messages');
                }
                const data = await response.json();
                this.messages = data.messages.map(msg => ({
                    text: msg.content,
                    isUser: msg.type === 'human',
                    ragTrace: msg.rag_trace || null
                }));

                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            } catch (error) {
                alert('加载会话失败：' + error.message);
                this.messages = [];
            }
        },

        async deleteSession(sessionId) {
            if (!confirm(`确定要删除会话 "${sessionId}" 吗？`)) {
                return;
            }

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`, {
                    method: 'DELETE'
                });

                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.detail || 'Delete failed');
                }

                this.sessions = this.sessions.filter(s => s.session_id !== sessionId);

                if (this.sessionId === sessionId) {
                    this.messages = [];
                    this.sessionId = 'session_' + Date.now();
                    this.activeNav = 'newChat';
                }

                if (payload.message) {
                    alert(payload.message);
                }
            } catch (error) {
                alert('删除会话失败：' + error.message);
            }
        },

        handleSettings() {
            if (!this.isAdmin) {
                alert('仅管理员可访问文档管理');
                return;
            }
            this.activeNav = 'settings';
            this.showHistorySidebar = false;
            this.loadDocuments();
        },

        mergeDocumentsWithActiveDeletes(nextDocuments) {
            const merged = Array.isArray(nextDocuments) ? [...nextDocuments] : [];
            Object.keys(this.deleteJobs).forEach(filename => {
                const job = this.deleteJobs[filename];
                if (!job || job.status === 'failed') return;
                const exists = merged.some(doc => doc.filename === filename);
                if (!exists) {
                    const currentDoc = this.documents.find(doc => doc.filename === filename);
                    if (currentDoc) {
                        merged.push(currentDoc);
                    }
                }
            });
            return merged;
        },

        async loadDocuments() {
            this.documentsLoading = true;
            try {
                const response = await this.authFetch('/documents');
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || 'Failed to load documents');
                }
                const data = await response.json();
                const merged = this.mergeDocumentsWithActiveDeletes(data.documents);
                const map = new Map();
                merged.forEach(item => {
                    const key = (item.filename || '').trim();
                    if (!key) return;
                    const prev = map.get(key);
                    if (!prev || (Number(item.chunk_count || 0) > Number(prev.chunk_count || 0))) {
                        map.set(key, item);
                    }
                });
                this.documents = Array.from(map.values());
                this.resourcePage = 1;
            } catch (error) {
                alert('加载文档列表失败：' + error.message);
            } finally {
                this.documentsLoading = false;
            }
        },

        handleFileSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                this.selectedFile = files[0];
                this.uploadProgress = '';
                this.uploadSteps = this.createUploadSteps();
                this.uploadProgressCollapsed = false;
                this.activeUploadJobId = '';
            }
        },

        clearSelectedFile() {
            this.selectedFile = null;
            if (this.$refs.settingsFileInput) {
                this.$refs.settingsFileInput.value = '';
            }
        },

        handleOcrFileSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                this.ocrSelectedFile = files[0];
                this.ocrResult = '';
                this.ocrMessage = '';
            }
        },

        createUploadSteps() {
            return [
                { key: 'upload', label: '文档上传', percent: 0, status: 'pending', message: '' },
                { key: 'cleanup', label: '清理旧版本', percent: 0, status: 'pending', message: '' },
                { key: 'parse', label: '解析与分块', percent: 0, status: 'pending', message: '' },
                { key: 'parent_store', label: '父级分块入库', percent: 0, status: 'pending', message: '' },
                { key: 'vector_store', label: '向量化入库', percent: 0, status: 'pending', message: '' },
            ];
        },

        async uploadImageOcr(item = null) {
            if (!this.ocrSelectedFile || this.isOcrUploading) {
                return;
            }
            if (!this.isAuthenticated) {
                alert('请先登录');
                return;
            }

            this.isOcrUploading = true;
            this.ocrMessage = '正在上传并识别...';
            this.ocrResult = '';

            try {
                const formData = new FormData();
                formData.append('file', this.ocrSelectedFile);

                const response = await this.authFetch(this.isAdmin ? '/ocr/upload/admin' : '/ocr/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(data.detail || 'OCR 上传失败');
                }

                this.ocrMessage = `${data.message || 'OCR 完成'}${data.provider ? `（${data.provider}）` : ''}`;
                this.ocrResult = data.text || '';
                this.previewOcrText = this.ocrResult;
                this.previewOcrMessage = this.ocrMessage;
                this.imageContextText = this.ocrResult;
                if (item) {
                    item.text = this.ocrResult;
                    item.status = 'done';
                }
                this.chatAttachments = [
                    ...(item ? [item] : []),
                    ...this.chatAttachments.filter(v => !item || v.id !== item.id)
                ];
                this.ocrSelectedFile = null;
            } catch (error) {
                this.ocrMessage = 'OCR 失败：' + error.message;
                this.ocrResult = '';
                if (item) item.status = 'failed';
            } finally {
                this.isOcrUploading = false;
            }
        },

        prevResourcePage() {
            if (this.resourcePage > 1) this.resourcePage -= 1;
        },

        nextResourcePage() {
            if (this.resourcePage < this.resourceTotalPages) this.resourcePage += 1;
        },

        getResourcePageLabel() {
            return `第 ${this.resourcePage} / ${this.resourceTotalPages} 页`;
        },

        updateUploadStep(key, percent, status = 'running', message = '') {
            if (!this.uploadSteps.length) {
                this.uploadSteps = this.createUploadSteps();
            }
            const idx = this.uploadSteps.findIndex(step => step.key === key);
            if (idx === -1) return;
            this.uploadSteps[idx] = {
                ...this.uploadSteps[idx],
                percent: Math.max(0, Math.min(100, Math.round(percent || 0))),
                status,
                message
            };
        },

        uploadFileWithProgress(file) {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                const formData = new FormData();
                formData.append('file', file);

                xhr.open('POST', '/documents/upload/async');
                const headers = this.authHeaders();
                Object.entries(headers).forEach(([key, value]) => xhr.setRequestHeader(key, value));

                xhr.upload.onprogress = (event) => {
                    if (!event.lengthComputable) return;
                    const percent = Math.round((event.loaded / event.total) * 100);
                    this.updateUploadStep('upload', percent, 'running', `已上传 ${percent}%`);
                };

                xhr.onload = () => {
                    if (xhr.status === 401) {
                        this.handleLogout();
                        reject(new Error('登录已过期，请重新登录'));
                        return;
                    }

                    let data = {};
                    try {
                        data = JSON.parse(xhr.responseText || '{}');
                    } catch (e) {
                        reject(new Error('上传响应解析失败'));
                        return;
                    }

                    if (xhr.status < 200 || xhr.status >= 300) {
                        reject(new Error(data.detail || `HTTP ${xhr.status}`));
                        return;
                    }

                    this.updateUploadStep('upload', 100, 'completed', '文档上传完成');
                    resolve(data);
                };

                xhr.onerror = () => reject(new Error('上传请求失败'));
                xhr.onabort = () => reject(new Error('上传已取消'));
                xhr.send(formData);
            });
        },

        syncUploadJob(job) {
            this.activeUploadJobId = job.job_id;
            this.uploadProgress = job.message || '';
            if (Array.isArray(job.steps)) {
                this.uploadSteps = job.steps.map(step => ({
                    key: step.key,
                    label: step.label,
                    percent: step.percent,
                    status: step.status,
                    message: step.message || ''
                }));
            }
            // 入库成功后自动收起步骤明细，保留摘要供用户再次展开查看。
            if (job.status === 'completed') {
                this.uploadProgressCollapsed = true;
            }
        },

        toggleUploadProgressCollapsed() {
            this.uploadProgressCollapsed = !this.uploadProgressCollapsed;
        },

        stopUploadJobPolling() {
            if (this.uploadPollTimer) {
                clearInterval(this.uploadPollTimer);
                this.uploadPollTimer = null;
            }
        },

        startUploadJobPolling(jobId) {
            this.stopUploadJobPolling();

            const poll = async () => {
                try {
                    const response = await this.authFetch(`/documents/upload/jobs/${encodeURIComponent(jobId)}`);
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error.detail || 'Failed to load upload job');
                    }

                    const job = await response.json();
                    this.syncUploadJob(job);

                    if (job.status === 'completed') {
                        this.stopUploadJobPolling();
                        this.isUploading = false;
                        this.selectedFile = null;
                        if (this.$refs.settingsFileInput) {
                            this.$refs.settingsFileInput.value = '';
                        }
                        await this.loadDocuments();
                    } else if (job.status === 'failed') {
                        this.stopUploadJobPolling();
                        this.isUploading = false;
                    }
                } catch (error) {
                    this.uploadProgress = '进度查询失败：' + error.message;
                    this.stopUploadJobPolling();
                    this.isUploading = false;
                }
            };

            poll();
            this.uploadPollTimer = setInterval(poll, 1000);
        },

        async uploadDocument() {
            if (!this.selectedFile) {
                alert('请先选择文件');
                return;
            }

            this.isUploading = true;
            this.uploadProgress = '正在上传...';
            this.uploadSteps = this.createUploadSteps();
            this.uploadProgressCollapsed = false;
            this.updateUploadStep('upload', 0, 'running', '准备上传');

            try {
                const data = await this.uploadFileWithProgress(this.selectedFile);
                this.uploadProgress = data.message;
                this.activeUploadJobId = data.job_id;
                this.startUploadJobPolling(data.job_id);
            } catch (error) {
                this.updateUploadStep('upload', 100, 'failed', error.message);
                this.uploadProgress = '上传失败：' + error.message;
                this.isUploading = false;
            }
        },

        createDeleteSteps() {
            return [
                { key: 'prepare', label: '准备删除', percent: 0, status: 'pending', message: '' },
                { key: 'bm25', label: '同步 BM25 统计', percent: 0, status: 'pending', message: '' },
                { key: 'milvus', label: '删除向量数据', percent: 0, status: 'pending', message: '' },
                { key: 'parent_store', label: '删除父级分块', percent: 0, status: 'pending', message: '' },
                { key: 'file_cleanup', label: '删除本地文件', percent: 0, status: 'pending', message: '' },
            ];
        },

        isDeletingDocument(filename) {
            const job = this.deleteJobs[filename];
            return job && job.status === 'running';
        },

        isDeleteActionLocked(filename) {
            const job = this.deleteJobs[filename];
            return job && (job.status === 'running' || job.status === 'completed');
        },

        getDeleteButtonIcon(filename) {
            const job = this.deleteJobs[filename];
            if (job?.status === 'running') return 'fas fa-spinner fa-spin';
            if (job?.status === 'completed') return 'fas fa-check';
            return 'fas fa-trash';
        },

        setDeleteJob(filename, nextJob) {
            this.deleteJobs = {
                ...this.deleteJobs,
                [filename]: {
                    ...(this.deleteJobs[filename] || {}),
                    ...nextJob
                }
            };
        },

        syncDeleteJob(filename, job) {
            const current = this.deleteJobs[filename] || {};
            // 后端返回统一的步骤结构，前端只负责同步到当前文档行内卡片。
            this.setDeleteJob(filename, {
                jobId: job.job_id,
                status: job.status,
                message: job.message || '',
                collapsed: job.status === 'completed' ? true : Boolean(current.collapsed),
                steps: Array.isArray(job.steps) ? job.steps.map(step => ({
                    key: step.key,
                    label: step.label,
                    percent: step.percent,
                    status: step.status,
                    message: step.message || ''
                })) : this.createDeleteSteps()
            });
        },

        toggleDeleteJobCollapsed(filename) {
            const job = this.deleteJobs[filename];
            if (!job) return;
            this.setDeleteJob(filename, { collapsed: !job.collapsed });
        },

        stopDeleteJobPolling(filename) {
            const timer = this.deletePollTimers[filename];
            if (!timer) return;
            clearInterval(timer);
            const { [filename]: _removed, ...rest } = this.deletePollTimers;
            this.deletePollTimers = rest;
        },

        stopAllDeleteJobPolling() {
            Object.keys(this.deletePollTimers).forEach(filename => this.stopDeleteJobPolling(filename));
        },

        clearDeleteRemovalTimer(filename) {
            const timer = this.deleteRemoveTimers[filename];
            if (!timer) return;
            clearTimeout(timer);
            const { [filename]: _removed, ...rest } = this.deleteRemoveTimers;
            this.deleteRemoveTimers = rest;
        },

        scheduleDeleteJobDismiss(filename, delayMs = 4000) {
            this.clearDeleteRemovalTimer(filename);
            const timer = setTimeout(() => {
                const { [filename]: _job, ...jobs } = this.deleteJobs;
                const { [filename]: _timer, ...timers } = this.deleteRemoveTimers;
                this.deleteJobs = jobs;
                this.deleteRemoveTimers = timers;
            }, delayMs);
            this.deleteRemoveTimers = {
                ...this.deleteRemoveTimers,
                [filename]: timer
            };
        },

        scheduleDeletedDocumentRemoval(filename) {
            this.clearDeleteRemovalTimer(filename);
            // 立即从列表中移除已删除的文档
            this.documents = this.documents.filter(doc => doc.filename !== filename);
            // 延迟清理 deleteJobs 数据，不再自动刷新列表
            const timer = setTimeout(() => {
                const { [filename]: _job, ...jobs } = this.deleteJobs;
                const { [filename]: _timer, ...timers } = this.deleteRemoveTimers;
                this.deleteJobs = jobs;
                this.deleteRemoveTimers = timers;
            }, 3000);
            this.deleteRemoveTimers = {
                ...this.deleteRemoveTimers,
                [filename]: timer
            };
        },

        startDeleteJobPolling(filename, jobId) {
            this.stopDeleteJobPolling(filename);

            const poll = async () => {
                try {
                    const response = await this.authFetch(`/documents/delete/jobs/${encodeURIComponent(jobId)}`);
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error.detail || 'Failed to load delete job');
                    }

                    const job = await response.json();
                    this.syncDeleteJob(filename, job);

                    if (job.status === 'completed') {
                        this.stopDeleteJobPolling(filename);
                        this.scheduleDeletedDocumentRemoval(filename);
                    } else if (job.status === 'failed') {
                        this.stopDeleteJobPolling(filename);
                        await this.loadDocuments();
                        this.scheduleDeleteJobDismiss(filename);
                    }
                } catch (error) {
                    this.setDeleteJob(filename, {
                        status: 'failed',
                        message: '删除进度查询失败：' + error.message,
                        collapsed: false,
                        steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps()
                    });
                    this.stopDeleteJobPolling(filename);
                }
            };

            poll();
            this.deletePollTimers = {
                ...this.deletePollTimers,
                [filename]: setInterval(poll, 1000)
            };
        },

        async deleteDocument(filename) {
            if (this.isDeletingDocument(filename)) {
                return;
            }
            if (!confirm(`确定要删除文档 "${filename}" 吗？这将同时删除 Milvus 中的所有相关向量。`)) {
                return;
            }

            this.clearDeleteRemovalTimer(filename);
            this.setDeleteJob(filename, {
                status: 'running',
                message: '正在提交删除任务...',
                collapsed: false,
                steps: this.createDeleteSteps().map(step => (
                    step.key === 'prepare'
                        ? { ...step, percent: 1, status: 'running', message: '正在提交删除任务' }
                        : step
                ))
            });

            try {
                const response = await this.authFetch(`/documents/delete/async/${encodeURIComponent(filename)}`, {
                    method: 'DELETE'
                });

                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || 'Delete failed');
                }

                const data = await response.json();
                this.setDeleteJob(filename, {
                    jobId: data.job_id,
                    status: 'running',
                    message: data.message || `正在删除 ${filename}`,
                    collapsed: false
                });
                this.startDeleteJobPolling(filename, data.job_id);

            } catch (error) {
                this.setDeleteJob(filename, {
                    status: 'failed',
                    message: '删除文档失败：' + error.message,
                    collapsed: false,
                    steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps()
                });
            }
        },

        getFileIcon(fileType) {
            if (fileType === 'PDF') {
                return 'fas fa-file-pdf';
            } else if (fileType === 'Word') {
                return 'fas fa-file-word';
            } else if (fileType === 'Excel') {
                return 'fas fa-file-excel';
            }
            return 'fas fa-file';
        },

        getFileExtension(filename) {
            if (!filename) return '';
            const idx = filename.lastIndexOf('.');
            return idx >= 0 ? filename.slice(idx + 1).toUpperCase() : '';
        },

        getAttachmentBaseName(filename) {
            if (!filename) return '';
            const idx = filename.lastIndexOf('.');
            return idx >= 0 ? filename.slice(0, idx) : filename;
        },

        getAttachmentFileIcon(filename) {
            const ext = (this.getFileExtension(filename) || '').toLowerCase();
            if (ext === 'pdf') return 'fas fa-file-pdf';
            if (['doc', 'docx'].includes(ext)) return 'fas fa-file-word';
            if (['xls', 'xlsx'].includes(ext)) return 'fas fa-file-excel';
            if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'].includes(ext)) return 'fas fa-file-image';
            if (['txt', 'md'].includes(ext)) return 'fas fa-file-lines';
            return 'fas fa-file';
        },

        formatFileSize(bytes) {
            if (bytes === undefined || bytes === null) return '';
            if (bytes < 1024) return `${bytes} B`;
            if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
            return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
        },

        getImageStatusLabel(item) {
            if (!item || item.kind !== 'image') return '';
            if (item.status === 'pending') return '识别中...';
            if (item.status === 'done') return '已识别';
            if (item.status === 'error') return '识别失败';
            return '图片';
        },

        removeAttachment(id) {
            const item = this.chatAttachments.find(v => v.id === id);
            if (item?.previewUrl) {
                URL.revokeObjectURL(item.previewUrl);
            }
            this.chatAttachments = this.chatAttachments.filter(v => v.id !== id);
            if (item?.kind === 'image') {
                this.previewImageUrl = '';
                this.previewImageName = '';
                this.previewOcrText = '';
                this.previewOcrMessage = '';
                this.imageContextText = '';
                this.ocrSelectedFile = null;
                this.ocrResult = '';
                this.ocrMessage = '';
            }
        },
    },
    watch: {
        messages: {
            handler() {
                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            },
            deep: true
        }
    }
};
