/* WhaleClaw WebChat SPA — Vue 3 Composition API */

const { createApp, ref, reactive, computed, watch, nextTick, onMounted } = Vue;

/* ── markdown-it + highlight.js setup ── */
const md = window.markdownit({
  html: true,
  breaks: true,
  linkify: true,
  highlight(str, lang) {
    if (lang && typeof hljs !== 'undefined' && hljs.getLanguage(lang)) {
      try {
        const html = hljs.highlight(str, { language: lang }).value;
        return `<pre class="hljs"><code>${html}</code><button class="copy-btn" onclick="copyCode(this)">复制</button></pre>`;
      } catch (_) { /* ignore */ }
    }
    return `<pre class="hljs"><code>${md.utils.escapeHtml(str)}</code><button class="copy-btn" onclick="copyCode(this)">复制</button></pre>`;
  },
});

/* Rewrite local file paths in markdown image tokens to /api/local-file?path= */
const _defaultImageRender = md.renderer.rules.image ||
  function (tokens, idx, options, env, self) { return self.renderToken(tokens, idx, options); };

md.renderer.rules.image = function (tokens, idx, options, env, self) {
  const token = tokens[idx];
  const srcIdx = token.attrIndex('src');
  if (srcIdx >= 0) {
    let src = token.attrs[srcIdx][1];
    if (src && /^(\/|~\/|\.\/|\.\.\/)/.test(src) && !src.startsWith('/api/')) {
      token.attrs[srcIdx][1] = `/api/local-file?path=${encodeURIComponent(src)}`;
    } else if (src && src.includes('/api/local-file?path=')) {
      /* LLM already included the proxy prefix — ensure path portion is encoded */
      const parts = src.split('path=');
      if (parts.length === 2 && !parts[1].includes('%')) {
        token.attrs[srcIdx][1] = `${parts[0]}path=${encodeURIComponent(decodeURIComponent(parts[1]))}`;
      }
    }
  }
  return _defaultImageRender(tokens, idx, options, env, self);
};

window.copyCode = function (btn) {
  const code = btn.previousElementSibling.textContent;
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = '已复制';
    setTimeout(() => (btn.textContent = '复制'), 1500);
  });
};

/* ── App ── */
createApp({
  setup() {
    const theme = ref(
      localStorage.getItem('wc-theme') ||
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
    );
    watch(theme, (v) => {
      document.documentElement.setAttribute('data-theme', v);
      localStorage.setItem('wc-theme', v);
    }, { immediate: true });

    const token = ref(localStorage.getItem('wc-token') || '');
    const needLogin = ref(false);
    const authMode = ref('none');
    const loginPassword = ref('');
    const loginToken = ref('');
    const loginError = ref('');
    const compressionReady = ref(true);
    const compressionRunning = ref(false);

    const sessions = ref([]);
    const activeSessionId = ref('');
    const messages = ref([]);
    const inputText = ref('');
    const isStreaming = ref(false);
    const showSettings = ref(false);
    const showSidebar = ref(false);
    const pendingImages = ref([]);
    const pendingFiles = ref([]);
    const memoryStyle = ref('');
    const memoryStyleEnabled = ref(true);
    const memoryStyleSource = ref('none');
    const memoryStyleLoading = ref(false);
    const memoryStyleSaving = ref(false);
    const memoryStyleLastRefresh = ref('');
    const memoryStyleRefreshNote = ref('');
    const evomapEnabled = ref(false);
    const evomapLoading = ref(false);
    const browserVisible = ref(true);
    const browserVisibleLoading = ref(false);

    const currentModel = ref('');
    const thinkingLevel = ref('off');
    const availableModels = ref([]);
    const defaultModel = ref('');
    const sessionTokens = ref({ input_tokens: 0, output_tokens: 0 });
    const totalTokens = ref({ input_tokens: 0, output_tokens: 0 });

    const activeTab = ref('chat');
    const skills = ref([]);
    const tools = ref([]);
    const multiAgentLoading = ref(false);
    const multiAgentSaving = ref(false);
    const multiAgentEnabled = ref(false);
    const multiAgentScenario = ref('software_development');
    const multiAgentCustomScenarios = ref([]);
    const multiAgentCustomScenarioInput = ref('');
    const multiAgentMode = ref('parallel');
    const multiAgentMaxRounds = ref(3);
    const multiAgentRoles = ref([]);
    const skillSourceTab = ref('local');
    const skillInstallSource = ref('');
    const skillInstalling = ref(false);
    const skillDetail = ref(null);
    const pendingUninstallSkillId = ref('');
    const clawhubDetail = ref(null);
    const toolDetail = ref(null);
    const clawhubConfigLoading = ref(false);
    const clawhubConfigSaving = ref(false);
    const clawhubCliInstalling = ref(false);
    const clawhubLoggingIn = ref(false);
    const clawhubLoggingOut = ref(false);
    const clawhubAuthLoading = ref(false);
    const clawhubLoggedIn = ref(false);
    const clawhubAuthMessage = ref('');
    const clawhubEnabled = ref(false);
    const clawhubRegistryUrl = ref('https://clawhub.ai');
    const clawhubCliAvailable = ref(false);
    const clawhubQuery = ref('');
    const clawhubSearching = ref(false);
    const clawhubResults = ref([]);
    const clawhubSearched = ref(false);
    const clawhubPage = ref(1);
    const clawhubPageSize = 8;
    const clawhubInstalling = ref({});
    const clawhubPublishing = ref({});
    const publishDialog = ref(null);
    const publishSlugInput = ref('');
    const publishVersionInput = ref('0.1.0');
    const uiAlertMessage = ref('');
    const pendingClearMemoryStyle = ref(false);
    const mcpServers = ref([]);
    const mcpLoading = ref(false);
    const mcporterAvailable = ref(false);

    const activeSession = computed(() =>
      sessions.value.find((s) => s.id === activeSessionId.value)
    );

    function _bumpMessageCount() {
      const s = sessions.value.find((s) => s.id === activeSessionId.value);
      if (s) s.message_count = messages.value.length;
    }

    const _PROVIDER_LABELS = {
      anthropic: 'Anthropic', openai: 'OpenAI', deepseek: 'DeepSeek',
      qwen: '通义千问', zhipu: '智谱 GLM', minimax: 'MiniMax',
      moonshot: '月之暗面', google: 'Google', nvidia: 'NVIDIA NIM',
    };

    const groupedModels = computed(() => {
      const groups = {};
      for (const m of availableModels.value) {
        const p = m.provider || 'other';
        if (!groups[p]) groups[p] = [];
        groups[p].push(m);
      }
      return Object.entries(groups).map(([provider, models]) => ({
        provider,
        label: _PROVIDER_LABELS[provider] || provider,
        models,
      }));
    });

    let ws = null;
    let streamingMessage = null;
    let _compressionPollTimer = null;

    /* ── API helpers ── */
    const apiBase = window.location.origin;

    async function apiFetch(path, opts = {}) {
      const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
      if (token.value) headers['Authorization'] = `Bearer ${token.value}`;
      const res = await fetch(`${apiBase}${path}`, { ...opts, headers });
      if (res.status === 401) {
        if (!needLogin.value) needLogin.value = true;
        throw new Error('auth');
      }
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || data.detail || `请求失败 (${res.status})`);
      }
      return data;
    }

    function showUiAlert(message) {
      uiAlertMessage.value = String(message || '操作失败');
    }

    function closeUiAlert() {
      uiAlertMessage.value = '';
    }

    function openClearMemoryStyleDialog() {
      pendingClearMemoryStyle.value = true;
    }

    function closeClearMemoryStyleDialog() {
      pendingClearMemoryStyle.value = false;
    }

    /* ── Auth ── */
    async function checkAuth() {
      try {
        await apiFetch('/api/auth/verify');
        needLogin.value = false;
      } catch {
        /* needLogin already set by apiFetch */
      }
    }

    async function doLogin() {
      loginError.value = '';

      if (authMode.value === 'token') {
        const t = loginToken.value.trim();
        if (!t) { loginError.value = '请输入 Token'; return; }
        token.value = t;
        localStorage.setItem('wc-token', t);
        try {
          await apiFetch('/api/auth/verify');
          needLogin.value = false;
          await init();
        } catch {
          token.value = '';
          localStorage.removeItem('wc-token');
          loginError.value = 'Token 无效';
        }
        return;
      }

      try {
        const res = await fetch(`${apiBase}/api/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ password: loginPassword.value }),
        });
        const data = await res.json();
        if (data.token) {
          token.value = data.token;
          localStorage.setItem('wc-token', data.token);
          needLogin.value = false;
          await init();
        } else {
          loginError.value = data.error || '登录失败';
        }
      } catch (e) {
        loginError.value = '网络错误';
      }
    }

    function doLogout() {
      token.value = '';
      localStorage.removeItem('wc-token');
      loginPassword.value = '';
      loginToken.value = '';
      loginError.value = '';
      needLogin.value = true;
      if (ws) {
        ws._intentionalClose = true;
        ws.close();
        ws = null;
      }
    }

    /* ── Models ── */
    async function loadModels() {
      try {
        const data = await apiFetch('/api/models');
        availableModels.value = data.models || [];
        defaultModel.value = data.default || '';
        if (!currentModel.value && defaultModel.value) {
          currentModel.value = defaultModel.value;
        }
        if (data.thinking_level) {
          thinkingLevel.value = data.thinking_level;
        }
      } catch { /* ignore */ }
    }

    let _tokenUsageTimer = null;
    function loadTokenUsage(sessionId) {
      if (needLogin.value) return;
      if (!token.value && authMode.value !== 'none') return;
      clearTimeout(_tokenUsageTimer);
      _tokenUsageTimer = setTimeout(async () => {
        if (needLogin.value) return;
        try {
          const [sData, tData] = await Promise.all([
            apiFetch(`/api/sessions/${sessionId}/token-usage`),
            apiFetch('/api/token-usage'),
          ]);
          sessionTokens.value = sData;
          totalTokens.value = tData.total || { input_tokens: 0, output_tokens: 0 };
        } catch { /* ignore */ }
      }, 500);
    }

    /* ── Skills & Tools ── */
    async function loadSkills() {
      try {
        skills.value = await apiFetch('/api/skills');
      } catch { /* ignore */ }
    }

    async function loadClawhubConfig() {
      clawhubConfigLoading.value = true;
      try {
        const data = await apiFetch('/api/plugins/clawhub');
        clawhubEnabled.value = data.enabled === true;
        clawhubRegistryUrl.value = data.registry_url || 'https://clawhub.ai';
        clawhubCliAvailable.value = data.cli_available === true;
        await loadClawhubAuthStatus();
      } catch { /* ignore */ }
      finally {
        clawhubConfigLoading.value = false;
      }
    }

    async function installClawhubCli() {
      clawhubCliInstalling.value = true;
      try {
        await apiFetch('/api/clawhub/install-cli', { method: 'POST' });
        await loadClawhubConfig();
      } catch (e) {
        showUiAlert('安装 CLI 失败: ' + (e.message || e));
      } finally {
        clawhubCliInstalling.value = false;
      }
    }

    async function saveClawhubConfig() {
      clawhubConfigSaving.value = true;
      try {
        const body = {
          enabled: clawhubEnabled.value === true,
          registry_url: clawhubRegistryUrl.value.trim() || 'https://clawhub.ai',
        };
        const data = await apiFetch('/api/plugins/clawhub', {
          method: 'POST',
          body: JSON.stringify(body),
        });
        clawhubEnabled.value = data.enabled === true;
        clawhubRegistryUrl.value = data.registry_url || clawhubRegistryUrl.value;
        clawhubCliAvailable.value = data.cli_available === true;
        await loadClawhubAuthStatus();
      } catch (e) {
        showUiAlert('ClawHub 配置保存失败: ' + (e.message || e));
      } finally {
        clawhubConfigSaving.value = false;
      }
    }

    async function onClawhubEnabledChange(checked) {
      clawhubEnabled.value = checked === true;
      await saveClawhubConfig();
    }

    async function onClawhubRegistryBlur() {
      await saveClawhubConfig();
    }

    async function loadClawhubAuthStatus() {
      clawhubAuthLoading.value = true;
      try {
        const data = await apiFetch('/api/clawhub/auth-status');
        clawhubLoggedIn.value = data.logged_in === true;
        clawhubAuthMessage.value = data.message || '';
      } catch {
        clawhubLoggedIn.value = false;
        clawhubAuthMessage.value = '';
      } finally {
        clawhubAuthLoading.value = false;
      }
    }

    async function doClawhubLogin() {
      if (!clawhubCliAvailable.value) {
        showUiAlert('请先安装 CLI，再登录');
        return;
      }
      clawhubLoggingIn.value = true;
      try {
        const data = await apiFetch('/api/clawhub/login', { method: 'POST' });
        if (!data.ok) {
          showUiAlert(data.message || '登录未完成');
        }
        await loadClawhubAuthStatus();
      } catch (e) {
        showUiAlert('登录失败: ' + (e.message || e));
      } finally {
        clawhubLoggingIn.value = false;
      }
    }

    async function doClawhubRelogin() {
      const raw = (clawhubRegistryUrl.value || 'https://clawhub.ai').trim();
      const base = /^https?:\/\//i.test(raw) ? raw : `https://${raw}`;
      const loginUrl = `${base.replace(/\/+$/, '')}/login`;
      window.open(loginUrl, '_blank', 'noopener,noreferrer');
      await loadClawhubAuthStatus();
    }

    async function doClawhubLogout() {
      if (!clawhubCliAvailable.value) {
        showUiAlert('请先安装 CLI');
        return;
      }
      clawhubLoggingOut.value = true;
      try {
        const data = await apiFetch('/api/clawhub/logout', { method: 'POST' });
        if (!data.ok) showUiAlert(data.message || '退出登录未完成');
        await loadClawhubAuthStatus();
      } catch (e) {
        showUiAlert('退出登录失败: ' + (e.message || e));
      } finally {
        clawhubLoggingOut.value = false;
      }
    }

    async function searchClawhub() {
      const q = clawhubQuery.value.trim();
      if (!q) return;
      clawhubSearched.value = true;
      clawhubSearching.value = true;
      try {
        const data = await apiFetch(`/api/clawhub/search?q=${encodeURIComponent(q)}&limit=24`);
        clawhubResults.value = data.items || [];
        clawhubPage.value = 1;
      } catch (e) {
        clawhubResults.value = [];
        showUiAlert('执行失败，请重试');
      } finally {
        clawhubSearching.value = false;
      }
    }

    async function installClawhubSkill(slug, version = '', repoUrl = '') {
      if (!slug) return;
      clawhubInstalling.value[slug] = true;
      try {
        await apiFetch('/api/clawhub/install', {
          method: 'POST',
          body: JSON.stringify({ slug, version, repo_url: repoUrl }),
        });
        await loadSkills();
      } catch (e) {
        showUiAlert('ClawHub 安装失败: ' + (e.message || e));
      } finally {
        clawhubInstalling.value[slug] = false;
      }
    }

    function openPublishDialog(skillId) {
      const sid = String(skillId || '').trim();
      if (!sid) return;
      publishDialog.value = { skillId: sid };
      publishSlugInput.value = sid;
      publishVersionInput.value = '0.1.0';
    }

    function closePublishDialog() {
      publishDialog.value = null;
      publishSlugInput.value = '';
      publishVersionInput.value = '0.1.0';
    }

    async function publishInstalledSkill(skillId, publishSlug = '', publishVersion = '') {
      const sid = String(skillId || '').trim();
      if (!sid) return;
      clawhubPublishing.value[sid] = true;
      try {
        await loadClawhubConfig();
        await loadClawhubAuthStatus();
        if (!clawhubEnabled.value) {
          showUiAlert('请先在 ClawHub 标签启用 ClawHub');
          return;
        }
        if (!clawhubCliAvailable.value) {
          showUiAlert('请先安装 ClawHub CLI');
          return;
        }
        if (!clawhubLoggedIn.value) {
          showUiAlert('请先登录 ClawHub');
          return;
        }
        await apiFetch('/api/clawhub/publish-installed', {
          method: 'POST',
          body: JSON.stringify({
            skill_id: sid,
            publish_slug: String(publishSlug || '').trim(),
            publish_version: String(publishVersion || '').trim(),
          }),
        });
        showUiAlert(`发布成功: ${sid}`);
      } catch (e) {
        showUiAlert('发布失败: ' + (e.message || e));
      } finally {
        clawhubPublishing.value[sid] = false;
      }
    }

    async function confirmPublishDialog() {
      if (!publishDialog.value || !publishDialog.value.skillId) return;
      const skillId = publishDialog.value.skillId;
      const slug = publishSlugInput.value.trim();
      const version = publishVersionInput.value.trim();
      if (!slug) {
        showUiAlert('slug 不能为空');
        return;
      }
      if (!version) {
        showUiAlert('版本号不能为空（例如 0.1.0）');
        return;
      }
      closePublishDialog();
      await publishInstalledSkill(skillId, slug, version);
    }

    function openClawhubDetail(item) {
      if (!item) return;
      clawhubDetail.value = {
        ...item,
        detail_url: item.detail_url || `${(clawhubRegistryUrl.value || 'https://clawhub.ai').replace(/\/$/, '')}/skills/${item.slug}`,
      };
    }

    const clawhubTotalPages = computed(() =>
      Math.max(1, Math.ceil(clawhubResults.value.length / clawhubPageSize))
    );

    const pagedClawhubResults = computed(() => {
      const page = Math.min(Math.max(1, clawhubPage.value), clawhubTotalPages.value);
      const start = (page - 1) * clawhubPageSize;
      return clawhubResults.value.slice(start, start + clawhubPageSize);
    });

    const clawhubPageButtons = computed(() => {
      const total = clawhubTotalPages.value;
      if (total <= 1) return [1];
      const page = Math.min(Math.max(1, clawhubPage.value), total);
      const start = Math.max(1, page - 2);
      const end = Math.min(total, start + 4);
      const btns = [];
      for (let p = start; p <= end; p += 1) btns.push(p);
      return btns;
    });

    async function loadTools() {
      try {
        tools.value = await apiFetch('/api/tools');
      } catch { /* ignore */ }
    }

    async function loadMcpServers() {
      mcpLoading.value = true;
      try {
        const data = await apiFetch('/api/mcp/servers');
        mcpServers.value = data.servers || [];
        mcporterAvailable.value = !!data.mcporter_available;
      } catch { /* ignore */ }
      mcpLoading.value = false;
    }

    async function removeMcpServer(serverId, source, configPath) {
      try {
        let qs = `source=${encodeURIComponent(source)}`;
        if (configPath) qs += `&config_path=${encodeURIComponent(configPath)}`;
        await apiFetch(`/api/mcp/servers/${encodeURIComponent(serverId)}?${qs}`, { method: 'DELETE' });
        mcpServers.value = mcpServers.value.filter(s => (s.id || s.name) !== serverId);
      } catch (e) {
        showUiAlert(e.message || '删除 MCP 服务失败');
      }
    }

    const _MA_SCENARIO_BUILTINS = [
      { value: 'product_design', label: '产品设计' },
      { value: 'content_creation', label: '内容创作' },
      { value: 'software_development', label: '软件开发' },
      { value: 'data_analysis_decision', label: '数据分析决策' },
      { value: 'scientific_research', label: '科研' },
      { value: 'intelligent_assistant', label: '智能助理' },
      { value: 'workflow_automation', label: '自动化工作流' },
    ];
    const _MA_SCENARIO_CUSTOM_ENTRY = { value: '__custom__', label: '自定义' };
    const _MA_SCENARIO_DEFAULT = 'software_development';

    const _ROLE_PRESETS_BY_SCENARIO = {
      product_design: [
        {
          id: 'product_manager',
          name: '产品经理',
          enabled: true,
          model: '',
          system_prompt: '你负责把需求变成可执行产品方案。输出：目标用户、核心问题、价值主张、优先级与里程碑。',
        },
        {
          id: 'ux_researcher',
          name: '用户研究',
          enabled: true,
          model: '',
          system_prompt: '你负责验证用户需求与场景。输出：关键用户画像、使用路径、痛点证据、调研与验证计划。',
        },
        {
          id: 'interaction_designer',
          name: '交互设计',
          enabled: true,
          model: '',
          system_prompt: '你负责设计信息架构与交互流程。输出：页面结构、关键流程、状态切换、异常处理与可用性原则。',
        },
        {
          id: 'product_reviewer',
          name: '产品评审',
          enabled: true,
          model: '',
          system_prompt: '你负责从业务、技术、体验与风险角度评审。输出：问题清单、优化建议、上线门槛与验收标准。',
        },
      ],
      content_creation: [
        {
          id: 'content_strategist',
          name: '内容策划',
          enabled: true,
          model: '',
          system_prompt: '你负责内容定位与选题策略。输出：目标受众、传播目标、内容结构、选题矩阵与发布节奏。',
        },
        {
          id: 'copywriter',
          name: '文案主笔',
          enabled: true,
          model: '',
          system_prompt: '你负责撰写主文案。输出：标题方案、正文初稿、核心论点、金句与行动号召。',
        },
        {
          id: 'editor',
          name: '编辑润色',
          enabled: true,
          model: '',
          system_prompt: '你负责语言优化与一致性。输出：修订稿、表达问题清单、结构优化点、风格统一建议。',
        },
        {
          id: 'distribution_operator',
          name: '分发运营',
          enabled: true,
          model: '',
          system_prompt: '你负责渠道分发与增长。输出：渠道改写版本、发布时间建议、互动策略、复盘指标与迭代计划。',
        },
      ],
      software_development: [
        {
          id: 'planner',
          name: '规划师',
          enabled: true,
          model: '',
          system_prompt: '你是规划师。先澄清目标与约束，再拆解任务边界。输出：目标定义、范围/非范围、里程碑、优先级与验收标准。',
        },
        {
          id: 'architect',
          name: '架构师',
          enabled: true,
          model: '',
          system_prompt: '你是架构师。基于规划给出技术方案与关键决策。输出：总体架构、模块边界、关键接口、技术选型、风险与降级方案。',
        },
        {
          id: 'implementer',
          name: '执行者',
          enabled: true,
          model: '',
          system_prompt: '你是执行者。把方案落地为可执行步骤或代码改动。输出：分步实施计划、关键实现细节、命令/代码片段与完成定义。',
        },
        {
          id: 'reviewer',
          name: '评审者',
          enabled: true,
          model: '',
          system_prompt: '你是评审者。独立审视可行性与质量风险。输出：问题清单、回归点、测试策略、监控指标与上线检查项。',
        },
      ],
      data_analysis_decision: [
        {
          id: 'business_analyst',
          name: '业务分析',
          enabled: true,
          model: '',
          system_prompt: '你负责定义业务问题与指标口径。输出：问题定义、指标体系、分析假设、成功标准与决策边界。',
        },
        {
          id: 'data_analyst',
          name: '数据分析',
          enabled: true,
          model: '',
          system_prompt: '你负责数据拆解与结论生成。输出：数据观察、关键对比、异常点、因果假设与可视化建议。',
        },
        {
          id: 'decision_advisor',
          name: '决策顾问',
          enabled: true,
          model: '',
          system_prompt: '你负责给出可执行决策方案。输出：备选方案、收益风险、资源成本、执行优先级与时间表。',
        },
        {
          id: 'risk_controller',
          name: '风险控制',
          enabled: true,
          model: '',
          system_prompt: '你负责识别风险并设监控。输出：风险清单、预警阈值、应急预案、复盘机制与治理建议。',
        },
      ],
      scientific_research: [
        {
          id: 'research_director',
          name: '研究总监',
          enabled: true,
          model: '',
          system_prompt: '你负责定义研究问题与边界。输出：研究目标、核心假设、创新点、评估指标与阶段计划。',
        },
        {
          id: 'literature_analyst',
          name: '文献分析',
          enabled: true,
          model: '',
          system_prompt: '你负责梳理相关工作。输出：关键文献脉络、方法对比、空白点、可复现要点与引用建议。',
        },
        {
          id: 'experiment_engineer',
          name: '实验工程',
          enabled: true,
          model: '',
          system_prompt: '你负责实验设计与执行方案。输出：实验变量、对照设置、数据采集方案、复现实验步骤与失败兜底。',
        },
        {
          id: 'peer_reviewer',
          name: '同行评议',
          enabled: true,
          model: '',
          system_prompt: '你负责批判性评审。输出：方法缺陷、统计风险、结论边界、补充实验建议与投稿准备清单。',
        },
      ],
      intelligent_assistant: [
        {
          id: 'intent_analyst',
          name: '意图分析',
          enabled: true,
          model: '',
          system_prompt: '你负责解析用户目标与上下文。输出：用户意图、约束条件、信息缺口与任务拆解。',
        },
        {
          id: 'task_planner',
          name: '任务规划',
          enabled: true,
          model: '',
          system_prompt: '你负责制定执行路径。输出：步骤计划、依赖条件、时间预算、优先级与中断恢复策略。',
        },
        {
          id: 'tool_executor',
          name: '工具执行',
          enabled: true,
          model: '',
          system_prompt: '你负责工具调用与动作落地。输出：执行结果、关键产物、失败重试策略与状态更新。',
        },
        {
          id: 'quality_guard',
          name: '质量把关',
          enabled: true,
          model: '',
          system_prompt: '你负责结果质量与用户体验。输出：结果核对、风险提示、可读性优化、后续建议与确认项。',
        },
      ],
      workflow_automation: [
        {
          id: 'workflow_designer',
          name: '流程设计',
          enabled: true,
          model: '',
          system_prompt: '你负责定义自动化流程。输出：触发条件、流程图节点、输入输出规范与异常分支。',
        },
        {
          id: 'integration_engineer',
          name: '集成工程',
          enabled: true,
          model: '',
          system_prompt: '你负责系统对接与接口实现。输出：接口映射、鉴权方案、重试机制、幂等设计与部署要求。',
        },
        {
          id: 'operator',
          name: '运行维护',
          enabled: true,
          model: '',
          system_prompt: '你负责上线运行与监控。输出：监控指标、告警策略、运行手册、回滚策略与值班流程。',
        },
        {
          id: 'compliance_auditor',
          name: '合规审计',
          enabled: true,
          model: '',
          system_prompt: '你负责安全合规与审计。输出：权限边界、数据合规点、审计日志要求、风险项与整改清单。',
        },
      ],
    };

    function _customScenarioValue(name) {
      return `custom::${encodeURIComponent(String(name || '').trim())}`;
    }

    function _customScenarioNameFromValue(value) {
      const raw = String(value || '');
      if (!raw.startsWith('custom::')) return '';
      return decodeURIComponent(raw.slice('custom::'.length));
    }

    const multiAgentScenarioOptions = computed(() => {
      const list = [..._MA_SCENARIO_BUILTINS];
      for (const name of multiAgentCustomScenarios.value) {
        list.push({
          value: _customScenarioValue(name),
          label: name,
        });
      }
      list.push(_MA_SCENARIO_CUSTOM_ENTRY);
      return list;
    });

    function _buildCustomScenarioPresets(name) {
      const label = String(name || '自定义场景').trim() || '自定义场景';
      return [
        {
          id: 'lead',
          name: '主持人',
          enabled: true,
          model: '',
          system_prompt: `你是「${label}」主持人。负责澄清目标、约束与成功标准，并形成执行议程。`,
        },
        {
          id: 'analyst',
          name: '分析者',
          enabled: true,
          model: '',
          system_prompt: `你是「${label}」分析者。负责提出关键分析框架、证据点与判断依据。`,
        },
        {
          id: 'executor',
          name: '执行者',
          enabled: true,
          model: '',
          system_prompt: `你是「${label}」执行者。负责产出可落地步骤、关键动作与交付清单。`,
        },
        {
          id: 'reviewer',
          name: '复核者',
          enabled: true,
          model: '',
          system_prompt: `你是「${label}」复核者。负责检查风险、补充边界条件并给出最终确认建议。`,
        },
      ];
    }

    function _presetsByScenarioValue(value) {
      const scenario = String(value || '');
      if (_ROLE_PRESETS_BY_SCENARIO[scenario]) {
        return _ROLE_PRESETS_BY_SCENARIO[scenario];
      }
      const customName = _customScenarioNameFromValue(scenario);
      if (customName) {
        return _buildCustomScenarioPresets(customName);
      }
      return _ROLE_PRESETS_BY_SCENARIO[_MA_SCENARIO_DEFAULT];
    }

    function _buildRolesFromScenario(value) {
      return _presetsByScenarioValue(value).map((role) => ({
        id: String(role.id || ''),
        name: String(role.name || ''),
        enabled: role.enabled !== false,
        model: String(role.model || ''),
        system_prompt: String(role.system_prompt || ''),
      }));
    }

    function _normalizeScenarioValue(value) {
      const str = String(value || '').trim();
      if (!str) return _MA_SCENARIO_DEFAULT;
      if (str === '__custom__') return '__custom__';
      if (_MA_SCENARIO_BUILTINS.some((x) => x.value === str)) return str;
      if (str.startsWith('custom::')) return str;
      return _MA_SCENARIO_DEFAULT;
    }

    function _scenarioValueForPersist(value) {
      const normalized = _normalizeScenarioValue(value);
      if (normalized !== '__custom__') return normalized;
      if (multiAgentCustomScenarios.value.length > 0) {
        return _customScenarioValue(multiAgentCustomScenarios.value[0]);
      }
      return _MA_SCENARIO_DEFAULT;
    }

    function _buildRoleDraft(order = 1) {
      const presets = _presetsByScenarioValue(multiAgentScenario.value);
      const preset = presets[(Math.max(1, order) - 1) % presets.length];
      const existingIds = new Set(
        (Array.isArray(multiAgentRoles.value) ? multiAgentRoles.value : []).map((r) =>
          String((r && r.id) || '').trim().toLowerCase()
        )
      );
      const baseId = String(preset.id || `role_${order}`).trim().toLowerCase();
      let roleId = baseId || `role_${order}`;
      let n = 2;
      while (existingIds.has(roleId)) {
        roleId = `${baseId}_${n}`;
        n += 1;
      }
      return {
        id: roleId,
        name: String(preset.name || `角色${order}`),
        enabled: preset.enabled !== false,
        model: String(preset.model || ''),
        system_prompt: String(preset.system_prompt || ''),
      };
    }

    function _defaultRoleDraft(index = 1) {
      return _buildRoleDraft(index);
    }

    function onMultiAgentScenarioChange() {
      const scenario = _normalizeScenarioValue(multiAgentScenario.value);
      multiAgentScenario.value = scenario;
      if (scenario === '__custom__') return;
      multiAgentRoles.value = _buildRolesFromScenario(scenario);
    }

    function addCustomMultiAgentScenario() {
      const name = String(multiAgentCustomScenarioInput.value || '').trim();
      if (!name) {
        showUiAlert('请先输入自定义模式名称');
        return;
      }
      if (multiAgentCustomScenarios.value.includes(name)) {
        const value = _customScenarioValue(name);
        multiAgentScenario.value = value;
        multiAgentRoles.value = _buildRolesFromScenario(value);
        multiAgentCustomScenarioInput.value = '';
        return;
      }
      multiAgentCustomScenarios.value.push(name);
      const value = _customScenarioValue(name);
      multiAgentScenario.value = value;
      multiAgentRoles.value = _buildRolesFromScenario(value);
      multiAgentCustomScenarioInput.value = '';
    }

    function removeCustomMultiAgentScenario(name) {
      const target = String(name || '').trim();
      if (!target) return;
      multiAgentCustomScenarios.value = multiAgentCustomScenarios.value.filter((x) => x !== target);
      if (multiAgentScenario.value === _customScenarioValue(target)) {
        multiAgentScenario.value = '__custom__';
      }
    }

    async function loadMultiAgentConfig() {
      multiAgentLoading.value = true;
      try {
        if (!availableModels.value.length) {
          await loadModels();
        }
        const data = await apiFetch('/api/plugins/multi-agent');
        multiAgentEnabled.value = data.enabled === true;
        const customScenariosRaw = Array.isArray(data.custom_scenarios) ? data.custom_scenarios : [];
        multiAgentCustomScenarios.value = customScenariosRaw
          .map((x) => String(x || '').trim())
          .filter((x, idx, arr) => x && arr.indexOf(x) === idx);
        multiAgentScenario.value = _normalizeScenarioValue(
          String(data.scenario || _MA_SCENARIO_DEFAULT)
        );
        multiAgentMode.value = data.mode === 'serial' ? 'serial' : 'parallel';
        multiAgentMaxRounds.value = Number.isFinite(Number(data.max_rounds))
          ? Math.max(1, Math.min(10, Number(data.max_rounds)))
          : 3;
        const rawRoles = Array.isArray(data.roles) ? data.roles : [];
        multiAgentRoles.value = rawRoles.map((r, idx) => ({
          id: String(r.id || `role_${idx + 1}`),
          name: String(r.name || `角色${idx + 1}`),
          enabled: r.enabled !== false,
          model: String(r.model || ''),
          system_prompt: String(r.system_prompt || ''),
        }));
        const customName = _customScenarioNameFromValue(multiAgentScenario.value);
        if (customName && !multiAgentCustomScenarios.value.includes(customName)) {
          multiAgentCustomScenarios.value.push(customName);
        }
        if (!multiAgentRoles.value.length) {
          multiAgentRoles.value = _buildRolesFromScenario(multiAgentScenario.value);
        }
      } catch (e) {
        showUiAlert('加载多Agent配置失败: ' + (e.message || e));
      } finally {
        multiAgentLoading.value = false;
      }
    }

    function addMultiAgentRole() {
      multiAgentRoles.value.push(_buildRoleDraft(multiAgentRoles.value.length + 1));
    }

    function removeMultiAgentRole(index) {
      if (index < 0 || index >= multiAgentRoles.value.length) return;
      multiAgentRoles.value.splice(index, 1);
      if (!multiAgentRoles.value.length) multiAgentRoles.value.push(_defaultRoleDraft(1));
    }

    async function toggleMultiAgentEnabled(enabled) {
      multiAgentEnabled.value = enabled;
      multiAgentSaving.value = true;
      try {
        const data = await apiFetch('/api/plugins/multi-agent/toggle', {
          method: 'POST',
          body: JSON.stringify({ enabled }),
        });
        multiAgentEnabled.value = data.enabled === true;
        showUiAlert(data.enabled ? '多Agent编排 已启用' : '多Agent编排 已关闭');
      } catch (e) {
        multiAgentEnabled.value = !enabled;
        showUiAlert('切换多Agent模式失败: ' + (e.message || e));
      } finally {
        multiAgentSaving.value = false;
      }
    }

    async function saveMultiAgentConfig() {
      const payload = {
        enabled: multiAgentEnabled.value === true,
        scenario: _scenarioValueForPersist(multiAgentScenario.value),
        custom_scenarios: multiAgentCustomScenarios.value.map((x) => String(x || '').trim()).filter(Boolean),
        mode: multiAgentMode.value === 'serial' ? 'serial' : 'parallel',
        max_rounds: Number.isFinite(Number(multiAgentMaxRounds.value))
          ? Math.max(1, Math.min(10, Number(multiAgentMaxRounds.value)))
          : 3,
        roles: multiAgentRoles.value.map((r, idx) => ({
          id: String(r.id || `role_${idx + 1}`).trim(),
          name: String(r.name || `角色${idx + 1}`).trim(),
          enabled: r.enabled !== false,
          model: String(r.model || '').trim(),
          system_prompt: String(r.system_prompt || '').trim(),
        })),
      };
      multiAgentSaving.value = true;
      try {
        const data = await apiFetch('/api/plugins/multi-agent', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
        multiAgentEnabled.value = data.enabled === true;
        const customScenariosRaw = Array.isArray(data.custom_scenarios) ? data.custom_scenarios : [];
        multiAgentCustomScenarios.value = customScenariosRaw
          .map((x) => String(x || '').trim())
          .filter((x, idx, arr) => x && arr.indexOf(x) === idx);
        multiAgentScenario.value = _normalizeScenarioValue(
          String(data.scenario || _MA_SCENARIO_DEFAULT)
        );
        multiAgentMode.value = data.mode === 'serial' ? 'serial' : 'parallel';
        multiAgentMaxRounds.value = Number(data.max_rounds || 3);
        multiAgentRoles.value = (Array.isArray(data.roles) ? data.roles : []).map((r, idx) => ({
          id: String(r.id || `role_${idx + 1}`),
          name: String(r.name || `角色${idx + 1}`),
          enabled: r.enabled !== false,
          model: String(r.model || ''),
          system_prompt: String(r.system_prompt || ''),
        }));
        showUiAlert('多Agent配置已保存');
      } catch (e) {
        showUiAlert('保存多Agent配置失败: ' + (e.message || e));
      } finally {
        multiAgentSaving.value = false;
      }
    }

    /* ── Global Memory Style ── */
    function _formatClock(d) {
      const hh = String(d.getHours()).padStart(2, '0');
      const mm = String(d.getMinutes()).padStart(2, '0');
      const ss = String(d.getSeconds()).padStart(2, '0');
      return `${hh}:${mm}:${ss}`;
    }

    async function loadMemoryStyle(showFeedback = false) {
      memoryStyleLoading.value = true;
      try {
        const data = await apiFetch('/api/memory/style');
        memoryStyleEnabled.value = data.enabled !== false;
        memoryStyle.value = data.style_directive || '';
        memoryStyleSource.value = String(data.source || 'none');
        memoryStyleLastRefresh.value = _formatClock(new Date());
        if (showFeedback) {
          if (memoryStyleSource.value === 'cleared') {
            memoryStyleRefreshNote.value = '已刷新（当前回复风格已清空，后续可由新对话重新生成）';
          } else if (!memoryStyle.value.trim()) {
            memoryStyleRefreshNote.value = '已刷新（当前还没有可用的回复风格）';
          } else if (memoryStyleSource.value === 'manual') {
            memoryStyleRefreshNote.value = '已刷新（当前显示的是手动覆盖的回复风格）';
          } else if (memoryStyleSource.value === 'saved') {
            memoryStyleRefreshNote.value = '已刷新（当前显示的是已保存的回复风格）';
          } else {
            memoryStyleRefreshNote.value = '已刷新（当前显示的是从长期记忆画像自动派生的回复风格）';
          }
        }
      } catch (e) {
        if (showFeedback) {
          memoryStyleRefreshNote.value = '刷新失败：' + (e.message || e);
        }
      }
      finally {
        memoryStyleLoading.value = false;
      }
    }

    async function onMemoryStyleRefresh() {
      await loadMemoryStyle(true);
    }

    async function saveMemoryStyle() {
      if (!memoryStyle.value.trim()) {
        showUiAlert('风格指令不能为空');
        return;
      }
      memoryStyleSaving.value = true;
      try {
        await apiFetch('/api/memory/style', {
          method: 'POST',
          body: JSON.stringify({ style_directive: memoryStyle.value.trim() }),
        });
        memoryStyleSource.value = 'manual';
        showUiAlert('保存成功');
      } catch (e) {
        showUiAlert('保存失败: ' + (e.message || e));
      } finally {
        memoryStyleSaving.value = false;
      }
    }

    async function confirmClearMemoryStyle() {
      memoryStyleSaving.value = true;
      try {
        await apiFetch('/api/memory/style', { method: 'DELETE' });
        memoryStyle.value = '';
        memoryStyleSource.value = 'cleared';
        closeClearMemoryStyleDialog();
        showUiAlert('清空成功');
      } catch (e) {
        showUiAlert('清除失败: ' + (e.message || e));
      } finally {
        memoryStyleSaving.value = false;
      }
    }

    /* ── EvoMap Toggle ── */
    async function loadEvomapSetting() {
      evomapLoading.value = true;
      try {
        const data = await apiFetch('/api/plugins/evomap');
        evomapEnabled.value = data.enabled === true;
      } catch { /* ignore */ }
      finally {
        evomapLoading.value = false;
      }
    }

    async function setEvomapEnabled(enabled) {
      if (evomapLoading.value) return;
      evomapLoading.value = true;
      try {
        const data = await apiFetch('/api/plugins/evomap', {
          method: 'POST',
          body: JSON.stringify({ enabled }),
        });
        evomapEnabled.value = data.enabled === true;
        await loadTools();
      } catch (e) {
        showUiAlert('EvoMap 开关更新失败: ' + (e.message || e));
      } finally {
        evomapLoading.value = false;
      }
    }

    async function loadBrowserVisibilitySetting() {
      browserVisibleLoading.value = true;
      try {
        const data = await apiFetch('/api/plugins/browser');
        browserVisible.value = data.visible !== false;
      } catch { /* ignore */ }
      finally {
        browserVisibleLoading.value = false;
      }
    }

    async function setBrowserVisible(enabled) {
      if (browserVisibleLoading.value) return;
      browserVisibleLoading.value = true;
      try {
        const data = await apiFetch('/api/plugins/browser', {
          method: 'POST',
          body: JSON.stringify({ visible: enabled }),
        });
        browserVisible.value = data.visible !== false;
      } catch (e) {
        showUiAlert('浏览器可视开关更新失败: ' + (e.message || e));
      } finally {
        browserVisibleLoading.value = false;
      }
    }

    const _SKILL_CATEGORIES = {
      bundled: '内置技能',
      user: '已安装技能',
    };

    const groupedSkills = computed(() => {
      const groups = {};
      for (const s of skills.value) {
        const src = s.source || 'bundled';
        if (!groups[src]) groups[src] = { label: _SKILL_CATEGORIES[src] || src, items: [] };
        groups[src].items.push(s);
      }
      return Object.values(groups);
    });

    function normalizeSkillKey(v) {
      return String(v || '')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '');
    }

    const installedSkillKeySet = computed(() => {
      const out = new Set();
      for (const s of skills.value) {
        if (!s || s.source !== 'user') continue;
        out.add(normalizeSkillKey(s.id));
        out.add(normalizeSkillKey(s.name));
      }
      return out;
    });

    function isClawhubSkillInstalled(item) {
      if (!item) return false;
      const slugKey = normalizeSkillKey(item.slug);
      const nameKey = normalizeSkillKey(item.name);
      return (
        (slugKey && installedSkillKeySet.value.has(slugKey)) ||
        (nameKey && installedSkillKeySet.value.has(nameKey))
      );
    }

    const _TOOL_CATEGORY_LABELS = {
      system: '系统',
      file: '文件操作',
      browser: '浏览器',
      device: '设备与界面',
      session: '会话管理',
      automation: '自动化',
      memory: '记忆',
      skill: '技能',
      integration: '集成与插件',
      other: '其他',
    };
    const _TOOL_CATEGORY_ORDER = [
      'system',
      'file',
      'browser',
      'device',
      'session',
      'automation',
      'memory',
      'skill',
      'integration',
      'other',
    ];

    const groupedTools = computed(() => {
      const groups = {};
      for (const t of tools.value) {
        const cat = t.category || 'other';
        if (!groups[cat]) groups[cat] = { label: _TOOL_CATEGORY_LABELS[cat] || cat, items: [] };
        groups[cat].items.push(t);
      }
      return _TOOL_CATEGORY_ORDER
        .filter((k) => groups[k])
        .map((k) => groups[k]);
    });

    async function installSkill() {
      const src = skillInstallSource.value.trim();
      if (!src) return;
      skillInstalling.value = true;
      try {
        await apiFetch('/api/skills/install', {
          method: 'POST',
          body: JSON.stringify({ source: src }),
        });
        skillInstallSource.value = '';
        await loadSkills();
      } catch (e) {
        showUiAlert('安装失败: ' + (e.message || e));
      } finally {
        skillInstalling.value = false;
      }
    }

    function requestUninstallSkill(skillId) {
      pendingUninstallSkillId.value = String(skillId || '').trim();
    }

    function cancelUninstallSkill() {
      pendingUninstallSkillId.value = '';
    }

    async function uninstallSkill(skillId) {
      try {
        await apiFetch(`/api/skills/${encodeURIComponent(skillId)}`, { method: 'DELETE' });
        pendingUninstallSkillId.value = '';
        await loadSkills();
      } catch (e) {
        showUiAlert('卸载失败: ' + (e.message || e));
      }
    }

    async function showSkillDetail(skillId) {
      try {
        const data = await apiFetch(`/api/skills/${encodeURIComponent(skillId)}`);
        skillDetail.value = data;
      } catch (e) {
        try {
          const fallback = await apiFetch(`/api/skills/${encodeURIComponent(skillId)}/raw`);
          skillDetail.value = {
            id: fallback.id || skillId,
            name: fallback.name || skillId,
            raw_markdown: fallback.raw_markdown || '',
          };
        } catch (e2) {
          skillDetail.value = null;
          showUiAlert('加载技能详情失败: ' + ((e2 && e2.message) || (e && e.message) || e2 || e));
        }
      }
    }

    function formatTokens(n) {
      if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
      if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
      return String(n);
    }

    function formatCount(n) {
      if (n === null || n === undefined || n === '') return '—';
      const v = Number(n);
      if (!Number.isFinite(v) || v < 0) return '—';
      if (v === 0) return '0';
      if (v >= 1_000_000) return (v / 1_000_000).toFixed(1) + 'M';
      if (v >= 1_000) return (v / 1_000).toFixed(1) + 'k';
      return String(v);
    }

    function hasClawhubStats(item) {
      if (!item) return false;
      return ['stars', 'downloads', 'current_installs', 'all_time_installs'].some((k) => {
        const v = item[k];
        return !(v === null || v === undefined || v === '');
      });
    }

    function formatClawhubStats(item) {
      if (!hasClawhubStats(item)) return '统计数据暂不可用';
      return `⭐${formatCount(item.stars)} · 📦${formatCount(item.downloads)} · ${formatCount(item.current_installs)} 当前安装 · ${formatCount(item.all_time_installs)} 总安装`;
    }

    /* ── Sessions ── */
    async function loadSessions() {
      try {
        sessions.value = await apiFetch('/api/sessions');
      } catch { /* ignore auth redirect */ }
    }

    async function createSession() {
      const data = await apiFetch('/api/sessions', { method: 'POST' });
      await loadSessions();
      await switchSession(data.id);
    }

    async function deleteSession(id) {
      await apiFetch(`/api/sessions/${id}`, { method: 'DELETE' });
      if (activeSessionId.value === id) {
        activeSessionId.value = '';
        messages.value = [];
      }
      await loadSessions();
    }

    async function switchSession(id) {
      activeSessionId.value = id;
      try {
        const data = await apiFetch(`/api/sessions/${id}`);
        messages.value = (data.messages || []).map((m, i) => ({
          id: `hist-${i}`,
          role: m.role,
          content: m.content,
          rendered: renderMarkdown(m.content),
          toolCalls: [],
        }));
        currentModel.value = data.model || '';
        thinkingLevel.value = data.thinking_level || 'off';
      } catch { /* ignore */ }
      loadTokenUsage(id);
      connectWS(id);
      await nextTick();
      scrollToBottom();
    }

    async function syncActiveSessionHistory() {
      const sid = activeSessionId.value;
      if (!sid) return;
      try {
        const data = await apiFetch(`/api/sessions/${sid}`);
        const serverMessages = (data.messages || []).map((m, i) => ({
          id: `hist-sync-${i}-${Date.now()}`,
          role: m.role,
          content: m.content,
          rendered: renderMarkdown(m.content),
          toolCalls: [],
        }));
        if (serverMessages.length >= messages.value.length) {
          messages.value = serverMessages;
        }
        currentModel.value = data.model || currentModel.value;
        thinkingLevel.value = data.thinking_level || thinkingLevel.value;
        _bumpMessageCount();
        await nextTick();
        scrollToBottom();
      } catch { /* ignore */ }
    }

    /* ── WebSocket ── */
    let _wsGeneration = 0;

    function connectWS(sessionId) {
      if (ws) {
        ws._intentionalClose = true;
        ws.close();
        ws = null;
      }
      const gen = ++_wsGeneration;
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      let url = `${protocol}//${location.host}/ws`;
      if (token.value) url += `?token=${encodeURIComponent(token.value)}`;
      const socket = new WebSocket(url);
      ws = socket;

      socket.onopen = () => {
        socket._ping = setInterval(() => {
          if (socket.readyState === 1) {
            socket.send(JSON.stringify({ type: 'ping' }));
          }
        }, 25000);
        if (activeSessionId.value) {
          syncActiveSessionHistory();
        }
      };

      socket.onmessage = (evt) => {
        if (gen !== _wsGeneration) return;
        let msg;
        try { msg = JSON.parse(evt.data); } catch { return; }
        const incomingSid = msg.session_id || '';
        if (
          incomingSid &&
          incomingSid !== activeSessionId.value &&
          isStreaming.value
        ) {
          activeSessionId.value = incomingSid;
          const exists = sessions.value.some((s) => s.id === incomingSid);
          if (!exists) {
            sessions.value.unshift({
              id: incomingSid,
              model: currentModel.value || '',
              message_count: messages.value.length,
              created_at: new Date().toISOString(),
            });
          }
        }

        if (msg.type === 'stream') {
          if (!streamingMessage) {
            streamingMessage = {
              id: `msg-${Date.now()}`,
              role: 'assistant',
              content: '',
              rendered: '',
              toolCalls: [],
            };
            messages.value.push(streamingMessage);
          }
          streamingMessage.content += msg.payload.content || '';
          streamingMessage.rendered = renderMarkdown(streamingMessage.content);
          scrollToBottom();
        } else if (msg.type === 'message') {
          const msgSid = msg.session_id || '';
          const isOwnSession = msgSid && msgSid === activeSessionId.value;
          isStreaming.value = false;
          if (streamingMessage) {
            streamingMessage.content = msg.payload.content || streamingMessage.content;
            streamingMessage.rendered = renderMarkdown(streamingMessage.content);
            streamingMessage = null;
          } else if (isOwnSession) {
            messages.value.push({
              id: `msg-${Date.now()}`,
              role: 'assistant',
              content: msg.payload.content,
              rendered: renderMarkdown(msg.payload.content),
              toolCalls: [],
            });
          }
          _bumpMessageCount();
          if (activeSessionId.value) loadTokenUsage(activeSessionId.value);
          scrollToBottom();
        } else if (msg.type === 'tool_call') {
          if (!streamingMessage) {
            streamingMessage = {
              id: `msg-${Date.now()}`,
              role: 'assistant',
              content: '',
              rendered: '',
              toolCalls: [],
            };
            messages.value.push(streamingMessage);
            isStreaming.value = true;
          }
          const tc = {
            name: msg.payload.name,
            args: JSON.stringify(msg.payload.arguments, null, 2),
            result: null,
            loading: true,
            collapsed: false,
          };
          streamingMessage.toolCalls.push(tc);
          scrollToBottom();
        } else if (msg.type === 'tool_result') {
          if (streamingMessage) {
            const tc = streamingMessage.toolCalls.find(
              (t) => t.name === msg.payload.name && t.loading
            );
            if (tc) {
              tc.result = msg.payload.output;
              tc.loading = false;
              tc.collapsed = true;
            }
            scrollToBottom();
          }
        } else if (msg.type === 'error') {
          isStreaming.value = false;
          streamingMessage = null;
          messages.value.push({
            id: `err-${Date.now()}`,
            role: 'assistant',
            content: `**Error:** ${msg.payload.error}`,
            rendered: renderMarkdown(`**Error:** ${msg.payload.error}`),
            toolCalls: [],
          });
          scrollToBottom();
        } else if (msg.type === 'status') {
          const text = msg.payload?.text || '';
          const model = msg.payload?.model || '';
          const sid = msg.session_id || activeSessionId.value;
          if (model) {
            if (sid && sid === activeSessionId.value) {
              currentModel.value = model;
            }
            const s = sessions.value.find((x) => x.id === sid);
            if (s) s.model = model;
          }
          if (text && sid && sid === activeSessionId.value) {
            messages.value.push({
              id: `status-${Date.now()}`,
              role: 'assistant',
              content: text,
              rendered: renderMarkdown(text),
              toolCalls: [],
            });
            _bumpMessageCount();
            scrollToBottom();
          }
        }
      };

      socket.onclose = () => {
        clearInterval(socket._ping);
        if (socket._intentionalClose) return;
        if (gen !== _wsGeneration) return;
        setTimeout(() => {
          if (gen === _wsGeneration && activeSessionId.value === sessionId) {
            connectWS(sessionId);
          }
        }, 2000);
      };
    }

    /* ── Send message ── */
    function sendMessage() {
      const text = inputText.value.trim();
      const imgs = pendingImages.value;
      if (!compressionReady.value) return;
      if ((!text && !imgs.length) || !ws || ws.readyState !== 1) return;

      let displayHtml = renderMarkdown(text);
      if (imgs.length) {
        const imgHtml = imgs.map((img) => `<img src="${img.dataUrl}" style="max-width:200px;max-height:160px;border-radius:6px;margin:4px 2px">`).join('');
        displayHtml = imgHtml + (displayHtml ? '<br>' + displayHtml : '');
      }

      messages.value.push({
        id: `user-${Date.now()}`,
        role: 'user',
        content: text,
        rendered: displayHtml,
        toolCalls: [],
      });

      const attachments = pendingFiles.value.map((f) => ({
        name: f.name,
        url: f.url,
        filename: f.filename,
        size: f.size,
      }));
      const fallbackText = imgs.length ? '(用户发送了图片)' : attachments.length ? '(用户发送了文件)' : '';
      const payload = { content: text || fallbackText };
      if (imgs.length) {
        payload.images = imgs.map((img) => ({
          data: img.dataUrl.split(',')[1],
          mime: img.mime,
          name: img.name,
        }));
      }
      if (attachments.length) {
        payload.attachments = attachments;
      }

      ws.send(JSON.stringify({
        type: 'message',
        session_id: activeSessionId.value,
        payload,
      }));

      inputText.value = '';
      pendingImages.value = [];
      pendingFiles.value = [];
      isStreaming.value = true;
      streamingMessage = null;
      _bumpMessageCount();
      nextTick(scrollToBottom);
    }

    function handleKeydown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }

    /* ── Image handling ── */
    function addImageFiles(files) {
      for (const file of files) {
        if (!file.type.startsWith('image/')) continue;
        if (pendingImages.value.length >= 4) break;
        const reader = new FileReader();
        reader.onload = (e) => {
          pendingImages.value.push({
            dataUrl: e.target.result,
            mime: file.type,
            name: file.name || 'image.png',
          });
        };
        reader.readAsDataURL(file);
      }
    }

    function removeImage(idx) {
      pendingImages.value.splice(idx, 1);
    }

    function onPaste(e) {
      const items = e.clipboardData?.items;
      if (!items) return;
      const imageFiles = [];
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const f = item.getAsFile();
          if (f) imageFiles.push(f);
        }
      }
      if (imageFiles.length) {
        e.preventDefault();
        addImageFiles(imageFiles);
      }
    }

    function onDrop(e) {
      e.preventDefault();
      if (!e.dataTransfer?.files) return;
      const imgFiles = [];
      const otherFiles = [];
      for (const f of e.dataTransfer.files) {
        if (f.type.startsWith('image/')) imgFiles.push(f);
        else otherFiles.push(f);
      }
      if (imgFiles.length) addImageFiles(imgFiles);
      if (otherFiles.length) addFiles(otherFiles);
    }

    function onDragOver(e) { e.preventDefault(); }

    function triggerFileInput() {
      const input = document.createElement('input');
      input.type = 'file';
      input.multiple = true;
      input.onchange = (e) => {
        if (!e.target.files) return;
        const imgFiles = [];
        const otherFiles = [];
        for (const f of e.target.files) {
          if (f.type.startsWith('image/')) imgFiles.push(f);
          else otherFiles.push(f);
        }
        if (imgFiles.length) addImageFiles(imgFiles);
        if (otherFiles.length) addFiles(otherFiles);
      };
      input.click();
    }

    async function addFiles(files) {
      for (const file of files) {
        if (pendingFiles.value.length >= 5) break;
        const formData = new FormData();
        formData.append('file', file);
        try {
          const res = await fetch('/api/upload', { method: 'POST', body: formData });
          const data = await res.json();
          if (data.error) { showUiAlert('文件上传失败: ' + data.error); continue; }
          pendingFiles.value.push({
            name: file.name,
            size: data.size || file.size,
            url: data.url,
            filename: data.filename,
          });
        } catch (e) {
          showUiAlert('文件上传失败: ' + (e.message || e));
        }
      }
    }

    function removeFile(idx) {
      pendingFiles.value.splice(idx, 1);
    }

    /* ── Settings ── */
    async function switchModel(model) {
      if (model === currentModel.value) return;
      currentModel.value = model;
      const modelInfo = availableModels.value.find((m) => m.id === model);
      if (modelInfo && modelInfo.thinking) {
        thinkingLevel.value = modelInfo.thinking;
      }
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({
          type: 'message',
          session_id: activeSessionId.value,
          payload: { content: `/model ${model}` },
        }));
        if (modelInfo && modelInfo.thinking && modelInfo.thinking !== 'off') {
          ws.send(JSON.stringify({
            type: 'message',
            session_id: activeSessionId.value,
            payload: { content: `/thinking ${modelInfo.thinking}` },
          }));
        }
      }
    }

    /* ── Utils ── */
    const _FILE_ICONS = {
      pptx: '📊', ppt: '📊',
      xlsx: '📗', xls: '📗', csv: '📗',
      docx: '📝', doc: '📝',
      pdf: '📕',
      zip: '📦', rar: '📦', '7z': '📦', tar: '📦', gz: '📦',
      mp3: '🎵', wav: '🎵', flac: '🎵',
      mp4: '🎬', mov: '🎬', avi: '🎬',
      py: '🐍', js: '📜', ts: '📜', json: '📜',
      txt: '📄', md: '📄', log: '📄',
    };
    const _IMAGE_EXTS = new Set(['jpg','jpeg','png','gif','webp','bmp','svg','ico']);

    function _appendToken(url) {
      if (!token.value) return url;
      const sep = url.includes('?') ? '&' : '?';
      return `${url}${sep}token=${encodeURIComponent(token.value)}`;
    }

    function _fileCard(filePath) {
      const name = filePath.split('/').pop();
      const ext = (name.split('.').pop() || '').toLowerCase();
      const icon = _FILE_ICONS[ext] || '📎';
      const encPath = encodeURIComponent(filePath);
      const downloadUrl = _appendToken(`/api/local-file?path=${encPath}&download=true`);
      const openUrl = _appendToken(`/api/local-file?path=${encPath}`);
      return `<div class="file-card" onclick="window.open('${openUrl}','_blank')">` +
        `<div class="file-card-icon">${icon}</div>` +
        `<div class="file-card-info">` +
          `<div class="file-card-name">${name}</div>` +
          `<div class="file-card-meta" data-path="${encPath}">加载中...</div>` +
        `</div>` +
        `<a class="file-card-dl" href="${downloadUrl}" onclick="event.stopPropagation()" title="下载">⬇</a>` +
      `</div>`;
    }

    let _metaTimer = null;
    function _loadFileCardMeta() {
      clearTimeout(_metaTimer);
      _metaTimer = setTimeout(() => {
        document.querySelectorAll('.file-card-meta[data-path]').forEach(async (el) => {
          const p = el.dataset.path;
          if (!p || el.dataset.loaded) return;
          el.dataset.loaded = '1';
          try {
            const resp = await fetch(_appendToken(`/api/file-info?path=${p}`));
            if (resp.ok) {
              const info = await resp.json();
              el.textContent = `${info.ext.toUpperCase()} 文件 · ${info.size_human}`;
            } else {
              el.textContent = '文件不可用';
            }
          } catch { el.textContent = ''; }
        });
      }, 300);
    }

    function renderMarkdown(text) {
      if (!text) return '';
      let html = md.render(text);

      /* Rewrite <img src="/local/path"> to use /api/local-file proxy */
      html = html.replace(
        /(<img\s[^>]*src=["'])(\/(tmp|home|Users|var|opt|etc)[^"']+)(["'][^>]*>)/gi,
        (m, pre, src, _d, post) => {
          if (m.includes('/api/local-file')) return m;
          return `${pre}/api/local-file?path=${encodeURIComponent(src)}${post}`;
        }
      );

      /* Convert file paths to file cards (non-image files only) */
      html = html.replace(
        /(\/(?:tmp|home|Users|var|opt|etc)\/[^\n<"']+?\.(\w{2,5}))(?=[\s<"']|$)/gi,
        (m, filePath, ext) => {
          if (_IMAGE_EXTS.has(ext.toLowerCase())) return m;
          if (m.includes('file-card')) return m;
          return _fileCard(filePath);
        }
      );

      /* Append auth token to /api/ URLs so <img> and file cards pass auth */
      if (token.value) {
        html = html.replace(
          /((?:src|href)=["'])(\/api\/[^"']+)(["'])/gi,
          (m, pre, url, post) => `${pre}${_appendToken(url)}${post}`
        );
      }

      nextTick(_loadFileCardMeta);
      return html;
    }

    function renderSkillMarkdown(text) {
      if (!text) return '';
      // Skill detail should not trigger file-card conversion/window.open side effects.
      let html = md.render(text);
      if (token.value) {
        html = html.replace(
          /((?:src|href)=["'])(\/api\/[^"']+)(["'])/gi,
          (m, pre, url, post) => `${pre}${_appendToken(url)}${post}`
        );
      }
      return html;
    }

    function _escapeHtml(text) {
      return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function safeRenderSkillMarkdown(text) {
      try {
        return renderSkillMarkdown(text);
      } catch (e) {
        console.error('skill_markdown_render_failed', e);
        return `<pre style="white-space:pre-wrap;word-break:break-word">${_escapeHtml(text || '')}</pre>`;
      }
    }

    const messagesEl = ref(null);
    function scrollToBottom() {
      nextTick(() => {
        if (messagesEl.value) {
          messagesEl.value.scrollTop = messagesEl.value.scrollHeight;
        }
      });
    }

    function toggleTheme() {
      theme.value = theme.value === 'dark' ? 'light' : 'dark';
    }

    function formatTime(iso) {
      if (!iso) return '';
      const d = new Date(iso);
      return d.toLocaleDateString();
    }

    /* ── Init ── */
    async function init() {
      await loadModels();
      await loadSessions();
      if (sessions.value.length > 0) {
        await switchSession(sessions.value[0].id);
      } else {
        await createSession();
      }
      loadSkills();
      loadClawhubConfig();
      loadTools();
      loadMemoryStyle();
      loadEvomapSetting();
      loadBrowserVisibilitySetting();
    }

    onMounted(async () => {
      const updateCompressionState = (status) => {
        compressionReady.value = status.compression_ready !== false;
        compressionRunning.value = status.compression_running === true;
      };
      const pollCompressionState = () => {
        clearTimeout(_compressionPollTimer);
        if (compressionReady.value) return;
        _compressionPollTimer = setTimeout(async () => {
          try {
            const s = await fetch(`${apiBase}/api/status`).then((r) => r.json());
            updateCompressionState(s);
          } catch { /* ignore */ }
          pollCompressionState();
        }, 3000);
      };
      try {
        const status = await fetch(`${apiBase}/api/status`).then((r) => r.json());
        updateCompressionState(status);
        if (status.status === 'ok') {
          authMode.value = status.auth_mode || 'none';
          if (authMode.value === 'none') {
            needLogin.value = false;
            await init();
          } else {
            await checkAuth();
            if (!needLogin.value) await init();
          }
        }
        pollCompressionState();
      } catch {
        /* Gateway unreachable — assume no auth, try to init */
        try { await init(); } catch { /* ignore */ }
      }
    });

    return {
      theme, token, needLogin, authMode, loginPassword, loginToken, loginError, doLogin, doLogout,
      compressionReady, compressionRunning,
      sessions, activeSessionId, activeSession, messages,
      inputText, isStreaming, showSettings, showSidebar, pendingImages, pendingFiles,
      currentModel, thinkingLevel, availableModels, defaultModel, groupedModels, messagesEl,
      sessionTokens, totalTokens, formatTokens,
      formatCount, formatClawhubStats,
      activeTab, skills, tools, groupedSkills, groupedTools,
      multiAgentLoading, multiAgentSaving, multiAgentEnabled, multiAgentMode, multiAgentMaxRounds, multiAgentRoles,
      multiAgentScenario, multiAgentScenarioOptions, multiAgentCustomScenarios, multiAgentCustomScenarioInput,
      skillSourceTab,
      skillInstallSource, skillInstalling, skillDetail, pendingUninstallSkillId,
      installSkill, uninstallSkill, requestUninstallSkill, cancelUninstallSkill, showSkillDetail,
      clawhubConfigLoading, clawhubConfigSaving,
      clawhubCliInstalling, installClawhubCli,
      clawhubLoggingIn, clawhubLoggingOut, clawhubAuthLoading, clawhubLoggedIn, clawhubAuthMessage, doClawhubLogin, doClawhubRelogin, doClawhubLogout, loadClawhubAuthStatus, onClawhubRegistryBlur,
      clawhubEnabled, clawhubRegistryUrl, clawhubCliAvailable,
      clawhubQuery, clawhubSearching, clawhubResults, clawhubSearched, clawhubInstalling, clawhubPublishing,
      clawhubPage, clawhubPageSize, clawhubTotalPages, pagedClawhubResults, clawhubPageButtons,
      clawhubDetail, openClawhubDetail,
      isClawhubSkillInstalled,
      loadClawhubConfig, saveClawhubConfig, onClawhubEnabledChange, searchClawhub, installClawhubSkill, publishInstalledSkill,
      publishDialog, publishSlugInput, publishVersionInput, openPublishDialog, closePublishDialog, confirmPublishDialog,
      loadMultiAgentConfig, saveMultiAgentConfig, toggleMultiAgentEnabled, addMultiAgentRole, removeMultiAgentRole,
      onMultiAgentScenarioChange, addCustomMultiAgentScenario, removeCustomMultiAgentScenario,
      toolDetail,
      memoryStyle, memoryStyleEnabled, memoryStyleSource, memoryStyleLoading, memoryStyleSaving, memoryStyleLastRefresh, memoryStyleRefreshNote,
      loadMemoryStyle, onMemoryStyleRefresh, saveMemoryStyle, pendingClearMemoryStyle, openClearMemoryStyleDialog, closeClearMemoryStyleDialog, confirmClearMemoryStyle,
      evomapEnabled, evomapLoading, loadEvomapSetting, setEvomapEnabled,
      browserVisible, browserVisibleLoading, loadBrowserVisibilitySetting, setBrowserVisible,
      createSession, deleteSession, switchSession,
      sendMessage, handleKeydown, switchModel, loadModels,
      toggleTheme, formatTime, renderMarkdown,
      renderSkillMarkdown, safeRenderSkillMarkdown,
      addImageFiles, removeImage, addFiles, removeFile, onPaste, onDrop, onDragOver, triggerFileInput,
      uiAlertMessage, closeUiAlert,
      mcpServers, mcpLoading, mcporterAvailable, loadMcpServers, removeMcpServer,
      loadSkills, loadTools,
    };
  },

  template: `
    <!-- Login Overlay -->
    <div v-if="needLogin" class="login-overlay">
      <div class="login-card">
        <h2>WhaleClaw</h2>
        <p v-if="loginError" class="login-error">{{ loginError }}</p>
        <template v-if="authMode === 'token'">
          <input
            v-model="loginToken"
            type="password"
            placeholder="输入 Token"
            @keydown.enter="doLogin"
          />
        </template>
        <template v-else>
          <input
            v-model="loginPassword"
            type="password"
            placeholder="输入密码"
            @keydown.enter="doLogin"
          />
        </template>
        <button class="btn-send" @click="doLogin">登录</button>
      </div>
    </div>

    <!-- Main App -->
    <template v-else>
      <!-- Sidebar -->
      <aside class="sidebar" :class="{ open: showSidebar }">
        <div class="sidebar-header">
          <h1>WhaleClaw</h1>
          <div class="sidebar-actions">
            <button class="btn-icon" @click="createSession" title="新建会话">+</button>
            <button class="btn-icon" @click="toggleTheme" title="切换主题">
              {{ theme === 'dark' ? '☀' : '🌙' }}
            </button>
            <button class="btn-icon" @click="showSettings = !showSettings" title="设置">⚙</button>
          </div>
        </div>
        <div class="session-list" v-show="activeTab === 'chat'">
          <div
            v-for="s in sessions"
            :key="s.id"
            class="session-item"
            :class="{ active: s.id === activeSessionId }"
            @click="switchSession(s.id); showSidebar = false"
          >
            <div class="session-info">
              <div class="session-title">{{ s.model || '会话' }}</div>
              <div class="session-meta">{{ formatTime(s.created_at) }} · {{ s.message_count || 0 }} 条<template v-if="s.tokens"> · 🪙{{ formatTokens(s.tokens) }}</template></div>
            </div>
            <button class="session-delete" @click.stop="deleteSession(s.id)">✕</button>
          </div>
        </div>
        <div class="sidebar-tab-bar">
          <button class="sidebar-tab" :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'">💬 聊天</button>
          <button class="sidebar-tab" :class="{ active: activeTab === 'skills' }" @click="activeTab = 'skills'; loadSkills(); loadClawhubConfig()">🧩 技能</button>
          <button class="sidebar-tab" :class="{ active: activeTab === 'multi-agent' }" @click="activeTab = 'multi-agent'; loadModels(); loadMultiAgentConfig()">🧠 多Agent</button>
          <button class="sidebar-tab" :class="{ active: activeTab === 'tools' }" @click="activeTab = 'tools'; loadTools()">🔧 工具</button>
          <button class="sidebar-tab" :class="{ active: activeTab === 'mcp' }" @click="activeTab = 'mcp'; loadMcpServers()">🔌 MCP</button>
        </div>
      </aside>

      <!-- Main Area -->
      <div class="main">
        <!-- Chat Header (only in chat tab) -->
        <div class="chat-header" v-if="activeTab === 'chat'">
          <div style="display:flex;align-items:center;gap:8px">
            <button class="btn-icon mobile-menu" @click="showSidebar = !showSidebar">☰</button>
            <select
              class="model-selector"
              :value="currentModel"
              @change="switchModel($event.target.value)"
            >
              <template v-if="groupedModels.length">
                <optgroup v-for="g in groupedModels" :key="g.provider" :label="g.label">
                  <option
                    v-for="m in g.models"
                    :key="m.id"
                    :value="m.id"
                  >{{ m.name }}{{ m.thinking && m.thinking !== 'off' ? ' 💭' : '' }}{{ m.tools ? '' : ' ⚠无工具' }}</option>
                </optgroup>
              </template>
              <template v-else>
                <option :value="currentModel">{{ currentModel || '未配置模型' }}</option>
              </template>
            </select>
            <span v-if="thinkingLevel !== 'off'" class="thinking-badge">💭 {{ thinkingLevel }}</span>
          </div>
          <div class="chat-header-info" v-if="activeSession">
            <span>{{ activeSession.message_count || messages.length }} 条消息</span>
            <span class="token-badge"
              :title="'本会话: ↑' + sessionTokens.input_tokens.toLocaleString() + ' ↓' + sessionTokens.output_tokens.toLocaleString() + '\\n总计: ↑' + totalTokens.input_tokens.toLocaleString() + ' ↓' + totalTokens.output_tokens.toLocaleString()"
            >🪙 ↑{{ formatTokens(sessionTokens.input_tokens) }} ↓{{ formatTokens(sessionTokens.output_tokens) }} · 总{{ formatTokens(totalTokens.input_tokens + totalTokens.output_tokens) }}</span>
          </div>
        </div>

        <!-- Page Header (skills / tools) -->
        <div class="page-header" v-if="activeTab !== 'chat'">
          <button class="btn-icon mobile-menu" @click="showSidebar = !showSidebar">☰</button>
          <h2 v-if="activeTab === 'skills'">🧩 技能管理</h2>
          <h2 v-if="activeTab === 'multi-agent'">🧠 多Agent 编排</h2>
          <h2 v-if="activeTab === 'tools'">🔧 工具列表 <small>({{ tools.length }})</small></h2>
          <h2 v-if="activeTab === 'mcp'">🔌 MCP 服务 <small>({{ mcpServers.length }})</small></h2>
        </div>

        <!-- Chat Tab -->
        <template v-if="activeTab === 'chat'">
          <div class="messages" ref="messagesEl">
            <div
              v-for="msg in messages"
              :key="msg.id"
              class="message-row"
              :class="msg.role"
            >
              <div class="bubble" :class="msg.role">
                <div v-if="msg.rendered" v-html="msg.rendered"></div>
                <div v-if="msg.toolCalls && msg.toolCalls.length" class="tool-calls">
                  <div v-for="(tc, ti) in msg.toolCalls" :key="ti" class="tool-card" :class="{ loading: tc.loading }">
                    <div class="tool-card-header" @click="tc.collapsed = !tc.collapsed">
                      <span class="tool-card-icon">{{ tc.loading ? '⏳' : '✅' }}</span>
                      <span class="tool-card-name">{{ tc.name }}</span>
                      <span class="tool-card-status">{{ tc.loading ? '执行中...' : '完成' }}</span>
                      <span class="tool-card-toggle">{{ tc.collapsed ? '▸' : '▾' }}</span>
                    </div>
                    <div v-if="!tc.collapsed" class="tool-card-body">
                      <pre class="tool-card-args">{{ tc.args }}</pre>
                      <pre v-if="tc.result" class="tool-card-result">{{ tc.result.length > 500 ? tc.result.slice(0, 500) + '...' : tc.result }}</pre>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div v-if="isStreaming && !messages.length" class="message-row assistant">
              <div class="bubble assistant">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
              </div>
            </div>
          </div>

          <div class="input-area" @drop="onDrop" @dragover="onDragOver">
            <div v-if="!compressionReady" style="font-size:12px;color:#f59e0b;padding:0 12px 8px;">
              {{ compressionRunning ? '会话压缩中，请稍后发送消息…' : '压缩尚未就绪，请稍后重试。' }}
            </div>
            <div v-if="pendingImages.length" class="image-preview-strip">
              <div v-for="(img, idx) in pendingImages" :key="idx" class="image-preview-item">
                <img :src="img.dataUrl" />
                <button class="image-remove-btn" @click="removeImage(idx)">✕</button>
              </div>
            </div>
            <div v-if="pendingFiles.length" class="file-preview-strip">
              <div v-for="(file, idx) in pendingFiles" :key="'f-'+idx" class="file-preview-item">
                <span class="file-preview-icon">📎</span>
                <span class="file-preview-name" :title="file.name">{{ file.name }}</span>
                <span class="file-preview-size">{{ (file.size / 1024).toFixed(0) }}KB</span>
                <button class="image-remove-btn" @click="removeFile(idx)">✕</button>
              </div>
            </div>
            <div class="input-wrapper">
              <button class="btn-attach" @click="triggerFileInput" title="添加图片或文件">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
              </button>
              <textarea
                v-model="inputText"
                placeholder="输入消息... (Enter 发送, Shift+Enter 换行, 可粘贴/拖拽图片)"
                @keydown="handleKeydown"
                @paste="onPaste"
                :disabled="!compressionReady"
                rows="1"
              ></textarea>
              <button class="btn-send" :disabled="!compressionReady || isStreaming || (!inputText.trim() && !pendingImages.length && !pendingFiles.length)" @click="sendMessage">
                发送
              </button>
            </div>
          </div>
        </template>

        <!-- Skills Tab -->
        <template v-if="activeTab === 'skills'">
          <div class="tab-content">
            <div class="skill-subtabs">
              <button class="skill-subtab" :class="{ active: skillSourceTab === 'local' }" @click="skillSourceTab = 'local'">本地技能</button>
              <button class="skill-subtab" :class="{ active: skillSourceTab === 'clawhub' }" @click="skillSourceTab = 'clawhub'; loadClawhubConfig()">ClawHub</button>
            </div>
            <template v-if="skillSourceTab === 'local'">
              <div class="panel-toolbar">
                <input
                  v-model="skillInstallSource"
                  class="panel-search"
                  placeholder="输入 GitHub 地址或本地路径安装技能…"
                  @keydown.enter="installSkill"
                />
                <button class="btn-pill" :disabled="skillInstalling || !skillInstallSource.trim()" @click="installSkill">
                  {{ skillInstalling ? '安装中…' : '+ 安装' }}
                </button>
              </div>
              <div class="tab-content-body">
                <div v-if="!skills.length" class="tab-empty">暂无技能</div>
                <details v-for="group in groupedSkills" :key="group.label" class="panel-group" open>
                  <summary class="panel-group-header">
                    <span>{{ group.label }}</span>
                    <span class="panel-group-count">{{ group.items.length }}</span>
                  </summary>
                  <div class="panel-grid skills-grid">
                    <div v-for="s in group.items" :key="s.id" class="panel-card skill-card" @click.prevent="showSkillDetail(s.id)">
                      <div class="panel-card-main">
                        <div class="panel-card-title">🧩 {{ s.name }}</div>
                        <div v-if="s.trigger_description" class="panel-card-desc">{{ s.trigger_description }}</div>
                        <div class="chip-row">
                          <span class="chip" :class="s.source === 'user' ? 'chip-accent' : 'chip-ok'">{{ s.source === 'user' ? '已安装' : '内置' }}</span>
                          <span class="chip" v-for="t in (s.triggers || []).slice(0, 3)" :key="t">{{ t }}</span>
                        </div>
                        <div v-if="s.tools && s.tools.length" class="panel-card-tools">
                          <code v-for="t in s.tools" :key="t" class="tool-code">{{ t }}</code>
                        </div>
                      </div>
                      <div class="panel-card-action skill-card-action">
                        <button v-if="s.source === 'user'" class="btn-danger-sm" @click.stop="requestUninstallSkill(s.id)" title="卸载">✕</button>
                        <button
                          v-if="s.source === 'user'"
                          class="btn-outline"
                          :disabled="clawhubPublishing[s.id]"
                          @click.stop="openPublishDialog(s.id)"
                          title="发布到clawhub"
                        >{{ clawhubPublishing[s.id] ? '发布中…' : '发布到clawhub' }}</button>
                      </div>
                    </div>
                  </div>
                </details>
              </div>
            </template>
            <template v-else>
              <div class="panel-toolbar clawhub-toolbar">
                <label class="switch">
                  <input type="checkbox" :checked="clawhubEnabled" :disabled="clawhubConfigSaving || clawhubConfigLoading" @change="onClawhubEnabledChange($event.target.checked)">
                  <span class="switch-slider"></span>
                </label>
                <span class="clawhub-enable-text">启用 ClawHub</span>
                <span class="chip" :class="clawhubCliAvailable ? 'chip-ok' : ''">{{ clawhubCliAvailable ? 'CLI 已安装' : 'CLI 未安装' }}</span>
                <span class="chip" :class="clawhubLoggedIn ? 'chip-accent' : ''">{{ clawhubLoggedIn ? '已登录' : '未登录' }}</span>
                <button
                  class="btn-outline"
                  :disabled="clawhubCliInstalling || clawhubCliAvailable"
                  @click="installClawhubCli"
                >{{ clawhubCliAvailable ? 'CLI 已安装' : (clawhubCliInstalling ? '安装 CLI 中…' : '安装 CLI') }}</button>
                <button
                  class="btn-outline"
                  :disabled="clawhubLoggingIn || !clawhubCliAvailable || clawhubLoggedIn"
                  @click="doClawhubLogin"
                >{{ clawhubLoggedIn ? '已登录' : (clawhubLoggingIn ? '登录中…' : '登录 ClawHub') }}</button>
                <button class="btn-outline" :disabled="clawhubLoggingOut || !clawhubCliAvailable || !clawhubLoggedIn" @click="doClawhubLogout">{{ clawhubLoggingOut ? '退出中…' : '退出登录' }}</button>
                <span class="clawhub-auth-msg">若查询失败，请退出后重新登录！</span>
              </div>
              <div class="panel-toolbar clawhub-config-row">
                <input
                  v-model="clawhubRegistryUrl"
                  class="panel-search"
                  placeholder="ClawHub 源地址 (默认: https://clawhub.ai)"
                  @blur="onClawhubRegistryBlur"
                />
                <span class="clawhub-auth-msg">{{ clawhubAuthMessage || '登录后可使用账号态能力（如 whoami/list/update/publish）' }}</span>
              </div>
              <div class="panel-toolbar">
                <input
                  v-model="clawhubQuery"
                  class="panel-search"
                  placeholder="搜索 ClawHub 技能，例如: browser, ppt, feishu..."
                  @keydown.enter="searchClawhub"
                />
                <button class="btn-pill" :disabled="clawhubSearching || clawhubConfigSaving || !clawhubEnabled || !clawhubQuery.trim()" @click="searchClawhub">
                  {{ clawhubSearching ? '搜索中…' : '搜索' }}
                </button>
              </div>
              <div class="tab-content-body">
                <div v-if="!clawhubEnabled" class="tab-empty">请先启用 ClawHub</div>
                <div v-else-if="clawhubSearching" class="tab-empty">搜索中，请耐心等待30秒！</div>
                <div v-else-if="!clawhubResults.length" class="tab-empty">{{ clawhubSearched ? '未找到记录（0 条）' : '输入关键词后点击搜索，请耐心等待，最多获取24条记录。' }}</div>
                <div v-else class="clawhub-results-wrap">
                  <div class="panel-grid skills-grid clawhub-results-grid">
                    <div v-for="item in pagedClawhubResults" :key="item.slug" class="panel-card skill-card" @click="openClawhubDetail(item)">
                      <div class="panel-card-main">
                        <div class="panel-card-title">🛍 {{ item.name || item.slug }}</div>
                        <div class="panel-card-desc">{{ item.summary || '暂无描述' }}</div>
                        <div class="clawhub-stats">
                          {{ formatClawhubStats(item) }}
                        </div>
                        <div class="chip-row">
                          <span class="chip chip-accent">{{ item.slug }}</span>
                          <span v-if="item.version" class="chip">v{{ item.version }}</span>
                        </div>
                      </div>
                      <div class="panel-card-action">
                        <button class="btn-pill btn-pill-sm" :disabled="clawhubInstalling[item.slug] || isClawhubSkillInstalled(item)" @click.stop="installClawhubSkill(item.slug, item.version || '', item.repo_url || '')">
                          {{ isClawhubSkillInstalled(item) ? '已安装' : (clawhubInstalling[item.slug] ? '安装中…' : '安装') }}
                        </button>
                      </div>
                    </div>
                  </div>
                  <div v-if="clawhubResults.length >= clawhubPageSize" class="pager">
                    <button class="pager-btn" :disabled="clawhubPage <= 1" @click="clawhubPage = Math.max(1, clawhubPage - 1)">上一页</button>
                    <button
                      v-for="p in clawhubPageButtons"
                      :key="'cp' + p"
                      class="pager-btn"
                      :class="{ active: p === clawhubPage }"
                      @click="clawhubPage = p"
                    >{{ p }}</button>
                    <button class="pager-btn" :disabled="clawhubPage >= clawhubTotalPages" @click="clawhubPage = Math.min(clawhubTotalPages, clawhubPage + 1)">下一页</button>
                    <span class="pager-total">共 {{ clawhubResults.length }} 条</span>
                  </div>
                </div>
              </div>
            </template>
          </div>
        </template>

        <!-- Multi-Agent Tab -->
        <template v-if="activeTab === 'multi-agent'">
          <div class="tab-content">
            <div class="panel-toolbar multi-agent-toolbar">
              <label class="switch">
                <input type="checkbox" :checked="multiAgentEnabled" :disabled="multiAgentSaving || multiAgentLoading" @change="toggleMultiAgentEnabled($event.target.checked)">
                <span class="switch-slider"></span>
              </label>
              <span class="clawhub-enable-text">启用多Agent编排</span>
              <span class="multi-agent-label mode-label">模式</span>
              <select class="model-selector scenario-select" v-model="multiAgentScenario" :disabled="multiAgentSaving || multiAgentLoading" @change="onMultiAgentScenarioChange()">
                <option v-for="item in multiAgentScenarioOptions" :key="'ma-scenario-' + item.value" :value="item.value">{{ item.label }}</option>
              </select>
              <span class="multi-agent-label">协作</span>
              <select class="model-selector mini-select" v-model="multiAgentMode" :disabled="multiAgentSaving || multiAgentLoading">
                <option value="parallel">并行</option>
                <option value="serial">串行</option>
              </select>
              <span class="multi-agent-label">最大回合</span>
              <input class="panel-search mini-input" type="number" min="1" max="10" v-model.number="multiAgentMaxRounds" :disabled="multiAgentSaving || multiAgentLoading" />
              <button class="btn-pill" :disabled="multiAgentSaving || multiAgentLoading" @click="saveMultiAgentConfig()">{{ multiAgentSaving ? '保存中…' : '保存配置' }}</button>
              <span class="multi-agent-feishu-note">备注：飞书里使用/multi on/off 来切换单agent和多agent模式</span>
            </div>
            <div class="tab-content-body">
              <div class="panel-toolbar multi-agent-custom-toolbar" v-if="multiAgentScenario === '__custom__' || multiAgentCustomScenarios.length">
                <input
                  v-if="multiAgentScenario === '__custom__'"
                  v-model="multiAgentCustomScenarioInput"
                  class="panel-search custom-scenario-input"
                  placeholder="输入自定义模式名称，例如：读书会议"
                  :disabled="multiAgentSaving || multiAgentLoading"
                  @keydown.enter="addCustomMultiAgentScenario()"
                />
                <button
                  v-if="multiAgentScenario === '__custom__'"
                  class="btn-outline"
                  :disabled="multiAgentSaving || multiAgentLoading || !multiAgentCustomScenarioInput.trim()"
                  @click="addCustomMultiAgentScenario()"
                >加入列表</button>
                <div class="multi-agent-custom-list" v-if="multiAgentCustomScenarios.length">
                  <span v-for="name in multiAgentCustomScenarios" :key="'ma-custom-' + name" class="multi-agent-custom-chip">
                    {{ name }}
                    <button class="multi-agent-custom-remove" title="删除自定义模式" @click="removeCustomMultiAgentScenario(name)">✕</button>
                  </span>
                </div>
              </div>
              <div class="panel-toolbar">
                <button class="btn-outline" :disabled="multiAgentSaving || multiAgentLoading" @click="addMultiAgentRole()">+ 新增角色</button>
              </div>
              <div v-if="multiAgentLoading" class="tab-empty">配置加载中…</div>
              <div v-else class="panel-grid multi-agent-grid">
                <div v-for="(role, idx) in multiAgentRoles" :key="role.id + '-' + idx" class="panel-card multi-agent-role-card">
                  <div class="panel-card-main">
                    <div class="multi-agent-role-head">
                      <label class="switch">
                        <input type="checkbox" :checked="role.enabled" :disabled="multiAgentSaving" @change="role.enabled = $event.target.checked">
                        <span class="switch-slider"></span>
                      </label>
                      <span class="chip" :class="role.enabled ? 'chip-ok' : ''">{{ role.enabled ? '启用' : '忽略' }}</span>
                    </div>
                    <div class="multi-agent-role-fields">
                      <input class="panel-search" v-model="role.id" placeholder="角色ID (slug)" :disabled="multiAgentSaving" />
                      <input class="panel-search" v-model="role.name" placeholder="角色名" :disabled="multiAgentSaving" />
                      <select class="panel-search role-model-select" v-model="role.model" :disabled="multiAgentSaving">
                        <option value="">默认模型（当前: {{ defaultModel || '未设置' }}）</option>
                        <template v-if="groupedModels.length">
                          <optgroup v-for="g in groupedModels" :key="'ma-role-' + g.provider" :label="g.label">
                            <option v-for="m in g.models" :key="'ma-role-opt-' + m.id" :value="m.id">
                              {{ m.name }}
                            </option>
                          </optgroup>
                        </template>
                      </select>
                      <textarea class="setting-textarea" rows="4" v-model="role.system_prompt" placeholder="角色系统提示词..." :disabled="multiAgentSaving"></textarea>
                    </div>
                  </div>
                  <div class="panel-card-action">
                    <button class="btn-danger-sm" @click="removeMultiAgentRole(idx)" title="删除角色">✕</button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- Tools Tab -->
        <template v-if="activeTab === 'tools'">
          <div class="tab-content">
            <div class="tab-content-body">
              <div v-if="!tools.length" class="tab-empty">暂无工具</div>
              <details v-for="group in groupedTools" :key="group.label" class="panel-group" open>
                <summary class="panel-group-header">
                  <span>{{ group.label }}</span>
                  <span class="panel-group-count">{{ group.items.length }}</span>
                </summary>
                <div class="panel-grid tools-grid">
                  <div v-for="t in group.items" :key="t.name" class="panel-card tool-item tool-card" @click="toolDetail = t">
                    <div class="panel-card-main">
                      <div class="panel-card-title"><code class="tool-name-code">{{ t.name }}</code></div>
                      <div class="panel-card-desc">{{ t.description }}</div>
                    </div>
                    <div v-if="t.parameters && t.parameters.length" class="tool-params-strip">
                      <span v-for="p in t.parameters" :key="p.name" class="tool-param-chip" :class="{ required: p.required }">{{ p.name }}<em>{{ p.type }}</em></span>
                    </div>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </template>

        <!-- MCP Tab -->
        <template v-if="activeTab === 'mcp'">
          <div class="tab-content">
            <div class="tab-content-body">
              <div v-if="mcpLoading" class="tab-empty">加载中…</div>
              <div v-else-if="!mcpServers.length" class="tab-empty">
                暂无 MCP 服务。<br>
                通过 Agent 对话安装 MCP，或使用 mcporter CLI 配置后，服务会自动出现此列表。
              </div>
              <div v-else class="panel-grid">
                <div v-for="s in mcpServers" :key="s.id || s.name" class="panel-card">
                  <div class="panel-card-main">
                    <div class="panel-card-title" style="display:flex;align-items:center;gap:6px">
                      {{ s.name || s.id }}
                      <span class="source-badge" :class="s.source">{{ s.source === 'mcporter' ? 'mcporter' : '内置' }}</span>
                    </div>
                    <div class="panel-card-desc" v-if="s.transport">{{ s.transport }}</div>
                    <div class="panel-card-desc" v-if="s.status" style="margin-top:2px;font-size:12px">状态: {{ s.status }}</div>
                  </div>
                  <div style="display:flex;gap:6px;margin-top:8px">
                    <button class="btn-sm btn-danger" @click="removeMcpServer(s.id || s.name, s.source, s.config_path)">删除</button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </template>
      </div>

      <!-- Settings Panel -->
      <div class="settings-panel" :class="{ open: showSettings }">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
          <h3>设置</h3>
          <button class="btn-icon" @click="showSettings = false">✕</button>
        </div>
        <div class="setting-group">
          <label>当前模型</label>
          <select :value="currentModel" @change="switchModel($event.target.value)">
            <template v-if="groupedModels.length">
              <optgroup v-for="g in groupedModels" :key="g.provider" :label="g.label">
                <option v-for="m in g.models" :key="m.id" :value="m.id">
                  {{ m.name }}{{ m.tools ? '' : ' ⚠无工具' }}
                </option>
              </optgroup>
            </template>
            <template v-else>
              <option :value="currentModel">{{ currentModel || '未配置' }}</option>
            </template>
          </select>
          <p style="font-size:12px;color:var(--text-secondary);margin-top:4px">
            在「修改配置.command」中添加更多 API Key 后刷新页面
          </p>
        </div>
        <div class="setting-group">
          <label>主题</label>
          <select :value="theme" @change="theme = $event.target.value">
            <option value="light">亮色</option>
            <option value="dark">暗色</option>
          </select>
        </div>
        <div class="setting-group">
          <div class="setting-row">
            <div>
              <label style="margin-bottom:2px">EvoMap 插件</label>
              <p class="setting-hint" style="margin:0">控制 EvoMap 工具与经验检索开关（即时生效）</p>
            </div>
            <label class="switch">
              <input
                type="checkbox"
                :checked="evomapEnabled"
                :disabled="evomapLoading"
                @change="setEvomapEnabled($event.target.checked)"
              >
              <span class="switch-slider"></span>
            </label>
          </div>
        </div>
        <div class="setting-group">
          <div class="setting-row">
            <div>
              <label style="margin-bottom:2px">浏览器可视操作</label>
              <p class="setting-hint" style="margin:0">开启后可看到 browser 工具操作窗口；关闭后后台无界面运行（下次启动 browser 生效）</p>
            </div>
            <label class="switch">
              <input
                type="checkbox"
                :checked="browserVisible"
                :disabled="browserVisibleLoading"
                @change="setBrowserVisible($event.target.checked)"
              >
              <span class="switch-slider"></span>
            </label>
          </div>
        </div>
        <div class="setting-group">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <label style="margin:0">当前回复风格（自动派生，可手动覆盖）</label>
            <button class="btn-mini" @click="onMemoryStyleRefresh" :disabled="memoryStyleLoading">{{ memoryStyleLoading ? '刷新中…' : '刷新' }}</button>
          </div>
          <p class="setting-hint" v-if="memoryStyleLastRefresh" style="margin-top:0">上次刷新：{{ memoryStyleLastRefresh }}</p>
          <p class="setting-hint" v-if="memoryStyleRefreshNote" style="margin-top:0">{{ memoryStyleRefreshNote }}</p>
          <p class="setting-hint" v-if="memoryStyleEnabled">这里显示的是当前实际生效的回复风格：你可以手动写入并覆盖当前风格；也可能由系统根据后续对话重新生成。每轮默认应用，用户本轮明确要求优先。</p>
          <p class="setting-hint">“刷新”会重新读取当前生效风格；手动保存后会直接覆盖当前风格，清空后当前不再应用回复风格，直到后续对话生成出新的风格。</p>
          <p class="setting-hint" v-if="!memoryStyleEnabled">全局风格功能已关闭（config: agent.memory.global_style_enabled=false）。</p>
          <textarea
            class="setting-textarea"
            v-model="memoryStyle"
            rows="4"
            placeholder="这里会显示当前生效的回复风格；也可手动输入覆盖，例如：普通问答默认简洁紧凑，避免冗余客套和过多空行；需要展开时可自然展开。"
          ></textarea>
          <div class="setting-actions">
            <button class="btn-pill" @click="saveMemoryStyle" :disabled="memoryStyleSaving || !memoryStyleEnabled || !memoryStyle.trim()">
              {{ memoryStyleSaving ? '保存中…' : '保存并覆盖当前风格' }}
            </button>
            <button class="btn-outline" @click="openClearMemoryStyleDialog" :disabled="memoryStyleSaving || !memoryStyleEnabled || !memoryStyle.trim()">
              清空当前风格
            </button>
          </div>
        </div>
        <div v-if="token" class="setting-group" style="margin-top:24px;border-top:1px solid var(--border);padding-top:16px">
          <button class="btn-logout" @click="doLogout">退出登录</button>
        </div>
        <div class="setting-footer-note">
          B站飞翔鲸祝您马年大吉！财源广进！<br>
          WhaleClaw 免费开源！
        </div>
      </div>

      <!-- Skill Detail Modal -->
      <div v-if="uiAlertMessage" class="detail-overlay" @click.self="closeUiAlert()">
        <div class="detail-modal confirm-modal">
          <div class="detail-modal-header">
            <h2>提示</h2>
            <button class="btn-icon" @click="closeUiAlert()" title="关闭">✕</button>
          </div>
          <div class="detail-modal-body">
            <p>{{ uiAlertMessage }}</p>
            <div class="confirm-actions">
              <button class="btn-pill" @click="closeUiAlert()">确定</button>
            </div>
          </div>
        </div>
      </div>

      <div v-if="pendingClearMemoryStyle" class="detail-overlay" @click.self="closeClearMemoryStyleDialog()">
        <div class="detail-modal confirm-modal">
          <div class="detail-modal-header">
            <h2>确认清空</h2>
            <button class="btn-icon" @click="closeClearMemoryStyleDialog()" title="关闭">✕</button>
          </div>
          <div class="detail-modal-body">
            <p>确定清空当前回复风格吗？清空后当前不再应用回复风格，后续可由新对话重新生成。</p>
            <div class="confirm-actions">
              <button class="btn-outline" @click="closeClearMemoryStyleDialog()">取消</button>
              <button class="btn-pill" @click="confirmClearMemoryStyle()" :disabled="memoryStyleSaving">
                {{ memoryStyleSaving ? '清空中…' : '确定清空' }}
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Skill Detail Modal -->
      <div v-if="skillDetail" class="detail-overlay" @click.self="skillDetail = null">
        <div class="detail-modal">
          <div class="detail-modal-header">
            <h2>🧩 {{ skillDetail.name }}</h2>
            <button class="btn-icon" @click="skillDetail = null" title="关闭">✕</button>
          </div>
          <div class="detail-modal-body markdown-body" v-html="safeRenderSkillMarkdown(skillDetail.raw_markdown || '')"></div>
        </div>
      </div>

      <!-- Uninstall Confirm Modal -->
      <div v-if="pendingUninstallSkillId" class="detail-overlay" @click.self="cancelUninstallSkill()">
        <div class="detail-modal confirm-modal">
          <div class="detail-modal-header">
            <h2>确认卸载</h2>
            <button class="btn-icon" @click="cancelUninstallSkill()" title="关闭">✕</button>
          </div>
          <div class="detail-modal-body">
            <p>确定卸载技能「{{ pendingUninstallSkillId }}」？</p>
            <div class="confirm-actions">
              <button class="btn-outline" @click="cancelUninstallSkill()">取消</button>
              <button class="btn-pill" @click="uninstallSkill(pendingUninstallSkillId)">确定</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Publish Modal -->
      <div v-if="publishDialog" class="detail-overlay" @click.self="closePublishDialog()">
        <div class="detail-modal confirm-modal">
          <div class="detail-modal-header">
            <h2>发布到clawhub</h2>
            <button class="btn-icon" @click="closePublishDialog()" title="关闭">✕</button>
          </div>
          <div class="detail-modal-body">
            <label class="setting-hint" style="display:block;margin-bottom:6px">slug（可修改）</label>
            <input v-model="publishSlugInput" class="panel-search" placeholder="例如: flywhale-pdf-skill" />
            <label class="setting-hint" style="display:block;margin:10px 0 6px">版本号（semver）</label>
            <input v-model="publishVersionInput" class="panel-search" placeholder="例如: 0.1.0" />
            <div class="confirm-actions">
              <button class="btn-outline" @click="closePublishDialog()">取消</button>
              <button class="btn-pill" @click="confirmPublishDialog()">确认发布</button>
            </div>
          </div>
        </div>
      </div>

      <!-- ClawHub Detail Modal -->
      <div v-if="clawhubDetail" class="detail-overlay" @click.self="clawhubDetail = null">
        <div class="detail-modal">
          <div class="detail-modal-header">
            <h2>🛍 {{ clawhubDetail.name || clawhubDetail.slug }}</h2>
            <button class="btn-icon" @click="clawhubDetail = null" title="关闭">✕</button>
          </div>
          <div class="detail-modal-body">
            <p class="tool-detail-desc">{{ clawhubDetail.summary || '暂无介绍' }}</p>
            <div class="clawhub-stats">
              {{ formatClawhubStats(clawhubDetail) }}
            </div>
            <div class="chip-row" style="margin-bottom:10px">
              <span class="chip chip-accent">{{ clawhubDetail.slug }}</span>
              <span v-if="clawhubDetail.version" class="chip">v{{ clawhubDetail.version }}</span>
            </div>
            <div class="clawhub-links">
              <a class="btn-outline" :href="clawhubDetail.detail_url" target="_blank" rel="noopener noreferrer">打开技能页</a>
              <a v-if="clawhubDetail.repo_url" class="btn-outline" :href="clawhubDetail.repo_url" target="_blank" rel="noopener noreferrer">打开源码仓库</a>
              <button class="btn-pill btn-pill-sm" :disabled="clawhubInstalling[clawhubDetail.slug] || isClawhubSkillInstalled(clawhubDetail)" @click="installClawhubSkill(clawhubDetail.slug, clawhubDetail.version || '', clawhubDetail.repo_url || '')">
                {{ isClawhubSkillInstalled(clawhubDetail) ? '已安装' : (clawhubInstalling[clawhubDetail.slug] ? '安装中…' : '安装此技能') }}
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Tool Detail Modal -->
      <div v-if="toolDetail" class="detail-overlay" @click.self="toolDetail = null">
        <div class="detail-modal">
          <div class="detail-modal-header">
            <h2><code style="font-size:16px">🔧 {{ toolDetail.name }}</code></h2>
            <button class="btn-icon" @click="toolDetail = null" title="关闭">✕</button>
          </div>
          <div class="detail-modal-body">
            <p class="tool-detail-desc">{{ toolDetail.description }}</p>
            <div v-if="toolDetail.parameters && toolDetail.parameters.length" class="tool-detail-params">
              <h3>参数</h3>
              <div v-for="p in toolDetail.parameters" :key="p.name" class="tool-detail-param">
                <div class="tool-detail-param-header">
                  <code>{{ p.name }}</code>
                  <span class="tool-detail-param-type">{{ p.type }}</span>
                  <span v-if="p.required" class="chip chip-accent" style="font-size:11px;padding:1px 6px">必填</span>
                  <span v-else class="chip" style="font-size:11px;padding:1px 6px">可选</span>
                </div>
                <div v-if="p.description" class="tool-detail-param-desc">{{ p.description }}</div>
                <div v-if="p.enum && p.enum.length" class="tool-detail-param-enum">
                  可选值：<code v-for="e in p.enum" :key="e">{{ e }}</code>
                </div>
              </div>
            </div>
            <div v-else class="tool-detail-no-params">此工具无需参数</div>
          </div>
        </div>
      </div>
    </template>
  `,
}).mount('#app');
