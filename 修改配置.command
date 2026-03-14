#!/bin/bash
# ═══════════════════════════════════════════════
#  WhaleClaw — 修改配置
# ═══════════════════════════════════════════════
cd "$(dirname "$0")"

PYTHON="./python/bin/python3.12"
PROJECT_META_FILE=".whaleclaw.project.json"
DEFAULT_CONFIG_DIR="$HOME/.whaleclaw"
CONFIG_DIR="$DEFAULT_CONFIG_DIR"

if [ -f "$PROJECT_META_FILE" ] && [ -f "$PYTHON" ]; then
    saved_home=$("$PYTHON" -c "
import json, pathlib
p = pathlib.Path('$PROJECT_META_FILE')
try:
    data = json.loads(p.read_text(encoding='utf-8'))
except Exception:
    data = {}
home = data.get('whaleclaw_home') if isinstance(data, dict) else ''
print(home or '')
" 2>/dev/null)
    if [ -n "$saved_home" ]; then
        CONFIG_DIR="$saved_home"
    fi
fi

export WHALECLAW_HOME="$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/whaleclaw.json"

save_project_home_binding() {
    local home_dir="$1"
    "$PYTHON" << PYEOF
import json, pathlib
p = pathlib.Path('$PROJECT_META_FILE')
p.write_text(
    json.dumps({"whaleclaw_home": '$home_dir'}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print("  ✅ 当前项目已绑定根目录: " + '$home_dir')
PYEOF
}

# 自动检测本地代理（用于访问海外 API）
if [ -z "$https_proxy" ]; then
    for port in 7897 7890 1087 8080; do
        if nc -z 127.0.0.1 $port 2>/dev/null; then
            export https_proxy="http://127.0.0.1:$port"
            export http_proxy="http://127.0.0.1:$port"
            break
        fi
    done
fi

mkdir -p "$CONFIG_DIR"

# 如果配置文件不存在，创建默认配置
if [ ! -f "$CONFIG_FILE" ]; then
    cat > "$CONFIG_FILE" << 'DEFAULTCFG'
{
  "gateway": {
    "port": 18666,
    "bind": "127.0.0.1",
    "verbose": false,
    "auth": { "mode": "none", "password": null }
  },
  "agent": {
    "model": "deepseek/deepseek-chat",
    "max_tool_rounds": 50,
    "thinking_level": "off",
    "summarizer": {
      "model": "zhipu/glm-4.7-flash",
      "enabled": true
    }
  },
  "models": {
    "anthropic": { "api_key": null, "base_url": null },
    "openai":    { "api_key": null, "base_url": null },
    "deepseek":  { "api_key": null, "base_url": null },
    "qwen":      { "api_key": null, "base_url": null },
    "zhipu":     { "api_key": null, "base_url": null },
    "minimax":   { "api_key": null, "base_url": null },
    "moonshot":  { "api_key": null, "base_url": null },
    "google":    { "api_key": null, "base_url": null },
    "nvidia":    { "api_key": null, "base_url": null }
  },
  "channels": {
    "feishu": {
      "mode": "ws",
      "app_id": "",
      "app_secret": "",
      "verification_token": null,
      "encrypt_key": null,
      "webhook_path": "/webhook/feishu",
      "dm_policy": "pairing"
    }
  },
  "security": {
    "sandbox_mode": "non-main",
    "dm_policy": "pairing",
    "audit": true
  }
}
DEFAULTCFG
    echo ""
    echo "  ✅ 已创建默认配置文件: $CONFIG_FILE"
fi

# ── 用 Python 读取当前状态 ──
read_status() {
    "$PYTHON" << 'PYEOF'
import json, pathlib, os

p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
gw = cfg.get('gateway', {})
ag = cfg.get('agent', {})
au = gw.get('auth', {})
models = cfg.get('models', {})

port = gw.get('port', 18666)
bind = gw.get('bind', '127.0.0.1')
auth_mode = au.get('mode', 'none')
model = ag.get('model', '未设置')
thinking = ag.get('thinking_level', 'off')

# 找出已配置的提供商和模型
configured = []
for name, conf in models.items():
    key = conf.get('api_key')
    is_oauth = name == 'openai' and conf.get('auth_mode') == 'oauth' and conf.get('oauth_access')
    if not key and not is_oauth:
        continue
    cm = conf.get('configured_models', [])
    verified = [m for m in cm if m.get('verified')]
    total = len(cm)
    if is_oauth:
        auth_label = '🔑 ChatGPT OAuth'
        if verified:
            names = ', '.join(m.get('name') or m['id'] for m in verified[:3])
            extra = f' +{len(verified)-3}' if len(verified) > 3 else ''
            configured.append(f'{name} | {auth_label} | {len(verified)}/{total} 模型 | {names}{extra}')
        else:
            configured.append(f'{name} | {auth_label} (未配置模型)')
    else:
        key_preview = key[:8] + '...' + key[-4:] if len(str(key)) > 16 else str(key)
        if verified:
            names = ', '.join(m.get('name') or m['id'] for m in verified[:3])
            extra = f' +{len(verified)-3}' if len(verified) > 3 else ''
            configured.append(f'{name} | {len(verified)}/{total} 模型 | {names}{extra}')
        else:
            configured.append(f'{name} | Key: {key_preview} (未配置模型)')

summ = ag.get('summarizer', {})
summ_model = summ.get('model', 'zhipu/glm-4.7-flash')
summ_enabled = summ.get('enabled', True)
summ_provider = summ_model.split('/')[0] if '/' in summ_model else ''
summ_has_key = bool(models.get(summ_provider, {}).get('api_key')) if summ_provider else False
summ_status = '开启' if summ_enabled else '关闭'
if summ_enabled and not summ_has_key:
    summ_status = '⚠️ 未配置 Key'

feishu = cfg.get('channels', {}).get('feishu', {})
feishu_appid = feishu.get('app_id', '')
feishu_secret = feishu.get('app_secret', '')
if feishu_appid and feishu_secret:
    fid_preview = feishu_appid[:6] + '...' + feishu_appid[-4:] if len(feishu_appid) > 14 else feishu_appid
    feishu_status = f'已配置 (App: {fid_preview})'
else:
    feishu_status = '未配置'
feishu_mode = feishu.get('mode', 'ws')
feishu_dm = feishu.get('dm_policy', 'pairing')
feishu_webhook = feishu.get('webhook_path', '/webhook/feishu')

def q(s: str) -> str:
    return "'" + str(s).replace("'", "'\\''") + "'"

print(f'PORT={q(port)}')
print(f'BIND={q(bind)}')
print(f'AUTH_MODE={q(auth_mode)}')
print(f'MODEL={q(model)}')
print(f'THINKING={q(thinking)}')
print(f'SUMM_MODEL={q(summ_model)}')
print(f'SUMM_ENABLED={q(summ_status)}')
print(f'NUM_KEYS={q(len(configured))}')
for i, c in enumerate(configured):
    print(f'KEY_{i}={q(c)}')
print(f'FEISHU_STATUS={q(feishu_status)}')
print(f'FEISHU_MODE={q(feishu_mode)}')
print(f'FEISHU_DM={q(feishu_dm)}')
print(f'FEISHU_WEBHOOK={q(feishu_webhook)}')
PYEOF
}

eval "$(read_status)"

show_menu() {
    echo "  ═══════════════════════════════════════════════════"
    echo "  🐋 WhaleClaw 配置管理"
    echo "  🎉 B站飞翔鲸祝您马年大吉！财源广进！WhaleClaw 免费开源！"
    echo "  ═══════════════════════════════════════════════════"
    echo ""
    echo "  📁 配置文件: $CONFIG_FILE"
    echo "  🗂️  根目录:   $CONFIG_DIR"
    echo ""
    echo "  ┌─────────────── 当前状态 ───────────────┐"
    echo "  │  端口:     ${PORT}"
    echo "  │  地址:     http://${BIND}:${PORT}"
    echo "  │  认证:     ${AUTH_MODE}"
    echo "  │  默认模型: ${MODEL}"
    echo "  │  思考深度: ${THINKING}"
    echo "  │  压缩模型: ${SUMM_MODEL} (${SUMM_ENABLED})"
    echo "  │  已配置:   ${NUM_KEYS} 个提供商"

    i=0
    while [ $i -lt ${NUM_KEYS:-0} ]; do
        eval "info=\$KEY_$i"
        echo "  │    → $info"
        i=$((i+1))
    done

    echo "  │"
    echo "  │  飞书渠道: ${FEISHU_STATUS}"
    echo "  │  连接模式: ${FEISHU_MODE}"
    echo "  │  飞书 DM:  ${FEISHU_DM}"
    echo "  └────────────────────────────────────────┘"
    echo ""
    echo "  ─── 操作菜单 ────────────────────────────"
    echo ""
    echo "  1) 配置 AI 模型"
    echo "  2) 删除 AI 模型"
    echo "  3) 修改 Gateway 端口"
    echo "  4) 设置登录密码"
    echo "  5) 配置上下文压缩"
    echo "  6) 配置飞书渠道"
    echo "  7) 编辑配置文件 (系统编辑器)"
    echo "  8) 运行诊断 (doctor)"
    echo "  9) 查看完整配置"
    echo "  0) 退出"
    echo ""
}

# ══════════════════════════════════════════════
#  通用 API 验证函数
#  参数: provider model api_key base_url
#  返回: 0=成功, 1=失败
# ══════════════════════════════════════════════
verify_api() {
    local v_provider="$1" v_model="$2" v_apikey="$3" v_base_url="$4"

    "$PYTHON" << PYEOF
import httpx, json, socket, sys, os, pathlib

_raw_key = '''$v_apikey'''
base_url = '''$v_base_url'''
provider = '''$v_provider'''
model = '''$v_model'''

# For OpenAI OAuth mode, read the token from config to avoid bash escaping issues
api_key = _raw_key
_is_oauth = False
if provider == 'openai':
    _cp = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
    if _cp.exists():
        _cc = json.loads(_cp.read_text())
        _oc = _cc.get('models', {}).get('openai', {})
        if _oc.get('auth_mode') == 'oauth':
            _is_oauth = True
            _tok = _oc.get('oauth_access', '')
            if _tok:
                api_key = _tok

def find_proxy():
    for port in (7897, 7890, 1087, 8080):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.3)
            s.connect(('127.0.0.1', port))
            s.close()
            return f'http://127.0.0.1:{port}'
        except Exception:
            pass
    return None

NEEDS_PROXY = {'anthropic', 'openai', 'google'}

headers = {}
url = ''
body = {}

if provider == 'anthropic':
    url = f'{base_url}/v1/messages'
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json',
    }
    body = {'model': model, 'max_tokens': 5, 'messages': [{'role': 'user', 'content': 'hi'}]}
elif provider == 'google':
    url = f'{base_url}/models/{model}:generateContent?key={api_key}'
    headers = {'Content-Type': 'application/json'}
    body = {'contents': [{'parts': [{'text': 'hi'}]}], 'generationConfig': {'maxOutputTokens': 5}}
else:
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    if _is_oauth and _oc.get('oauth_account_id'):
        headers['ChatGPT-Account-Id'] = _oc['oauth_account_id']
    if _is_oauth and provider == 'openai':
        print('  ℹ️  OAuth 模式下 OpenAI 模型跳过验证 (使用 ChatGPT OAuth 链路)')
        sys.exit(0)
    elif 'codex' in model:
        url = f'{base_url}/responses'
        body = {'model': model, 'input': 'hi', 'max_output_tokens': 5}
    else:
        url = f'{base_url}/chat/completions'
        body = {'model': model, 'max_tokens': 5, 'messages': [{'role': 'user', 'content': 'hi'}]}

def try_request(proxy_url=None):
    kwargs = {'timeout': 15}
    if proxy_url:
        kwargs['proxy'] = proxy_url
    with httpx.Client(**kwargs) as client:
        return client.post(url, json=body, headers=headers)

resp = None

if provider in NEEDS_PROXY:
    proxy = os.environ.get('https_proxy') or os.environ.get('HTTP_PROXY') or find_proxy()
    if proxy:
        print(f'  📡 代理: {proxy}')
    try:
        resp = try_request(proxy)
    except (httpx.ConnectError, httpx.TimeoutException):
        if proxy:
            print('  ⚠️  代理失败，尝试直连...')
            try:
                resp = try_request(None)
            except Exception:
                print(f'  ❌ 无法连接 {base_url}')
                sys.exit(1)
        else:
            print(f'  ❌ 无法连接 {base_url}')
            sys.exit(1)
else:
    print('  📡 直连')
    try:
        resp = try_request(None)
    except (httpx.ConnectError, httpx.TimeoutException):
        proxy = os.environ.get('https_proxy') or os.environ.get('HTTP_PROXY') or find_proxy()
        if proxy:
            print(f'  ⚠️  直连失败，尝试代理 {proxy} ...')
            try:
                resp = try_request(proxy)
            except Exception:
                print(f'  ❌ 代理也失败')
                sys.exit(1)
        else:
            print(f'  ❌ 无法连接 {base_url}')
            sys.exit(1)

if resp.status_code in (200, 201):
    print('  \033[32m✅ 验证成功\033[0m')
elif resp.status_code == 401:
    print('  \033[31m❌ API Key 无效 (401)\033[0m')
    sys.exit(1)
elif resp.status_code == 403:
    print('  \033[31m❌ 权限不足 (403)\033[0m')
    sys.exit(1)
elif resp.status_code == 404:
    print(f'  \033[31m❌ 模型不存在: {model} (404)\033[0m')
    sys.exit(1)
elif resp.status_code == 429:
    msg429 = ''
    try:
        msg429 = resp.json().get('error', {}).get('code', '')
    except Exception:
        pass
    if _is_oauth and msg429 == 'insufficient_quota':
        print('  \033[32m✅ OAuth 认证有效 (ChatGPT Plus 不含 API 额度，实际使用无影响)\033[0m')
    else:
        print('  \033[33m⚠️  速率限制 (429)，Key 有效\033[0m')
else:
    print(f'  \033[33m⚠️  返回 {resp.status_code}\033[0m')
    try:
        err = resp.json()
        if 'error' in err:
            msg = err['error'].get('message', '') if isinstance(err['error'], dict) else str(err['error'])
            print(f'  ⚠️  {msg[:120]}')
    except Exception:
        pass
PYEOF
}

# ══════════════════════════════════════════════
#  通用：保存模型到 configured_models
#  参数: provider model_id display_name base_url verified(0/1) thinking
# ══════════════════════════════════════════════
_save_model_entry() {
    local s_provider="$1" s_mid="$2" s_name="$3" s_url="$4" s_verified="$5" s_thinking="$6"

    "$PYTHON" << PYEOF
import json, pathlib, os

p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
prov = cfg.setdefault('models', {}).setdefault('$s_provider', {})
models = prov.setdefault('configured_models', [])

entry = {
    'id': '$s_mid',
    'name': '$s_name',
    'verified': $s_verified == 0,
    'thinking': '$s_thinking',
}
provider_base = prov.get('base_url', '')
if '$s_url' and '$s_url' != provider_base:
    entry['base_url'] = '$s_url'

existing = [i for i, m in enumerate(models) if m.get('id') == '$s_mid']
if existing:
    models[existing[0]] = entry
else:
    models.append(entry)

p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

v = entry['verified']
c = '\033[32m' if v else '\033[31m'
r = '\033[0m'
tag = '✅' if v else '❌'
tk = f' [thinking={entry["thinking"]}]' if entry['thinking'] != 'off' else ''
print(f'  {c}{tag} {entry["id"]} — {entry["name"]}{tk}{r}')
PYEOF
}

# ══════════════════════════════════════════════
#  每个提供商的推荐模型列表
#  格式: "model_id|display_name|thinking_default"
#  thinking_default: off / low / medium / high
# ══════════════════════════════════════════════

_MODELS_anthropic=(
    "claude-sonnet-4-20250514|Claude Sonnet 4|off"
    "claude-opus-4-20250514|Claude Opus 4|off"
    "claude-sonnet-4-20250514|Claude Sonnet 4 (思考)|medium"
    "claude-opus-4-20250514|Claude Opus 4 (思考)|high"
)
_URL_anthropic="https://api.anthropic.com"

_MODELS_openai=(
    "gpt-5.4|GPT-5.4|off"
    "gpt-5.3-codex|GPT-5.3 Codex|off"
    "gpt-5.2|GPT-5.2|off"
)
_URL_openai="https://api.openai.com/v1"

_MODELS_deepseek=(
    "deepseek-chat|DeepSeek Chat|off"
    "deepseek-reasoner|DeepSeek Reasoner (思考)|high"
)
_URL_deepseek="https://api.deepseek.com"

_MODELS_qwen=(
    "qwen3.5-plus|Qwen 3.5 Plus|off"
    "qwen3-max|Qwen 3 Max|off"
)
_URL_qwen="https://dashscope.aliyuncs.com/compatible-mode/v1"

_MODELS_zhipu=(
    "glm-4.7-flash|GLM-4.7 Flash (免费)|off"
    "glm-4.7|GLM-4.7|off"
    "glm-5|GLM-5|off"
)
_URL_zhipu="https://open.bigmodel.cn/api/paas/v4"

_MODELS_minimax=(
    "MiniMax-M2.5|MiniMax M2.5|off"
    "MiniMax-M2.1|MiniMax M2.1|off"
)
_URL_minimax="https://api.minimax.chat/v1"

_MODELS_moonshot=(
    "kimi-k2.5|Kimi K2.5|off"
)
_URL_moonshot="https://api.moonshot.cn/v1"

_MODELS_google=(
    "gemini-3-flash-preview|Gemini 3 Flash|off"
    "gemini-3.1-pro-preview|Gemini 3.1 Pro|off"
    "gemini-3-flash-thinking|Gemini 3 Flash (思考)|medium"
)
_URL_google="https://generativelanguage.googleapis.com/v1beta"

_MODELS_nvidia=(
    "qwen/qwen3.5-397b-a17b|Qwen 3.5 397B 🔧|off"
    "z-ai/glm5|GLM-5 🔧|off"
    "z-ai/glm4.7|GLM-4.7 🔧|off"
    "minimaxai/minimax-m2.1|MiniMax M2.1|off"
    "moonshotai/kimi-k2.5|Kimi K2.5 🔧|off"
    "meta/llama-3.1-405b-instruct|Llama 3.1 405B 🔧|off"
)
_URL_nvidia="https://integrate.api.nvidia.com/v1"

# ══════════════════════════════════════════════
#  通用提供商配置流程
#  参数: provider_name display_label
# ══════════════════════════════════════════════
configure_provider() {
    local pname="$1"
    local plabel="$2"

    echo ""
    echo "  ═══ 配置 $plabel ═══"
    echo ""

    # 获取该 provider 的推荐模型数组和默认 URL (bash 3.2 兼容)
    eval "local default_url=\"\$_URL_${pname}\""

    # ── Step 1: API Key (复用已保存的) ──
    local saved_key
    saved_key=$("$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
print(cfg.get('models', {}).get('$pname', {}).get('api_key', '') or '')
" 2>/dev/null)

    if [ -n "$saved_key" ]; then
        local preview="${saved_key:0:12}...${saved_key: -4}"
        echo "  已保存的 API Key: $preview"
        read -p "  回车复用，或输入新 Key: " new_key
        if [ -n "$new_key" ]; then
            saved_key="$new_key"
        fi
    else
        read -p "  请输入 $plabel API Key: " saved_key
        if [ -z "$saved_key" ]; then
            echo "  ⚠️  API Key 为空，跳过"
            return
        fi
    fi

    local apikey="$saved_key"

    # ── Step 2: Base URL ──
    echo ""
    echo "  默认 Base URL: $default_url"
    read -p "  如需修改请输入，回车保持默认: " custom_url
    local base_url="${custom_url:-$default_url}"

    # 保存 API Key + Base URL
    "$PYTHON" << PYEOF
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
prov = cfg.setdefault('models', {}).setdefault('$pname', {})
prov['api_key'] = '''$apikey'''
prov['base_url'] = '$base_url'
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ API Key 和 Base URL 已保存')
PYEOF

    # ── Step 3: 显示已配置的模型 ──
    echo ""
    echo "  ─── 已配置的模型 ────────────────"
    "$PYTHON" << PYEOF
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
models = cfg.get('models', {}).get('$pname', {}).get('configured_models', [])
if models:
    for m in models:
        v = m.get('verified', False)
        c = '\033[32m' if v else '\033[31m'
        r = '\033[0m'
        tag = '✅' if v else '❌'
        tk = f' [thinking={m.get("thinking","off")}]' if m.get('thinking','off') != 'off' else ''
        burl = m.get('base_url', '')
        url_info = f'  (URL: {burl})' if burl else ''
        print(f'  {c}{tag} {m["id"]} — {m.get("name","")}{tk}{url_info}{r}')
else:
    print('  (暂无已配置模型)')
PYEOF

    # ── Step 4: 选择要配置的模型 ──
    _provider_model_menu "$pname" "$plabel" "$apikey" "$base_url"
}

# ══════════════════════════════════════════════
#  显示模型菜单并处理选择
# ══════════════════════════════════════════════
_provider_model_menu() {
    local pname="$1" plabel="$2" apikey="$3" base_url="$4"
    local oauth_openai_only="no"
    if [ "$pname" = "openai" ]; then
        local _mode
        _mode=$("$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
if p.exists():
    cfg = json.loads(p.read_text())
    print(cfg.get('models', {}).get('openai', {}).get('auth_mode', 'api_key'))
else:
    print('api_key')
" 2>/dev/null)
        if [ "$_mode" = "oauth" ]; then
            oauth_openai_only="yes"
        fi
    fi

    # bash 3.2: 用 eval 间接获取数组
    local arr_name="_MODELS_${pname}"
    eval "local count=\${#${arr_name}[@]}"

    echo ""
    echo "  ─── 可选模型 ─────────────────────"
    echo ""

    local i=0
    while [ $i -lt $count ]; do
        eval "local entry=\"\${${arr_name}[$i]}\""
        local mid=$(echo "$entry" | cut -d'|' -f1)
        local mname=$(echo "$entry" | cut -d'|' -f2)
        local mthink=$(echo "$entry" | cut -d'|' -f3)
        local think_label=""
        if [ "$mthink" != "off" ]; then
            think_label=" 💭"
        fi
        echo "    $((i+1))) ${mname}${think_label}"
        i=$((i+1))
    done
    if [ "$oauth_openai_only" != "yes" ]; then
        echo "    c) 自定义输入"
        echo "    a) 批量验证以上全部"
    fi
    echo "    0) 返回"
    echo ""
    if [ "$oauth_openai_only" = "yes" ]; then
        read -p "  选择 [1-${count}/0]: " mchoice
    else
        read -p "  选择 [1-${count}/c/a/0]: " mchoice
    fi

    case $mchoice in
        0) return ;;
        a|A)
            if [ "$oauth_openai_only" = "yes" ]; then
                echo "  ❌ OAuth 模式下 OpenAI 仅支持推荐列表中的模型"
                return
            fi
            _batch_verify_provider "$pname" "$apikey" "$base_url"; return ;;
        c|C)
            if [ "$oauth_openai_only" = "yes" ]; then
                echo "  ❌ OAuth 模式下 OpenAI 仅支持推荐列表中的模型"
                return
            fi
            read -p "  输入模型 ID: " custom_mid
            read -p "  显示名称: " custom_name
            [ -z "$custom_mid" ] && return
            _configure_single_model "$pname" "$custom_mid" "$custom_name" "$apikey" "$base_url" "off"
            return
            ;;
    esac

    # 数字选择
    if [ "$mchoice" -ge 1 ] 2>/dev/null && [ "$mchoice" -le $count ] 2>/dev/null; then
        local idx=$((mchoice - 1))
        eval "local entry=\"\${${arr_name}[$idx]}\""
        local mid=$(echo "$entry" | cut -d'|' -f1)
        local mname=$(echo "$entry" | cut -d'|' -f2)
        local mthink=$(echo "$entry" | cut -d'|' -f3)
        _configure_single_model "$pname" "$mid" "$mname" "$apikey" "$base_url" "$mthink"
    else
        echo "  ❌ 无效选择"
    fi
}

# ══════════════════════════════════════════════
#  配置单个模型 (验证 + 思考设置 + 保存)
# ══════════════════════════════════════════════
_configure_single_model() {
    local pname="$1" mid="$2" mname="$3" apikey="$4" base_url="$5" default_think="$6"

    echo ""
    echo "  模型: $mid ($mname)"
    echo "  Base URL: $base_url"
    read -p "  此模型使用不同的 Base URL? 回车保持默认: " model_url
    local use_url="${model_url:-$base_url}"

    # 思考模式设置
    local thinking="$default_think"
    if [ "$default_think" != "off" ]; then
        echo ""
        echo "  此模型支持深度思考 (当前: $default_think)"
        echo "    0) off — 关闭"
        echo "    1) low — 轻度"
        echo "    2) medium — 中度"
        echo "    3) high — 深度"
        read -p "  选择思考深度 [0-3, 默认 $default_think]: " tchoice
        case $tchoice in
            0) thinking="off" ;;
            1) thinking="low" ;;
            2) thinking="medium" ;;
            3) thinking="high" ;;
        esac
    else
        echo ""
        echo "  是否启用深度思考? (适用于复杂推理任务)"
        echo "    0) off — 关闭 (默认)"
        echo "    1) low — 轻度"
        echo "    2) medium — 中度"
        echo "    3) high — 深度"
        read -p "  选择 [0-3, 默认 off]: " tchoice
        case $tchoice in
            1) thinking="low" ;;
            2) thinking="medium" ;;
            3) thinking="high" ;;
        esac
    fi

    echo ""
    echo "  ⏳ 验证 $mid ..."
    verify_api "$pname" "$mid" "$apikey" "$use_url"
    local vresult=$?

    _save_model_entry "$pname" "$mid" "$mname" "$use_url" "$vresult" "$thinking"

    if [ "$vresult" -eq 0 ]; then
        echo ""
        read -p "  是否设为默认模型? [y/N]: " set_default
        if echo "$set_default" | grep -qi '^y'; then
            "$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
cfg.setdefault('agent', {})['model'] = '$pname/$mid'
cfg['agent']['thinking_level'] = '$thinking'
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ 默认模型: $pname/$mid (thinking=$thinking)')
"
        fi
    fi

    echo ""
    read -p "  继续配置其他 $pname 模型? [Y/n]: " cont
    if ! echo "$cont" | grep -qi '^n'; then
        _provider_model_menu "$pname" "" "$apikey" "$base_url"
    fi
}

# ══════════════════════════════════════════════
#  批量验证某 provider 的全部推荐模型
# ══════════════════════════════════════════════
_batch_verify_provider() {
    local pname="$1" apikey="$2" base_url="$3"

    local arr_name="_MODELS_${pname}"
    eval "local count=\${#${arr_name}[@]}"

    echo ""
    echo "  ═══ 批量验证 $count 个模型 ═══"
    echo ""

    local i=0
    local pass_count=0
    while [ $i -lt $count ]; do
        eval "local entry=\"\${${arr_name}[$i]}\""
        local mid=$(echo "$entry" | cut -d'|' -f1)
        local mname=$(echo "$entry" | cut -d'|' -f2)
        local mthink=$(echo "$entry" | cut -d'|' -f3)

        echo "  [$((i+1))/$count] $mid ..."
        verify_api "$pname" "$mid" "$apikey" "$base_url" 2>/dev/null
        local vresult=$?

        _save_model_entry "$pname" "$mid" "$mname" "$base_url" "$vresult" "$mthink"

        if [ $vresult -eq 0 ]; then
            pass_count=$((pass_count + 1))
        fi
        i=$((i + 1))
        echo ""
    done

    echo "  ─────────────────────────────────"
    echo "  验证完成: $pass_count / $count 个模型可用"
    echo ""

    if [ $pass_count -gt 0 ]; then
        read -p "  将第一个可用模型设为默认? [y/N]: " set_def
        if echo "$set_def" | grep -qi '^y'; then
            "$PYTHON" << PYEOF
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
models = cfg.get('models', {}).get('$pname', {}).get('configured_models', [])
for m in models:
    if m.get('verified'):
        cfg.setdefault('agent', {})['model'] = f'$pname/{m["id"]}'
        cfg['agent']['thinking_level'] = m.get('thinking', 'off')
        p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        tk = m.get('thinking', 'off')
        print(f'  ✅ 默认模型: $pname/{m["id"]} (thinking={tk})')
        break
PYEOF
        fi
    fi
}

# ══════════════════════════════════════════════
#  配置上下文压缩模型
# ══════════════════════════════════════════════
configure_summarizer() {
    echo ""
    echo "  ═══ 配置上下文压缩 ═══"
    echo ""
    echo "  当对话过长时，WhaleClaw 会用便宜的小模型自动生成 L0/L1 分层摘要，"
    echo "  持久化到数据库，后续对话按需加载，节省 token 开销。"
    echo ""
    echo "  ┌─────────────── 当前配置 ───────────────┐"
    echo "  │  压缩模型:   ${SUMM_MODEL}"
    echo "  │  状态:       ${SUMM_ENABLED}"
    echo "  └────────────────────────────────────────┘"
    echo ""
    echo "  ─── 操作 ──────────────────────────────"
    echo ""
    echo "  1) 选择压缩模型"
    echo "  2) 开启/关闭压缩"
    echo "  0) 返回"
    echo ""
    read -p "  选择 [0-2]: " schoice

    case $schoice in
        1)
            echo ""
            echo "  推荐使用便宜、快速的模型 (几乎不花钱):"
            echo ""
            echo "  1) zhipu/glm-4.7-flash  — 智谱 GLM (免费)"
            echo "  2) deepseek/deepseek-chat — DeepSeek Chat"
            echo "  3) qwen/qwen-turbo       — 通义千问 Turbo"
            echo "  4) moonshot/kimi-k2.5    — Kimi K2.5"
            echo "  5) 自定义输入"
            echo ""
            read -p "  选择 [1-5]: " msel
            local new_model="" new_provider="" new_model_short=""
            case $msel in
                1) new_model="zhipu/glm-4.7-flash"; new_provider="zhipu"; new_model_short="glm-4.7-flash" ;;
                2) new_model="deepseek/deepseek-chat"; new_provider="deepseek"; new_model_short="deepseek-chat" ;;
                3) new_model="qwen/qwen-turbo"; new_provider="qwen"; new_model_short="qwen-turbo" ;;
                4) new_model="moonshot/kimi-k2.5"; new_provider="moonshot"; new_model_short="kimi-k2.5" ;;
                5)
                    read -p "  输入模型 ID (格式: provider/model): " new_model
                    new_provider=$(echo "$new_model" | cut -d'/' -f1)
                    new_model_short=$(echo "$new_model" | cut -d'/' -f2-)
                    ;;
                *) echo "  ❌ 无效选择"; return ;;
            esac

            if [ -z "$new_model" ] || [ -z "$new_provider" ]; then
                return
            fi

            # ── Step 1: 获取该 provider 的默认 URL ──
            eval "local default_url=\"\$_URL_${new_provider}\""
            if [ -z "$default_url" ]; then
                default_url="https://api.example.com/v1"
            fi

            # ── Step 2: 检查已有 API Key ──
            local apikey=""
            local existing_key
            existing_key=$("$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
print(cfg.get('models', {}).get('$new_provider', {}).get('api_key', '') or '')
" 2>/dev/null)

            local existing_url
            existing_url=$("$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
print(cfg.get('models', {}).get('$new_provider', {}).get('base_url', '') or '')
" 2>/dev/null)

            if [ -n "$existing_key" ]; then
                local preview="${existing_key:0:8}...${existing_key: -4}"
                echo ""
                echo "  $new_provider 已有 API Key: $preview"
                read -p "  回车复用，或输入新 Key: " new_key
                if [ -n "$new_key" ]; then
                    apikey="$new_key"
                else
                    apikey="$existing_key"
                fi
            else
                echo ""
                echo "  ⚠️  提供商 $new_provider 尚未配置 API Key"
                echo ""
                read -p "  请输入 $new_provider 的 API Key: " apikey
                if [ -z "$apikey" ]; then
                    echo "  ❌ API Key 为空，已取消"
                    return
                fi
            fi

            # ── Step 3: Base URL ──
            local use_url="${existing_url:-$default_url}"
            echo ""
            echo "  默认 Base URL: $use_url"
            read -p "  如需修改请输入，回车保持默认: " custom_url
            local base_url="${custom_url:-$use_url}"

            # ── Step 4: 保存 API Key + Base URL ──
            "$PYTHON" << PYEOF
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
prov = cfg.setdefault('models', {}).setdefault('$new_provider', {})
prov['api_key'] = '''$apikey'''
prov['base_url'] = '$base_url'
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ $new_provider API Key 和 Base URL 已保存')
PYEOF

            # ── Step 5: 验证 API 连通性 ──
            echo ""
            echo "  ─── 验证 $new_model ───"
            verify_api "$new_provider" "$new_model_short" "$apikey" "$base_url"
            local verify_result=$?

            if [ $verify_result -ne 0 ]; then
                echo ""
                echo "  ⚠️  验证失败，但配置已保存。你可以稍后重试。"
                read -p "  仍然设为压缩模型? [y/N]: " force
                if ! echo "$force" | grep -qi '^y'; then
                    echo "  已取消"
                    return
                fi
            fi

            # ── Step 6: 保存压缩模型选择 ──
            "$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
summ = cfg.setdefault('agent', {}).setdefault('summarizer', {})
summ['model'] = '$new_model'
summ['enabled'] = True
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print(f'  ✅ 压缩模型已设置为 $new_model')
"
            # 刷新状态变量
            eval "$(read_status)"
            ;;
        2)
            echo ""
            "$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
summ = cfg.setdefault('agent', {}).setdefault('summarizer', {})
current = summ.get('enabled', True)
summ['enabled'] = not current
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
state = '开启' if not current else '关闭'
print(f'  ✅ 上下文压缩已{state}')
"
            # 刷新状态变量
            eval "$(read_status)"
            ;;
        0) return ;;
        *) echo "  ❌ 无效选择" ;;
    esac
}

# ══════════════════════════════════════════════
#  配置飞书渠道
# ══════════════════════════════════════════════
configure_feishu() {
    echo ""
    echo "  ═══ 配置飞书渠道 ═══"
    echo ""
    echo "  飞书机器人需要在飞书开放平台创建应用并获取凭证。"
    echo "  开放平台地址: https://open.feishu.cn"
    echo ""
    echo "  ┌─────────────── 当前配置 ───────────────┐"
    echo "  │  状态:       ${FEISHU_STATUS}"
    echo "  │  连接模式:   ${FEISHU_MODE} (ws=长连接, webhook=回调)"
    echo "  │  DM 策略:    ${FEISHU_DM}"
    echo "  └────────────────────────────────────────┘"
    echo ""
    echo "  ─── 操作 ──────────────────────────────"
    echo ""
    echo "  1) 设置 App ID 和 App Secret"
    echo "  2) 设置连接模式"
    echo "  3) 设置 DM 策略"
    echo "  0) 返回"
    echo ""
    read -p "  选择 [0-3]: " fchoice

    case $fchoice in
        1)
            echo ""
            local saved_aid saved_secret
            saved_aid=$("$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
print(cfg.get('channels', {}).get('feishu', {}).get('app_id', '') or '')
" 2>/dev/null)
            saved_secret=$("$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
print(cfg.get('channels', {}).get('feishu', {}).get('app_secret', '') or '')
" 2>/dev/null)

            if [ -n "$saved_aid" ]; then
                local aid_preview="${saved_aid:0:10}...${saved_aid: -4}"
                echo "  已保存的 App ID: $aid_preview"
                read -p "  回车复用，或输入新 App ID: " new_aid
                if [ -n "$new_aid" ]; then
                    saved_aid="$new_aid"
                fi
            else
                read -p "  请输入 App ID (cli_ 开头): " saved_aid
                if [ -z "$saved_aid" ]; then
                    echo "  ⚠️  App ID 为空，跳过"
                    return
                fi
            fi

            if [ -n "$saved_secret" ]; then
                local sec_preview="${saved_secret:0:8}...${saved_secret: -4}"
                echo "  已保存的 App Secret: $sec_preview"
                read -p "  回车复用，或输入新 App Secret: " new_secret
                if [ -n "$new_secret" ]; then
                    saved_secret="$new_secret"
                fi
            else
                read -p "  请输入 App Secret: " saved_secret
                if [ -z "$saved_secret" ]; then
                    echo "  ⚠️  App Secret 为空，跳过"
                    return
                fi
            fi

            "$PYTHON" << PYEOF
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
feishu = cfg.setdefault('channels', {}).setdefault('feishu', {})
feishu['app_id'] = '''$saved_aid'''
feishu['app_secret'] = '''$saved_secret'''
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ 飞书 App ID 和 App Secret 已保存')
print('  ⚠️  需要重启 Gateway 生效')
PYEOF
            eval "$(read_status)"
            ;;
        2)
            echo ""
            echo "  连接模式决定 WhaleClaw 如何接收飞书消息:"
            echo ""
            echo "  a) ws      — 长连接 (推荐，无需公网 IP)"
            echo "  b) webhook — HTTP 回调 (需要公网可访问地址)"
            echo ""
            read -p "  选择 [a-b, 当前: ${FEISHU_MODE}]: " modechoice
            local mode_val=""
            case $modechoice in
                a) mode_val="ws" ;;
                b) mode_val="webhook" ;;
                *) echo "  ❌ 无效选择"; return ;;
            esac
            "$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
cfg.setdefault('channels', {}).setdefault('feishu', {})['mode'] = '$mode_val'
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ 连接模式已设置为 $mode_val')
print('  ⚠️  需要重启 Gateway 生效')
"
            eval "$(read_status)"
            ;;
        3)
            echo ""
            echo "  DM 策略控制谁可以与机器人私聊:"
            echo ""
            echo "  a) pairing — 需要配对码验证 (默认，最安全)"
            echo "  b) open    — 任何人可直接对话"
            echo "  c) closed  — 禁止私聊"
            echo ""
            read -p "  选择 [a-c]: " dmpolicy
            local dm_val=""
            case $dmpolicy in
                a) dm_val="pairing" ;;
                b) dm_val="open" ;;
                c) dm_val="closed" ;;
                *) echo "  ❌ 无效选择"; return ;;
            esac
            "$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
cfg.setdefault('channels', {}).setdefault('feishu', {})['dm_policy'] = '$dm_val'
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ 飞书 DM 策略已设置为 $dm_val')
"
            eval "$(read_status)"
            ;;
        0) return ;;
        *) echo "  ❌ 无效选择" ;;
    esac
}

# ══════════════════════════════════════════════
#  配置 AI 模型 — 选择提供商
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
#  OpenAI 专用配置 (API Key / ChatGPT 账号登录)
# ══════════════════════════════════════════════
configure_openai() {
    echo ""
    echo "  ═══ 配置 OpenAI ═══"
    echo ""

    # 检查当前认证模式
    local current_mode
    current_mode=$("$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
if p.exists():
    cfg = json.loads(p.read_text())
    print(cfg.get('models', {}).get('openai', {}).get('auth_mode', 'api_key'))
else:
    print('api_key')
" 2>/dev/null)

    if [ "$current_mode" = "oauth" ]; then
        echo "  当前模式: 🔑 ChatGPT 账号登录 (OAuth)"
        "$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
oc = cfg.get('models', {}).get('openai', {})
aid = oc.get('oauth_account_id', '')
exp = oc.get('oauth_expires', 0)
import time
if exp > time.time():
    print(f'  Account ID: {aid}')
    print(f'  Token 有效期至: {time.strftime(\"%Y-%m-%d %H:%M\", time.localtime(exp))}')
else:
    print(f'  Account ID: {aid}')
    print('  ⚠️  Token 已过期，需要重新登录')
" 2>/dev/null
    else
        echo "  当前模式: 🔐 API Key"
    fi

    echo ""
    echo "  请选择认证方式:"
    echo "  ─────────────────────────────────────────────────"
    echo "  1) API Key         — 使用 OpenAI API Key (需付费)"
    echo "  2) ChatGPT 账号登录 — 使用 ChatGPT Plus/Pro 账号 (免费)"
    echo "  0) 返回"
    echo ""
    read -p "  选择 [1/2/0]: " auth_choice

    case $auth_choice in
        1)
            configure_provider "openai" "OpenAI"
            # 确保 auth_mode 设为 api_key
            "$PYTHON" << 'PYEOF'
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
cfg.setdefault('models', {}).setdefault('openai', {})['auth_mode'] = 'api_key'
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
PYEOF
            ;;
        2)
            # 检查是否已有有效 OAuth token
            local need_login="yes"
            local oauth_token
            oauth_token=$("$PYTHON" -c "
import json, pathlib, os, time
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
if p.exists():
    cfg = json.loads(p.read_text())
    oc = cfg.get('models', {}).get('openai', {})
    token = oc.get('oauth_access', '')
    exp = oc.get('oauth_expires', 0)
    if token and exp > time.time():
        print(token)
" 2>/dev/null)

            if [ -n "$oauth_token" ]; then
                need_login="no"
                echo ""
                echo "  ✅ 已有有效的 ChatGPT 登录，无需重新登录"
                echo ""
                echo "  是否重新登录？[y/N]: "
                read -p "  " relogin_choice
                if [ "$relogin_choice" = "y" ] || [ "$relogin_choice" = "Y" ]; then
                    need_login="yes"
                fi
            fi

            if [ "$need_login" = "yes" ]; then
                echo ""
                echo "  将打开浏览器登录你的 ChatGPT 账号"
                echo "  (需要 ChatGPT Plus / Pro / Team 订阅)"
                echo ""
                "$PYTHON" << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('whaleclaw')))
from whaleclaw.utils.openai_oauth import login, save_oauth_to_config
try:
    result = login()
    save_oauth_to_config(result)
    print(f'\n  ✅ ChatGPT 账号已关联')
    print(f'  Account ID: {result.account_id}')
except KeyboardInterrupt:
    print('\n  ❌ 已取消')
except Exception as e:
    print(f'\n  ❌ 登录失败: {e}')
PYEOF
                # 重新读取 token
                oauth_token=$("$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
print(cfg.get('models', {}).get('openai', {}).get('oauth_access', '') or '')
" 2>/dev/null)
            fi

            if [ -n "$oauth_token" ]; then
                "$PYTHON" << 'PYEOF'
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
oc = cfg.setdefault('models', {}).setdefault('openai', {})
oc['auth_mode'] = 'oauth'
oc['base_url'] = 'https://chatgpt.com/backend-api/codex'
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
PYEOF
                echo ""
                echo "  现在配置要使用的模型:"
                _provider_model_menu "openai" "OpenAI" "$oauth_token" "https://chatgpt.com/backend-api/codex"
            fi
            ;;
        0) return ;;
        *) echo "  ❌ 无效选择" ;;
    esac
}

configure_custom_provider() {
    echo ""
    echo "  ═══ 自定义 AI 模型 ═══"
    echo ""
    echo "  适用于任何兼容 OpenAI Chat Completions 接口的服务。"
    echo ""

    read -p "  提供商名称 (英文，如 my-provider): " custom_pname
    if [ -z "$custom_pname" ]; then
        echo "  ⚠️  名称为空，已取消"
        return
    fi

    read -p "  显示名称 (如 我的模型): " custom_label
    [ -z "$custom_label" ] && custom_label="$custom_pname"

    read -p "  API Key: " custom_key
    if [ -z "$custom_key" ]; then
        echo "  ⚠️  API Key 为空，已取消"
        return
    fi

    read -p "  Base URL (如 https://api.example.com/v1): " custom_url
    if [ -z "$custom_url" ]; then
        echo "  ⚠️  Base URL 为空，已取消"
        return
    fi

    read -p "  模型 ID (如 my-model-v1): " custom_mid
    if [ -z "$custom_mid" ]; then
        echo "  ⚠️  模型 ID 为空，已取消"
        return
    fi

    read -p "  模型显示名称 (回车同模型 ID): " custom_mname
    [ -z "$custom_mname" ] && custom_mname="$custom_mid"

    # 保存到配置
    "$PYTHON" << PYEOF
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
prov = cfg.setdefault('models', {}).setdefault('$custom_pname', {})
prov['api_key'] = '''$custom_key'''
prov['base_url'] = '$custom_url'
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ API Key 和 Base URL 已保存')
PYEOF

    echo ""
    echo "  ⏳ 验证 $custom_mid ..."
    verify_api "$custom_pname" "$custom_mid" "$custom_key" "$custom_url"
    local vresult=$?

    _save_model_entry "$custom_pname" "$custom_mid" "$custom_mname" "$custom_url" "$vresult" "off"

    if [ "$vresult" -eq 0 ]; then
        echo ""
        read -p "  是否设为默认模型? [y/N]: " set_default
        if echo "$set_default" | grep -qi '^y'; then
            "$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
cfg.setdefault('agent', {})['model'] = '$custom_pname/$custom_mid'
cfg['agent']['thinking_level'] = 'off'
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ 默认模型: $custom_pname/$custom_mid')
"
        fi
    fi
}

# ══════════════════════════════════════════════
#  删除 AI 模型 / 提供商
# ══════════════════════════════════════════════
delete_models() {
    echo ""
    echo "  ═══ 删除 AI 模型 ═══"
    echo ""

    # 列出所有已配置的提供商及其模型
    "$PYTHON" << 'PYEOF'
import json, pathlib, os

p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
models_cfg = cfg.get('models', {})

entries = []
idx = 1
for pname, pconf in models_cfg.items():
    key = pconf.get('api_key')
    is_oauth = pname == 'openai' and pconf.get('auth_mode') == 'oauth' and pconf.get('oauth_access')
    if not key and not is_oauth:
        continue
    cm = pconf.get('configured_models', [])
    if not cm:
        print(f'  {idx}) [{pname}] (无模型，仅有 API Key)')
        entries.append((pname, None))
        idx += 1
        continue
    for m in cm:
        v = '✅' if m.get('verified') else '❌'
        tk = f' [thinking={m.get("thinking","off")}]' if m.get('thinking', 'off') != 'off' else ''
        print(f'  {idx}) [{pname}] {v} {m["id"]} — {m.get("name","")}{tk}')
        entries.append((pname, m['id']))
        idx += 1

if not entries:
    print('  (没有已配置的模型)')

# Write entries to temp file for bash to read
tmp = pathlib.Path('/tmp/.whaleclaw_delete_list.json')
tmp.write_text(json.dumps(entries, ensure_ascii=False))
print(f'\n  共 {len(entries)} 项')
PYEOF

    echo ""
    echo "  ─── 操作 ──────────────────────────────"
    echo "  输入编号删除单个模型"
    echo "  输入 p 删除整个提供商 (含所有模型和 Key)"
    echo "  输入 0 返回"
    echo ""
    read -p "  选择: " del_choice

    if [ "$del_choice" = "0" ] || [ -z "$del_choice" ]; then
        return
    fi

    if [ "$del_choice" = "p" ] || [ "$del_choice" = "P" ]; then
        read -p "  输入要删除的提供商名称 (如 deepseek): " del_pname
        if [ -z "$del_pname" ]; then
            echo "  已取消"
            return
        fi
        "$PYTHON" << PYEOF
import json, pathlib, os

p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
models_cfg = cfg.get('models', {})
pname = '$del_pname'

if pname not in models_cfg:
    print(f'  ❌ 提供商 {pname} 不存在')
else:
    cm = models_cfg[pname].get('configured_models', [])
    count = len(cm)
    del models_cfg[pname]
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f'  ✅ 已删除提供商 {pname} (含 {count} 个模型)')

    # 检查默认模型是否被删了
    agent_model = cfg.get('agent', {}).get('model', '')
    if agent_model.startswith(pname + '/'):
        print(f'  ⚠️  默认模型 {agent_model} 已被删除，请重新设置')
PYEOF
        eval "$(read_status)"
        return
    fi

    # 数字选择 — 删除单个模型
    "$PYTHON" << PYEOF
import json, pathlib, os

p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())

tmp = pathlib.Path('/tmp/.whaleclaw_delete_list.json')
if not tmp.exists():
    print('  ❌ 列表已失效，请重试')
    exit(1)

entries = json.loads(tmp.read_text())
choice = int('$del_choice') - 1

if choice < 0 or choice >= len(entries):
    print('  ❌ 无效编号')
    exit(1)

pname, mid = entries[choice]
models_cfg = cfg.get('models', {})
pconf = models_cfg.get(pname, {})

if mid is None:
    # 删除仅有 Key 无模型的提供商
    del models_cfg[pname]
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f'  ✅ 已删除提供商 {pname}')
else:
    cm = pconf.get('configured_models', [])
    cm = [m for m in cm if m.get('id') != mid]
    pconf['configured_models'] = cm
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    remaining = len(cm)
    print(f'  ✅ 已删除模型 {pname}/{mid}')
    if remaining == 0:
        print(f'  ℹ️  {pname} 已无模型 (API Key 保留)')

    # 检查默认模型
    agent_model = cfg.get('agent', {}).get('model', '')
    if agent_model == f'{pname}/{mid}':
        print(f'  ⚠️  默认模型 {agent_model} 已被删除，请重新设置')

tmp.unlink(missing_ok=True)
PYEOF
    eval "$(read_status)"
}

configure_model() {
    echo ""
    echo "  ═══ 配置 AI 模型 ═══"
    echo ""
    echo "  选择提供商:"
    echo "  ─────────────────────────────────────────────────"
    echo "  1) Anthropic  — Claude Sonnet 4 / Opus 4"
    echo "  2) OpenAI     — GPT-5.4 / GPT-5.2"
    echo "  3) 通义千问   — qwen3.5-plus / qwen3-max / qwq-plus"
    echo "  4) 智谱 GLM   — glm-5 / glm-4.7 / glm-4.7-flash"
    echo "  5) MiniMax    — MiniMax-M2.5 / M2.1"
    echo "  6) 月之暗面   — kimi-k2.5"
    echo "  7) Google     — Gemini 3 Flash / 3.1 Pro"
    echo "  8) NVIDIA NIM — Qwen 3.5 / GLM / Kimi / Llama"
    echo "  9) 自定义     — 任意 OpenAI 兼容接口"
    echo "  0) 返回"
    echo ""
    read -p "  选择提供商 [0-9]: " provider_choice

    case $provider_choice in
        1) configure_provider "anthropic" "Anthropic" ;;
        2) configure_openai ;;
        3) configure_provider "qwen" "通义千问" ;;
        4) configure_provider "zhipu" "智谱 GLM" ;;
        5) configure_provider "minimax" "MiniMax" ;;
        6) configure_provider "moonshot" "月之暗面" ;;
        7) configure_provider "google" "Google" ;;
        8) configure_provider "nvidia" "NVIDIA NIM" ;;
        9) configure_custom_provider ;;
        0) return ;;
        *) echo "  ❌ 无效选择" ;;
    esac
}

# ══════════════════════════════════════════════
#  主循环
# ══════════════════════════════════════════════
while true; do
    eval "$(read_status)"
    show_menu
    read -p "  请输入选项 [0-9]: " choice

    case $choice in
        1)
            configure_model
            echo ""
            read -p "  按回车键继续..."
            ;;
        2)
            delete_models
            echo ""
            read -p "  按回车键继续..."
            ;;
        3)
            echo ""
            echo "  当前端口: ${PORT}"
            read -p "  请输入新端口号: " port
            if [ -n "$port" ]; then
                "$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
cfg.setdefault('gateway', {})['port'] = $port
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ Gateway 端口已设置为 $port')
print('  ⚠️  需要重启 Gateway 生效')
"

                echo ""
                echo "  当前项目根目录: $CONFIG_DIR"
                read -p "  是否为当前项目创建独立根目录? [y/N]: " new_home_choice
                if echo "$new_home_choice" | grep -qi '^y'; then
                    default_home="$HOME/.whaleclaw-${port}"
                    read -p "  输入新根目录 (回车使用默认: ${default_home}): " input_home
                    picked_home="${input_home:-$default_home}"
                    resolved_home=$("$PYTHON" -c "
from pathlib import Path
print(str(Path('$picked_home').expanduser().resolve()))
")
                    new_config_file="${resolved_home}/whaleclaw.json"

                    if [ "$resolved_home" = "$CONFIG_DIR" ]; then
                        echo "  ℹ️  新根目录与当前一致，保持不变。"
                    else
                        mkdir -p "$resolved_home"
                        if [ ! -f "$new_config_file" ]; then
                            cp "$CONFIG_FILE" "$new_config_file"
                            echo "  ✅ 已复制当前配置到: $new_config_file"
                        else
                            echo "  ℹ️  检测到已有配置文件: $new_config_file"
                        fi

                        save_project_home_binding "$resolved_home"
                        CONFIG_DIR="$resolved_home"
                        CONFIG_FILE="$new_config_file"
                        export WHALECLAW_HOME="$CONFIG_DIR"
                        echo "  ✅ 当前项目后续将使用独立根目录"
                    fi
                fi
            fi
            echo ""
            read -p "  按回车键继续..."
            ;;
        4)
            echo ""
            echo "  当前认证模式: ${AUTH_MODE}"
            echo ""
            echo "  1) 设置密码认证"
            echo "  2) 设置 Token 认证"
            echo "  3) 关闭认证 (删除密码和 Token)"
            echo "  0) 返回"
            echo ""
            read -p "  选择 [0-3]: " authmode

            case $authmode in
                1)
                    read -p "  请设置登录密码: " pw
                    if [ -z "$pw" ]; then
                        echo "  ⚠️  密码为空，已取消"
                    else
                        "$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
auth = cfg.setdefault('gateway', {}).setdefault('auth', {})
auth['mode'] = 'password'
auth['password'] = '$pw'
auth.pop('token', None)
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ 密码认证已启用')
print('  ⚠️  需要重启 Gateway 生效')
"
                    fi
                    ;;
                2)
                    read -p "  请设置 Token: " tk
                    if [ -z "$tk" ]; then
                        echo "  ⚠️  Token 为空，已取消"
                    else
                        "$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
auth = cfg.setdefault('gateway', {}).setdefault('auth', {})
auth['mode'] = 'token'
auth['token'] = '$tk'
auth.pop('password', None)
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ Token 认证已启用')
print('  ⚠️  需要重启 Gateway 生效')
"
                    fi
                    ;;
                3)
                    "$PYTHON" -c "
import json, pathlib, os
p = pathlib.Path(os.environ.get('WHALECLAW_HOME', os.path.expanduser('~/.whaleclaw'))) / 'whaleclaw.json'
cfg = json.loads(p.read_text())
auth = cfg.setdefault('gateway', {}).setdefault('auth', {})
auth['mode'] = 'none'
auth.pop('password', None)
auth.pop('token', None)
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('  ✅ 已关闭认证 (密码和 Token 已清除)')
print('  ⚠️  需要重启 Gateway 生效')
"
                    ;;
                0) ;;
                *) echo "  ❌ 无效选择" ;;
            esac
            echo ""
            read -p "  按回车键继续..."
            ;;
        5)
            configure_summarizer
            echo ""
            read -p "  按回车键继续..."
            ;;
        6)
            configure_feishu
            echo ""
            read -p "  按回车键继续..."
            ;;
        7)
            open -t "$CONFIG_FILE"
            echo "  📝 已用系统编辑器打开配置文件"
            echo ""
            read -p "  按回车键继续..."
            ;;
        8)
            echo ""
            "$PYTHON" -c "
import asyncio
from whaleclaw.doctor.runner import Doctor

async def main():
    d = Doctor()
    results = await d.run_all()
    print(d.format_report(results))

asyncio.run(main())
"
            echo ""
            read -p "  按回车键继续..."
            ;;
        9)
            echo ""
            echo "  完整配置内容:"
            echo "  ─────────────────────────────────"
            "$PYTHON" -m json.tool "$CONFIG_FILE" 2>/dev/null || cat "$CONFIG_FILE"
            echo ""
            read -p "  按回车键继续..."
            ;;
        0)
            echo "  👋 再见"
            exit 0
            ;;
        *)
            echo "  ❌ 无效选择"
            sleep 1
            ;;
    esac
done
