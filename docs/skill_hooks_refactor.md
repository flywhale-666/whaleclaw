# SkillHooks 重构总结

## 目标

将 `single_agent.py`、`skill_lock.py`、`bot.py`、`tool_guards.py`、`tool_execution.py`、`bash.py` 中所有 `if skill_id == "nano-banana-image-t8"` / `if skill_id == "xiaohongshu_publish"` 的硬编码分支，抽取为 **SkillHooks 协议**，实现技能逻辑即插即用。

## 架构设计

```
whaleclaw/skills/hooks.py          ← SkillHooks Protocol + DefaultSkillHooks 基类 + get_skill_hooks()
whaleclaw/skills/bundled/
  nano-banana-image-t8/hooks.py    ← Nano Banana 的全部技能专属逻辑 (~1050 行)
  xiaohongshu_publish/hooks.py     ← 小红书发布的全部技能专属逻辑 (~170 行)
```

**核心思路**：每个技能目录下放一个 `hooks.py`，导出 `class Hooks(DefaultSkillHooks)`，只覆盖自己关心的方法。`SkillManager.discover()` 自动加载并挂到 `Skill.hooks` 字段上。业务代码通过 `get_skill_hooks(skill)` 获取 hooks 实例，用通用循环分发调用。

## SkillHooks 协议方法清单

| 方法 | 用途 | nano-banana | xhs |
|---|---|---|---|
| `build_param_guard_reply(state)` | 自定义参数守卫展示文案 | ✅ | ✅ |
| `missing_required(state, control_message_only)` | 判断是否缺必填参数 | ✅ | ✅ |
| `update_guard_state(state, message, images, ...)` | 自定义 guard 状态更新 | ✅ | ✅ |
| `is_execution_request(message, ...)` | 判断是否为执行请求 | ✅ | - |
| `is_control_message(message)` | 判断是否为纯控制消息 | ✅ | - |
| `is_activation_message(message)` | 判断是否为激活消息 | ✅ | - |
| `build_command(state, session)` | 构建 bash 执行命令 | ✅ | - |
| `build_execution_system_message(session)` | 执行阶段约束 system message | ✅ | - |
| `build_command_template_system_message(cmd)` | 命令模板 system message | ✅ | - |
| `postprocess_reply(text, session)` | 回复后处理 | ✅ | - |
| `build_lock_status_extra(session)` | 锁定状态附加信息 | ✅ | - |
| `build_already_locked_reply(session)` | 已锁定时的回复 | ✅ | - |
| `image_buffer_enabled` (property) | 是否启用图片缓冲 | ✅ | ✅ |
| `image_buffer_hint(labels)` | 图片缓冲提示文案 | ✅ | ✅ |
| `on_tool_failure(tc, result)` | 工具失败自定义守卫 | ✅ | - |
| `repair_tool_call(command)` | 修复工具调用参数 | ✅ | - |
| `on_bash_success(tc, result, session)` | bash 成功后状态更新 | ✅ | - |
| `handle_control_message(message, state, session)` | 控制消息直接返回 | ✅ | - |
| `extra_tool_names()` | 额外工具名 | ✅ | - |
| `stage_rules` (property) | 阶段性 system message 规则 | ✅ | ✅ |
| `long_running_script_pattern` (property) | 长时脚本正则 | ✅ | - |
| `long_running_timeout_seconds` (property) | 长时脚本超时 | ✅ | - |
| `parallel_limit` / `batch_delay_seconds` | 批量并行参数 | ✅ | - |

## 变更文件清单

### 新增文件（8 个）

| 文件 | 行数 | 说明 |
|---|---|---|
| `whaleclaw/skills/hooks.py` | 318 | SkillHooks Protocol + DefaultSkillHooks + get_skill_hooks() |
| `whaleclaw/skills/bundled/nano-banana-image-t8/hooks.py` | 1049 | NB 全部专属逻辑（从 single_agent/skill_lock 迁入） |
| `whaleclaw/skills/bundled/xiaohongshu_publish/hooks.py` | 168 | XHS 全部专属逻辑 |
| `whaleclaw/skills/bundled/xiaohongshu_publish/SKILL.md` | 285 | XHS 技能描述文件 |
| `tests/test_agent/loop_helpers.py` | - | 测试辅助函数提取 |
| `tests/test_agent/test_loop_guards.py` | - | tool_guards 测试拆分 |
| `tests/test_agent/test_loop_nano_banana.py` | - | nano-banana 专项测试拆分 |
| `tests/test_agent/test_loop_skills.py` | - | 技能锁定/守卫测试拆分 |

### 修改文件（核心 6 个 + 测试/其他）

| 文件 | 改动要点 |
|---|---|
| `whaleclaw/skills/manager.py` | `discover()` 新增 `_load_hooks()` 自动加载 hooks.py |
| `whaleclaw/skills/parser.py` | `Skill` model 新增 `hooks: Any = None` 字段 |
| `whaleclaw/agent/single_agent.py` | 替换所有 `if skill_id == "xxx"` 为通用 hooks 循环分发 |
| `whaleclaw/agent/helpers/skill_lock.py` | `build_skill_param_guard_reply` 新增 `hooks` 参数分发 |
| `whaleclaw/agent/helpers/tool_guards.py` | `_apply_failure_guard` 新增 `skill_hooks` 参数 |
| `whaleclaw/agent/helpers/tool_execution.py` | `repair_tool_call` 新增 `skill_hooks` 参数 |
| `whaleclaw/tools/bash.py` | 长时脚本检测/批量改写泛化为可配置参数 |
| `whaleclaw/channels/feishu/bot.py` | `_IMAGE_BUFFER_SKILL_IDS` 改为从 hooks 动态获取 |

## single_agent.py 核心改动模式

**Before（硬编码）：**

```python
if "nano-banana-image-t8" in locked_skill_ids:
    cmd = _build_nano_banana_command(state, session)
    system_messages.append(_build_nano_banana_execution_hint(session))
```

**After（通用分发）：**

```python
for skill in guards:
    hooks = get_skill_hooks(skill)
    if hooks is not None and _hooks_execution_request:
        cmd = hooks.build_command(state_map.get(skill.id, {}), session)
        exec_msg = hooks.build_execution_system_message(session)
        if exec_msg: system_messages.append(exec_msg)
```

## 守卫路径的关键交互逻辑

1. **hooks 存在且 `update_guard_state` 返回非 None** → 完全替代通用 `_update_guard_state`
2. **hooks 存在但 `update_guard_state` 返回 None** → 回退到通用 `_update_guard_state`，再用 hooks 的 `missing_required` 覆盖判断
3. **api_key 类型参数** → 无论走哪条路径，都先用通用逻辑自动填充凭证
4. **`control_message_only` 与 `execution_request` 冲突** → 当消息同时是控制消息和执行请求时，以 `control_message_only=False` 传入
5. **`is_execution_request` 返回 `None`** → 不跳过守卫（仅 `False` 时跳过）

## 重构过程中修复的 Bug

1. **`_extract_input_image_paths_from_text` 中 `match.group(2)` 应为 `group(1)`** — 正则只有一个捕获组
2. **hooks 的 `update_guard_state` 与通用守卫逻辑的交互** — 返回 None 时需回退到通用逻辑
3. **api_key 凭证自动加载** — hooks 替代通用逻辑后需单独处理 api_key 类型参数
4. **`_hooks_skip_guard_ids` 误判** — `is_execution_request` 返回 `None` 时不应跳过守卫
5. **xhs hooks 缺少 `missing_required`** — 需要实现确认流程的 missing 判断
6. **ratio 提取遗漏** — hooks 的 `update_guard_state` 需调用 `extract_ratio_or_size`
7. **`__mode__` 和 `__input_paths__` 推断** — 在 `update_guard_state` 中预计算并存入 state
8. **激活消息不应设为 prompt** — 当消息是激活消息且无历史 prompt 时跳过 prompt 设置
9. **text 模式下图片复用** — 通过 hooks state 判断 mode 决定是否复用
10. **`control_message_only` 与 `execution_request` 冲突** — 同时为控制消息和执行请求时不阻止

## 测试结果

```
551 passed, 0 failed
```
