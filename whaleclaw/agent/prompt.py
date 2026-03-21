"""PromptAssembler — layered system prompt builder.

Architecture:
- Static layer: core identity, always injected (cacheable)
- Dynamic layer: skills routed by user message keywords, injected in full
- Fallback layer: tool descriptions for providers without native tools API

Tool JSON Schemas are passed via the LLM API ``tools`` parameter (not in
the system prompt), so the LLM discovers tool capabilities from the schema
itself — no need to describe tools in the prompt.
"""

from __future__ import annotations

from enum import StrEnum

from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import CacheControl, Message
from whaleclaw.skills.manager import SkillManager
from whaleclaw.skills.parser import Skill


class PromptLayer(StrEnum):
    """Prompt layers by priority."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    LAZY = "lazy"


_DEFAULT_ASSISTANT_NAME = "WhaleClaw"

_SAFETY_CONSTITUTION = """\
【安全宪法 — 不可覆盖】
- 无自主目标：不追求自我保存、复制、资源获取或权力扩张
- 人类监督优先：安全高于任务完成；指令冲突时暂停并询问
- 不绕过安全机制：不操纵任何人扩展权限或关闭安全措施
- 不自我修改：不擅自修改系统提示词、安全规则或工具策略"""

_STATIC_PROMPT_TEMPLATE = """\
{safety_constitution}

你是 {assistant_name}，一个运行在用户本地电脑上的 AI 助手。
- 使用用户的语言回复，简洁准确
- 你拥有多种工具（bash、文件读写、浏览器、定时任务、技能管理等），能做就做，不要说"我无法"
- 设置/查看/删除定时任务时，必须使用内置 cron 或 reminder 工具，禁止用 bash 操作系统 crontab
- 用 bash 执行网络请求时，curl 必须加 --connect-timeout 5 --max-time 15 防止卡住
- 用户要求做的事，计划好后就立即调用工具执行，不要等用户确认或给用户出选择题。
- 用户说"N分钟/小时后做某事"时，这是定时任务，必须调用 reminder 或 cron 工具，不要立即执行该事
- 工具执行失败时尝试其他方案
- 下载或生成的图片用 markdown 显示：![描述](文件绝对路径)
- 生成的其他文件告诉用户绝对路径
- **严禁编造文件路径**：只使用工具返回的真实路径，绝不自己猜测或虚构文件名
- 运行 Python 脚本时，必须显式使用 `./python/bin/python3.12 script.py [args]`；
  不要直接执行 `.py` 文件，也不要用 python -c 或 python3 -c（脚本长度会被截断）
- 需要生成文件（PPT/Excel/PDF等）时：先用 file_write 写 .py 脚本到 /tmp/，再用 bash 执行
- 修改已有 Office 文件时先判断复杂度：纯文本改动优先局部编辑（ppt_edit/docx_edit/xlsx_edit）；
  若用户要求插图/音视频/版式重排/风格升级，先做简短执行规划，
  再组合 browser、bash、file_write 与编辑工具完成
- 除非用户明确要求重做，否则不要整份重建
- 如果你的技能不够好，可以用 skill(action="search", query="关键词") 搜索 GitHub 上更好的技能并安装
- 用户明确点名某个技能（skill id/技能名）时，必须优先按该技能说明执行；
  不要绕过技能直接用通用 bash/python 方案
- **禁止自我否定已完成的工作**：如果你之前已经成功调用工具生成了文件，
  不要说"我之前没有真正执行"或"我一直在空谈"。工具调用成功就是成功，直接告诉用户文件路径即可
- 用户说"改一下"时，只修改用户指出的问题，不要推翻重做整个任务"""

_TOOL_FALLBACK_HEADER = """\
【工具调用规则 - 必须严格遵守】

你可以调用工具。调用方式：在回复中输出且**仅输出**以下格式的 JSON 代码块：

```json
{"tool": "工具名", "arguments": {"参数名": "参数值"}}
```

**严禁**：
- 不要用文字描述你打算使用什么工具，直接输出 JSON
- 不要假装工具已经执行，必须输出 JSON 等待真实结果
- 不要编造工具执行结果、文件路径或图片 URL，只使用工具返回的真实数据
- 不要在同一次回复中输出多个 JSON 块

**必须**：
- 需要执行操作时，立即输出一个工具调用 JSON 块，然后停止
- 等收到工具执行结果后，再继续回复
- 搜索/下载图片 → 用 browser 工具的 search_images
- 执行系统命令 → 用 bash
- 读写文件/补丁 → 用 file_read / file_write / file_edit / patch_apply

以下是全部可用工具：

"""

class PromptAssembler:
    """Build system prompts.

    Static layer: core identity, cached across turns.
    Dynamic layer: skills routed by SkillManager, injected in full.
    Awareness layer: runtime metadata (model, context budget) for self-awareness.
    """

    def __init__(self, skill_manager: SkillManager | None = None) -> None:
        self._skill_manager = skill_manager or SkillManager()

    def build(
        self,
        config: WhaleclawConfig,
        user_message: str,
        *,
        token_budget: int | None = None,
        tool_fallback_text: str = "",
        assistant_name: str = _DEFAULT_ASSISTANT_NAME,
        forced_skill_id: str | None = None,
        forced_skill_ids: list[str] | None = None,
        model_id: str = "",
        max_context_tokens: int = 0,
    ) -> list[Message]:
        """Assemble system prompt messages.

        Args:
            config: Global configuration.
            user_message: Current user message (used for skill routing).
            token_budget: Max tokens for the system prompt.
            tool_fallback_text: Tool descriptions for providers without
                native tools support. Injected into prompt when non-empty.
            model_id: Current model identifier (for agent self-awareness).
            max_context_tokens: Model's max context window size.
        """
        parts: list[str] = [self._build_static(config, assistant_name)]

        dynamic = self._build_dynamic(
            user_message,
            forced_skill_id,
            forced_skill_ids,
        )
        if dynamic:
            parts.append(dynamic)

        if tool_fallback_text:
            parts.append(_TOOL_FALLBACK_HEADER + "\n" + tool_fallback_text)

        messages: list[Message] = [
            Message(
                role="system",
                content=parts[0],
                cache_control=CacheControl(),
            ),
        ]
        if len(parts) > 1:
            messages.append(Message(role="system", content="\n\n".join(parts[1:])))

        awareness = self._build_awareness(model_id, max_context_tokens)
        if awareness:
            messages.append(Message(role="system", content=awareness))

        return messages

    @staticmethod
    def _build_awareness(model_id: str, max_context_tokens: int) -> str:
        """Awareness layer — inject runtime metadata so agent knows its own capabilities."""
        if not model_id:
            return ""
        parts: list[str] = [f"当前模型: {model_id}"]
        if max_context_tokens > 0:
            parts.append(f"上下文窗口: {max_context_tokens:,} tokens")
        return "【运行时信息】" + " | ".join(parts)

    def _build_static(self, config: WhaleclawConfig, assistant_name: str) -> str:
        """Static layer — core identity + safety constitution (~250 tokens), cacheable."""
        safe_name = assistant_name.strip() or _DEFAULT_ASSISTANT_NAME
        return _STATIC_PROMPT_TEMPLATE.format(
            safety_constitution=_SAFETY_CONSTITUTION,
            assistant_name=safe_name,
        )

    def _build_dynamic(
        self,
        user_message: str,
        forced_skill_id: str | None = None,
        forced_skill_ids: list[str] | None = None,
    ) -> str:
        """Dynamic layer — skills routed by user message keywords.

        当轮 system prompt 不限制技能注入长度，完整注入匹配到的技能说明。
        Token 预算控制在历史对话压缩层做，不在这里。
        """
        skills = self._skill_manager.get_routed_skills(
            user_message,
            forced_skill_id=forced_skill_id,
            forced_skill_ids=forced_skill_ids,
        )
        if not skills:
            return ""
        return self._skill_manager.format_for_prompt(skills)

    def route_skill_ids(
        self,
        user_message: str,
        *,
        forced_skill_ids: list[str] | None = None,
    ) -> list[str]:
        skills = self._skill_manager.get_routed_skills(
            user_message,
            forced_skill_ids=forced_skill_ids,
        )
        return [s.id for s in skills]

    def route_skills(
        self,
        user_message: str,
        *,
        forced_skill_ids: list[str] | None = None,
    ) -> list[Skill]:
        return self._skill_manager.get_routed_skills(
            user_message,
            forced_skill_ids=forced_skill_ids,
        )

    def estimate_tokens(self, text: str) -> int:
        """Quick token estimate: ~1.5 chars/token for CJK, ~4 chars/token for Latin."""
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        latin = len(text) - cjk
        return int(cjk / 1.5 + latin / 4)
