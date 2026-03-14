"""Context window management with fixed-token hierarchical compression.

Policy (user-defined):
- Keep the latest 5 conversation groups unchanged.
- Groups ranked from 6th to 12th latest use L1 compression.
- Groups ranked from 13th to 25th latest use L0 compression.
- Target total input budget is ~1600 tokens (best effort).
- If still over budget after physical truncation, compact dropped range into
  a single summary message and feed it back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whaleclaw.providers.base import Message

if TYPE_CHECKING:
    from whaleclaw.sessions.store import SummaryRow

MODEL_MAX_CONTEXT: dict[str, int] = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.2": 1_000_000,
    "gpt-5.2-codex": 1_000_000,
    "gpt-5.3-codex": 1_000_000,
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
    "qwen3.5-plus": 1_000_000,
    "qwen3-max": 262_144,
    "qwen-max": 32_000,
    "qwen-plus": 131_072,
    "qwen-turbo": 131_072,
    "glm-5": 200_000,
    "glm-4.7": 200_000,
    "glm-4.7-flash": 200_000,
    "MiniMax-M2.5": 1_000_000,
    "MiniMax-M2.1": 1_000_000,
    "kimi-k2.5": 256_000,
    "kimi-k2-thinking": 128_000,
    "gemini-3.1-pro-preview": 1_000_000,
    "gemini-3-pro-preview": 1_000_000,
    "gemini-3-flash-preview": 1_000_000,
    "meta/llama-3.1-8b-instruct": 128_000,
}

_DEFAULT_CONTEXT = 128_000
TARGET_CONTENT_TOKENS = 1600
RECENT_PROTECTED = 5
_L1_WINDOW = 7
_MIN_CONTENT_BUDGET = 200


def estimate_tokens(text: str) -> int:
    """Quick token estimate: ~1.5 chars/token CJK, ~4 chars/token Latin."""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = len(text) - cjk
    return max(1, int(cjk / 1.5 + latin / 4))


_estimate_tokens = estimate_tokens


def _total_tokens(msgs: list[Message]) -> int:
    return sum(_estimate_tokens(m.content) for m in msgs)


def _clip_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    char_cap = max_tokens * 3
    if len(text) <= char_cap:
        return text
    return text[:char_cap].rstrip() + " ..."


def _compress_l1(msg: Message) -> Message:
    text = msg.content.strip()
    if _estimate_tokens(text) <= 120:
        return msg

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    kept: list[str] = []
    for ln in lines:
        if ln.startswith("/") or "路径" in ln or "文件" in ln or "成功" in ln or "失败" in ln:
            kept.append(ln[:160])
        elif len(kept) < 4:
            kept.append(ln[:180])
        if len(kept) >= 6:
            break

    if not kept:
        kept = [text[:240]]

    body = "\n".join(kept)
    return Message(role=msg.role, content=f"[L1压缩] {body} ...")


def _compress_l0(msg: Message) -> Message:
    text = msg.content.strip()
    if _estimate_tokens(text) <= 80:
        return msg

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    first = lines[0][:140] if lines else text[:140]
    hints: list[str] = []
    for ln in lines[1:]:
        if ln.startswith("/") or "路径" in ln or "文件" in ln:
            hints.append(ln[:120])
        if len(hints) >= 2:
            break

    body = first if not hints else first + " | " + " ; ".join(hints)
    return Message(role=msg.role, content=f"[L0压缩] {body} ...")


def _build_summary_message(summaries: list[SummaryRow], budget: int) -> str:
    if budget <= 20 or not summaries:
        return ""

    l0_parts: list[str] = []
    l1_parts: list[str] = []
    for s in summaries:
        if s.level == "L0" and s.content.strip():
            l0_parts.append(s.content.strip())
        elif s.level == "L1" and s.content.strip():
            l1_parts.append(s.content.strip())

    preferred = "\n".join(l1_parts) if l1_parts else "\n".join(l0_parts)
    if not preferred:
        return ""
    return _clip_tokens(preferred, budget)


def _build_compacted_range_message(
    dropped: list[Message],
    *,
    budget: int,
    summaries: list[SummaryRow],
) -> Message | None:
    if budget <= 40 or not dropped:
        return None

    prefix = _build_summary_message(summaries, max(0, budget // 2))
    parts: list[str] = []
    if prefix:
        parts.append(prefix)

    lines: list[str] = []
    for m in dropped[-20:]:
        role = "用户" if m.role == "user" else "助手" if m.role == "assistant" else m.role
        text = m.content.split("\n", 1)[0].strip()
        if not text:
            continue
        lines.append(f"- [{role}] {text[:100]}")
    if lines:
        parts.append("\n".join(lines))

    if not parts:
        return None

    body = _clip_tokens("\n".join(parts), budget)
    if not body:
        return None
    return Message(role="system", content=f"[以下是本会话的历史摘要，供你了解上下文背景]\n{body}")


def _keep_recent_with_budget(  # pyright: ignore[reportUnusedFunction]
    recent: list[Message], budget: int
) -> tuple[list[Message], list[Message]]:
    kept_rev: list[Message] = []
    used = 0
    for msg in reversed(recent):
        cost = _estimate_tokens(msg.content)
        if used + cost > budget:
            break
        kept_rev.append(msg)
        used += cost
    kept = list(reversed(kept_rev))
    dropped = recent[: len(recent) - len(kept)]
    return kept, dropped


def _group_by_turn(messages: list[Message]) -> list[list[Message]]:
    """Group non-system messages by user turns.

    A new group starts when a ``user`` message appears; following assistant/tool
    messages belong to that same group until next user message.
    """
    groups: list[list[Message]] = []
    current: list[Message] = []
    for msg in messages:
        if msg.role == "user":
            if current:
                groups.append(current)
            current = [msg]
            continue
        if not current:
            current = [msg]
        else:
            current.append(msg)
    if current:
        groups.append(current)
    return groups


def _flatten_groups(groups: list[list[Message]]) -> list[Message]:
    out: list[Message] = []
    for g in groups:
        out.extend(g)
    return out


def _compress_group(group: list[Message], *, level: str) -> list[Message]:
    if level == "L1":
        return [_compress_l1(m) for m in group]
    return [_compress_l0(m) for m in group]


def _keep_recent_groups_with_budget(
    recent_groups: list[list[Message]],
    budget: int,
) -> tuple[list[list[Message]], list[list[Message]]]:
    kept_rev: list[list[Message]] = []
    used = 0
    for group in reversed(recent_groups):
        cost = _total_tokens(group)
        if used + cost > budget:
            break
        kept_rev.append(group)
        used += cost
    kept = list(reversed(kept_rev))
    dropped = recent_groups[: len(recent_groups) - len(kept)]
    return kept, dropped


class ContextWindow:
    """Fixed-budget hierarchical context trimmer."""

    @staticmethod
    def get_max_context(model: str) -> int:
        return MODEL_MAX_CONTEXT.get(model, _DEFAULT_CONTEXT)

    def trim(self, messages: list[Message], model: str) -> list[Message]:
        del model  # fixed policy does not depend on model max context
        return self._trim_core(messages, summaries=[])

    def trim_with_summaries(
        self,
        messages: list[Message],
        model: str,
        summaries: list[SummaryRow],
    ) -> list[Message]:
        del model
        return self._trim_core(messages, summaries=summaries)

    def _trim_core(
        self,
        messages: list[Message],
        *,
        summaries: list[SummaryRow],
    ) -> list[Message]:
        system: list[Message] = []
        non_system: list[Message] = []
        for m in messages:
            (system if m.role == "system" else non_system).append(m)

        if not non_system:
            return system

        # User requirement: only cap message content (user/assistant/tool),
        # not system prompt and not tool schema payload in API body.
        non_budget = max(_MIN_CONTENT_BUDGET, TARGET_CONTENT_TOKENS)

        if _total_tokens(non_system) <= non_budget:
            return [*system, *non_system]

        groups = _group_by_turn(non_system)
        recent_n = min(RECENT_PROTECTED, len(groups))
        recent_groups = list(groups[-recent_n:])

        middle_end = max(0, len(groups) - recent_n)
        middle_start = max(0, middle_end - _L1_WINDOW)
        middle_groups = list(groups[middle_start:middle_end])
        old_groups = list(groups[:middle_start])

        old_c_groups = [_compress_group(g, level="L0") for g in old_groups]
        middle_c_groups = [_compress_group(g, level="L1") for g in middle_groups]
        core_groups = [*old_c_groups, *middle_c_groups]

        recent_kept_groups, recent_dropped_groups = _keep_recent_groups_with_budget(
            recent_groups, non_budget
        )
        recent_kept = _flatten_groups(recent_kept_groups)
        used = _total_tokens(recent_kept)

        core_kept_rev: list[list[Message]] = []
        for group in reversed(core_groups):
            cost = _total_tokens(group)
            if used + cost > non_budget:
                break
            core_kept_rev.append(group)
            used += cost
        core_kept_groups = list(reversed(core_kept_rev))
        core_kept = _flatten_groups(core_kept_groups)

        dropped_group_count = len(core_groups) - len(core_kept_groups)
        dropped_groups = [*core_groups[:dropped_group_count], *recent_dropped_groups]
        dropped = _flatten_groups(dropped_groups)

        trimmed = [*core_kept, *recent_kept]

        headroom = non_budget - _total_tokens(trimmed)
        compact_budget = min(220, max(0, headroom))
        if dropped and compact_budget < 80:
            target = 120
            while compact_budget < target and len(trimmed) > len(recent_kept):
                removed = trimmed.pop(0)
                compact_budget += _estimate_tokens(removed.content)
            compact_budget = min(220, compact_budget)

        compacted = _build_compacted_range_message(
            dropped,
            budget=compact_budget,
            summaries=summaries,
        )
        has_compacted = compacted is not None
        if compacted is not None:
            trimmed = [compacted, *trimmed]

        while _total_tokens(trimmed) > non_budget and len(trimmed) > len(recent_kept):
            if has_compacted and len(trimmed) > len(recent_kept) + 1:
                trimmed.pop(1)
                continue
            trimmed.pop(0)

        return [*system, *trimmed]
