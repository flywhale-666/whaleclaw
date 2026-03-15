"""Skill routing by keyword matching."""

from __future__ import annotations

import re

from whaleclaw.skills.parser import Skill


class SkillRouter:
    """Route user messages to skills by keyword matching."""

    def route(
        self,
        user_message: str,
        available_skills: list[Skill],
    ) -> list[Skill]:
        """Select all skills that match the user message by keyword."""
        msg = user_message.strip()
        lower = msg.lower()
        if msg.startswith("/use "):
            skill_id = msg[5:].strip().lower()
            for s in available_skills:
                if s.id.lower() == skill_id:
                    return [s]

        if any(marker in lower for marker in ("技能", "skill")):
            explicit: list[Skill] = []
            for s in available_skills:
                if self._mentions_skill(msg, s):
                    explicit.append(s)
            if explicit:
                return explicit

        return [s for s in available_skills if self._has_trigger_hit(lower, s)]

    @staticmethod
    def _has_trigger_hit(lower_message: str, skill: Skill) -> bool:
        """Return True if any trigger keyword appears in the message."""
        return any(t.lower() in lower_message for t in skill.triggers)

    @staticmethod
    def _norm_text(text: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())

    def _mentions_skill(self, message: str, skill: Skill) -> bool:
        lower = message.lower()
        msg_norm = self._norm_text(message)
        for raw in (skill.id, skill.name):
            token = raw.strip().lower()
            if not token:
                continue
            if token in lower:
                return True
            norm = self._norm_text(token)
            if len(norm) >= 5 and norm in msg_norm:
                return True
        return False
