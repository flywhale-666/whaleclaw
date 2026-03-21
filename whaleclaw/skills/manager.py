"""Skill discovery, loading, routing and prompt formatting."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from whaleclaw.config.paths import WORKSPACE_DIR
from whaleclaw.skills.hooks import DefaultSkillHooks, SkillHooks, register_hooks
from whaleclaw.skills.parser import Skill, SkillParser
from whaleclaw.skills.router import SkillRouter
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

_BUNDLED_DIR = Path(__file__).resolve().parent / "bundled"
_USER_SKILLS_DIR = WORKSPACE_DIR / "skills"
_DEFAULT_SKILLS_DIRS = [_BUNDLED_DIR, _USER_SKILLS_DIR]

_DYNAMIC_BUDGET = 6000


def _load_hooks(skill_dir: Path, skill_id: str) -> SkillHooks | None:
    """尝试从技能目录加载 hooks.py，优先技能自身目录，再查 bundled。"""
    candidates = [
        skill_dir / "hooks.py",
        _BUNDLED_DIR / skill_id / "hooks.py",
    ]
    for hooks_path in candidates:
        if not hooks_path.is_file():
            continue
        try:
            mod_name = f"whaleclaw.skills.hooks.{skill_id}"
            sys.modules.pop(mod_name, None)
            spec = importlib.util.spec_from_file_location(
                mod_name,
                hooks_path,
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            hooks_cls: Any = getattr(module, "Hooks", None)
            if hooks_cls is not None:
                instance = hooks_cls()
                if isinstance(instance, (SkillHooks, DefaultSkillHooks)):
                    register_hooks(instance)
                    return instance
        except Exception as exc:
            log.warning(
                "skill.hooks_load_failed",
                skill_id=skill_id,
                path=str(hooks_path),
                error=str(exc),
            )
    return None




class SkillManager:
    """Discover, route and format skills for prompt injection."""

    def __init__(
        self,
        skills_dirs: list[Path] | None = None,
    ) -> None:
        self._skills_dirs = skills_dirs or _DEFAULT_SKILLS_DIRS
        self._parser = SkillParser()
        self._router = SkillRouter()

    def discover(self) -> list[Skill]:
        """Scan skill dirs for SKILL.md files and parse them."""
        skills: list[Skill] = []
        for d in self._skills_dirs:
            if not d.exists():
                continue
            for path in d.rglob("SKILL.md"):
                with contextlib.suppress(Exception):
                    skill = self._parser.parse(path)
                    hooks = _load_hooks(path.parent, skill.id)
                    if hooks is not None:
                        skill.hooks = hooks
                    skills.append(skill)
        return skills

    def get_routed_skills(
        self,
        user_message: str,
        forced_skill_id: str | None = None,
        forced_skill_ids: list[str] | None = None,
    ) -> list[Skill]:
        """Route user message to matched skills."""
        available = self.discover()
        if forced_skill_ids:
            id_to_skill = {s.id: s for s in available}
            forced: list[Skill] = []
            for sid in forced_skill_ids:
                skill = id_to_skill.get(sid)
                if skill is not None:
                    forced.append(skill)
            if forced:
                return forced
        if forced_skill_id:
            forced_skill = next((s for s in available if s.id == forced_skill_id), None)
            if forced_skill is not None:
                return [forced_skill]
        return self._router.route(user_message, available)

    def format_for_prompt(
        self,
        skills: list[Skill],
        total_budget: int = _DYNAMIC_BUDGET,
    ) -> str:
        """Format skills for prompt injection with budget control.

        Args:
            skills: 本轮匹配到的技能列表。
            total_budget: 所有技能共享的 token 总预算（默认 _DYNAMIC_BUDGET）。

        预算分配：total_budget 平均分给 N 个技能，每个技能实际注入量 =
        min(分到的预算, 该技能 max_tokens)。超出部分硬截断（按字符，1 token ≈ 4 字符）。
        """
        if not skills:
            return ""

        n = len(skills)
        per_skill_budget = total_budget // n

        parts: list[str] = []
        for s in skills:
            block = f"## 技能: {s.name}\n\n{s.instructions}"
            if s.examples:
                block += "\n\n### 示例\n" + "\n".join(s.examples)
            cap_tokens = min(per_skill_budget, s.max_tokens)
            char_cap = cap_tokens * 4
            if len(block) > char_cap:
                block = block[:char_cap]
                log.info(
                    "skill.prompt_truncated",
                    skill_id=s.id,
                    original_chars=len(f"## 技能: {s.name}\n\n{s.instructions}"),
                    cap_tokens=cap_tokens,
                    cap_chars=char_cap,
                )
            parts.append(block)
        return "\n\n---\n\n".join(parts)

    def list_installed(self) -> list[Skill]:
        """List all user-installed skills (excluding bundled)."""
        skills: list[Skill] = []
        if _USER_SKILLS_DIR.exists():
            for path in _USER_SKILLS_DIR.rglob("SKILL.md"):
                with contextlib.suppress(Exception):
                    skills.append(self._parser.parse(path))
        return skills

    def install(self, source: str) -> Skill:
        """Install skill from GitHub URL, local path, or directory name.

        Supported sources:
        - GitHub: ``user/repo``, ``user/repo/path/to/skill``,
          ``https://github.com/user/repo/tree/main/skill``
        - Local path: ``/absolute/path/to/skill_dir``
        """
        _USER_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

        src_path = Path(source)
        if src_path.is_dir() and (src_path / "SKILL.md").is_file():
            return self._install_from_local(src_path)

        if source.startswith("http://") or source.startswith("https://"):
            return self._install_from_github_url(source)

        if "/" in source:
            return self._install_from_github_shorthand(source)

        raise ValueError(
            f"无法识别的技能来源: {source}\n支持: GitHub (user/repo/path), URL, 或本地目录路径"
        )

    def _install_from_local(self, src: Path) -> Skill:
        """Copy a local skill directory into user skills."""
        skill_id = src.name
        dest = _USER_SKILLS_DIR / skill_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        log.info("skill.installed", skill_id=skill_id, source=str(src))
        return self._parser.parse(dest / "SKILL.md")

    def _install_from_github_url(self, url: str) -> Skill:
        """Clone repo and extract skill from GitHub URL."""
        url = url.rstrip("/")
        parts = url.replace("https://github.com/", "").replace("http://github.com/", "")
        segments = parts.split("/")
        if len(segments) < 2:
            raise ValueError(f"无效的 GitHub URL: {url}")

        user, repo = segments[0], segments[1]
        sub_path = ""
        if "tree" in segments:
            tree_idx = segments.index("tree")
            if tree_idx + 2 < len(segments):
                sub_path = "/".join(segments[tree_idx + 2 :])
        elif len(segments) > 2:
            sub_path = "/".join(segments[2:])

        return self._clone_and_install(user, repo, sub_path)

    def _install_from_github_shorthand(self, shorthand: str) -> Skill:
        """Install from ``user/repo`` or ``user/repo/sub/path``."""
        parts = shorthand.split("/")
        if len(parts) < 2:
            raise ValueError(f"格式错误: {shorthand} (需要 user/repo)")
        user, repo = parts[0], parts[1]
        sub_path = "/".join(parts[2:]) if len(parts) > 2 else ""
        return self._clone_and_install(user, repo, sub_path)

    @staticmethod
    def _git_env() -> dict[str, str]:
        """Build env dict with proxy for git subprocess."""
        env = os.environ.copy()
        if env.get("https_proxy") or env.get("HTTPS_PROXY"):
            return env
        for port in (7897, 7890, 1087, 8080):
            with contextlib.suppress(OSError):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.3)
                s.connect(("127.0.0.1", port))
                s.close()
                proxy = f"http://127.0.0.1:{port}"
                env["https_proxy"] = proxy
                env["http_proxy"] = proxy
                log.info("git.proxy_detected", proxy=proxy)
                return env
        return env

    def _clone_and_install(self, user: str, repo: str, sub_path: str) -> Skill:
        """Clone GitHub repo to temp dir, copy skill into user skills."""
        clone_url = f"https://github.com/{user}/{repo}.git"

        with tempfile.TemporaryDirectory(prefix="whaleclaw-skill-") as tmp:
            tmp_dir = Path(tmp)
            result = subprocess.run(
                ["git", "clone", "--depth=1", clone_url, str(tmp_dir / "repo")],
                capture_output=True,
                text=True,
                timeout=60,
                env=self._git_env(),
            )
            if result.returncode != 0:
                raise RuntimeError(f"git clone 失败: {result.stderr.strip()}")

            src = tmp_dir / "repo"
            if sub_path:
                src = src / sub_path

            skill_md = src / "SKILL.md"
            if not skill_md.is_file():
                candidates = list(src.rglob("SKILL.md"))
                if candidates:
                    src = candidates[0].parent
                else:
                    raise FileNotFoundError(f"在 {user}/{repo}/{sub_path} 中未找到 SKILL.md")

            return self._install_from_local(src)

    def uninstall(self, skill_id: str) -> bool:
        """Remove an installed skill by ID. Returns True if removed."""
        target = _USER_SKILLS_DIR / skill_id
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
            log.info("skill.uninstalled", skill_id=skill_id)
            return True
        return False
