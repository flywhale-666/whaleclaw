"""Tests for SkillRouter."""

from __future__ import annotations

from pathlib import Path

from whaleclaw.skills.parser import Skill
from whaleclaw.skills.router import SkillRouter


def _make_skill(
    skill_id: str,
    triggers: list[str],
) -> Skill:
    return Skill(
        id=skill_id,
        name=skill_id,
        triggers=triggers,
        trigger_description="",
        instructions="use tool",
        tools=["browser"],
        examples=[],
        max_tokens=800,
        source_path=Path("/fake"),
    )


def test_keyword_matching_returns_correct_skills() -> None:
    router = SkillRouter()
    browser = _make_skill("browser-control", ["浏览器", "截图", "打开网页"])
    sandbox = _make_skill("code-sandbox", ["运行代码", "Python", "计算"])
    skills = [browser, sandbox]

    out = router.route("帮我打开网页并截图", skills)
    assert len(out) >= 1
    assert any(s.id == "browser-control" for s in out)


def test_use_command_activates_skill() -> None:
    router = SkillRouter()
    browser = _make_skill("browser-control", ["浏览器"])
    sandbox = _make_skill("code-sandbox", ["运行代码"])
    skills = [browser, sandbox]

    out = router.route("/use browser-control", skills)
    assert len(out) == 1
    assert out[0].id == "browser-control"


def test_no_match_returns_empty() -> None:
    router = SkillRouter()
    browser = _make_skill("browser-control", ["浏览器", "截图"])
    skills = [browser]

    out = router.route("今天天气怎么样", skills)
    assert out == []


def test_explicit_skill_id_mention_activates_skill() -> None:
    router = SkillRouter()
    ppt = _make_skill("ppt-generator", ["生成PPT", "演示文稿"])
    browser = _make_skill("browser-control", ["浏览器", "截图"])
    skills = [browser, ppt]

    out = router.route("请用 ppt-generator 这个技能生成一份PPT", skills)
    assert len(out) >= 1
    assert out[0].id == "ppt-generator"


def test_explicit_skill_name_mention_activates_skill() -> None:
    router = SkillRouter()
    ppt = _make_skill("ppt-generator", ["生成PPT", "演示文稿"])
    ppt.name = "PPT Generator"
    browser = _make_skill("browser-control", ["浏览器", "截图"])
    skills = [browser, ppt]

    out = router.route("我想用 PPT Generator 这个 skill 来做汇报", skills)
    assert len(out) >= 1
    assert out[0].id == "ppt-generator"
