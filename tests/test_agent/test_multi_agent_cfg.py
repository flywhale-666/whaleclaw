"""Tests for multi-agent effective config resolution."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from whaleclaw.agent.loop import (
    _attach_rounds_marker,
    _build_multi_agent_requirement_baseline,
    _can_auto_create_parent_for_failure,
    _extract_delivery_artifact_paths,
    _extract_multi_agent_rounds,
    _extract_office_paths,
    _extract_requested_deliverables,
    _extract_round_delivery_section,
    _extract_rounds_marker,
    _looks_like_bad_coordinator_output,
    _looks_like_role_stall_output,
    _need_image_output,
    _resolve_multi_agent_cfg,
    _scenario_delivery_focus,
    _scenario_discuss_focus,
    _snapshot_round_artifacts,
    _sync_multi_agent_compression_boundary,
    _with_round_version_suffix,
    run_agent,
)
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.tools.base import ToolResult


def _make_session(metadata: dict[str, object]) -> Session:
    now = datetime.now(UTC)
    return Session(
        id="s1",
        channel="feishu",
        peer_id="u1",
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata=metadata,
    )


def test_resolve_multi_agent_cfg_uses_global_by_default() -> None:
    cfg = WhaleclawConfig()
    cfg.plugins["multi_agent"] = {
        "enabled": True,
        "mode": "parallel",
        "max_rounds": 3,
        "roles": [],
    }

    out = _resolve_multi_agent_cfg(cfg, _make_session({}))

    assert out["enabled"] is True
    assert out["mode"] == "parallel"
    assert out["max_rounds"] == 3


def test_resolve_multi_agent_cfg_session_override_wins() -> None:
    cfg = WhaleclawConfig()
    cfg.plugins["multi_agent"] = {
        "enabled": False,
        "mode": "parallel",
        "max_rounds": 2,
        "roles": [],
    }
    session = _make_session(
        {
            "multi_agent_enabled": True,
            "multi_agent_mode": "serial",
            "multi_agent_max_rounds": 8,
        }
    )

    out = _resolve_multi_agent_cfg(cfg, session)

    assert out["enabled"] is True
    assert out["mode"] == "serial"
    assert out["max_rounds"] == 8


def test_multi_agent_scenario_focuses_cover_all_modes() -> None:
    scenarios = [
        "product_design",
        "content_creation",
        "software_development",
        "data_analysis_decision",
        "scientific_research",
        "intelligent_assistant",
        "workflow_automation",
        "custom::study-meeting",
    ]
    for scenario in scenarios:
        discuss = _scenario_discuss_focus(scenario)
        delivery = _scenario_delivery_focus(scenario)
        assert discuss.strip()
        assert delivery.strip()
        assert "交付物" in discuss or "交付物" in delivery


def test_bad_coordinator_output_detection() -> None:
    bad = (
        "我会把你提供的各角色多轮输出当作输入材料。\n"
        "请把下面这些内容贴出来即可。"
    )
    assert _looks_like_bad_coordinator_output(bad) is True


def test_bad_coordinator_output_detection_rejects_user_intervention_prompt() -> None:
    bad = (
        "为了我马上开工，请你任选其一补充信息：\n"
        "你回复我后我就能继续。"
    )
    assert _looks_like_bad_coordinator_output(bad) is True


def test_role_stall_output_detection() -> None:
    assert _looks_like_role_stall_output("我将使用 cron_reminder、doc 技能继续完成任务。") is True
    assert _looks_like_role_stall_output("请你补充 3 个信息点？") is True
    ok = "1) 结论：可执行。\n2) 行动项：A/B/C。\n3) 配合点：评审。"
    assert _looks_like_role_stall_output(ok) is False


def test_need_image_output_detects_image_intent() -> None:
    assert _need_image_output("给我一版图文，并配图") is True
    assert _need_image_output("做一个封面海报文案") is True


def test_need_image_output_false_for_text_only() -> None:
    assert _need_image_output("只输出PRD文档，不要图片") is False
    assert _need_image_output("做一份接口设计说明和测试清单") is False


def test_extract_requested_deliverables_detection() -> None:
    out = _extract_requested_deliverables("要一版图文，并导出word、ppt和html")
    assert "image" in out
    assert "word" in out
    assert "ppt" in out
    assert "html" in out


def test_extract_requested_deliverables_honors_negative() -> None:
    out = _extract_requested_deliverables("给我PRD， 不要图片，不要ppt")
    assert "image" not in out
    assert "ppt" not in out


def test_build_multi_agent_requirement_baseline_contains_original_requirement() -> None:
    baseline = _build_multi_agent_requirement_baseline(
        message="做6页商务PPT并配图，基于杨超越写真主题",
        scenario="product_design",
        mode="parallel",
        max_rounds=3,
        requested_deliverables=["ppt", "image"],
    )
    assert "冻结需求基线" in baseline
    assert "做6页商务PPT并配图" in baseline
    assert "交付类型: ppt, image" in baseline


def test_extract_multi_agent_rounds_from_sentence() -> None:
    assert _extract_multi_agent_rounds("先按你建议推进，改成3轮再执行") == 3
    assert _extract_multi_agent_rounds("改为 2 轮") == 2
    assert _extract_multi_agent_rounds("这句话不改回合") is None


def test_rounds_marker_attach_and_extract() -> None:
    topic = "做一个方案并输出文档"
    marked = _attach_rounds_marker(topic, 3)
    clean, rounds = _extract_rounds_marker(marked)
    assert rounds == 3
    assert clean == topic


def test_can_auto_create_parent_for_missing_target() -> None:
    result = ToolResult(
        success=False,
        output=(
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'/Users/flywhale/output/demo.pptx'"
        ),
        error="",
    )
    assert _can_auto_create_parent_for_failure(result) == "/Users/flywhale/output/demo.pptx"


def test_extract_office_paths_from_round_delivery() -> None:
    text = (
        "第1轮已生成文件：\n"
        "/private/tmp/刘亦菲_商务介绍_5页_Round1.pptx\n"
        "/private/tmp/report.docx"
    )
    paths = _extract_office_paths(text)
    assert "/private/tmp/刘亦菲_商务介绍_5页_Round1.pptx" in paths
    assert "/private/tmp/report.docx" in paths


def test_extract_delivery_paths_only_from_final_delivery_section() -> None:
    text = (
        "执行脚本：/private/tmp/build_round2.py\n"
        "3) 本轮可直接交付结果（必须给出具体内容）\n"
        "- /private/tmp/final_round2.pptx\n"
        "- /private/tmp/cover.png\n"
        "4) 结果协调者执行记录\n"
        "- /private/tmp/another.py"
    )
    section = _extract_round_delivery_section(text)
    assert "final_round2.pptx" in section
    paths = _extract_delivery_artifact_paths(text)
    assert "/private/tmp/final_round2.pptx" in paths
    assert "/private/tmp/cover.png" in paths
    assert "/private/tmp/build_round2.py" not in paths
    assert "/private/tmp/another.py" not in paths


def test_extract_delivery_paths_supports_script_when_requested() -> None:
    text = (
        "3) 本轮可直接交付结果（必须给出具体内容）\n"
        "- /private/tmp/final_round2.py\n"
        "- /private/tmp/deploy.sh\n"
        "4) 结果协调者执行记录\n"
        "- /private/tmp/another.py"
    )
    paths = _extract_delivery_artifact_paths(text, include_scripts=True)
    assert "/private/tmp/final_round2.py" in paths
    assert "/private/tmp/deploy.sh" in paths
    assert "/private/tmp/another.py" not in paths


def test_extract_round_delivery_section_accepts_multiple_number_formats() -> None:
    text = (
        "2. 角色观点\n"
        "3、本轮可直接交付结果\n"
        "- /private/tmp/final_round2.html\n"
        "4：结果协调者执行记录\n"
        "- /private/tmp/noise.log"
    )
    section = _extract_round_delivery_section(text)
    assert "final_round2.html" in section
    assert "noise.log" not in section


def test_with_round_version_suffix_rewrites_or_appends() -> None:
    assert (
        _with_round_version_suffix("/tmp/demo.pptx", 2)
        == "/tmp/demo_V2.pptx"
    )
    assert (
        _with_round_version_suffix("/tmp/demo_V1.pptx", 3)
        == "/tmp/demo_V3.pptx"
    )


def test_snapshot_round_artifacts_creates_versioned_copy(tmp_path: Path) -> None:
    src = tmp_path / "report.pptx"
    src.write_text("v1", encoding="utf-8")
    out = _snapshot_round_artifacts([str(src)], 1)
    assert len(out) == 1
    v1 = Path(out[0])
    assert v1.name == "report_V1.pptx"
    assert v1.read_text(encoding="utf-8") == "v1"


class _StubStore:
    def __init__(self) -> None:
        self.saved_metadata: dict[str, object] | None = None

    async def update_session_field(self, session_id: str, **fields: object) -> None:  # noqa: ARG002
        md = fields.get("metadata")
        if isinstance(md, dict):
            self.saved_metadata = md


class _StubSessionManager:
    def __init__(self) -> None:
        self._store = _StubStore()

    async def update_metadata(self, session: Any, metadata: dict[str, object]) -> None:
        session.metadata = metadata
        self._store.saved_metadata = metadata


class _StubGroupCompressor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    async def set_session_suspended(self, *, session_id: str, suspended: bool) -> None:
        self.calls.append((session_id, suspended))


@pytest.mark.asyncio
async def test_multi_agent_requires_confirm_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = WhaleclawConfig()
    cfg.plugins["multi_agent"] = {
        "enabled": True,
        "mode": "parallel",
        "max_rounds": 3,
        "roles": [
            {
                "id": "planner",
                "name": "规划师",
                "enabled": True,
                "model": "",
                "system_prompt": "负责规划",
            }
        ],
    }
    session = _make_session({})
    sm = _StubSessionManager()

    called: dict[str, Any] = {"count": 0, "message": ""}
    discuss_called: dict[str, int] = {"count": 0}
    discuss_intro_flags: list[bool] = []

    async def _fake_discuss(**kwargs: Any) -> str:
        discuss_called["count"] += 1
        discuss_intro_flags.append(bool(kwargs.get("include_intro", False)))
        return "主控讨论中：请补充目标与约束。"

    async def _fake_exec(**kwargs: Any) -> str:
        called["count"] += 1
        called["message"] = str(kwargs.get("message", ""))
        return "EXEC_OK"

    monkeypatch.setattr(
        "whaleclaw.agent.loop._run_multi_agent_controller_discussion",
        _fake_discuss,
    )
    monkeypatch.setattr("whaleclaw.agent.loop._run_multi_agent_executor", _fake_exec)

    first = await run_agent(
        message="帮我做一个发布计划",
        session_id=session.id,
        config=cfg,
        session=session,
        session_manager=cast(SessionManager, sm),
    )
    assert "主控讨论中" in first
    assert discuss_called["count"] == 1
    assert discuss_intro_flags == [True]
    assert called["count"] == 0

    follow = await run_agent(
        message="目标用户是资深玩家",
        session_id=session.id,
        config=cfg,
        session=session,
        session_manager=cast(SessionManager, sm),
    )
    assert "主控讨论中" in follow
    assert discuss_called["count"] == 2
    assert discuss_intro_flags == [True, False]

    second = await run_agent(
        message="确认需求",
        session_id=session.id,
        config=cfg,
        session=session,
        session_manager=cast(SessionManager, sm),
    )
    assert second == "EXEC_OK"
    assert called["count"] == 1
    assert called["message"] == "帮我做一个发布计划\n补充要求: 目标用户是资深玩家"


@pytest.mark.asyncio
async def test_multi_agent_waiting_stage_can_adjust_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = WhaleclawConfig()
    cfg.plugins["multi_agent"] = {
        "enabled": True,
        "mode": "parallel",
        "max_rounds": 3,
        "roles": [
            {
                "id": "planner",
                "name": "规划师",
                "enabled": True,
                "model": "",
                "system_prompt": "负责规划",
            }
        ],
    }
    session = _make_session({})
    sm = _StubSessionManager()

    discuss_cfg: dict[str, Any] = {}

    async def _fake_discuss(**kwargs: Any) -> str:
        discuss_cfg.update(kwargs)
        return "主控讨论中"

    monkeypatch.setattr(
        "whaleclaw.agent.loop._run_multi_agent_controller_discussion",
        _fake_discuss,
    )

    _ = await run_agent(
        message="做个增长方案",
        session_id=session.id,
        config=cfg,
        session=session,
        session_manager=cast(SessionManager, sm),
    )
    out = await run_agent(
        message="改为2轮",
        session_id=session.id,
        config=cfg,
        session=session,
        session_manager=cast(SessionManager, sm),
    )
    assert "主控讨论中" in out
    assert session.metadata.get("multi_agent_pending_rounds") == 2
    cfg_used = discuss_cfg.get("cfg", {})
    assert isinstance(cfg_used, dict)
    assert cfg_used.get("max_rounds") == 2


@pytest.mark.asyncio
async def test_multi_agent_disable_sets_compression_resume_boundary() -> None:
    session = _make_session({"multi_agent_active_prev": True})
    session.messages = [cast(Any, object()) for _ in range(5)]
    sm = _StubSessionManager()
    gc = _StubGroupCompressor()

    await _sync_multi_agent_compression_boundary(
        session,
        cast(SessionManager, sm),
        cast(Any, gc),
        ma_enabled=False,
    )

    assert session.metadata.get("multi_agent_active_prev") is False
    assert session.metadata.get("compression_resume_message_index") == 5
    assert isinstance(sm._store.saved_metadata, dict)
    assert gc.calls == [("s1", False)]


@pytest.mark.asyncio
async def test_multi_agent_enable_suspends_compressor() -> None:
    session = _make_session({})
    sm = _StubSessionManager()
    gc = _StubGroupCompressor()

    await _sync_multi_agent_compression_boundary(
        session,
        cast(SessionManager, sm),
        cast(Any, gc),
        ma_enabled=True,
    )

    assert session.metadata.get("multi_agent_active_prev") is True
    assert gc.calls == [("s1", True)]
