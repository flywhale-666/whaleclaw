"""Tests for the Feishu bot message handler."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from whaleclaw.channels.feishu.bot import (
    FeishuBot,
    format_exception_text,
    processing_reaction_emojis,
)
from whaleclaw.channels.feishu.config import FeishuConfig
from whaleclaw.config.schema import ProviderModelEntry, WhaleclawConfig


class _StubClient:
    def __init__(self) -> None:
        self.replies: list[tuple[str, str, str]] = []
        self.reactions_created: list[tuple[str, str]] = []
        self.reactions_deleted: list[tuple[str, str]] = []
        self._reaction_seq = 0

    async def reply_message(  # noqa: D401
        self, message_id: str, msg_type: str, content: str
    ) -> dict[str, Any]:
        self.replies.append((message_id, msg_type, content))
        return {"data": {"message_id": "reply-1"}}

    async def create_message_reaction(
        self,
        message_id: str,
        emoji_type: str,
    ) -> dict[str, Any]:
        self.reactions_created.append((message_id, emoji_type))
        self._reaction_seq += 1
        return {
            "data": {
                "reaction_id": f"reaction-{self._reaction_seq}",
                "reaction_type": {"emoji_type": emoji_type},
            }
        }

    async def delete_message_reaction(
        self,
        message_id: str,
        reaction_id: str,
    ) -> dict[str, Any]:
        self.reactions_deleted.append((message_id, reaction_id))
        return {"data": {}}

    async def upload_image(
        self, image: bytes, image_type: str = "message"
    ) -> str:
        return "img_uploaded_key"

    async def upload_file(
        self, file: bytes, filename: str, file_type: str
    ) -> str:
        return "file_uploaded_key"

    async def send_message(
        self,
        receive_id: str,
        msg_type: str,
        content: str,
        receive_id_type: str = "open_id",
    ) -> dict[str, Any]:
        self.replies.append((receive_id, msg_type, content))
        return {"data": {"message_id": "sent-1"}}

    async def download_resource(
        self, message_id: str, file_key: str, *, resource_type: str = "file"
    ) -> bytes:
        assert resource_type == "image"
        return b"fake-image-binary"


class _StubSessionManager:
    def __init__(self) -> None:
        self.updated_to: str | None = None
        self.updated_metadata: dict[str, Any] | None = None
        self.session = SimpleNamespace(
            id="s-feishu-1",
            channel="feishu",
            peer_id="ou_xxx",
            metadata={},
            model="openai/gpt-5.2",
        )

    async def get_or_create(self, channel: str, peer_id: str) -> Any:
        self.session.channel = channel
        self.session.peer_id = peer_id
        return self.session

    async def update_model(self, session: Any, model: str) -> None:
        session.model = model
        self.updated_to = model

    async def update_metadata(self, session: Any, metadata: dict[str, Any]) -> None:
        session.metadata = metadata
        self.updated_metadata = metadata


class TestExtractText:
    def test_text_message(self) -> None:
        message = {
            "message_type": "text",
            "content": json.dumps({"text": "hello world"}),
        }
        assert FeishuBot.extract_text(message) == "hello world"

    def test_post_message(self) -> None:
        message = {
            "message_type": "post",
            "content": json.dumps({
                "content": [
                    [
                        {"tag": "text", "text": "line one"},
                        {"tag": "text", "text": "line two"},
                    ]
                ]
            }),
        }
        assert "line one" in FeishuBot.extract_text(message)

    def test_empty_content(self) -> None:
        message = {"message_type": "text", "content": "{}"}
        assert FeishuBot.extract_text(message) == ""

    def test_invalid_json(self) -> None:
        message = {"message_type": "text", "content": "not json"}
        assert FeishuBot.extract_text(message) == ""


class TestFormatExceptionText:
    def test_empty_exception_message(self) -> None:
        assert format_exception_text(TimeoutError()) == "TimeoutError"

    def test_non_empty_exception_message(self) -> None:
        assert format_exception_text(RuntimeError("boom")) == "boom"


def test_prepare_reply_payload_extracts_audio_markdown_file(tmp_path: Path) -> None:
    audio = tmp_path / "reply.mp3"
    audio.write_bytes(b"audio")
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))

    text, images, files = bot.prepare_reply_payload(
        f"已生成音频: [reply]({audio})"
    )

    assert "📎 reply" in text
    assert images == []
    assert files == [audio]


def test_prepare_reply_payload_collapses_gap_left_by_removed_image(tmp_path: Path) -> None:
    image = tmp_path / "result.png"
    image.write_bytes(b"image")
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))

    text, images, files = bot.prepare_reply_payload(
        "旺财🐶，当前使用模型：香蕉2\n"
        f"![结果图]({image})\n"
        "回复“任务完成”以解除技能锁定；若继续修改请直接说需求。"
    )

    assert text == (
        "旺财🐶，当前使用模型：香蕉2\n"
        "回复“任务完成”以解除技能锁定；若继续修改请直接说需求。"
    )
    assert images == [image]
    assert files == []


def test_prepare_reply_payload_removes_numbering_left_by_removed_images(tmp_path: Path) -> None:
    image1 = tmp_path / "result-1.png"
    image1.write_bytes(b"image-1")
    image2 = tmp_path / "result-2.png"
    image2.write_bytes(b"image-2")
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))

    text, images, files = bot.prepare_reply_payload(
        "旺财🐶，已生成 2 张。\n"
        "1.\n"
        f"![结果图1]({image1})\n"
        "2.\n"
        f"![结果图2]({image2})\n"
        "回复“任务完成”以解除技能锁定；若继续修改请直接说需求。"
    )

    assert text == (
        "旺财🐶，已生成 2 张。\n"
        "回复“任务完成”以解除技能锁定；若继续修改请直接说需求。"
    )
    assert images == [image1, image2]
    assert files == []


def test_prepare_reply_payload_extracts_audio_bare_path(tmp_path: Path) -> None:
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"audio")
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))

    text, images, files = bot.prepare_reply_payload(
        f"音频文件在这里: {audio}"
    )

    assert "📎 voice.wav" in text
    assert images == []
    assert files == [audio]


def test_prepare_reply_payload_extracts_aiff_bare_path(tmp_path: Path) -> None:
    audio = tmp_path / "test_joke.aiff"
    audio.write_bytes(b"audio")
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))

    text, images, files = bot.prepare_reply_payload(
        f"文件路径: `{audio}`"
    )

    assert "📎 test_joke.aiff" in text
    assert images == []
    assert files == [audio]


def test_prepare_reply_payload_extracts_txt_bare_path(tmp_path: Path) -> None:
    text_file = tmp_path / "note.txt"
    text_file.write_text("hello", encoding="utf-8")
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))

    text, images, files = bot.prepare_reply_payload(
        f"输出在: {text_file}"
    )

    assert "📎 note.txt" in text
    assert images == []
    assert files == [text_file]


def test_prepare_reply_payload_extracts_mp4_bare_path(tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))

    text, images, files = bot.prepare_reply_payload(
        f"视频: `{video}`"
    )

    assert "📎 demo.mp4" in text
    assert images == []
    assert files == [video]


def test_prepare_reply_payload_extracts_chinese_pptx_bare_path(tmp_path: Path) -> None:
    pptx = tmp_path / "贵州2日游.pptx"
    pptx.write_bytes(b"pptx")
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))

    text, images, files = bot.prepare_reply_payload(
        f"PPT 文件在这里: {pptx}"
    )

    assert "📎 贵州2日游.pptx" in text
    assert images == []
    assert files == [pptx]


def test_prepare_reply_payload_extracts_bold_pptx_path(tmp_path: Path) -> None:
    pptx = tmp_path / "沙特两日游.pptx"
    pptx.write_bytes(b"pptx")
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))

    text, images, files = bot.prepare_reply_payload(
        f"PPT 文件路径：\n- **{pptx}**"
    )

    assert "📎 沙特两日游.pptx" in text
    assert images == []
    assert files == [pptx]


@pytest.mark.asyncio
async def test_handle_image_message_passes_images_to_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))
    captured: dict[str, Any] = {}
    media_dir = Path("/tmp/whaleclaw-test-feishu-images")

    async def _fake_run(
        text: str,
        peer_id: str,
        card_msg_id: str,
        *,
        images: list[Any] | None = None,
    ) -> None:
        captured["text"] = text
        captured["peer_id"] = peer_id
        captured["card_msg_id"] = card_msg_id
        captured["images"] = images or []

    monkeypatch.setattr("whaleclaw.channels.feishu.bot._FEISHU_MEDIA_DIR", media_dir)
    monkeypatch.setattr(bot, "_run_agent_and_reply", _fake_run)

    await bot.handle_message({
        "sender": {"sender_id": {"open_id": "ou_xxx"}},
        "message": {
            "message_id": "msg-image-1",
            "chat_type": "p2p",
            "message_type": "image",
            "content": json.dumps({"image_key": "img_key_1"}),
        },
    })

    assert captured["text"].startswith("(用户发送了图片)\n![飞书图片1](")
    assert captured["peer_id"] == "ou_xxx"
    assert len(captured["images"]) == 1
    assert captured["images"][0].mime == "image/png"
    assert captured["images"][0].data == base64.b64encode(b"fake-image-binary").decode("ascii")
    saved_path = Path(captured["text"].split("](", 1)[1].rstrip(")"))
    assert saved_path.is_file()
    assert saved_path.read_bytes() == b"fake-image-binary"


@pytest.mark.asyncio
async def test_handle_image_message_buffers_until_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _StubClient()
    bot = FeishuBot(client, FeishuConfig(dm_policy="open"))
    sm = _StubSessionManager()
    sm.session.metadata = {"locked_skill_ids": ["xiaohongshu_publish"]}
    monkeypatch.setattr(bot, "_session_manager", sm)
    media_dir = Path("/tmp/whaleclaw-test-feishu-buffer")
    captured: dict[str, Any] = {}

    async def _fake_run(
        text: str,
        peer_id: str,
        card_msg_id: str,
        *,
        images: list[Any] | None = None,
    ) -> None:
        captured["text"] = text
        captured["peer_id"] = peer_id
        captured["card_msg_id"] = card_msg_id
        captured["images"] = images or []

    monkeypatch.setattr("whaleclaw.channels.feishu.bot._FEISHU_MEDIA_DIR", media_dir)
    monkeypatch.setattr(bot, "_run_agent_and_reply", _fake_run)

    await bot.handle_message({
        "sender": {"sender_id": {"open_id": "ou_xxx"}},
        "message": {
            "message_id": "msg-buffer-1",
            "chat_type": "p2p",
            "message_type": "image",
            "content": json.dumps({"image_key": "img_key_1"}),
        },
    })

    assert "已收到图1" in client.replies[-1][2]
    assert captured == {}

    await bot.handle_message({
        "sender": {"sender_id": {"open_id": "ou_xxx"}},
        "message": {
            "message_id": "msg-buffer-2",
            "chat_type": "p2p",
            "message_type": "text",
            "content": json.dumps({"text": "让图1的男孩骑马 提交"}),
        },
    })

    assert captured["peer_id"] == "ou_xxx"
    assert captured["text"].startswith("让图1的男孩骑马")
    assert "![飞书图片1](" in captured["text"]
    assert len(captured["images"]) == 1
    assert client.reactions_created == [("msg-buffer-2", processing_reaction_emojis()[0])]
    assert client.reactions_deleted == [("msg-buffer-2", "reaction-1")]


@pytest.mark.asyncio
async def test_handle_images_with_prompt_runs_immediately_without_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _StubClient()
    bot = FeishuBot(client, FeishuConfig(dm_policy="open"))
    monkeypatch.setattr(bot, "_session_manager", _StubSessionManager())
    media_dir = Path("/tmp/whaleclaw-test-feishu-direct-run")
    captured: dict[str, Any] = {}

    async def _fake_run(
        text: str,
        peer_id: str,
        card_msg_id: str,
        *,
        images: list[Any] | None = None,
    ) -> None:
        captured["text"] = text
        captured["peer_id"] = peer_id
        captured["card_msg_id"] = card_msg_id
        captured["images"] = images or []

    monkeypatch.setattr("whaleclaw.channels.feishu.bot._FEISHU_MEDIA_DIR", media_dir)
    monkeypatch.setattr(bot, "_run_agent_and_reply", _fake_run)

    await bot.handle_message({
        "sender": {"sender_id": {"open_id": "ou_xxx"}},
        "message": {
            "message_id": "msg-direct-run-1",
            "chat_type": "p2p",
            "message_type": "post",
            "content": json.dumps({
                "content": [
                    [
                        {"tag": "text", "text": "让图1的男孩骑着图2的马"},
                    ]
                ],
                "image_keys": ["img_key_1", "img_key_2", "img_key_3"],
            }),
        },
    })

    assert captured["peer_id"] == "ou_xxx"
    assert captured["text"].startswith("让图1的男孩骑着图2的马")
    assert captured["text"].count("![飞书图片") == 3
    assert len(captured["images"]) == 3
    assert client.reactions_created == [
        ("msg-direct-run-1", processing_reaction_emojis()[0])
    ]
    assert client.reactions_deleted == [("msg-direct-run-1", "reaction-1")]


@pytest.mark.asyncio
async def test_handle_buffered_images_then_prompt_runs_immediately_without_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _StubClient()
    bot = FeishuBot(client, FeishuConfig(dm_policy="open"))
    sm = _StubSessionManager()
    sm.session.metadata = {"locked_skill_ids": ["xiaohongshu_publish"]}
    monkeypatch.setattr(bot, "_session_manager", sm)
    media_dir = Path("/tmp/whaleclaw-test-feishu-buffer-then-run")
    captured: dict[str, Any] = {}

    async def _fake_run(
        text: str,
        peer_id: str,
        card_msg_id: str,
        *,
        images: list[Any] | None = None,
    ) -> None:
        captured["text"] = text
        captured["peer_id"] = peer_id
        captured["card_msg_id"] = card_msg_id
        captured["images"] = images or []

    monkeypatch.setattr("whaleclaw.channels.feishu.bot._FEISHU_MEDIA_DIR", media_dir)
    monkeypatch.setattr(bot, "_run_agent_and_reply", _fake_run)

    await bot.handle_message({
        "sender": {"sender_id": {"open_id": "ou_xxx"}},
        "message": {
            "message_id": "msg-buffer-run-1",
            "chat_type": "p2p",
            "message_type": "image",
            "content": json.dumps({"image_key": "img_key_1"}),
        },
    })
    await bot.handle_message({
        "sender": {"sender_id": {"open_id": "ou_xxx"}},
        "message": {
            "message_id": "msg-buffer-run-2",
            "chat_type": "p2p",
            "message_type": "image",
            "content": json.dumps({"image_key": "img_key_2"}),
        },
    })

    assert "已收到图2" in client.replies[-1][2]
    assert captured == {}

    await bot.handle_message({
        "sender": {"sender_id": {"open_id": "ou_xxx"}},
        "message": {
            "message_id": "msg-buffer-run-3",
            "chat_type": "p2p",
            "message_type": "text",
            "content": json.dumps({"text": "图2的巨型三花猫正好奇地拨弄着图1的寺庙"}),
        },
    })

    assert captured["peer_id"] == "ou_xxx"
    assert captured["text"].startswith("图2的巨型三花猫正好奇地拨弄着图1的寺庙")
    assert captured["text"].count("![飞书图片") == 2
    assert len(captured["images"]) == 2
    assert client.reactions_created == [
        ("msg-buffer-run-3", processing_reaction_emojis()[0])
    ]
    assert client.reactions_deleted == [("msg-buffer-run-3", "reaction-1")]


@pytest.mark.asyncio
async def test_model_command_lists_configured_models() -> None:
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))
    cfg = WhaleclawConfig()
    cfg.models.qwen.api_key = "test-key"
    cfg.models.qwen.configured_models = [
        ProviderModelEntry(id="qwen3.5-plus", verified=True),
        ProviderModelEntry(id="qwen3-max", verified=False),
    ]
    object.__setattr__(bot, "_whaleclaw_config", cfg)
    object.__setattr__(bot, "_session_manager", _StubSessionManager())
    session = SimpleNamespace(id="s1", model="qwen/qwen3.5-plus")

    out = await bot._handle_command("/models", session)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    assert out is not None
    assert "1. qwen/qwen3.5-plus (当前)" in out
    assert "qwen3-max" not in out


@pytest.mark.asyncio
async def test_model_command_switch_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))
    cfg = WhaleclawConfig()
    cfg.models.qwen.api_key = "test-key"
    cfg.models.qwen.configured_models = [
        ProviderModelEntry(id="qwen3.5-plus", verified=True),
        ProviderModelEntry(id="qwen3-max", verified=True),
    ]
    persisted: list[str] = []

    def _fake_persist(model: str) -> None:
        persisted.append(model)

    monkeypatch.setattr("whaleclaw.channels.feishu.bot.set_default_agent_model", _fake_persist)
    sm = _StubSessionManager()
    monkeypatch.setattr(bot, "_whaleclaw_config", cfg)
    monkeypatch.setattr(bot, "_session_manager", sm)
    session = SimpleNamespace(id="s1", model="qwen/qwen3.5-plus")

    out = await bot._handle_command("/model 2", session)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    assert out == "已切换模型到: qwen/qwen3-max"
    assert session.model == "qwen/qwen3-max"
    assert sm.updated_to == "qwen/qwen3-max"
    assert cfg.agent.model == "qwen/qwen3-max"
    assert persisted == ["qwen/qwen3-max"]


@pytest.mark.asyncio
async def test_multi_command_toggle_session_override() -> None:
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))
    cfg = WhaleclawConfig()
    cfg.plugins["multi_agent"] = {
        "enabled": False,
        "mode": "parallel",
        "max_rounds": 2,
        "roles": [],
    }
    sm = _StubSessionManager()
    object.__setattr__(bot, "_whaleclaw_config", cfg)
    object.__setattr__(bot, "_session_manager", sm)
    session = SimpleNamespace(id="s1", model="openai/gpt-5.2", metadata={})

    on_out = await bot._handle_command("/multi on", session)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    off_out = await bot._handle_command("/multi off", session)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    assert on_out is not None and "已开启本会话多Agent" in on_out
    assert off_out is not None and "已关闭本会话多Agent" in off_out
    assert sm.updated_metadata is not None
    assert sm.updated_metadata.get("multi_agent_enabled") is False


@pytest.mark.asyncio
async def test_multi_command_status_shows_effective_state() -> None:
    bot = FeishuBot(_StubClient(), FeishuConfig(dm_policy="open"))
    cfg = WhaleclawConfig()
    cfg.plugins["multi_agent"] = {
        "enabled": False,
        "mode": "parallel",
        "max_rounds": 2,
        "roles": [],
    }
    object.__setattr__(bot, "_whaleclaw_config", cfg)
    object.__setattr__(bot, "_session_manager", _StubSessionManager())
    session = SimpleNamespace(
        id="s1",
        model="openai/gpt-5.2",
        metadata={
            "multi_agent_enabled": True,
            "multi_agent_mode": "serial",
            "multi_agent_max_rounds": 4,
        },
    )

    out = await bot._handle_command("/multi status", session)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    assert out is not None
    assert "全局: 关闭" in out
    assert "当前生效: 开启" in out
    assert "串行（serial）" in out
    assert "回合=4" in out
