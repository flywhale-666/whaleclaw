from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from whaleclaw.tools.browser import BrowserTool, _BrowserCfg, _normalize_image_query


# ── _normalize_image_query tests ──────────────────────────────────


def test_normalize_image_query_expand_person_name() -> None:
    q = _normalize_image_query("杨幂")
    assert q == "杨幂 近照 高清 人像"


def test_normalize_image_query_keep_specific_query() -> None:
    q = _normalize_image_query("杨幂 2025 机场 近照 高清")
    assert q == "杨幂 2025 机场 近照 高清"


def test_normalize_image_query_reject_generic() -> None:
    with pytest.raises(ValueError) as exc:
        _normalize_image_query("2")
    assert "无效" in str(exc.value) or "泛化" in str(exc.value)


def test_normalize_image_query_strips_control_chars() -> None:
    q = _normalize_image_query("\x10\x10刘亦菲\x10 写真 高清\x10")
    assert q == "刘亦菲 写真 高清"


def test_normalize_image_query_strips_escaped_noise() -> None:
    q = _normalize_image_query("杨幂 \\n0\\n0\\n0 高清 写真")
    assert q == "杨幂 高清 写真"


# ── _BrowserCfg tests ─────────────────────────────────────────────


def test_browser_cfg_defaults() -> None:
    cfg = _BrowserCfg()
    assert cfg.mode == "launch"
    assert cfg.cdp_url == "http://localhost:9222"
    assert cfg.headless is False


def test_browser_cfg_from_config_no_file(tmp_path: Path) -> None:
    with patch("whaleclaw.tools.browser.CONFIG_FILE", tmp_path / "nope.json"):
        cfg = _BrowserCfg.from_config_file()
    assert cfg.mode == "launch"


def test_browser_cfg_from_config_cdp_mode(tmp_path: Path) -> None:
    cfg_file = tmp_path / "whaleclaw.json"
    cfg_file.write_text(json.dumps({
        "browser": {"mode": "cdp", "cdp_url": "http://127.0.0.1:9333", "visible": False}
    }))
    with patch("whaleclaw.tools.browser.CONFIG_FILE", cfg_file):
        cfg = _BrowserCfg.from_config_file()
    assert cfg.mode == "cdp"
    assert cfg.cdp_url == "http://127.0.0.1:9333"
    assert cfg.headless is True


def test_browser_cfg_from_config_launch_mode(tmp_path: Path) -> None:
    cfg_file = tmp_path / "whaleclaw.json"
    cfg_file.write_text(json.dumps({"browser": {"mode": "launch"}}))
    with patch("whaleclaw.tools.browser.CONFIG_FILE", cfg_file):
        cfg = _BrowserCfg.from_config_file()
    assert cfg.mode == "launch"
    assert cfg.headless is False


def test_browser_cfg_legacy_plugins_format(tmp_path: Path) -> None:
    """兼容旧的 plugins.browser.visible 格式。"""
    cfg_file = tmp_path / "whaleclaw.json"
    cfg_file.write_text(json.dumps({"plugins": {"browser": {"visible": False}}}))
    with patch("whaleclaw.tools.browser.CONFIG_FILE", cfg_file):
        cfg = _BrowserCfg.from_config_file()
    assert cfg.mode == "launch"
    assert cfg.headless is True


# ── BrowserTool CDP close message test ────────────────────────────


@pytest.mark.asyncio
async def test_close_cdp_mode_message() -> None:
    """CDP 模式下 close 应提示辅助浏览器仍在运行。"""
    tool = BrowserTool()
    tool._mode = "cdp"
    tool._browser = AsyncMock()
    tool._playwright = AsyncMock()
    tool._page = MagicMock()

    result = await tool._close()
    assert result.success is True
    assert "仍在运行" in result.output


@pytest.mark.asyncio
async def test_close_launch_mode_message() -> None:
    """launch 模式下 close 应提示浏览器已关闭。"""
    tool = BrowserTool()
    tool._mode = "launch"
    tool._browser = AsyncMock()
    tool._playwright = AsyncMock()
    tool._page = MagicMock()

    result = await tool._close()
    assert result.success is True
    assert "已关闭" in result.output


def test_browser_cfg_cdp_mode_value(tmp_path: Path) -> None:
    """确认 cdp 配置正确解析后 mode 字段为 'cdp'。"""
    cfg_file = tmp_path / "whaleclaw.json"
    cfg_file.write_text(json.dumps({"browser": {"mode": "cdp"}}))
    with patch("whaleclaw.tools.browser.CONFIG_FILE", cfg_file):
        cfg = _BrowserCfg.from_config_file()
    assert cfg.mode == "cdp"
    assert cfg.cdp_url == "http://localhost:9222"


@pytest.mark.asyncio
async def test_upload_action_sets_input_files(tmp_path: Path) -> None:
    """upload action 应调用 set_input_files 上传本地文件。"""
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG")

    mock_el = AsyncMock()
    mock_page = AsyncMock()
    mock_page.query_selector = AsyncMock(return_value=mock_el)

    tool = BrowserTool()
    result = await tool._upload_files(mock_page, {
        "selector": "input[type='file']",
        "file_paths": str(img),
    })
    assert result.success is True
    assert "1 个文件" in result.output
    mock_el.set_input_files.assert_called_once_with([str(img)])


@pytest.mark.asyncio
async def test_upload_action_cdp_returns_error_with_hint(tmp_path: Path) -> None:
    """CDP 模式下 set_input_files 失败时应 fallback 提示用 osascript。"""
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG")

    mock_el = AsyncMock()
    mock_el.set_input_files = AsyncMock(side_effect=Exception("CDP not supported"))
    mock_page = AsyncMock()
    mock_page.query_selector = AsyncMock(return_value=mock_el)

    tool = BrowserTool()
    tool._mode = "cdp"
    result = await tool._upload_files(mock_page, {
        "selector": "input[type='file']",
        "file_paths": str(img),
    })
    assert result.success is False
    assert "osascript" in (result.error or "")


@pytest.mark.asyncio
async def test_upload_action_rejects_missing_file() -> None:
    """upload action 文件不存在时应返回错误。"""
    tool = BrowserTool()
    mock_page = AsyncMock()
    result = await tool._upload_files(mock_page, {
        "file_paths": "/nonexistent/fake.png",
    })
    assert result.success is False
    assert "文件不存在" in (result.error or "")


@pytest.mark.asyncio
async def test_upload_action_rejects_empty_paths() -> None:
    """upload action file_paths 为空时应返回错误。"""
    tool = BrowserTool()
    mock_page = AsyncMock()
    result = await tool._upload_files(mock_page, {"file_paths": ""})
    assert result.success is False
    assert "file_paths" in (result.error or "")
