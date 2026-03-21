"""Tests for the Feishu allowlist."""

from __future__ import annotations

from pathlib import Path

from whaleclaw.channels.feishu.allowlist import FeishuAllowList


def test_add_and_check(tmp_path: Path) -> None:
    al = FeishuAllowList(path=tmp_path / "allow.json")
    assert not al.is_allowed("ou_123")
    al.add("ou_123")
    assert al.is_allowed("ou_123")


def test_remove(tmp_path: Path) -> None:
    al = FeishuAllowList(path=tmp_path / "allow.json")
    al.add("ou_123")
    al.remove("ou_123")
    assert not al.is_allowed("ou_123")


def test_persistence(tmp_path: Path) -> None:
    path = tmp_path / "allow.json"
    al1 = FeishuAllowList(path=path)
    al1.add("ou_abc")

    al2 = FeishuAllowList(path=path)
    assert al2.is_allowed("ou_abc")


def test_list_all(tmp_path: Path) -> None:
    al = FeishuAllowList(path=tmp_path / "allow.json")
    al.add("ou_b")
    al.add("ou_a")
    assert al.list_all() == ["ou_a", "ou_b"]
