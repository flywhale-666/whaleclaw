"""Feishu media handling — image and file upload/download."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whaleclaw.channels.feishu.client import FeishuClient
from whaleclaw.config.paths import WHALECLAW_HOME

_MEDIA_DIR = WHALECLAW_HOME / "media"


async def handle_image_message(
    client: FeishuClient, message: dict[str, Any]
) -> str:
    """Download an image from a Feishu message and return the local path."""
    msg_id = message.get("message_id", "")
    content = json.loads(message.get("content", "{}"))
    image_key = content.get("image_key", "")
    if not image_key:
        return ""

    data = await client.download_resource(msg_id, image_key)
    _MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _MEDIA_DIR / f"{image_key}.png"
    dest.write_bytes(data)
    return str(dest)


async def handle_file_message(
    client: FeishuClient, message: dict[str, Any]
) -> str:
    """Download a file from a Feishu message and return the local path."""
    msg_id = message.get("message_id", "")
    content = json.loads(message.get("content", "{}"))
    file_key = content.get("file_key", "")
    file_name = content.get("file_name", file_key)
    if not file_key:
        return ""

    data = await client.download_resource(msg_id, file_key)
    _MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file_name).name
    dest = _MEDIA_DIR / safe_name
    if not dest.resolve().is_relative_to(_MEDIA_DIR.resolve()):
        return ""
    dest.write_bytes(data)
    return str(dest)


async def send_image(
    client: FeishuClient, peer_id: str, image_path: str
) -> None:
    """Upload a local image and send it as a Feishu message."""
    data = Path(image_path).read_bytes()
    image_key = await client.upload_image(data)
    content = json.dumps({"image_key": image_key})
    await client.send_message(peer_id, "image", content)


async def send_file(
    client: FeishuClient, peer_id: str, file_path: str
) -> None:
    """Upload a local file and send it as a Feishu message."""
    p = Path(file_path)
    data = p.read_bytes()
    file_key = await client.upload_file(data, p.name, "stream")
    content = json.dumps({"file_key": file_key})
    await client.send_message(peer_id, "file", content)
