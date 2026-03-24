"""Feishu Webhook endpoint — signature verification and event dispatch."""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
from base64 import b64decode
from typing import Any

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from whaleclaw.channels.feishu.config import FeishuConfig
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)


def verify_signature(
    timestamp: str,
    nonce: str,
    encrypt_key: str,
    body: bytes,
    signature: str,
) -> bool:
    """Verify Feishu Webhook signature (SHA-256)."""
    content = f"{timestamp}{nonce}{encrypt_key}".encode() + body
    expected = hashlib.sha256(content).hexdigest()
    return _hmac.compare_digest(expected, signature)


def decrypt_event(encrypt_key: str, encrypted: str) -> dict[str, Any]:
    """Decrypt an AES-256-CBC encrypted Feishu event body."""
    key = hashlib.sha256(encrypt_key.encode()).digest()
    raw = b64decode(encrypted)
    iv = raw[:16]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(raw[16:]) + decryptor.finalize()
    pad_len = padded[-1]
    return json.loads(padded[:-pad_len].decode("utf-8"))


EventHandler = Any


def create_feishu_router(
    config: FeishuConfig,
    on_event: EventHandler | None = None,
) -> APIRouter:
    """Create a FastAPI router for the Feishu webhook endpoint."""
    router = APIRouter()

    @router.post(config.webhook_path)
    async def feishu_webhook(request: Request) -> JSONResponse:
        raw_body = await request.body()
        body: dict[str, Any] = json.loads(raw_body)

        if config.encrypt_key:
            headers = request.headers
            sig = headers.get("x-lark-signature", "")
            ts = headers.get("x-lark-request-timestamp", "")
            nonce = headers.get("x-lark-request-nonce", "")
            if not verify_signature(ts, nonce, config.encrypt_key, raw_body, sig):
                log.warning("feishu.invalid_signature")
                return JSONResponse({"error": "invalid signature"}, status_code=403)

        if config.encrypt_key and "encrypt" in body:
            body = decrypt_event(config.encrypt_key, body["encrypt"])

        if "challenge" in body:
            return JSONResponse({"challenge": body["challenge"]})

        event_type = (
            body.get("header", {}).get("event_type", "")
            or body.get("event", {}).get("type", "")
        )
        log.info("feishu.event", event_type=event_type)

        if on_event:
            try:
                await on_event(event_type, body)
            except Exception as exc:
                log.error("feishu.event_handler_error", error=str(exc))

        return JSONResponse({"ok": True})

    return router
