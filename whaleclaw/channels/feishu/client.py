"""Feishu Open Platform API client."""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import httpx

from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

_BASE = "https://open.feishu.cn/open-apis"
_HTTP_RETRY_ATTEMPTS = 3
_HTTP_RETRY_BACKOFF_SECONDS = 0.4


class FeishuClient:
    """Async HTTP client for the Feishu Open Platform."""

    def __init__(self, app_id: str, app_secret: str, *, timeout: int = 30) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._timeout = timeout
        self._tenant_access_token: str | None = None
        self._token_expires_at: float = 0

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, _HTTP_RETRY_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                log.warning(
                    "feishu.http_retry",
                    method=method,
                    url=url,
                    attempt=attempt,
                    max_attempts=_HTTP_RETRY_ATTEMPTS,
                    error=repr(exc),
                )
                if attempt < _HTTP_RETRY_ATTEMPTS:
                    await asyncio.sleep(_HTTP_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
                    continue
                break

        msg = f"飞书接口请求失败: {method} {url}"
        raise RuntimeError(msg) from last_exc

    async def _ensure_token(self) -> str:
        if self._tenant_access_token and time.time() < self._token_expires_at:
            return self._tenant_access_token
        resp = await self._request_with_retry(
            "POST",
            f"{_BASE}/auth/v3/tenant_access_token/internal",
            json={"app_id": self._app_id, "app_secret": self._app_secret},
        )
        data = resp.json()
        if data.get("code") != 0:
            msg = f"获取 tenant_access_token 失败: {data}"
            raise RuntimeError(msg)
        self._tenant_access_token = data["tenant_access_token"]
        self._token_expires_at = time.time() + data.get("expire", 7200) - 60
        return cast(str, self._tenant_access_token)

    async def request(
        self, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Make an authenticated API request."""
        token = await self._ensure_token()
        extra_headers = kwargs.pop("headers", None)
        merged_headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
        }
        if isinstance(extra_headers, dict):
            extra_headers_dict = cast(dict[object, object], extra_headers)
            merged_headers.update(
                {
                    str(key): str(value)
                    for key, value in extra_headers_dict.items()
                }
            )
        resp = await self._request_with_retry(
            method,
            f"{_BASE}{path}",
            headers=merged_headers,
            **kwargs,
        )
        data: dict[str, Any] = resp.json()
        if data.get("code", 0) != 0:
            log.warning("feishu.api_error", path=path, code=data.get("code"), msg=data.get("msg"))
        return data

    # ── Message API ─────────────────────────────────────────

    async def send_message(
        self,
        receive_id: str,
        msg_type: str,
        content: str,
        receive_id_type: str = "open_id",
    ) -> dict[str, Any]:
        return await self.request(
            "POST",
            f"/im/v1/messages?receive_id_type={receive_id_type}",
            json={"receive_id": receive_id, "msg_type": msg_type, "content": content},
        )

    async def reply_message(
        self, message_id: str, msg_type: str, content: str
    ) -> dict[str, Any]:
        return await self.request(
            "POST",
            f"/im/v1/messages/{message_id}/reply",
            json={"msg_type": msg_type, "content": content},
        )

    async def create_message_reaction(
        self,
        message_id: str,
        emoji_type: str,
    ) -> dict[str, Any]:
        return await self.request(
            "POST",
            f"/im/v1/messages/{message_id}/reactions",
            json={"reaction_type": {"emoji_type": emoji_type}},
        )

    async def delete_message_reaction(
        self,
        message_id: str,
        reaction_id: str,
    ) -> dict[str, Any]:
        return await self.request(
            "DELETE",
            f"/im/v1/messages/{message_id}/reactions/{reaction_id}",
        )

    async def update_message(
        self, message_id: str, content: str
    ) -> dict[str, Any]:
        return await self.request(
            "PATCH",
            f"/im/v1/messages/{message_id}",
            json={"content": content},
        )

    async def get_message(self, message_id: str) -> dict[str, Any]:
        return await self.request("GET", f"/im/v1/messages/{message_id}")

    # ── Media API ───────────────────────────────────────────

    async def upload_image(
        self, image: bytes, image_type: str = "message"
    ) -> str:
        token = await self._ensure_token()
        resp = await self._request_with_retry(
            "POST",
            f"{_BASE}/im/v1/images",
            headers={"Authorization": f"Bearer {token}"},
            data={"image_type": image_type},
            files={"image": ("image.png", image, "image/png")},
        )
        data = resp.json()
        return data.get("data", {}).get("image_key", "")

    async def upload_file(
        self, file: bytes, filename: str, file_type: str
    ) -> str:
        token = await self._ensure_token()
        resp = await self._request_with_retry(
            "POST",
            f"{_BASE}/im/v1/files",
            headers={"Authorization": f"Bearer {token}"},
            data={"file_type": file_type, "file_name": filename},
            files={"file": (filename, file, "application/octet-stream")},
        )
        data = resp.json()
        return data.get("data", {}).get("file_key", "")

    async def download_resource(
        self, message_id: str, file_key: str, *, resource_type: str = "file"
    ) -> bytes:
        token = await self._ensure_token()
        resp = await self._request_with_retry(
            "GET",
            f"{_BASE}/im/v1/messages/{message_id}/resources/{file_key}",
            headers={"Authorization": f"Bearer {token}"},
            params={"type": resource_type},
        )
        return resp.content

    # ── User API ────────────────────────────────────────────

    async def get_user_info(
        self, user_id: str, user_id_type: str = "open_id"
    ) -> dict[str, Any]:
        return await self.request(
            "GET",
            f"/contact/v3/users/{user_id}?user_id_type={user_id_type}",
        )
