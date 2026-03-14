"""In-memory store with keyword matching and optional JSON persistence."""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from whaleclaw.memory.base import MemoryEntry, MemorySearchResult, MemoryStore


def _serialize_entry(entry: MemoryEntry) -> dict[str, Any]:
    d: dict[str, Any] = entry.model_dump(mode="json")
    d["created_at"] = entry.created_at.isoformat()
    d["last_accessed"] = entry.last_accessed.isoformat()
    return d


def _deserialize_entry(raw: dict[str, Any]) -> MemoryEntry:
    d: dict[str, Any] = dict(raw)
    d["created_at"] = datetime.fromisoformat(str(d["created_at"]))
    d["last_accessed"] = datetime.fromisoformat(str(d["last_accessed"]))
    return MemoryEntry.model_validate(d)


class SimpleMemoryStore(MemoryStore):
    """In-memory store with keyword matching, optional JSON persistence."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._persist_dir = persist_dir
        if persist_dir:
            self._load()

    def _load(self) -> None:
        if not self._persist_dir:
            return
        path = self._persist_dir / "memory.json"
        if not path.exists():
            return
        try:
            parsed: Any = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                return
            raw = cast(dict[str, Any], parsed)
            entries: list[Any] = raw.get("entries", [])
            for d in entries:
                if isinstance(d, dict):
                    entry = _deserialize_entry(cast(dict[str, Any], d))
                    self._entries[entry.id] = entry
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    def _save(self) -> None:
        if not self._persist_dir:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        path = self._persist_dir / "memory.json"
        data: dict[str, Any] = {"entries": [_serialize_entry(e) for e in self._entries.values()]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    async def add(self, content: str, source: str, tags: list[str] | None = None) -> MemoryEntry:
        now = datetime.now(UTC)
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            source=source,
            tags=tags or [],
            created_at=now,
            last_accessed=now,
        )
        self._entries[entry.id] = entry
        self._save()
        return entry

    def _keyword_score(self, query: str, content: str) -> float:
        words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 0]
        if not words:
            return 0.0
        content_lower = content.lower()
        found = sum(1 for w in words if w in content_lower)
        return found / len(words)

    async def search(
        self, query: str, limit: int = 5, min_score: float = 0.5
    ) -> list[MemorySearchResult]:
        results: list[MemorySearchResult] = []
        for entry in self._entries.values():
            score = self._keyword_score(query, entry.content)
            if score >= min_score:
                results.append(MemorySearchResult(entry=entry, score=score))
        results.sort(key=lambda r: (-r.score, -r.entry.created_at.timestamp()))
        return results[:limit]

    async def get(self, memory_id: str) -> MemoryEntry | None:
        return self._entries.get(memory_id)

    async def delete(self, memory_id: str) -> bool:
        if memory_id in self._entries:
            del self._entries[memory_id]
            self._save()
            return True
        return False

    async def list_recent(self, limit: int = 20) -> list[MemoryEntry]:
        entries = sorted(self._entries.values(), key=lambda e: e.created_at, reverse=True)
        return entries[:limit]
