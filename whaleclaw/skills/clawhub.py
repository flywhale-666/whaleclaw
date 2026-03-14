"""ClawHub CLI integration for searching and installing skills."""

from __future__ import annotations

import contextlib
import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import cast
from urllib.parse import quote

import httpx

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE)
_SEARCH_CACHE_TTL_SECONDS = 45.0
_DETAIL_CACHE_TTL_SECONDS = 900.0
_DETAIL_ENRICH_MAX_ITEMS = 24
_CLI_INSPECT_FALLBACK_MAX_ITEMS = 2
_DETAIL_ENRICH_MAX_WORKERS = 4
_TEXT_FILE_EXTENSIONS = {
    "md", "mdx", "txt", "json", "json5", "yaml", "yml", "toml",
    "js", "cjs", "mjs", "ts", "tsx", "jsx", "py", "sh", "rb", "go",
    "rs", "swift", "kt", "java", "cs", "cpp", "c", "h", "hpp", "sql",
    "csv", "ini", "cfg", "env", "xml", "html", "css", "scss", "sass", "svg",
}
_search_cache: dict[str, tuple[float, list[dict[str, object]]]] = {}
_detail_cache: dict[str, tuple[float, dict[str, object]]] = {}


class ClawHubCliError(RuntimeError):
    """Raised when ClawHub CLI command execution fails."""


def is_clawhub_cli_available() -> bool:
    """Return whether ``clawhub`` executable exists in PATH."""
    return _resolve_clawhub_bin() is not None


def get_clawhub_cli_status() -> dict[str, str | bool]:
    """Return CLI availability, path and version (if available)."""
    bin_path = _resolve_clawhub_bin()
    if not bin_path:
        return {"available": False, "path": "", "version": ""}
    version = ""
    with contextlib.suppress(Exception):
        version = _run([bin_path, "--cli-version"], env=os.environ.copy()).strip()
    return {"available": True, "path": bin_path, "version": version}


def _resolve_clawhub_bin() -> str | None:
    env_bin = os.environ.get("CLAWHUB_BIN", "").strip()
    if env_bin:
        p = Path(env_bin).expanduser()
        if p.is_file():
            return str(p)
    path_bin = shutil.which("clawhub")
    if path_bin:
        return path_bin
    repo_root = Path(__file__).resolve().parents[2]
    roots = [Path.cwd(), repo_root]
    for root in roots:
        for sub in ("node/bin/clawhub", ".local/npm-global/bin/clawhub"):
            candidate = root / sub
            if candidate.is_file():
                return str(candidate)
    return None


def _resolve_npm_bin() -> str | None:
    env_npm = os.environ.get("CLAWHUB_NPM_BIN", "").strip()
    if env_npm:
        p = Path(env_npm).expanduser()
        if p.is_file():
            return str(p)
    repo_root = Path(__file__).resolve().parents[2]
    roots = [Path.cwd(), repo_root]
    for root in roots:
        for sub in ("node/bin/npm", ".local/node/bin/npm"):
            candidate = root / sub
            if candidate.is_file():
                return str(candidate)
    path_npm = shutil.which("npm")
    if path_npm:
        return path_npm
    return None


def _build_env(
    *,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["CLAWHUB_REGISTRY"] = registry_url
    env["CLAWHUB_WORKDIR"] = str(workspace_dir)
    if api_token:
        env["CLAWHUB_TOKEN"] = api_token
    repo_root = Path(__file__).resolve().parents[2]
    embedded_node_bin = repo_root / "node" / "bin"
    local_node_bin = repo_root / ".local" / "node" / "bin"
    local_npm_bin = repo_root / ".local" / "npm-global" / "bin"
    path_parts = [
        str(embedded_node_bin),
        str(local_npm_bin),
        str(local_node_bin),
        env.get("PATH", ""),
    ]
    env["PATH"] = os.pathsep.join(part for part in path_parts if part)
    return env


def _run(args: list[str], *, env: dict[str, str]) -> str:
    bin_path = _resolve_clawhub_bin()
    if not bin_path:
        raise ClawHubCliError("未检测到 clawhub CLI")
    if args and args[0] == "clawhub":
        args = [bin_path, *args[1:]]
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=45,
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip() or "未知错误"
        raise ClawHubCliError(err)
    return (proc.stdout or "").strip()


def install_clawhub_cli() -> dict[str, str]:
    """Install clawhub CLI into project-embedded node/ prefix via npm."""
    npm_bin = _resolve_npm_bin()
    if not npm_bin:
        raise ClawHubCliError("未检测到 npm，请先安装 Node.js")

    repo_root = Path(__file__).resolve().parents[2]
    embedded_node = repo_root / "node"
    if embedded_node.is_dir():
        prefix_dir = embedded_node
    else:
        prefix_dir = repo_root / ".local" / "npm-global"
        prefix_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    path_parts = [
        str(embedded_node / "bin"),
        str(repo_root / ".local" / "npm-global" / "bin"),
        str(repo_root / ".local" / "node" / "bin"),
        env.get("PATH", ""),
    ]
    env["PATH"] = os.pathsep.join(p for p in path_parts if p)

    proc = subprocess.run(
        [npm_bin, "i", "-g", "clawhub", "--prefix", str(prefix_dir)],
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip() or "未知错误"
        raise ClawHubCliError(f"安装失败: {err}")

    status = get_clawhub_cli_status()
    if not bool(status["available"]):
        raise ClawHubCliError("安装完成但未找到 clawhub 可执行文件")
    return {
        "path": str(status["path"]),
        "version": str(status["version"]),
        "output": (proc.stdout or "").strip(),
    }


def get_clawhub_auth_status(
    *,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None = None,
) -> dict[str, str | bool]:
    """Return CLI auth status by running ``clawhub whoami``."""
    if not is_clawhub_cli_available():
        return {"logged_in": False, "message": "CLI 未安装"}
    env = _build_env(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    try:
        out = _run(["clawhub", "whoami"], env=env)
        msg = out.strip() or "已登录"
        return {"logged_in": True, "message": msg}
    except Exception as exc:
        raw = str(exc)
        cleaned = _clean_cli_error(raw)
        return {"logged_in": False, "message": cleaned}


def login_clawhub_cli(
    *,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None = None,
) -> dict[str, str | bool]:
    """Run ``clawhub login`` and return post-login status."""
    if not is_clawhub_cli_available():
        raise ClawHubCliError("CLI 未安装，请先点击“安装 CLI”")
    env = _build_env(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    proc = subprocess.run(
        [_resolve_clawhub_bin() or "clawhub", "login"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip() or "登录失败"
        raise ClawHubCliError(_clean_cli_error(err))
    status = get_clawhub_auth_status(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    return {
        "ok": bool(status["logged_in"]),
        "message": str(status["message"]),
        "output": (proc.stdout or "").strip(),
    }


def logout_clawhub_cli(
    *,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None = None,
) -> dict[str, str | bool]:
    """Run ``clawhub logout`` and return post-logout status."""
    if not is_clawhub_cli_available():
        raise ClawHubCliError("CLI 未安装，请先点击“安装 CLI”")
    env = _build_env(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    proc = subprocess.run(
        [_resolve_clawhub_bin() or "clawhub", "logout"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if proc.returncode != 0:
        err = _clean_cli_error((proc.stderr or proc.stdout).strip() or "退出登录失败")
        if "未登录" not in err and "Not logged in" not in err:
            raise ClawHubCliError(err)

    status = get_clawhub_auth_status(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    return {
        "ok": not bool(status["logged_in"]),
        "message": "已退出登录" if not bool(status["logged_in"]) else str(status["message"]),
        "output": (proc.stdout or "").strip(),
    }


def _clean_cli_error(message: str) -> str:
    lines = [ln.strip() for ln in message.splitlines() if ln.strip()]
    filtered = [
        ln
        for ln in lines
        if "ExperimentalWarning" not in ln and "--trace-warnings" not in ln
    ]
    text = "\n".join(filtered) if filtered else message.strip()
    if "Not logged in" in text or "Run: clawhub login" in text:
        return "未登录，请在终端执行: clawhub login"
    return text


def _decorate_results(
    *,
    items: list[dict[str, object]],
    registry_url: str,
) -> list[dict[str, object]]:
    base = registry_url.rstrip("/")
    out: list[dict[str, object]] = []
    for item in items:
        slug = str(item.get("slug", "")).strip()
        if not slug:
            continue
        detail_url = str(item.get("detail_url", "")).strip() or f"{base}/skills/{slug}"
        row = dict(item)
        row["detail_url"] = detail_url
        if "repo_url" not in row:
            row["repo_url"] = ""
        out.append(row)
    return out


def _sort_results_by_stats(items: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = list(items)
    ranked.sort(
        key=lambda row: (
            -_to_int(row.get("stars", 0)),
            -_to_int(row.get("downloads", 0)),
            str(row.get("slug", "")),
        )
    )
    return ranked


def _is_empty_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _merge_result_rows(  # pyright: ignore[reportUnusedFunction]
    *,
    base_items: list[dict[str, object]],
    extra_items: list[dict[str, object]],
) -> None:
    if not base_items or not extra_items:
        return
    by_slug: dict[str, dict[str, object]] = {}
    for row in extra_items:
        slug = str(row.get("slug", "")).strip()
        if slug:
            by_slug[slug] = row
    if not by_slug:
        return
    for row in base_items:
        slug = str(row.get("slug", "")).strip()
        if not slug:
            continue
        extra = by_slug.get(slug)
        if not extra:
            continue
        for key in (
            "name",
            "summary",
            "version",
            "detail_url",
            "repo_url",
            "stars",
            "downloads",
            "current_installs",
            "all_time_installs",
        ):
            if _is_empty_value(row.get(key)) and not _is_empty_value(extra.get(key)):
                row[key] = extra[key]


def _fill_guessed_install_stats(items: list[dict[str, object]]) -> None:
    for row in items:
        if _to_int_or_none(row.get("all_time_installs")) is not None:
            continue
        name = str(row.get("name", "")).strip()
        summary = str(row.get("summary", "")).strip()
        guessed = _parse_number_from_name(name) or _parse_number_from_text(summary)
        if guessed > 0:
            row["all_time_installs"] = guessed


def _search_cache_key(
    *,
    query: str,
    registry_url: str,
    api_token: str | None,
    limit: int,
) -> str:
    token_key = "token" if api_token else "anon"
    return f"{registry_url.rstrip('/')}\n{query.strip().lower()}\n{limit}\n{token_key}"


def _clone_rows(items: list[dict[str, object]]) -> list[dict[str, object]]:
    return [dict(item) for item in items]


def _cache_get(
    cache: dict[str, tuple[float, list[dict[str, object]]]],
    key: str,
    ttl_seconds: float,
) -> list[dict[str, object]] | None:
    entry = cache.get(key)
    if entry is None:
        return None
    expires_at, items = entry
    if expires_at < time.monotonic():
        cache.pop(key, None)
        return None
    return _clone_rows(items)


def _cache_put(
    cache: dict[str, tuple[float, list[dict[str, object]]]],
    key: str,
    items: list[dict[str, object]],
    ttl_seconds: float,
) -> None:
    cache[key] = (time.monotonic() + ttl_seconds, _clone_rows(items))


def _detail_cache_key(*, registry_url: str, slug: str, api_token: str | None) -> str:
    token_key = "token" if api_token else "anon"
    return f"{registry_url.rstrip('/')}\n{slug.strip().lower()}\n{token_key}"


def _detail_cache_get(
    *,
    registry_url: str,
    slug: str,
    api_token: str | None,
) -> dict[str, object] | None:
    cache_key = _detail_cache_key(
        registry_url=registry_url,
        slug=slug,
        api_token=api_token,
    )
    cached = _detail_cache.get(cache_key)
    if cached is None:
        return None
    expires_at, payload = cached
    if expires_at < time.monotonic():
        _detail_cache.pop(cache_key, None)
        return None
    return dict(payload)


def _detail_cache_put(
    *,
    registry_url: str,
    slug: str,
    api_token: str | None,
    payload: dict[str, object],
) -> None:
    cache_key = _detail_cache_key(
        registry_url=registry_url,
        slug=slug,
        api_token=api_token,
    )
    _detail_cache[cache_key] = (
        time.monotonic() + _DETAIL_CACHE_TTL_SECONDS,
        dict(payload),
    )


def _to_int(value: object) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        return int(float(str(value)))
    except Exception:
        return 0


def _to_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return _to_int(value)
    raw = str(value).strip()
    if not raw:
        return None
    scale = 1.0
    suffix = raw[-1].lower()
    if suffix in {"k", "m", "b"}:
        raw = raw[:-1].strip()
        scale = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}[suffix]
    raw = raw.replace(",", "")
    if raw.count(".") > 1:
        raw = raw.replace(".", "")
    try:
        return int(float(raw) * scale)
    except Exception:
        return None


def _pick_first_int(*values: object) -> int | None:
    for value in values:
        parsed = _to_int_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _walk_dict_values(value: object) -> list[tuple[list[str], object]]:
    out: list[tuple[list[str], object]] = []

    def _walk(node: object, path: list[str]) -> None:
        if isinstance(node, dict):
            node_dict = cast(dict[object, object], node)
            for k, v in node_dict.items():
                if isinstance(k, str):
                    _walk(v, [*path, k])
            return
        if isinstance(node, list):
            node_list = cast(list[object], node)
            for idx, v in enumerate(node_list):
                _walk(v, [*path, str(idx)])
            return
        out.append((path, node))

    _walk(value, [])
    return out


def _walk_dict_nodes(value: object) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            node_dict = cast(dict[object, object], node)
            normalized = {str(k): v for k, v in node_dict.items() if isinstance(k, str)}
            out.append(normalized)
            for v in node_dict.values():
                _walk(v)
        elif isinstance(node, list):
            node_list = cast(list[object], node)
            for v in node_list:
                _walk(v)

    _walk(value)
    return out


def _find_int_in_tree(source: dict[str, object], keys: set[str]) -> int | None:
    for path, value in _walk_dict_values(source):
        if not path:
            continue
        last = _norm_key(path[-1])
        if last in keys:
            parsed = _to_int_or_none(value)
            if parsed is not None:
                return parsed
    return None


def _extract_stats_from_metric_nodes(source: dict[str, object]) -> dict[str, int | None]:
    out: dict[str, int | None] = {
        "stars": None,
        "downloads": None,
        "current_installs": None,
        "all_time_installs": None,
    }
    for node in _walk_dict_nodes(source):
        keys = dict(node)
        if not keys:
            continue
        marker_parts: list[str] = []
        for marker_key in ("name", "key", "metric", "label", "title", "type"):
            marker_val = keys.get(marker_key)
            if isinstance(marker_val, str):
                marker_parts.append(marker_val.lower())
        marker = " ".join(marker_parts)
        if not marker:
            continue
        candidate = _pick_first_int(
            keys.get("value"),
            keys.get("count"),
            keys.get("total"),
            keys.get("current"),
            keys.get("allTime"),
            keys.get("all_time"),
            keys.get("installs"),
            keys.get("downloads"),
            keys.get("stars"),
        )
        if candidate is None:
            continue
        if "star" in marker and out["stars"] is None:
            out["stars"] = candidate
            continue
        if "download" in marker and out["downloads"] is None:
            out["downloads"] = candidate
            continue
        if "current" in marker and "install" in marker and out["current_installs"] is None:
            out["current_installs"] = candidate
            continue
        if ("alltime" in _norm_key(marker) or "all-time" in marker or "lifetime" in marker) and (
            "install" in marker
        ):
            if out["all_time_installs"] is None:
                out["all_time_installs"] = candidate
            continue
        if "install" in marker and out["all_time_installs"] is None:
            out["all_time_installs"] = candidate
    return out


def _extract_stats(source: dict[str, object]) -> dict[str, int | None]:
    stats_raw = source.get("stats")
    stats = cast(dict[str, object], stats_raw) if isinstance(stats_raw, dict) else {}
    installs = _pick_first_int(
        source.get("installs"),
        stats.get("installs"),
        _find_int_in_tree(
            stats,
            {
                "installs",
                "installcount",
                "totalinstalls",
                "installstotal",
                "installsalltime",
            },
        ),
    )
    stars = _pick_first_int(
        source.get("stars"),
        source.get("starCount"),
        source.get("starsCount"),
        source.get("stars_count"),
        stats.get("stars"),
        stats.get("starCount"),
        stats.get("starsCount"),
        _find_int_in_tree(
            stats,
            {"stars", "starcount", "starscount", "ratingcount"},
        ),
    )
    downloads = _pick_first_int(
        source.get("downloads"),
        source.get("downloadCount"),
        source.get("downloadsCount"),
        source.get("allDownloads"),
        stats.get("downloads"),
        stats.get("downloadCount"),
        stats.get("downloadsCount"),
        stats.get("allDownloads"),
        _find_int_in_tree(
            stats,
            {"downloads", "downloadcount", "downloadsalltime", "alldownloads"},
        ),
    )
    current_installs = _pick_first_int(
        source.get("current_installs"),
        source.get("currentInstalls"),
        source.get("currentInstallCount"),
        source.get("currentInstall"),
        source.get("activeInstalls"),
        stats.get("current_installs"),
        stats.get("currentInstalls"),
        stats.get("currentInstallCount"),
        stats.get("currentInstall"),
        stats.get("activeInstalls"),
        _find_int_in_tree(
            stats,
            {
                "currentinstalls",
                "installscurrent",
                "currentinstallcount",
                "activeinstalls",
                "current",
            },
        ),
        installs,
    )
    all_time_installs = _pick_first_int(
        source.get("all_time_installs"),
        source.get("allTimeInstalls"),
        source.get("allTimeInstallCount"),
        source.get("installsTotal"),
        source.get("totalInstalls"),
        source.get("totalInstallCount"),
        source.get("lifetimeInstalls"),
        stats.get("all_time_installs"),
        stats.get("allTimeInstalls"),
        stats.get("allTimeInstallCount"),
        stats.get("installsTotal"),
        stats.get("totalInstalls"),
        stats.get("totalInstallCount"),
        stats.get("lifetimeInstalls"),
        _find_int_in_tree(
            stats,
            {
                "alltimeinstalls",
                "installsalltime",
                "alltimeinstallcount",
                "totalinstalls",
                "installstotal",
                "lifetimeinstalls",
                "alltime",
                "total",
            },
        ),
        installs,
    )
    node_stats = _extract_stats_from_metric_nodes(source)
    return {
        "stars": stars if stars is not None else node_stats["stars"],
        "downloads": downloads if downloads is not None else node_stats["downloads"],
        "current_installs": (
            current_installs if current_installs is not None else node_stats["current_installs"]
        ),
        "all_time_installs": (
            all_time_installs if all_time_installs is not None else node_stats["all_time_installs"]
        ),
    }


def _has_any_stats(item: dict[str, object]) -> bool:  # pyright: ignore[reportUnusedFunction]
    for key in ("stars", "downloads", "current_installs", "all_time_installs"):
        if _to_int_or_none(item.get(key)) is not None:
            return True
    return False


def _has_primary_stats(item: dict[str, object]) -> bool:
    stars = _to_int_or_none(item.get("stars"))
    downloads = _to_int_or_none(item.get("downloads"))
    return stars is not None and downloads is not None


def _extract_stats_from_html(html: str) -> dict[str, int | None]:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return {
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        }
    m = re.search(
        (
            r"([0-9][0-9.,kKmMbB]*)\s*[·•]\s*([0-9][0-9.,kKmMbB]*)\s*[·•]\s*"
            r"([0-9][0-9.,kKmMbB]*)\s*current installs\s*[·•]\s*"
            r"([0-9][0-9.,kKmMbB]*)\s*all-time installs"
        ),
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return {
            "stars": _to_int_or_none(m.group(1)),
            "downloads": _to_int_or_none(m.group(2)),
            "current_installs": _to_int_or_none(m.group(3)),
            "all_time_installs": _to_int_or_none(m.group(4)),
        }
    m_zh = re.search(
        r"([0-9][0-9.,kKmMbB]*)\s*[·•]\s*([0-9][0-9.,kKmMbB]*)\s*[·•]\s*([0-9][0-9.,kKmMbB]*)\s*当前安装\s*[·•]\s*([0-9][0-9.,kKmMbB]*)\s*总安装",
        text,
        flags=re.IGNORECASE,
    )
    if m_zh:
        return {
            "stars": _to_int_or_none(m_zh.group(1)),
            "downloads": _to_int_or_none(m_zh.group(2)),
            "current_installs": _to_int_or_none(m_zh.group(3)),
            "all_time_installs": _to_int_or_none(m_zh.group(4)),
        }
    return {
        "stars": None,
        "downloads": None,
        "current_installs": None,
        "all_time_installs": None,
    }


def _parse_number_from_name(name: str) -> int:
    """Extract trailing '(3.671)' style counts from display name."""
    m = re.search(r"\(([\d.,]+)\)\s*$", name.strip())
    if not m:
        return 0
    raw = m.group(1).strip()
    # Handle common locale thousands separators used by registry labels.
    if "." in raw and "," not in raw and raw.count(".") >= 1:
        raw = raw.replace(".", "")
    raw = raw.replace(",", "")
    return _to_int(raw)


def _parse_number_from_text(text: str) -> int:
    """Extract '(3.671)' style count from any position in text."""
    m = re.search(r"\(([\d.,]+)\)", text)
    if not m:
        return 0
    raw = m.group(1).strip()
    if "." in raw and "," not in raw and raw.count(".") >= 1:
        raw = raw.replace(".", "")
    raw = raw.replace(",", "")
    return _to_int(raw)


def search_skills(
    *,
    query: str,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None = None,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Search ClawHub skills via HTTP API, then enrich via HTTP detail API."""
    resolved_api_token = _resolve_clawhub_api_token(api_token)
    search_limit = max(1, min(limit, 200))
    cache_key = _search_cache_key(
        query=query,
        registry_url=registry_url,
        api_token=resolved_api_token,
        limit=search_limit,
    )
    cached = _cache_get(_search_cache, cache_key, _SEARCH_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached[:limit]

    try:
        items = _search_via_http(
            query=query,
            registry_url=registry_url,
            api_token=resolved_api_token,
            limit=search_limit,
        )
    except Exception as exc:
        raise ClawHubCliError(_clean_cli_error(str(exc))) from exc
    if items:
        decorated = _decorate_results(items=items, registry_url=registry_url)[:limit]
        _cache_put(_search_cache, cache_key, decorated, _SEARCH_CACHE_TTL_SECONDS)
        return decorated
    _cache_put(_search_cache, cache_key, [], _SEARCH_CACHE_TTL_SECONDS)
    return []


def install_skill(
    *,
    slug: str,
    version: str | None,
    registry_url: str,
    workspace_dir: Path,
    install_dir: Path,
    api_token: str | None = None,
) -> str:
    """Install a ClawHub skill by slug (CLI first, HTTP fallback)."""
    _validate_slug(slug)
    resolved_api_token = _resolve_clawhub_api_token(api_token)
    if not is_clawhub_cli_available():
        _install_via_http(
            slug=slug,
            version=version,
            registry_url=registry_url,
            install_dir=install_dir,
            api_token=resolved_api_token,
        )
        return f"installed via http: {slug}"

    env = _build_env(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=resolved_api_token,
    )

    base_args = ["clawhub", "install", slug]
    if version:
        base_args.extend(["--version", version])

    attempts = [
        [*base_args, "--dir", str(install_dir), "--workdir", str(workspace_dir)],
        [*base_args, "--workdir", str(workspace_dir)],
        base_args,
    ]

    last_error = "未知错误"
    for args in attempts:
        try:
            return _run(args, env=env)
        except ClawHubCliError as exc:
            last_error = str(exc)

    _install_via_http(
        slug=slug,
        version=version,
        registry_url=registry_url,
        install_dir=install_dir,
        api_token=resolved_api_token,
    )
    return f"installed via http fallback: {slug} (cli_error={last_error})"


def publish_installed_skill(
    *,
    skill_dir: Path,
    skill_slug: str,
    skill_version: str | None,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None = None,
) -> str:
    """Publish one local installed skill to ClawHub via HTTP publish."""
    target = skill_dir.resolve()
    if not target.is_dir() or not (target / "SKILL.md").is_file():
        raise ClawHubCliError(f"无效技能目录: {target}")
    raw_slug = skill_slug.strip()
    _validate_slug(raw_slug)
    version = _normalize_publish_version(skill_version) or _resolve_publish_version(target)
    _ = workspace_dir
    return _publish_via_http(
        skill_dir=target,
        skill_slug=raw_slug,
        skill_version=version,
        registry_url=registry_url,
        api_token=api_token,
    )


def _resolve_publish_version(skill_dir: Path) -> str:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return "0.1.0"
    with contextlib.suppress(Exception):
        raw = skill_md.read_text(encoding="utf-8")
        fm_match = _FRONTMATTER_RE.match(raw)
        if not fm_match:
            return "0.1.0"
        frontmatter = fm_match.group(1)
        m = re.search(r"(?mi)^\s*version\s*:\s*['\"]?([^\n\"']+)['\"]?\s*$", frontmatter)
        if not m:
            return "0.1.0"
        ver = m.group(1).strip()
        if ver.lower().startswith("v"):
            ver = ver[1:].strip()
        if _SEMVER_RE.match(ver):
            return ver
    return "0.1.0"


def _normalize_publish_version(version: str | None) -> str | None:
    if version is None:
        return None
    raw = version.strip()
    if not raw:
        return None
    if raw.lower().startswith("v"):
        raw = raw[1:].strip()
    if _SEMVER_RE.match(raw):
        return raw
    raise ClawHubCliError(
        "发布失败：版本号不合法。请输入 x.y.z（例如 0.1.0）。"
    )


def _validate_slug(slug: str) -> None:
    raw = slug.strip()
    if not raw:
        raise ClawHubCliError("slug 不能为空")
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ClawHubCliError(f"非法 slug: {raw}")


def _title_case_slug(name: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"[-_]+", name) if part.strip())


def _resolve_clawhub_global_config_path() -> Path:
    override = os.environ.get("CLAWHUB_CONFIG_PATH", "").strip() or os.environ.get(
        "CLAWDHUB_CONFIG_PATH", ""
    ).strip()
    if override:
        return Path(override).expanduser()
    home = Path.home()
    system = platform.system()
    if system == "Darwin":
        base = home / "Library" / "Application Support"
    elif system == "Windows":
        app_data = os.environ.get("APPDATA", "").strip()
        base = Path(app_data).expanduser() if app_data else home / "AppData" / "Roaming"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
        base = Path(xdg).expanduser() if xdg else home / ".config"
    clawhub_path = base / "clawhub" / "config.json"
    legacy_path = base / "clawdhub" / "config.json"
    if clawhub_path.is_file():
        return clawhub_path
    if legacy_path.is_file():
        return legacy_path
    return clawhub_path


def _resolve_clawhub_global_config_paths() -> list[Path]:
    override = os.environ.get("CLAWHUB_CONFIG_PATH", "").strip() or os.environ.get(
        "CLAWDHUB_CONFIG_PATH", ""
    ).strip()
    if override:
        return [Path(override).expanduser()]
    home = Path.home()
    system = platform.system()
    if system == "Darwin":
        base = home / "Library" / "Application Support"
    elif system == "Windows":
        app_data = os.environ.get("APPDATA", "").strip()
        base = Path(app_data).expanduser() if app_data else home / "AppData" / "Roaming"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
        base = Path(xdg).expanduser() if xdg else home / ".config"
    return [
        base / "clawhub" / "config.json",
        base / "clawdhub" / "config.json",
    ]


def _read_clawhub_global_config() -> dict[str, object]:
    merged: dict[str, object] = {}
    for path in _resolve_clawhub_global_config_paths():
        if not path.is_file():
            continue
        with contextlib.suppress(Exception):
            raw = path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                continue
            parsed_map = {
                str(key): value
                for key, value in cast(dict[object, object], parsed).items()
                if isinstance(key, str)
            }
            if path.parent.name == "clawdhub":
                for key, value in parsed_map.items():
                    if key not in merged or _is_empty_value(merged[key]):
                        merged[key] = value
            else:
                merged.update(parsed_map)
    return merged


def _resolve_clawhub_api_token(api_token: str | None) -> str | None:
    token = (api_token or "").strip()
    if token:
        return token
    cfg = _read_clawhub_global_config()
    fallback = str(cfg.get("token", "")).strip()
    return fallback or None


def _iter_publish_files(skill_dir: Path) -> list[tuple[str, bytes, str]]:
    files: list[tuple[str, bytes, str]] = []
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(skill_dir).as_posix()
        if (
            rel_path.startswith(".git/")
            or rel_path.startswith("node_modules/")
            or rel_path.startswith(".clawhub/")
            or rel_path.startswith(".clawdhub/")
        ):
            continue
        suffix = path.suffix.lower().lstrip(".")
        if suffix not in _TEXT_FILE_EXTENSIONS:
            continue
        content_type = mimetypes.guess_type(path.name)[0] or "text/plain"
        files.append((rel_path, path.read_bytes(), content_type))
    return files


def _publish_via_http(
    *,
    skill_dir: Path,
    skill_slug: str,
    skill_version: str,
    registry_url: str,
    api_token: str | None,
) -> str:
    cfg = _read_clawhub_global_config()
    token = api_token or str(cfg.get("token", "")).strip()
    if not token:
        raise ClawHubCliError("发布失败：未找到 ClawHub 登录令牌，请先重新登录 ClawHub。")

    files = _iter_publish_files(skill_dir)
    if not files:
        raise ClawHubCliError("发布失败：技能目录中没有可上传的文本文件。")
    if not any(rel.lower() in {"skill.md", "skills.md"} for rel, _, _ in files):
        raise ClawHubCliError("发布失败：技能目录缺少 SKILL.md。")

    payload = {
        "slug": skill_slug,
        "displayName": _title_case_slug(skill_dir.name),
        "version": skill_version,
        "changelog": "",
        "tags": ["latest"],
        "acceptLicenseTerms": True,
    }
    upload_files: list[tuple[str, tuple[str, bytes, str]]] = []
    for rel_path, content, content_type in files:
        upload_files.append(("files", (rel_path, content, content_type)))

    with httpx.Client(timeout=45.0, follow_redirects=True) as client:
        response = client.post(
            f"{registry_url.rstrip('/')}/api/v1/skills",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            },
            data={"payload": json.dumps(payload, ensure_ascii=False)},
            files=upload_files,
        )
    if response.status_code >= 400:
        detail = response.text.strip() or f"HTTP {response.status_code}"
        raise ClawHubCliError(_clean_cli_error(detail))
    with contextlib.suppress(Exception):
        data = response.json()
        if isinstance(data, dict):
            version_id = str(data.get("versionId", "")).strip()
            if version_id:
                return f"OK. Published {skill_slug}@{skill_version} ({version_id})"
    return response.text.strip() or f"OK. Published {skill_slug}@{skill_version}"


def _search_via_http(
    *,
    query: str,
    registry_url: str,
    api_token: str | None,
    limit: int = 100,
) -> list[dict[str, object]]:
    base = registry_url.rstrip("/") + "/"
    headers: dict[str, str] = {"Accept": "application/json"}
    resolved_api_token = _resolve_clawhub_api_token(api_token)
    if resolved_api_token:
        headers["Authorization"] = f"Bearer {resolved_api_token}"
    with httpx.Client(timeout=12, follow_redirects=True) as client:
        resp = client.get(
            f"{base}api/v1/search",
            params={"q": query, "limit": max(1, min(limit, 200))},
            headers=headers,
        )
        if resp.status_code >= 400:
            raise ClawHubCliError(f"搜索失败: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json()
    raw_results = data.get("results", [])
    results: list[dict[str, object]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        item_map = {
            str(key): value
            for key, value in cast(dict[object, object], item).items()
            if isinstance(key, str)
        }
        slug = str(item_map.get("slug", "")).strip()
        if not slug:
            continue
        name = str(item_map.get("displayName", slug)).strip() or slug
        summary = str(item_map.get("summary", "")).strip()
        guessed_total = _parse_number_from_name(name) or _parse_number_from_text(summary)
        stats = _extract_stats(item_map)
        if stats["all_time_installs"] is None and guessed_total > 0:
            stats["all_time_installs"] = guessed_total
        results.append(
            {
                "slug": slug,
                "name": name,
                "summary": summary,
                "version": str(item_map.get("version", "")).strip(),
                "stars": stats["stars"],
                "downloads": stats["downloads"],
                "current_installs": stats["current_installs"],
                "all_time_installs": stats["all_time_installs"],
                "detail_url": str(
                    item_map.get("url", "")
                    or item_map.get("detailUrl", "")
                    or item_map.get("pageUrl", "")
                    or item_map.get("permalink", "")
                ).strip(),
                "repo_url": str(
                    item_map.get("repoUrl", "")
                    or item_map.get("repository", "")
                    or item_map.get("repo", "")
                ).strip(),
            }
        )
    _fill_guessed_install_stats(results)
    _enrich_results_with_skill_details(
        items=results,
        registry_url=registry_url,
        api_token=api_token,
        max_items=min(limit, _DETAIL_ENRICH_MAX_ITEMS),
    )
    return _sort_results_by_stats(results)


def _enrich_results_with_skill_details(
    *,
    items: list[dict[str, object]],
    registry_url: str,
    api_token: str | None,
    max_items: int,
) -> None:
    if not items or max_items <= 0:
        return
    candidates: list[dict[str, object]] = []
    for row in items:
        if len(candidates) >= max_items:
            break
        slug = str(row.get("slug", "")).strip()
        if not slug or _has_primary_stats(row):
            continue
        cached = _detail_cache_get(
            registry_url=registry_url,
            slug=slug,
            api_token=api_token,
        )
        if cached is not None:
            row.update(cached)
            continue
        candidates.append(row)

    if not candidates:
        return

    max_workers = min(_DETAIL_ENRICH_MAX_WORKERS, len(candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _fetch_skill_detail_enrichment,
                slug=str(row.get("slug", "")).strip(),
                detail_url=str(row.get("detail_url", "")).strip(),
                registry_url=registry_url,
                api_token=api_token,
                include_cli=False,
            ): row
            for row in candidates
        }
        for future in as_completed(future_map):
            row = future_map[future]
            try:
                detail_updates = future.result()
            except Exception:
                continue
            if not detail_updates:
                continue
            row.update(detail_updates)
            slug = str(row.get("slug", "")).strip()
            if slug:
                _detail_cache_put(
                    registry_url=registry_url,
                    slug=slug,
                    api_token=api_token,
                    payload=detail_updates,
                )

    cli_checked = 0
    for row in candidates:
        if _has_primary_stats(row):
            continue
        if cli_checked >= _CLI_INSPECT_FALLBACK_MAX_ITEMS:
            break
        cli_checked += 1
        slug = str(row.get("slug", "")).strip()
        if not slug:
            continue
        detail_updates: dict[str, object] = {}
        cli_stats = _inspect_skill_stats_via_cli(
            slug=slug,
            registry_url=registry_url,
            api_token=api_token,
        )
        for key, value in cli_stats.items():
            if value is not None:
                row[key] = value
                detail_updates[key] = value
        if detail_updates:
            _detail_cache_put(
                registry_url=registry_url,
                slug=slug,
                api_token=api_token,
                payload=detail_updates,
            )


def _inspect_skill_stats_via_cli(
    *,
    slug: str,
    registry_url: str,
    api_token: str | None,
) -> dict[str, int | None]:
    if not is_clawhub_cli_available():
        return {
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        }
    env = _build_env(
        registry_url=registry_url,
        workspace_dir=Path.cwd(),
        api_token=api_token,
    )
    try:
        raw = _run(["clawhub", "inspect", slug, "--json"], env=env)
        payload = json.loads(raw)
    except Exception:
        return {
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        }
    if not isinstance(payload, dict):
        return {
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        }
    payload_map = cast(dict[object, object], payload)
    skill = payload_map.get("skill")
    if not isinstance(skill, dict):
        return {
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        }
    skill_map = {
        str(key): value
        for key, value in cast(dict[object, object], skill).items()
        if isinstance(key, str)
    }
    return _extract_stats(skill_map)


def _fetch_skill_detail_enrichment(
    *,
    slug: str,
    detail_url: str,
    registry_url: str,
    api_token: str | None,
    include_cli: bool = False,
) -> dict[str, object]:
    if not slug:
        return {}
    base = registry_url.rstrip("/") + "/"
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    detail_updates: dict[str, object] = {}
    with httpx.Client(timeout=8, follow_redirects=True) as client:
        try:
            resp = client.get(
                f"{base}api/v1/skills/{quote(slug, safe='')}",
                headers=headers,
            )
            if resp.status_code < 400:
                payload = resp.json()
                if isinstance(payload, dict):
                    payload_map = cast(dict[object, object], payload)
                    skill = payload_map.get("skill")
                    if isinstance(skill, dict):
                        skill_map = {
                            str(key): value
                            for key, value in cast(dict[object, object], skill).items()
                            if isinstance(key, str)
                        }
                        stats = _extract_stats(skill_map)
                        for key, value in stats.items():
                            if value is not None:
                                detail_updates[key] = value
                    latest = payload_map.get("latestVersion")
                    if isinstance(latest, dict):
                        latest_map = {
                            str(key): value
                            for key, value in cast(dict[object, object], latest).items()
                            if isinstance(key, str)
                        }
                        ver = str(latest_map.get("version", "")).strip()
                        if ver:
                            detail_updates["version"] = ver
        except Exception:
            pass

        has_stats = any(
            detail_updates.get(k) is not None
            for k in ("stars", "downloads", "current_installs", "all_time_installs")
        )
        if has_stats:
            return detail_updates

        target_detail_url = detail_url or f"{base}skills/{quote(slug, safe='')}"
        try:
            page_resp = client.get(
                target_detail_url,
                headers={"Accept": "text/html,*/*"},
            )
            if page_resp.status_code < 400:
                page_stats = _extract_stats_from_html(page_resp.text)
                for key, value in page_stats.items():
                    if value is not None:
                        detail_updates[key] = value
        except Exception:
            pass

    if include_cli and not any(
        detail_updates.get(k) is not None
        for k in ("stars", "downloads", "current_installs", "all_time_installs")
    ):
        cli_stats = _inspect_skill_stats_via_cli(
            slug=slug,
            registry_url=registry_url,
            api_token=api_token,
        )
        for key, value in cli_stats.items():
            if value is not None:
                detail_updates[key] = value

    return detail_updates


def _install_via_http(
    *,
    slug: str,
    version: str | None,
    registry_url: str,
    install_dir: Path,
    api_token: str | None,
) -> None:
    base = registry_url.rstrip("/") + "/"
    headers: dict[str, str] = {}
    resolved_api_token = _resolve_clawhub_api_token(api_token)
    if resolved_api_token:
        headers["Authorization"] = f"Bearer {resolved_api_token}"

    with httpx.Client(timeout=25, follow_redirects=True) as client:
        params = {"slug": slug}
        if version:
            params["version"] = version
        resp = client.get(f"{base}api/v1/download", params=params, headers=headers)
        if resp.status_code >= 400:
            raise ClawHubCliError(f"下载失败: HTTP {resp.status_code} {resp.text[:200]}")
        payload = resp.content

    install_dir.mkdir(parents=True, exist_ok=True)
    target = install_dir / slug
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            for member in zf.infolist():
                name = member.filename
                if not name or name.endswith("/"):
                    continue
                rel = Path(name)
                parts = rel.parts
                if any(p in {"", ".", ".."} for p in parts):
                    continue
                # remove optional top-level directory in archive
                rel2 = Path(*parts[1:]) if len(parts) > 1 else rel
                if not rel2.parts:
                    continue
                dst = target / rel2
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, dst.open("wb") as out:
                    out.write(src.read())
    except zipfile.BadZipFile as exc:
        raise ClawHubCliError("下载内容不是有效 ZIP 包") from exc

    skill_md = target / "SKILL.md"
    if not skill_md.is_file():
        raise ClawHubCliError("安装失败: 压缩包中未找到 SKILL.md")
