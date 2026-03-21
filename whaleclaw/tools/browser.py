"""Browser control tool powered by Playwright (uses local Chrome).

Supports two modes:
- launch: start a fresh Chrome instance (default, for image search etc.)
- cdp: connect to an existing Chrome via CDP (for taking over logged-in sessions)
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from whaleclaw.config.paths import CONFIG_FILE, WHALECLAW_HOME
from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

_ACTIONS = [
    "navigate",
    "screenshot",
    "click",
    "type",
    "get_text",
    "evaluate",
    "upload",
    "search_images",
    "back",
    "close",
]

_SCREENSHOT_DIR = WHALECLAW_HOME / "screenshots"
_DOWNLOAD_DIR = WHALECLAW_HOME / "downloads"

_VIEWPORT = {"width": 1280, "height": 800}
_TIMEOUT = 15_000
_GENERIC_IMAGE_QUERIES = {
    "图",
    "图片",
    "照片",
    "近照",
    "高清",
    "最新",
    "随便",
    "来一张",
    "1",
    "2",
    "3",
    "?",
    "？",
}
_IMAGE_INTENT_HINTS = (
    "近照",
    "高清",
    "写真",
    "活动",
    "机场",
    "红毯",
    "肖像",
    "人像",
    "photo",
    "portrait",
    "recent",
    "latest",
    "hd",
)

_BING_JS = """
() => {
    const imgs = document.querySelectorAll('a.iusc');
    const urls = [];
    for (const a of imgs) {
        try {
            const m = JSON.parse(a.getAttribute('m') || '{}');
            if (m.murl) urls.push(m.murl);
        } catch {}
    }
    if (!urls.length) {
        for (const img of document.querySelectorAll('img.mimg, img[src^="http"]')) {
            const s = img.src || '';
            if (s.startsWith('http') && !s.includes('bing.com/th?') && img.naturalWidth > 60)
                urls.push(s);
        }
    }
    return urls.slice(0, 8);
}
"""

_BAIDU_JS = """
() => {
    const urls = [];
    for (const img of document.querySelectorAll('img.main_img, img[data-imgurl]')) {
        const u = img.getAttribute('data-imgurl') || img.src || '';
        if (u.startsWith('http') && !u.includes('baidu.com/img/'))
            urls.push(u);
    }
    if (!urls.length) {
        for (const a of document.querySelectorAll('a[href*="objurl"]')) {
            const m = new URLSearchParams(a.href.split('?')[1] || '');
            const ou = m.get('objurl');
            if (ou) urls.push(decodeURIComponent(ou));
        }
    }
    return urls.slice(0, 8);
}
"""

_GOOGLE_JS = """
() => {
    const imgs = document.querySelectorAll('img[src^="http"]');
    const urls = [];
    for (const img of imgs) {
        const src = img.src;
        if (src.includes('gstatic.com/images') || src.includes('google.com/logos'))
            continue;
        if (img.naturalWidth > 80 && img.naturalHeight > 80)
            urls.push(src);
    }
    return urls.slice(0, 5);
}
"""

_IMAGE_ENGINES: list[tuple[str, Callable[[str], str], str]] = [
    (
        "google",
        lambda q: f"https://www.google.com/search?q={q}&tbm=isch&udm=2",
        _GOOGLE_JS,
    ),
    (
        "bing",
        lambda q: f"https://www.bing.com/images/search?q={q}&form=HDRSC2",
        _BING_JS,
    ),
    (
        "baidu",
        lambda q: f"https://image.baidu.com/search/index?tn=baiduimage&word={q}",
        _BAIDU_JS,
    ),
]


class _BrowserCfg:
    """Parsed browser config from whaleclaw.json."""

    __slots__ = ("mode", "cdp_url", "headless")

    mode: Literal["launch", "cdp"]
    cdp_url: str
    headless: bool

    def __init__(
        self,
        mode: Literal["launch", "cdp"] = "launch",
        cdp_url: str = "http://localhost:9222",
        headless: bool = False,
    ) -> None:
        self.mode = mode
        self.cdp_url = cdp_url
        self.headless = headless

    @classmethod
    def from_config_file(cls) -> _BrowserCfg:
        """Load browser settings from whaleclaw.json.

        优先读 ``browser`` 顶层字段（新格式），
        兼容旧格式 ``plugins.browser.visible``。
        """
        try:
            if not CONFIG_FILE.is_file():
                return cls()
            raw_obj: object = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if not isinstance(raw_obj, dict):
                return cls()
            raw = cast(dict[str, object], raw_obj)

            browser_obj = raw.get("browser")
            if isinstance(browser_obj, dict):
                browser = cast(dict[str, object], browser_obj)
                mode_val = browser.get("mode", "launch")
                mode: Literal["launch", "cdp"] = (
                    "cdp" if mode_val == "cdp" else "launch"
                )
                cdp_url = str(browser.get("cdp_url", "http://localhost:9222"))
                visible = browser.get("visible", True)
                return cls(mode=mode, cdp_url=cdp_url, headless=not bool(visible))

            # 兼容旧 plugins.browser.visible 格式
            plugins_obj = raw.get("plugins")
            if isinstance(plugins_obj, dict):
                plugins = cast(dict[str, object], plugins_obj)
                old_cfg_obj = plugins.get("browser")
                if isinstance(old_cfg_obj, dict):
                    old_cfg = cast(dict[str, object], old_cfg_obj)
                    visible = old_cfg.get("visible", True)
                    return cls(headless=not bool(visible))

            return cls()
        except Exception:
            return cls()


class BrowserTool(Tool):
    """Browser control tool — uses local Chrome via Playwright.

    Supports two connection modes:
    - launch: start a fresh Chrome (for search, scraping public pages)
    - cdp: connect to user's existing Chrome via CDP (for logged-in sessions)
    """

    def __init__(self) -> None:
        self._browser: Any = None
        self._page: Any = None
        self._playwright: Any = None
        self._mode: Literal["launch", "cdp"] = "launch"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="browser",
            description=(
                "Control a real Chrome browser. Actions: "
                "navigate(url) — open URL; "
                "screenshot — capture current page; "
                "click(selector) — click element; "
                "type(selector, text) — type into input; "
                "get_text(selector?) — extract page/element text; "
                "evaluate(script) — run JavaScript; "
                "upload(selector, file_paths) — upload local files to a file input; "
                "search_images(query) — image search and download one image per query; "
                "back — go back; "
                "close — close browser."
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Action to perform.",
                    required=True,
                    enum=_ACTIONS,
                ),
                ToolParameter(
                    name="url",
                    type="string",
                    description="URL for navigate action.",
                    required=False,
                ),
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector for click/type/get_text.",
                    required=False,
                ),
                ToolParameter(
                    name="text",
                    type="string",
                    description=(
                        "Text to type, or search query for search_images. "
                        "For image search use explicit keywords, e.g. "
                        "'杨幂 近照 高清 人像' or 'cute cat photo hd'."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="script",
                    type="string",
                    description="JavaScript code for evaluate action.",
                    required=False,
                ),
                ToolParameter(
                    name="file_paths",
                    type="string",
                    description=(
                        "Comma-separated local file paths for upload action. "
                        "E.g. '/path/to/a.png,/path/to/b.jpg'"
                    ),
                    required=False,
                ),
            ],
        )

    async def _ensure_browser(self) -> Any:
        """Ensure a browser page is available.

        In ``cdp`` mode, connects to user's existing Chrome via CDP.
        In ``launch`` mode (or CDP fallback), starts a fresh Chrome.

        每次调用都会重新读取配置。如果配置切换到 cdp 而当前实例是
        launch 模式，会先关掉旧浏览器再连接 CDP，确保优先使用辅助浏览器。
        """
        cfg = _BrowserCfg.from_config_file()

        # 如果配置要求 CDP 但当前是 launch 模式的旧浏览器，先关掉
        if cfg.mode == "cdp" and self._mode == "launch" and self._page is not None:
            log.info("browser.switching_to_cdp", reason="配置为 cdp，关闭 launch 实例")
            await self._close()

        # 检查已有页面是否还活着
        if self._page is not None:
            try:
                await self._page.evaluate("1")
                return self._page
            except Exception:
                log.warning("browser.page_dead", reason="页面已关闭，重新初始化")
                self._browser = None
                self._page = None
                if self._playwright:
                    await self._playwright.stop()
                    self._playwright = None

        from whaleclaw.tools.deps import ensure_tool_dep

        if not ensure_tool_dep("playwright"):
            raise RuntimeError(
                "playwright 安装失败，请手动执行: "
                "pip install playwright && playwright install chromium"
            )

        from playwright.async_api import async_playwright

        # Playwright 启动 node 子进程，子进程继承环境变量。
        # 代理会拦截 localhost CDP 请求导致连接失败，
        # 必须在启动 playwright 之前清掉代理。
        import os
        _PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
        saved_proxies: dict[str, str] = {}
        if cfg.mode == "cdp":
            for key in _PROXY_KEYS:
                val = os.environ.pop(key, None)
                if val is not None:
                    saved_proxies[key] = val

        self._mode = cfg.mode
        try:
            if self._playwright is None:
                self._playwright = await async_playwright().start()
        finally:
            # 恢复代理，不影响其他网络请求（如 LLM API 调用）
            os.environ.update(saved_proxies)

        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

        if cfg.mode == "cdp":
            try:
                self._browser = await self._playwright.chromium.connect_over_cdp(
                    cfg.cdp_url
                )
                contexts = self._browser.contexts
                if contexts:
                    pages = contexts[0].pages
                    self._page = pages[0] if pages else await contexts[0].new_page()
                else:
                    context = await self._browser.new_context()
                    self._page = await context.new_page()
                # Playwright connect_over_cdp 会自动发送 Runtime.Enable，
                # 这会导致 Runtime.consoleAPICalled 泄露，被反自动化检测识别。
                # 立即发送 Runtime.Disable 关闭此泄露源。
                try:
                    cdp = await self._page.context.new_cdp_session(self._page)
                    await cdp.send("Runtime.disable")
                    await cdp.detach()
                except Exception:
                    log.debug("browser.cdp_runtime_disable_failed")
                log.info("browser.cdp_connected", cdp_url=cfg.cdp_url)
                return self._page
            except Exception as exc:
                log.warning(
                    "browser.cdp_fallback",
                    cdp_url=cfg.cdp_url,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    reason="辅助浏览器未运行，降级为 launch 模式",
                )
                self._mode = "launch"

        # launch 模式（含 CDP 降级）
        self._browser = await self._playwright.chromium.launch(
            channel="chrome",
            headless=cfg.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await self._browser.new_context(
            viewport=_VIEWPORT,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        self._page = await context.new_page()
        log.info("browser.launched", channel="chrome", headless=cfg.headless)
        return self._page

    async def _close(self) -> ToolResult:
        # CDP 模式只断开连接，不关闭用户的浏览器
        if self._browser:
            if self._mode == "cdp":
                await self._browser.close()
                msg = "已断开浏览器连接（辅助浏览器仍在运行）"
            else:
                await self._browser.close()
                msg = "浏览器已关闭"
            self._browser = None
            self._page = None
        else:
            msg = "浏览器未运行"
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        return ToolResult(success=True, output=msg)

    async def execute(self, **kwargs: Any) -> ToolResult:
        action: str = kwargs.get("action", "")
        if not action:
            return ToolResult(success=False, output="", error="action 参数为空")

        if action == "close":
            return await self._close()

        try:
            page = await self._ensure_browser()
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"浏览器启动失败: {exc}")

        try:
            return await self._dispatch(page, action, kwargs)
        except Exception as exc:
            log.error("browser.error", action=action, error=str(exc))
            return ToolResult(success=False, output="", error=str(exc))

    async def _dispatch(self, page: Any, action: str, kwargs: dict[str, Any]) -> ToolResult:
        if action == "navigate":
            url = kwargs.get("url", "")
            if not url:
                return ToolResult(success=False, output="", error="url 参数为空")
            await page.goto(url, timeout=_TIMEOUT, wait_until="domcontentloaded")
            title = await page.title()
            return ToolResult(
                success=True,
                output=f"已打开: {url}\n标题: {title}",
            )

        elif action == "screenshot":
            return await self._screenshot(page)

        elif action == "click":
            selector = kwargs.get("selector", "")
            if not selector:
                return ToolResult(success=False, output="", error="selector 参数为空")
            await page.click(selector, timeout=_TIMEOUT)
            return ToolResult(success=True, output=f"已点击: {selector}")

        elif action == "type":
            selector = kwargs.get("selector", "")
            text = kwargs.get("text", "")
            if not selector or not text:
                return ToolResult(
                    success=False, output="", error="selector 和 text 参数必填"
                )
            try:
                await page.fill(selector, text, timeout=_TIMEOUT)
            except Exception:
                # fill 对 contenteditable 富文本编辑器（TipTap/ProseMirror 等）
                # 通常会失败，用 select-all + delete 清空后再插入
                await page.evaluate(
                    """([sel, txt]) => {
                        const el = document.querySelector(sel);
                        if (!el) return;
                        el.focus();
                        const s = window.getSelection();
                        if (s) { s.selectAllChildren(el); s.deleteFromDocument(); }
                        if (el.textContent && el.textContent.trim()) {
                            document.execCommand('selectAll', false, null);
                            document.execCommand('delete', false, null);
                        }
                        document.execCommand('insertText', false, txt);
                    }""",
                    [selector, text],
                )
            return ToolResult(
                success=True, output=f"已输入到 {selector}（{len(text)} 字符）"
            )

        elif action == "get_text":
            selector = kwargs.get("selector", "")
            if selector:
                el = await page.query_selector(selector)
                if el is None:
                    return ToolResult(
                        success=False, output="", error=f"元素未找到: {selector}"
                    )
                text = await el.inner_text()
            else:
                text = await page.inner_text("body")
            truncated = text[:5000]
            if len(text) > 5000:
                truncated += f"\n...(截断，共 {len(text)} 字符)"
            return ToolResult(success=True, output=truncated)

        elif action == "evaluate":
            script = kwargs.get("script", "")
            if not script:
                return ToolResult(
                    success=False, output="", error="script 参数为空"
                )
            result = await page.evaluate(script)
            return ToolResult(success=True, output=str(result)[:5000])

        elif action == "upload":
            return await self._upload_files(page, kwargs)

        elif action == "search_images":
            query = kwargs.get("text", "")
            if not query:
                return ToolResult(
                    success=False, output="", error="text 参数为空（搜索关键词）"
                )
            try:
                normalized_query = _normalize_image_query(str(query))
            except ValueError as exc:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        f"{exc}。请使用明确关键词，如："
                        "“杨幂 近照 高清 人像”或“可爱猫咪 photo hd”"
                    ),
                )
            return await self._search_images(page, normalized_query)

        elif action == "back":
            await page.go_back(timeout=_TIMEOUT)
            title = await page.title()
            return ToolResult(success=True, output=f"已后退，当前页: {title}")

        return ToolResult(
            success=False, output="", error=f"未知 action: {action}"
        )

    async def _upload_files(self, page: Any, kwargs: dict[str, Any]) -> ToolResult:
        """上传本地文件到 <input type="file">。

        仅 launch 模式可靠。CDP 模式建议用 browser(evaluate) 点击 input
        触发对话框 + bash(osascript) 操控文件选择对话框。
        """
        selector = str(kwargs.get("selector", "")).strip()
        raw_paths = str(kwargs.get("file_paths", "")).strip()
        if not raw_paths:
            return ToolResult(
                success=False, output="", error="file_paths 参数为空"
            )
        if not selector:
            selector = 'input[type="file"]'

        paths: list[str] = []
        for p in raw_paths.split(","):
            resolved = Path(p.strip()).expanduser().resolve()
            if not resolved.is_file():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"文件不存在: {resolved}",
                )
            paths.append(str(resolved))

        el = await page.query_selector(selector)
        if el is None:
            return ToolResult(
                success=False,
                output="",
                error=f"未找到文件输入框: {selector}",
            )

        try:
            await el.set_input_files(paths)
        except Exception as exc:
            if self._mode == "cdp":
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        f"CDP 模式下 set_input_files 失败: {exc}\n"
                        "请改用：\n"
                        "1. browser(evaluate) 点击 input 触发文件对话框\n"
                        "2. bash(osascript) 在对话框中选择文件"
                    ),
                )
            raise
        names = [Path(p).name for p in paths]
        return ToolResult(
            success=True,
            output=f"已上传 {len(paths)} 个文件: {', '.join(names)}",
        )

    async def _screenshot(self, page: Any) -> ToolResult:
        filename = f"screenshot_{uuid.uuid4().hex[:8]}.png"
        path = _SCREENSHOT_DIR / filename
        try:
            await page.evaluate(
                'Object.defineProperty(document,"fonts",{value:{ready:Promise.resolve()}})'
            )
        except Exception:
            pass
        try:
            await page.screenshot(path=str(path), full_page=False, timeout=10000)
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"截图失败: {exc}",
            )
        return ToolResult(
            success=True,
            output=f"![截图]({path})",
        )

    async def _search_images(self, page: Any, query: str) -> ToolResult:
        """Image search -> download first result. Tries Bing then Google."""
        img_urls: list[str] = []

        for engine, url_fn, extract_js in _IMAGE_ENGINES:
            try:
                search_url = url_fn(query)
                log.info("browser.image_search", engine=engine, query=query)
                await page.goto(search_url, timeout=_TIMEOUT, wait_until="domcontentloaded")
                await page.wait_for_timeout(2500)
                img_urls = await page.evaluate(extract_js)
                if img_urls:
                    break
            except Exception as exc:
                log.warning("browser.image_search_failed", engine=engine, error=str(exc))
                continue

        if not img_urls:
            ss = await self._screenshot(page)
            return ToolResult(
                success=False,
                output=f"未找到图片结果。页面截图: {ss.output}",
                error="图片搜索未返回有效结果",
            )

        import httpx

        min_image_bytes = 50 * 1024  # prefer images >= 50 KB
        best_fallback: tuple[Path, int] | None = None

        for url in img_urls:
            try:
                async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        continue
                    ct = resp.headers.get("content-type", "")
                    if "image" not in ct:
                        continue

                    ext = "jpg"
                    if "png" in ct:
                        ext = "png"
                    elif "webp" in ct:
                        ext = "webp"
                    elif "gif" in ct:
                        ext = "gif"

                    safe_name = "".join(
                        c if c.isascii() and c.isalnum() or c in "-_" else ""
                        for c in query[:30]
                    ).strip("_") or "image"
                    filename = f"{safe_name}_{uuid.uuid4().hex[:8]}.{ext}"
                    path = _DOWNLOAD_DIR / filename
                    data = resp.content
                    path.write_bytes(data)

                    if len(data) >= min_image_bytes:
                        size_kb = len(data) / 1024
                        return ToolResult(
                            success=True,
                            output=(
                                f"搜索词: {query}\n"
                                f"图片已下载，请使用以下路径展示（禁止修改或编造路径）:\n"
                                f"![图片]({path})\n"
                                f"文件: {path}\n"
                                f"大小: {size_kb:.0f}KB"
                            ),
                        )
                    if best_fallback is None or len(data) > best_fallback[1]:
                        best_fallback = (path, len(data))
                    else:
                        path.unlink(missing_ok=True)
            except Exception:
                continue

        if best_fallback is not None:
            fb_path, fb_size = best_fallback
            size_kb = fb_size / 1024
            return ToolResult(
                success=True,
                output=(
                    f"搜索词: {query}\n"
                    f"图片已下载（未找到更大图片），请使用以下路径展示（禁止修改或编造路径）:\n"
                    f"![图片]({fb_path})\n"
                    f"文件: {fb_path}\n"
                    f"大小: {size_kb:.0f}KB"
                ),
            )

        return ToolResult(
            success=False,
            output="",
            error="所有图片 URL 下载失败",
        )


def _normalize_image_query(query: str) -> str:
    """Normalize and validate image-search query quality."""
    # strip ASCII control chars that may appear in malformed tool arguments
    q = "".join(ch for ch in query if ord(ch) >= 32 and ord(ch) != 127)
    # strip literal escaped noise like "\\n0\\n0\\x10"
    q = re.sub(r"(?:\\[nrt]\d*|\\x[0-9a-fA-F]{2})+", " ", q)
    q = " ".join(q.strip().split())
    if not q:
        raise ValueError("搜索关键词为空")
    if q.lower() in _GENERIC_IMAGE_QUERIES or q in _GENERIC_IMAGE_QUERIES:
        raise ValueError(f"搜索关键词过于泛化: {q}")
    if re.fullmatch(r"[\d\s\W_]+", q):
        raise ValueError(f"搜索关键词无效: {q}")
    if len(q) < 2:
        raise ValueError(f"搜索关键词过短: {q}")

    # Enforce one visual intent per search call.
    # If user passes multiple subjects (e.g. "花、瓶子、苹果"), caller should
    # split into multiple search_images calls.
    multi_intent_parts = re.split(r"[、，,;/|+]+", q)
    non_empty_parts = [p.strip() for p in multi_intent_parts if p.strip()]
    if len(non_empty_parts) >= 2:
        raise ValueError(
            "search_images 一次只支持一个主体关键词，请拆成多次调用"
        )

    has_hint = any(h in q.lower() for h in _IMAGE_INTENT_HINTS)
    if not has_hint:
        q = f"{q} 近照 高清 人像"
    return q
