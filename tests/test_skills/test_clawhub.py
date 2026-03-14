"""Tests for ClawHub skill search behavior."""

import contextlib
import json
from pathlib import Path

from whaleclaw.skills import clawhub


def test_search_skills_uses_http_search_and_http_detail(monkeypatch) -> None:  # noqa: ANN001
    clawhub._search_cache.clear()
    clawhub._detail_cache.clear()

    monkeypatch.setattr(
        clawhub,
        "_search_via_http",
        lambda **_: [
            {
                "slug": "python-executor",
                "name": "Python Executor",
                "summary": "demo",
                "version": "1.0.0",
                "stars": 10,
                "downloads": 20,
                "current_installs": 3,
                "all_time_installs": 50,
                "detail_url": "https://clawhub.ai/skills/python-executor",
                "repo_url": "",
            }
        ],
    )
    monkeypatch.setattr(
        clawhub,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cli should not run")),
    )

    items = clawhub.search_skills(
        query="python",
        registry_url="https://clawhub.ai",
        workspace_dir=Path.cwd(),
        limit=20,
    )
    assert len(items) == 1
    assert items[0]["slug"] == "python-executor"
    assert items[0]["stars"] == 10
    assert items[0]["downloads"] == 20


def test_search_via_http_does_not_re_filter_results(monkeypatch) -> None:  # noqa: ANN001
    clawhub._detail_cache.clear()
    monkeypatch.setattr(clawhub, "_enrich_results_with_skill_details", lambda **kwargs: None)

    class _Resp:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {
                "results": [
                    {
                        "slug": "x402-creative-resources",
                        "displayName": "x402 Creative Resources",
                        "summary": (
                            "Access Xona's x402 creative resource APIs "
                            "on api.xona-agent.com."
                        ),
                        "version": "1.0.0",
                    },
                    {
                        "slug": "nano-banana-2",
                        "displayName": "Nano Banana 2",
                        "summary": "Generate images with Gemini 3.1 Flash Image Preview.",
                        "version": "0.1.1",
                    },
                ]
            }

    class _Client:
        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN204
            return False

        def get(self, url: str, params: dict[str, object], headers: dict[str, str]) -> _Resp:  # noqa: ARG002
            return _Resp()

    monkeypatch.setattr(clawhub.httpx, "Client", lambda **kwargs: _Client())

    items = clawhub._search_via_http(
        query="banana",
        registry_url="https://clawhub.ai",
        api_token=None,
        limit=20,
    )
    assert [str(item["slug"]) for item in items] == [
        "nano-banana-2",
        "x402-creative-resources",
    ]


def test_enrich_results_with_skill_details_covers_more_than_first_page(monkeypatch) -> None:  # noqa: ANN001
    clawhub._detail_cache.clear()
    rows = [
        {"slug": f"skill-{i}", "detail_url": "", "stars": None, "downloads": None}
        for i in range(10)
    ]

    def _fake_fetch(**kwargs) -> dict[str, object]:  # noqa: ANN001
        slug = str(kwargs["slug"])
        index = int(slug.split("-")[-1])
        return {"stars": index, "downloads": index * 10}

    monkeypatch.setattr(clawhub, "_fetch_skill_detail_enrichment", _fake_fetch)
    monkeypatch.setattr(clawhub, "_inspect_skill_stats_via_cli", lambda **_: {})

    clawhub._enrich_results_with_skill_details(
        items=rows,
        registry_url="https://clawhub.ai",
        api_token=None,
        max_items=10,
    )
    assert all(row["stars"] is not None for row in rows)
    assert all(row["downloads"] is not None for row in rows)


def test_enrich_results_skips_detail_when_sort_keys_are_ready(monkeypatch) -> None:  # noqa: ANN001
    clawhub._detail_cache.clear()
    rows = [
        {
            "slug": "ready",
            "detail_url": "",
            "stars": 5,
            "downloads": 7,
            "current_installs": None,
            "all_time_installs": 9,
        },
        {
            "slug": "need-detail",
            "detail_url": "",
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        },
    ]
    called: list[str] = []

    def _fake_fetch(**kwargs) -> dict[str, object]:  # noqa: ANN001
        called.append(str(kwargs["slug"]))
        return {
            "stars": 1,
            "downloads": 2,
            "current_installs": 3,
            "all_time_installs": 4,
        }

    monkeypatch.setattr(clawhub, "_fetch_skill_detail_enrichment", _fake_fetch)
    monkeypatch.setattr(clawhub, "_inspect_skill_stats_via_cli", lambda **_: {})

    clawhub._enrich_results_with_skill_details(
        items=rows,
        registry_url="https://clawhub.ai",
        api_token=None,
        max_items=10,
    )
    assert called == ["need-detail"]
    assert rows[0]["downloads"] == 7
    assert rows[1]["downloads"] == 2


def test_publish_installed_skill_uses_http_publish_directly(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# demo\n", encoding="utf-8")

    called: dict[str, object] = {}

    def _fake_http_publish(**kwargs):  # noqa: ANN001
        called.update(kwargs)
        return "OK. Published demo-skill@1.0.1 (ver_123)"

    monkeypatch.setattr(clawhub, "_publish_via_http", _fake_http_publish)

    out = clawhub.publish_installed_skill(
        skill_dir=skill_dir,
        skill_slug="demo-skill",
        skill_version="1.0.1",
        registry_url="https://clawhub.ai",
        workspace_dir=tmp_path,
        api_token="token",
    )

    assert "OK. Published demo-skill@1.0.1" in out
    assert called["skill_slug"] == "demo-skill"
    assert called["skill_version"] == "1.0.1"


def test_publish_via_http_includes_accept_license_terms(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    skill_dir = tmp_path / "nano-banana-image-t8"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    (skill_dir / "run.py").write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setattr(clawhub, "_read_clawhub_global_config", lambda: {"token": "cfg-token"})

    captured: dict[str, object] = {}

    class _Resp:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"ok": True, "versionId": "ver_123"}

        @property
        def text(self) -> str:
            return json.dumps(self.json())

    class _Client:
        def __init__(self, **kwargs):  # noqa: ANN001
            captured["init"] = kwargs

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN204
            return False

        def post(
            self,
            url: str,
            headers: dict[str, str],
            data: dict[str, str],
            files: list[tuple[str, object]],
        ):  # noqa: ANN001
            captured["url"] = url
            captured["headers"] = headers
            captured["data"] = data
            captured["files"] = files
            return _Resp()

    monkeypatch.setattr(clawhub.httpx, "Client", _Client)

    out = clawhub._publish_via_http(
        skill_dir=skill_dir,
        skill_slug="nano-banana-image-t8",
        skill_version="1.0.1",
        registry_url="https://clawhub.ai",
        api_token=None,
    )

    assert "1.0.1" in out
    assert captured["url"] == "https://clawhub.ai/api/v1/skills"
    payload = json.loads(str(captured["data"]["payload"]))
    assert payload["acceptLicenseTerms"] is True


def test_read_clawhub_global_config_merges_legacy_token(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    monkeypatch.delenv("CLAWHUB_CONFIG_PATH", raising=False)
    monkeypatch.delenv("CLAWDHUB_CONFIG_PATH", raising=False)
    monkeypatch.setattr(clawhub.platform, "system", lambda: "Linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    main_dir = tmp_path / "clawhub"
    legacy_dir = tmp_path / "clawdhub"
    main_dir.mkdir()
    legacy_dir.mkdir()
    (main_dir / "config.json").write_text('{"token": "", "registry": "main"}', encoding="utf-8")
    (legacy_dir / "config.json").write_text(
        '{"token": "legacy-token", "legacy_only": true}',
        encoding="utf-8",
    )

    cfg = clawhub._read_clawhub_global_config()

    assert cfg["token"] == "legacy-token"
    assert cfg["registry"] == "main"
    assert cfg["legacy_only"] is True


def test_search_via_http_uses_global_token_when_api_token_missing(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(clawhub, "_read_clawhub_global_config", lambda: {"token": "cfg-token"})
    captured: dict[str, object] = {}

    class _Resp:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"results": []}

    class _Client:
        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN204
            return False

        def get(self, url: str, params: dict[str, object], headers: dict[str, str]) -> _Resp:
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return _Resp()

    monkeypatch.setattr(clawhub.httpx, "Client", lambda **kwargs: _Client())

    clawhub._search_via_http(
        query="banana",
        registry_url="https://clawhub.ai",
        api_token=None,
        limit=20,
    )

    assert captured["headers"] == {
        "Accept": "application/json",
        "Authorization": "Bearer cfg-token",
    }


def test_install_via_http_uses_global_token_when_api_token_missing(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    monkeypatch.setattr(clawhub, "_read_clawhub_global_config", lambda: {"token": "cfg-token"})
    captured: dict[str, object] = {}

    class _Resp:
        status_code = 200
        content = b"not-a-real-zip"
        text = ""

    class _Client:
        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN204
            return False

        def get(self, url: str, params: dict[str, object], headers: dict[str, str]) -> _Resp:
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return _Resp()

    monkeypatch.setattr(clawhub.httpx, "Client", lambda **kwargs: _Client())

    with contextlib.suppress(clawhub.ClawHubCliError):
        clawhub._install_via_http(
            slug="demo-skill",
            version=None,
            registry_url="https://clawhub.ai",
            install_dir=tmp_path,
            api_token=None,
        )

    assert captured["headers"] == {"Authorization": "Bearer cfg-token"}


def test_search_skills_does_not_wrap_http_error_prefix_twice(monkeypatch) -> None:  # noqa: ANN001
    clawhub._search_cache.clear()
    clawhub._detail_cache.clear()
    monkeypatch.setattr(
        clawhub,
        "_search_via_http",
        lambda **_: (_ for _ in ()).throw(
            clawhub.ClawHubCliError("搜索失败: HTTP 429 too many requests")
        ),
    )

    try:
        clawhub.search_skills(
            query="banana",
            registry_url="https://clawhub.ai",
            workspace_dir=Path.cwd(),
            limit=20,
        )
    except clawhub.ClawHubCliError as exc:
        assert str(exc) == "搜索失败: HTTP 429 too many requests"
    else:
        raise AssertionError("expected ClawHubCliError")
