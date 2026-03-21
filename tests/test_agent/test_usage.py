"""Tests for usage tracking and cost estimation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from whaleclaw.agent.usage import TokenUsage, UsageTracker, estimate_cost


def test_estimate_cost() -> None:
    """Verify cost calculation for known model."""
    cost = estimate_cost("claude-sonnet-4-20250514", 1_000_000, 500_000)
    assert abs(cost - (3.0 + 7.5)) < 0.001
    cost = estimate_cost("glm-4.7-flash", 1000, 2000)
    assert cost == 0.0


def test_estimate_cost_unknown() -> None:
    """Unknown model returns 0."""
    assert estimate_cost("unknown-model-xyz", 1000, 500) == 0.0


@pytest.mark.asyncio
async def test_record_and_query(tmp_path: Path) -> None:
    """Open tracker, record usage, query session usage, verify totals."""
    db_path = tmp_path / "usage.db"
    tracker = UsageTracker(db_path)
    await tracker.open()
    try:
        usage = TokenUsage(
            model="gpt-5.2",
            input_tokens=100,
            output_tokens=50,
            thinking_tokens=0,
        )
        await tracker.record("sess-1", usage)
        await tracker.record("sess-1", usage)
        sess = await tracker.get_session_usage("sess-1")
        assert sess.session_id == "sess-1"
        assert sess.total_input_tokens == 200
        assert sess.total_output_tokens == 100
        assert sess.request_count == 2
        assert sess.total_cost_usd > 0
    finally:
        await tracker.close()


@pytest.mark.asyncio
async def test_daily_usage(tmp_path: Path) -> None:
    """Record and query by date."""
    db_path = tmp_path / "usage.db"
    tracker = UsageTracker(db_path)
    await tracker.open()
    try:
        usage = TokenUsage(
            model="glm-4.7-flash",
            input_tokens=500,
            output_tokens=200,
        )
        await tracker.record("s1", usage)
        today = date.today().isoformat()
        daily = await tracker.get_daily_usage()
        assert daily.date == today
        assert daily.total_input_tokens == 500
        assert daily.total_output_tokens == 200
        assert daily.request_count == 1
        other = await tracker.get_daily_usage("1999-01-01")
        assert other.total_input_tokens == 0
        assert other.request_count == 0
    finally:
        await tracker.close()
