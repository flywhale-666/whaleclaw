"""Backward-compatible alias for the single-agent runtime module."""

from __future__ import annotations

import sys

from whaleclaw.agent import single_agent as _single_agent
from whaleclaw.agent.single_agent import run_agent as run_agent  # noqa: F401  # pyright re-export

sys.modules[__name__] = _single_agent
