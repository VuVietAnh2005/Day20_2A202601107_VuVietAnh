"""Tracing hooks and span instrumentation for multi-agent observability."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def setup_tracing(settings: Settings | None = None) -> None:
    """Configure external tracing providers (e.g. LangSmith) if keys are provided."""
    cfg = settings or get_settings()

    if cfg.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith_project
        logger.info("LangSmith tracing enabled for project: %s", cfg.langsmith_project)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager to measure and log trace span metrics."""
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "status": "in_progress",
        "duration_seconds": None,
    }
    try:
        yield span
        span["status"] = "success"
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 4)
        logger.debug(
            "Trace span '%s' finished in %.4fs [%s]",
            name,
            span["duration_seconds"],
            span["status"],
        )
