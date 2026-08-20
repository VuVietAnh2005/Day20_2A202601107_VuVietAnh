"""Benchmark implementation for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def evaluate_state(
    state: ResearchState, latency: float, run_name: str, notes: str = ""
) -> BenchmarkMetrics:
    """Evaluate quality, cost, and citation metrics on a completed ResearchState."""
    answer = state.final_answer or ""
    sources = state.sources

    # 1. Citation coverage calculation
    cited_indices: set[int] = set()
    for match in re.finditer(r"\[(\d+)\]", answer):
        idx = int(match.group(1))
        if 1 <= idx <= len(sources):
            cited_indices.add(idx)

    citation_coverage = len(cited_indices) / max(len(sources), 1) if sources else 0.0

    # 2. Quality score heuristic (0-10) based on depth, structure, citations
    quality_score = 0.0
    if answer:
        if len(answer) > 500:
            quality_score += 3.0
        elif len(answer) > 200:
            quality_score += 1.5

        if "##" in answer or "###" in answer:
            quality_score += 2.5
        if citation_coverage > 0.5:
            quality_score += 2.5
        elif citation_coverage > 0:
            quality_score += 1.5
        if state.analysis_notes or state.research_notes:
            quality_score += 2.0

    quality_score = min(10.0, max(0.0, quality_score))

    # 3. Estimated token cost
    total_chars = len(answer) + sum(len(s.snippet) for s in sources)
    if state.research_notes:
        total_chars += len(state.research_notes)
    if state.analysis_notes:
        total_chars += len(state.analysis_notes)

    est_tokens = total_chars // 4
    est_cost = round((est_tokens / 1_000_000) * 0.35, 5)

    failure_rate = 0.0 if bool(answer and not state.errors) else 1.0

    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 2),
        estimated_cost_usd=est_cost,
        quality_score=round(quality_score, 1),
        citation_coverage=round(citation_coverage, 2),
        failure_rate=failure_rate,
        notes=notes,
    )


def run_benchmark(
    run_name: str, query: str, runner: Runner, notes: str = ""
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, quality, citation coverage, and costs for a given runner."""
    started = perf_counter()
    try:
        state = runner(query)
        latency = perf_counter() - started
        metrics = evaluate_state(state, latency, run_name, notes=notes)
    except Exception as exc:
        latency = perf_counter() - started
        valid_query = query if len(query) >= 5 else "Default query"
        state = ResearchState(request=ResearchQuery(query=valid_query))
        state.errors.append(str(exc))
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=round(latency, 2),
            estimated_cost_usd=0.0,
            quality_score=0.0,
            citation_coverage=0.0,
            failure_rate=1.0,
            notes=f"Failed: {exc}",
        )
    return state, metrics
