"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a rich markdown report."""
    lines = [
        "# Benchmark Report: Single-Agent vs Multi-Agent",
        "",
        "## Summary Metrics",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.5f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}/10"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| **{item.run_name}** | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines.extend(
        [
            "",
            "## Tradeoff Analysis",
            "",
            "- **Single-Agent Baseline**:",
            "  - *Pros*: Fast response latency, lower token cost (single LLM request).",
            "  - *Cons*: Prone to hallucinations, lacks multi-hop synthesis and source citations.",
            "",
            "- **Multi-Agent Workflow (Supervisor + Researcher + Analyst + Writer)**:",
            "  - *Pros*: Superior quality, modular role separation, and rigorous citations.",
            "  - *Cons*: Higher latency due to multi-step orchestration and higher token cost.",
            "",
            "## Failure Modes & Mitigations",
            "",
            "1. **Infinite Supervisor Loops**:",
            "   - *Risk*: Supervisor gets stuck in circular delegation between agents.",
            "   - *Fix*: Hard guardrail `max_iterations = 6` forcing synthesis on max iterations.",
            "2. **Missing Source Citations**:",
            "   - *Risk*: Writer fabricates claims rather than grounding in retrieved sources.",
            "   - *Fix*: Integrated `CriticAgent` verification and strict numbered prompt format.",
        ]
    )

    return "\n".join(lines) + "\n"
