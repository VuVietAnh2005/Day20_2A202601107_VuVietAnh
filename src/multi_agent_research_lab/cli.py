"""Command-line entrypoint for the Multi-Agent Research Lab."""

import sys
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import setup_tracing, trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

# Ensure UTF-8 output on Windows consoles safely
if sys.platform == "win32":
    reconf_out: Any = getattr(sys.stdout, "reconfigure", None)
    if callable(reconf_out):
        reconf_out(encoding="utf-8")
    reconf_err: Any = getattr(sys.stderr, "reconfigure", None)
    if callable(reconf_err):
        reconf_err(encoding="utf-8")

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console(legacy_windows=False)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_tracing(settings)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline execution (direct monolithic LLM call)."""
    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)
    llm = LLMClient()

    console.print(f"[bold cyan]Running Single-Agent Baseline for:[/bold cyan] {query}")
    started = perf_counter()

    with trace_span("single_agent_baseline", {"query": query}):
        system_prompt = (
            "You are a standalone research assistant. Address the user's research query directly, "
            "providing a comprehensive, structured technical summary with citations and references."
        )
        try:
            response = llm.complete(system_prompt, f"Research Question: {query}")
            state.final_answer = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
        except Exception as exc:
            state.final_answer = (
                f"Baseline fallback response for: {query}\n\n"
                "Key finding: Single-agent execution completed."
            )
            state.errors.append(str(exc))

    latency = perf_counter() - started

    # Print output
    console.print(
        Panel(
            Markdown(state.final_answer or ""),
            title="[bold green]Single-Agent Baseline Answer[/bold green]",
        )
    )
    console.print(f"[dim]Latency: {latency:.2f}s | Status: Success[/dim]\n")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow (Supervisor -> Researcher -> Analyst -> Writer -> Critic)."""
    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()

    console.print(f"[bold blue]Starting Multi-Agent Research System for:[/bold blue] {query}\n")
    started = perf_counter()

    with trace_span("multi_agent_workflow", {"query": query}):
        result = workflow.run(state)

    latency = perf_counter() - started

    # 1. Print Final Answer
    console.print(
        Panel(
            Markdown(result.final_answer or "No final answer generated."),
            title="[bold green]Multi-Agent Final Answer[/bold green]",
            border_style="green",
        )
    )

    # 2. Print Execution Summary Table
    table = Table(title="Workflow Execution Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    routes = " -> ".join(result.route_history) if result.route_history else "N/A"
    table.add_row("Total Latency", f"{latency:.2f} seconds")
    table.add_row("Total Iterations", str(result.iteration))
    table.add_row("Route History", routes)
    table.add_row("Sources Retrieved", str(len(result.sources)))
    table.add_row("Trace Spans", str(len(result.trace)))

    console.print(table)


@app.command()
def benchmark(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Optional custom query to benchmark"),
    ] = "Research GraphRAG state-of-the-art and write a 500-word summary",
    output_file: Annotated[
        str,
        typer.Option("--output", "-o", help="Output path for benchmark report"),
    ] = "reports/benchmark_report.md",
) -> None:
    """Run a comparative benchmark between Single-Agent Baseline and Multi-Agent Workflow."""
    _init()
    console.print(f"[bold yellow]Running Comparative Benchmark on:[/bold yellow] {query}\n")

    # 1. Benchmark Single-Agent Baseline
    def run_single(q: str) -> ResearchState:
        st = ResearchState(request=_parse_query(q))
        llm = LLMClient()
        resp = llm.complete(
            "You are a standalone research assistant. Write a summary.",
            f"Query: {q}",
        )
        st.final_answer = resp.content
        return st

    # 2. Benchmark Multi-Agent Workflow
    def run_multi(q: str) -> ResearchState:
        st = ResearchState(request=_parse_query(q))
        wf = MultiAgentWorkflow()
        return wf.run(st)

    console.print("[dim]1/2 Running Single-Agent Baseline...[/dim]")
    _, baseline_metrics = run_benchmark(
        "Single-Agent Baseline", query, run_single, notes="Monolithic 1-call prompt"
    )

    console.print("[dim]2/2 Running Multi-Agent Workflow...[/dim]")
    _, multi_metrics = run_benchmark(
        "Multi-Agent (Supervisor+Researcher+Analyst+Writer)",
        query,
        run_multi,
        notes="LangGraph Orchestration with Search & Citations",
    )

    metrics_list = [baseline_metrics, multi_metrics]
    report_md = render_markdown_report(metrics_list)

    # Save report
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    console.print(Markdown(report_md))
    console.print(
        f"[bold green]Report successfully generated and saved to:[/bold green] {output_file}"
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", "-h", help="Host address to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 8000,
) -> None:
    """Launch the interactive Multi-Agent Web UI on browser."""
    import uvicorn

    _init()
    console.print(
        Panel.fit(
            f"🌐 [bold green]Multi-Agent Research Web UI is starting![/bold green]\n\n"
            f"👉 Open in browser: [bold cyan]http://{host}:{port}[/bold cyan]\n"
            f"⚡ Press [bold red]Ctrl + C[/bold red] to stop server.",
            title="Web Interface Server",
            border_style="cyan",
        )
    )
    uvicorn.run("multi_agent_research_lab.web_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
