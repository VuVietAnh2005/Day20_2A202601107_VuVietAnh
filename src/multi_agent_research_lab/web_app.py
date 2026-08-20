"""FastAPI web server providing interactive Web UI and REST API for Multi-Agent Research Lab."""

import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    reconf_out: Any = getattr(sys.stdout, "reconfigure", None)
    if callable(reconf_out):
        reconf_out(encoding="utf-8")

app = FastAPI(title="Multi-Agent Research Lab API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5)
    mode: Literal["multi-agent", "baseline"] = "multi-agent"
    max_sources: int = 4
    audience: str = "technical learners"


class BenchmarkRequest(BaseModel):
    query: str = Field(..., min_length=5)


@app.get("/")
def serve_index() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web UI static files not found")
    return FileResponse(str(index_file))


@app.post("/api/run")
def run_research(payload: ResearchRequest) -> dict[str, Any]:
    query = payload.query.strip()
    mode = payload.mode
    settings = get_settings()

    started = perf_counter()
    state = ResearchState(
        request=ResearchQuery(
            query=query,
            max_sources=payload.max_sources,
            audience=payload.audience,
        )
    )

    if mode == "baseline":
        llm = LLMClient(settings)
        system_prompt = (
            "You are a standalone research assistant. Address the user's research query directly, "
            "providing a comprehensive, structured technical summary with citations and references."
        )
        try:
            resp = llm.complete(system_prompt, f"Research Question: {query}")
            state.final_answer = resp.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=resp.content,
                    metadata={
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                    },
                )
            )
        except Exception as exc:
            state.final_answer = f"Baseline execution note: {exc}"
            state.errors.append(str(exc))
    else:
        workflow = MultiAgentWorkflow(settings)
        state = workflow.run(state)

    latency = round(perf_counter() - started, 2)

    return {
        "mode": mode,
        "query": query,
        "latency_seconds": latency,
        "iteration": state.iteration,
        "route_history": state.route_history,
        "final_answer": state.final_answer,
        "research_notes": state.research_notes,
        "analysis_notes": state.analysis_notes,
        "sources": [s.model_dump() for s in state.sources],
        "agent_results": [r.model_dump() for r in state.agent_results],
        "trace": state.trace,
        "errors": state.errors,
    }


@app.post("/api/benchmark")
def run_api_benchmark(payload: BenchmarkRequest) -> dict[str, Any]:
    query = payload.query.strip()
    settings = get_settings()

    def run_single(q: str) -> ResearchState:
        st = ResearchState(request=ResearchQuery(query=q))
        llm = LLMClient(settings)
        resp = llm.complete("You are a standalone research assistant.", f"Query: {q}")
        st.final_answer = resp.content
        return st

    def run_multi(q: str) -> ResearchState:
        st = ResearchState(request=ResearchQuery(query=q))
        wf = MultiAgentWorkflow(settings)
        return wf.run(st)

    _, single_metrics = run_benchmark("Single-Agent Baseline", query, run_single)
    _, multi_metrics = run_benchmark("Multi-Agent Workflow", query, run_multi)

    return {
        "query": query,
        "results": [
            single_metrics.model_dump(),
            multi_metrics.model_dump(),
        ],
    }


# Mount static assets if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
