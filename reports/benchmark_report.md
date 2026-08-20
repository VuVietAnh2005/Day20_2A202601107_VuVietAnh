# Benchmark Report: Single-Agent vs Multi-Agent

## Summary Metrics

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure | Notes |
|---|---:|---:|---:|---:|---:|---|
| **Single-Agent Baseline** | 7.27 | $0.00033 | 3.0/10 | 0% | 0% | Monolithic 1-call prompt |
| **Multi-Agent (Supervisor+Researcher+Analyst+Writer)** | 26.89 | $0.00119 | 10.0/10 | 100% | 0% | LangGraph Orchestration with Search & Citations |

## Tradeoff Analysis

- **Single-Agent Baseline**:
  - *Pros*: Fast response latency, lower token cost (single LLM request).
  - *Cons*: Prone to hallucinations, lacks multi-hop synthesis and source citations.

- **Multi-Agent Workflow (Supervisor + Researcher + Analyst + Writer)**:
  - *Pros*: Superior quality, modular role separation, and rigorous citations.
  - *Cons*: Higher latency due to multi-step orchestration and higher token cost.

## Failure Modes & Mitigations

1. **Infinite Supervisor Loops**:
   - *Risk*: Supervisor gets stuck in circular delegation between agents.
   - *Fix*: Hard guardrail `max_iterations = 6` forcing synthesis on max iterations.
2. **Missing Source Citations**:
   - *Risk*: Writer fabricates claims rather than grounding in retrieved sources.
   - *Fix*: Integrated `CriticAgent` verification and strict numbered prompt format.
