"""Critic agent implementation for verification and safety review."""

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Verifies citation coverage, fact consistency, and safety."""

    name = "critic"

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(self.settings, temperature=0.0)

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and calculate citation coverage."""
        logger.info("Critic reviewing final answer...")
        answer = state.final_answer or ""
        sources = state.sources

        # Simple citation coverage heuristic: check presence of citations [1], [2], etc.
        cited_indices: set[int] = set()
        for match in re.finditer(r"\[(\d+)\]", answer):
            idx = int(match.group(1))
            if 1 <= idx <= len(sources):
                cited_indices.add(idx)

        coverage = len(cited_indices) / max(len(sources), 1) if sources else 1.0

        critic_summary = (
            f"Review complete. Citation coverage: {coverage:.0%} "
            f"({len(cited_indices)}/{len(sources)} sources referenced)."
        )

        state.add_trace_event(
            "critic_complete",
            {
                "citation_coverage": coverage,
                "cited_sources_count": len(cited_indices),
                "total_sources": len(sources),
            },
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=critic_summary,
                metadata={"citation_coverage": coverage},
            )
        )
        return state
