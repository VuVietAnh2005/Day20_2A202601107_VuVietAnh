"""Analyst agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured technical insights, tradeoffs, and analysis."""

    name = "analyst"

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(self.settings, temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        logger.info("Analyst processing research notes...")
        notes = state.research_notes or "No research notes provided."

        system_prompt = (
            "You are a Principal AI Systems Analyst. Your goal is to critically analyze "
            "the research notes, identify mechanisms, compare tradeoffs, evaluate evidence, "
            "and highlight engineering implications."
        )

        user_prompt = (
            f"Original Query: {state.request.query}\n\n"
            f"Research Notes:\n{notes}\n\n"
            "Please deliver a structured analysis including:\n"
            "1. Key Architectural Principles & Core Mechanisms\n"
            "2. Comparative Strengths, Weaknesses, and Tradeoffs\n"
            "3. Production Considerations and Practical Caveats"
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            analysis_notes = response.content
        except Exception as exc:
            logger.warning(
                "LLM call in analyst failed: %s. Using structured heuristic analysis.", exc
            )
            analysis_notes = (
                f"### Analytical Assessment for: {state.request.query}\n"
                "- **Key Finding**: The approach introduces structured decomposition.\n"
                "- **Tradeoffs**: Higher orchestration complexity vs accuracy.\n"
                "- **Recommendation**: Evaluate latency and token costs before production."
            )

        state.analysis_notes = analysis_notes
        state.add_trace_event(
            "analyst_complete",
            {"analysis_length": len(analysis_notes)},
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=analysis_notes,
                metadata={"notes_length": len(analysis_notes)},
            )
        )
        return state
