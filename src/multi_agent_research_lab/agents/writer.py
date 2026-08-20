"""Writer agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces the final comprehensive answer from research and analysis notes with citations."""

    name = "writer"

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(self.settings, temperature=0.3)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        logger.info("Writer synthesizing final response...")
        query = state.request.query
        sources = state.sources
        research_notes = state.research_notes or "N/A"
        analysis_notes = state.analysis_notes or "N/A"

        sources_ref = "\n".join(
            f"[{idx + 1}] {s.title} ({s.url or 'N/A'}) - {s.snippet[:120]}..."
            for idx, s in enumerate(sources)
        )

        system_prompt = (
            "You are a Lead Technical Writer and Research Synthesizer. Your goal is to write "
            "a comprehensive, polished, and rigorous technical report addressing the user's query. "
            "You MUST use numbered citations (e.g. [1], [2]) corresponding to the provided sources "
            "and include a References section at the end."
        )

        user_prompt = (
            f"# Target Query: {query}\n"
            f"# Target Audience: {state.request.audience}\n\n"
            f"## Available Sources:\n{sources_ref}\n\n"
            f"## Research Notes:\n{research_notes}\n\n"
            f"## Analytical Insights:\n{analysis_notes}\n\n"
            "Produce an exhaustive, clearly structured response with:\n"
            "- Executive Summary\n"
            "- Key Concepts & Technical Architecture\n"
            "- Practical Evaluation & Tradeoffs\n"
            "- References (numbered list linking back to the citations)"
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            final_answer = response.content
        except Exception as exc:
            logger.warning(
                "LLM call in writer failed: %s. Using consolidated synthesis fallback.", exc
            )
            final_answer = (
                f"# Research Report: {query}\n\n"
                f"## Executive Summary\n"
                f"This report presents research and analytical findings regarding **{query}**.\n\n"
                f"## Analysis & Findings\n{analysis_notes}\n\n"
                f"## Key Research Notes\n{research_notes}\n\n"
                f"## References\n{sources_ref}\n"
            )

        state.final_answer = final_answer
        state.add_trace_event(
            "writer_complete",
            {"answer_length": len(final_answer)},
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=final_answer,
                metadata={"answer_length": len(final_answer)},
            )
        )
        return state
