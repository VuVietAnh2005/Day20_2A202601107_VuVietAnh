"""Researcher agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise, factual research notes."""

    name = "researcher"

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(self.settings, temperature=0.2)
        self.search_client = search_client or SearchClient(self.settings)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        logger.info("Researcher searching for: %s", query)

        # 1. Fetch search documents
        sources = self.search_client.search(query, max_results=state.request.max_sources)
        state.sources = sources

        # 2. Prepare structured notes synthesis
        sources_text = "\n\n".join(
            f"[{idx + 1}] Title: {s.title}\nURL: {s.url}\nSnippet: {s.snippet}"
            for idx, s in enumerate(sources)
        )

        system_prompt = (
            "You are an expert technical researcher. Extract verified facts, definitions, "
            "architectures, and findings from the sources. "
            "Organize into clear bullet points with references to [Source 1], etc."
        )

        user_prompt = (
            f"Research Question: {query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Retrieved Documents:\n{sources_text}\n\n"
            "Produce comprehensive, structured research notes focusing on key technical points."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            research_notes = response.content
        except Exception as exc:
            logger.warning(
                "LLM call in researcher failed: %s. Using source extraction fallback.", exc
            )
            research_notes = f"### Extracted Findings for: {query}\n" + "\n".join(
                f"- **{s.title}**: {s.snippet}" for s in sources
            )

        state.research_notes = research_notes
        state.add_trace_event(
            "researcher_complete",
            {"sources_count": len(sources), "notes_length": len(research_notes)},
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=research_notes,
                metadata={"sources_count": len(sources)},
            )
        )
        return state
