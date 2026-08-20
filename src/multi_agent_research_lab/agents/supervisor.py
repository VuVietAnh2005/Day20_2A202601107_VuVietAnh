"""Supervisor / router implementation."""

import logging
from typing import Literal

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

RouteDestination = Literal["researcher", "analyst", "writer", "FINISH"]


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def decide_next_route(self, state: ResearchState) -> RouteDestination:
        """Determine the next step based on shared state and guardrails."""
        # 1. Guardrail: Max iterations reached
        if state.iteration >= self.settings.max_iterations:
            logger.warning(
                "Supervisor reached max iterations (%d). Forcing termination/synthesis.",
                self.settings.max_iterations,
            )
            if not state.final_answer and (state.analysis_notes or state.research_notes):
                return "writer"
            return "FINISH"

        # 2. Sequential pipeline logic
        if not state.sources or not state.research_notes:
            return "researcher"

        if not state.analysis_notes:
            return "analyst"

        if not state.final_answer:
            return "writer"

        return "FINISH"

    def run(self, state: ResearchState) -> ResearchState:
        """Update state with the routing decision."""
        next_route = self.decide_next_route(state)
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_decision",
            {"next_route": next_route, "iteration": state.iteration},
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Decision: proceed to {next_route} (iteration {state.iteration})",
                metadata={"next_route": next_route, "iteration": state.iteration},
            )
        )
        return state
