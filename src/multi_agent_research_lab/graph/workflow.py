"""LangGraph workflow implementation."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and orchestrates the multi-agent graph with LangGraph."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.supervisor = SupervisorAgent(self.settings)
        self.researcher = ResearcherAgent(self.settings)
        self.analyst = AnalystAgent(self.settings)
        self.writer = WriterAgent(self.settings)
        self.critic = CriticAgent(self.settings)
        self._compiled_graph: Any = None

    def _node_supervisor(self, state: ResearchState) -> dict[str, Any]:
        updated = self.supervisor.run(state)
        return updated.model_dump()

    def _node_researcher(self, state: ResearchState) -> dict[str, Any]:
        updated = self.researcher.run(state)
        return updated.model_dump()

    def _node_analyst(self, state: ResearchState) -> dict[str, Any]:
        updated = self.analyst.run(state)
        return updated.model_dump()

    def _node_writer(self, state: ResearchState) -> dict[str, Any]:
        updated = self.writer.run(state)
        return updated.model_dump()

    def _node_critic(self, state: ResearchState) -> dict[str, Any]:
        updated = self.critic.run(state)
        return updated.model_dump()

    def _route_condition(self, state: ResearchState) -> str:
        if not state.route_history:
            return "FINISH"
        last_decision = state.route_history[-1]
        if last_decision in ("researcher", "analyst", "writer"):
            return last_decision
        return "FINISH"

    def build(self) -> Any:
        """Create and compile the LangGraph workflow."""
        builder = StateGraph(ResearchState)

        # 1. Register nodes
        builder.add_node("supervisor", self._node_supervisor)
        builder.add_node("researcher", self._node_researcher)
        builder.add_node("analyst", self._node_analyst)
        builder.add_node("writer", self._node_writer)
        builder.add_node("critic", self._node_critic)

        # 2. Add edges
        builder.add_edge(START, "supervisor")

        # Supervisor conditional branching
        builder.add_conditional_edges(
            "supervisor",
            self._route_condition,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "FINISH": "critic",
            },
        )

        # Workers loop back to supervisor to re-evaluate state
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")

        # Critic finishes the graph
        builder.add_edge("critic", END)

        return builder.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow graph and return updated ResearchState."""
        if self._compiled_graph is None:
            self._compiled_graph = self.build()

        result_dict: Any = self._compiled_graph.invoke(state.model_dump())
        if isinstance(result_dict, ResearchState):
            return result_dict
        return ResearchState.model_validate(result_dict)
