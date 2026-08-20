"""Unit tests for agents, supervisor routing, and workflow."""

from unittest.mock import MagicMock

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


def test_supervisor_routing_sequence() -> None:
    supervisor = SupervisorAgent()
    state = ResearchState(request=ResearchQuery(query="Test query about GraphRAG"))

    # Stage 1: Empty state -> should route to researcher
    assert supervisor.decide_next_route(state) == "researcher"
    supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    # Stage 2: Has sources & notes -> should route to analyst
    state.sources = [SourceDocument(title="Doc 1", snippet="Snippet 1")]
    state.research_notes = "Found key facts."
    assert supervisor.decide_next_route(state) == "analyst"
    supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    # Stage 3: Has analysis notes -> should route to writer
    state.analysis_notes = "Analyzed tradeoffs."
    assert supervisor.decide_next_route(state) == "writer"
    supervisor.run(state)
    assert state.route_history[-1] == "writer"

    # Stage 4: Has final answer -> should route to FINISH
    state.final_answer = "Final report with citations [1]."
    assert supervisor.decide_next_route(state) == "FINISH"


def test_supervisor_max_iterations_guardrail() -> None:
    supervisor = SupervisorAgent()
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.iteration = 6
    state.research_notes = "Some notes"
    # When reached max iterations without final answer, forces writer
    assert supervisor.decide_next_route(state) == "writer"

    state.final_answer = "Done"
    # When already has final answer and reached max iterations, finishes
    assert supervisor.decide_next_route(state) == "FINISH"


def test_researcher_agent_with_mock_services() -> None:
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.complete.return_value = LLMResponse(content="Synthesized research notes")
    mock_search = MagicMock(spec=SearchClient)
    mock_search.search.return_value = [
        SourceDocument(title="Doc A", url="http://a.com", snippet="Content A")
    ]

    agent = ResearcherAgent(llm_client=mock_llm, search_client=mock_search)
    state = ResearchState(request=ResearchQuery(query="Explain GraphRAG"))

    updated = agent.run(state)
    assert len(updated.sources) == 1
    assert updated.research_notes == "Synthesized research notes"
    assert any(res.agent == "researcher" for res in updated.agent_results)


def test_analyst_agent_with_mock_llm() -> None:
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.complete.return_value = LLMResponse(content="Tradeoffs and insights analysis")

    agent = AnalystAgent(llm_client=mock_llm)
    state = ResearchState(
        request=ResearchQuery(query="Explain GraphRAG"),
        research_notes="Raw notes",
    )

    updated = agent.run(state)
    assert updated.analysis_notes == "Tradeoffs and insights analysis"
    assert any(res.agent == "analyst" for res in updated.agent_results)


def test_writer_agent_with_mock_llm() -> None:
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.complete.return_value = LLMResponse(
        content="# Report\n\nContent based on [1].\n\n## References\n[1] Doc A"
    )

    agent = WriterAgent(llm_client=mock_llm)
    state = ResearchState(
        request=ResearchQuery(query="Explain GraphRAG"),
        sources=[SourceDocument(title="Doc A", snippet="Snippet A")],
        analysis_notes="Key analysis",
    )

    updated = agent.run(state)
    assert updated.final_answer is not None
    assert "[1]" in updated.final_answer


def test_critic_agent_citation_coverage() -> None:
    critic = CriticAgent()
    state = ResearchState(
        request=ResearchQuery(query="Test query with citation check"),
        sources=[
            SourceDocument(title="Source 1", snippet="Snippet 1"),
            SourceDocument(title="Source 2", snippet="Snippet 2"),
        ],
        final_answer="According to [1] and [2], the system is robust.",
    )

    updated = critic.run(state)
    critic_result = next(res for res in updated.agent_results if res.agent == "critic")
    assert critic_result.metadata["citation_coverage"] == 1.0


def test_multi_agent_workflow_end_to_end() -> None:
    workflow = MultiAgentWorkflow()
    # Mock LLM calls inside workflow components
    mock_response = LLMResponse(content="Structured output with citations [1].")
    workflow.researcher.llm_client.complete = MagicMock(return_value=mock_response)  # type: ignore[method-assign]
    workflow.analyst.llm_client.complete = MagicMock(return_value=mock_response)  # type: ignore[method-assign]
    workflow.writer.llm_client.complete = MagicMock(return_value=mock_response)  # type: ignore[method-assign]

    state = ResearchState(request=ResearchQuery(query="GraphRAG State of the Art"))
    final_state = workflow.run(state)

    assert final_state.final_answer is not None
    assert len(final_state.sources) > 0
    assert len(final_state.route_history) >= 3
    assert "researcher" in final_state.route_history
    assert "analyst" in final_state.route_history
    assert "writer" in final_state.route_history
