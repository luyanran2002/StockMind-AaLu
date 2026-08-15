from app.graph.state import create_initial_state


def test_initial_state_defaults():
    state = create_initial_state("Analyze NVDA", ticker="NVDA")
    assert state["user_query"] == "Analyze NVDA"
    assert state["ticker"] == "NVDA"
    assert state["messages"][0].type == "human"
    assert state["iteration_count"] == 0
    assert state["max_iterations"] == 8
    assert state["status"] == "running"
    assert state["final_output"] is None
    assert state["errors"] == []


def test_initial_state_custom_max_iterations():
    state = create_initial_state("Query", max_iterations=3)
    assert state["max_iterations"] == 3

