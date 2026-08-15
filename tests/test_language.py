from app.agents.prompts import get_finalize_prompt, get_system_prompt
from app.graph.state import create_initial_state


def test_language_in_state():
    state = create_initial_state("查询", language="zh")
    assert state["language"] == "zh"


def test_chinese_system_prompt():
    prompt = get_system_prompt("zh")
    assert "简体中文" in prompt


def test_chinese_finalize_prompt():
    prompt = get_finalize_prompt("zh")
    assert "简体中文" in prompt


def test_unknown_language_falls_back_to_english():
    assert get_system_prompt("xx") == get_system_prompt("en")

