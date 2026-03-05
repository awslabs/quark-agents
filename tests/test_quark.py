"""
Tests for quark.py.

Unit tests:        pytest tests/
Integration tests: pytest tests/ -m integration
"""

import json
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")
from quark import Agent, Workflow, _run, _schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(content=None, tool_calls=None):
    """Build a fake litellm completion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    response = MagicMock()
    response.choices[0].message = msg
    return response


def _mock_tool_call(name, arguments: dict, call_id="call_1"):
    """Build a fake tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


# ---------------------------------------------------------------------------
# _schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_name_and_description(self):
        def greet(name: str) -> str:
            """Say hello."""
            pass
        s = _schema("greet", greet)
        assert s["function"]["name"] == "greet"
        assert s["function"]["description"] == "Say hello."

    def test_type_hints_mapped(self):
        def add(a: int, b: float, flag: bool, label: str) -> None:
            pass
        props = _schema("add", add)["function"]["parameters"]["properties"]
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"
        assert props["label"]["type"] == "string"

    def test_required_only_includes_params_without_defaults(self):
        def fn(required: str, optional: str = "default") -> None:
            pass
        params = _schema("fn", fn)["function"]["parameters"]
        assert "required" in params["required"]
        assert "optional" not in params["required"]

    def test_no_type_hints_falls_back_to_string(self):
        def fn(x) -> None:
            pass
        props = _schema("fn", fn)["function"]["parameters"]["properties"]
        assert "x" in props
        assert props["x"]["type"] == "string"

    def test_missing_docstring_uses_empty_string(self):
        def fn(x: str) -> None:
            pass
        assert _schema("fn", fn)["function"]["description"] == ""


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------

class TestRun:
    def test_dispatches_to_run_method(self):
        node = MagicMock()
        node.run.return_value = "result"
        assert _run(node, "input") == "result"
        node.run.assert_called_once_with("input")

    def test_calls_plain_function_directly(self):
        fn = lambda x: f"processed: {x}"
        assert _run(fn, "hello") == "processed: hello"

    def test_coerces_non_string_return_to_str(self):
        fn = lambda x: 42
        assert _run(fn, "anything") == "42"


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class TestWorkflow:
    def test_rshift_builds_step_list(self):
        a, b = MagicMock(), MagicMock()
        w = Workflow([a]) >> b
        assert w.steps == [a, b]

    def test_chaining_multiple_rshift_builds_correct_steps(self):
        a, b, c = MagicMock(), MagicMock(), MagicMock()
        w = Workflow([a]) >> b >> c
        assert w.steps == [a, b, c]

    def test_rrshift_prepends(self):
        fn = lambda x: x
        w = Workflow([MagicMock()])
        result = fn >> w
        assert result.steps[0].run("x") == "x"

    def test_run_passes_output_to_next_step(self):
        step1 = MagicMock(); step1.run.return_value = "after_step1"
        step2 = MagicMock(); step2.run.return_value = "after_step2"
        w = Workflow([step1, step2])
        result = w.run("initial")
        step1.run.assert_called_once_with("initial")
        step2.run.assert_called_once_with("after_step1")
        assert result == "after_step2"

    def test_parallel_list_runs_all_nodes_with_same_input(self):
        a = MagicMock(); a.run.return_value = "from_a"; a.name = "a"
        b = MagicMock(); b.run.return_value = "from_b"; b.name = "b"
        w = Workflow([[a, b]])
        result = w.run("input")
        a.run.assert_called_once_with("input")
        b.run.assert_called_once_with("input")
        assert "from_a" in result
        assert "from_b" in result

    def test_parallel_includes_original_in_output(self):
        a = MagicMock(); a.run.return_value = "feedback"; a.name = "a"
        w = Workflow([[a]])
        result = w.run("original input")
        assert "original input" in result

    def test_workflow_as_step_in_another_workflow(self):
        inner = MagicMock(spec=Workflow); inner.run.return_value = "inner_out"
        outer = Workflow([inner])
        result = outer.run("start")
        inner.run.assert_called_once_with("start")
        assert result == "inner_out"

    def test_plain_function_as_step(self):
        w = Workflow([lambda x: x.upper()])
        assert w.run("hello") == "HELLO"


# ---------------------------------------------------------------------------
# Agent (unit — mocked LLM)
# ---------------------------------------------------------------------------

class TestAgent:
    def test_initial_history_has_system_prompt(self):
        a = Agent(system="You are a bot.")
        assert a.history[0] == {"role": "system", "content": "You are a bot."}

    def test_reset_clears_history_keeps_system(self):
        a = Agent(system="sys")
        a.history.append({"role": "user", "content": "hi"})
        a.reset()
        assert len(a.history) == 1
        assert a.history[0]["content"] == "sys"

    def test_rshift_returns_workflow(self):
        a = Agent(name="a")
        b = Agent(name="b")
        w = a >> b
        assert isinstance(w, Workflow)
        assert w.steps == [a, b]

    def test_rrshift_plain_function(self):
        fn = lambda x: x
        a = Agent(name="a")
        w = fn >> a
        assert isinstance(w, Workflow)
        assert w.steps[0].run("x") == "x"
        assert w.steps[1] == a

    @patch("quark.litellm.completion")
    def test_run_returns_content_when_no_tool_calls(self, mock_completion):
        mock_completion.return_value = _mock_response(content="hello world")
        a = Agent()
        result = a.run("say hello")
        assert result == "hello world"

    @patch("quark.litellm.completion")
    def test_run_appends_user_and_assistant_to_history(self, mock_completion):
        mock_completion.return_value = _mock_response(content="hi")
        a = Agent()
        a.run("hello")
        assert a.history[1]["role"] == "user"
        assert a.history[1]["content"] == "hello"

    @patch("quark.litellm.completion")
    def test_run_calls_tool_and_loops(self, mock_completion):
        tool_call = _mock_tool_call("double", {"x": 21})
        mock_completion.side_effect = [
            _mock_response(tool_calls=[tool_call]),
            _mock_response(content="the answer is 42"),
        ]
        def double(x: int) -> int:
            """Double a number."""
            return x * 2
        a = Agent(tools={"double": double})
        result = a.run("double 21")
        assert result == "the answer is 42"
        assert mock_completion.call_count == 2

    @patch("quark.litellm.completion")
    def test_run_stops_at_max_turns(self, mock_completion):
        tc = _mock_tool_call("noop", {})
        mock_completion.return_value = _mock_response(tool_calls=[tc])
        a = Agent(max_turns=3, tools={"noop": lambda: "ok"})
        a.run("go")
        assert mock_completion.call_count == 3

    @patch("quark.litellm.completion")
    def test_tool_error_is_caught_and_reported(self, mock_completion):
        tc = _mock_tool_call("boom", {})
        mock_completion.side_effect = [
            _mock_response(tool_calls=[tc]),
            _mock_response(content="handled"),
        ]
        def boom():
            """Explode."""
            raise ValueError("kaboom")
        a = Agent(tools={"boom": boom})
        a.run("trigger boom")
        tool_msg = next(m for m in a.history if m.get("role") == "tool")
        assert "Error" in tool_msg["content"]

    @patch("quark.litellm.completion")
    def test_memory_persists_across_run_calls(self, mock_completion):
        mock_completion.return_value = _mock_response(content="ok")
        a = Agent()
        a.run("first message")
        a.run("second message")
        user_msgs = [m for m in a.history if isinstance(m, dict) and m.get("role") == "user"]
        assert len(user_msgs) == 2


# ---------------------------------------------------------------------------
# Integration tests (real Bedrock — run with: pytest -m integration)
# ---------------------------------------------------------------------------

MODEL = "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0"

@pytest.mark.integration
def test_integration_basic():
    a = Agent(model=MODEL)
    result = a.run("Reply with exactly: hello world")
    assert "hello" in result.lower()

@pytest.mark.integration
def test_integration_tool_call():
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
    agent = Agent(model=MODEL, tools={"multiply": multiply})
    result = agent.run("What is 6 times 7?")
    assert "42" in result

@pytest.mark.integration
def test_integration_memory():
    a = Agent(model=MODEL)
    a.run("My name is Alice.")
    result = a.run("What is my name?")
    assert "alice" in result.lower()

@pytest.mark.integration
def test_integration_streaming():
    a = Agent(model=MODEL)
    chunks = list(a.stream("Count to 3, numbers only, one per line."))
    assert len(chunks) > 0
    full = "".join(chunks)
    assert "1" in full and "2" in full and "3" in full

@pytest.mark.integration
def test_integration_workflow_pipeline():
    def shout(x: str) -> str:
        return x.upper()
    summarizer = Agent(system="Repeat the input back in exactly 5 words.", model=MODEL, name="summarizer")
    pipeline = shout >> summarizer
    result = pipeline.run("black holes are fascinating")
    assert len(result.split()) <= 10
