"""
Tests for quark_ray.py.

Unit tests (local Ray, mocked LLM):  pytest tests/test_quark_ray.py
Integration tests (real LLM):        pytest tests/test_quark_ray.py -m integration
"""

import sys
import pytest

sys.path.insert(0, ".")

ray = pytest.importorskip("ray", reason="ray not installed — skip quark_ray tests")

# Initialize Ray once before importing quark_ray so @ray.remote decorators
# register against an active cluster.
if not ray.is_initialized():
    ray.init(
        num_cpus=4,
        ignore_reinit_error=True,
        runtime_env={"working_dir": None},
        resources={"worker": 10},  # required for tasks using resources={"worker": 1}
    )

from quark import Agent, Workflow, tool
from quark_ray import ray_run, _build_chain, _dispatch, _combine_fanout


# ---------------------------------------------------------------------------
# _dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_dispatch_agent_returns_object_ref(self):
        @tool
        def upper(x): return x.upper()
        ref = ray.put("hello")
        result_ref = _dispatch(upper, ref)
        assert isinstance(result_ref, ray.ObjectRef)
        assert ray.get(result_ref) == "HELLO"

    def test_dispatch_tool_node_returns_object_ref(self):
        @tool
        def upper(x): return x.upper()
        ref = ray.put("hello")
        result_ref = _dispatch(upper, ref)
        assert isinstance(result_ref, ray.ObjectRef)

    def test_dispatch_plain_function_returns_object_ref(self):
        from quark import _wrap
        node = _wrap(lambda x: x + "!")
        ref = ray.put("hello")
        result_ref = _dispatch(node, ref)
        assert isinstance(result_ref, ray.ObjectRef)

    def test_dispatch_nested_workflow_returns_object_ref(self):
        @tool
        def upper(x): return x.upper()
        @tool
        def exclaim(x): return x + "!"
        nested = upper >> exclaim
        ref = ray.put("hello")
        result_ref = _dispatch(nested, ref)
        assert isinstance(result_ref, ray.ObjectRef)


# ---------------------------------------------------------------------------
# _combine_fanout
# ---------------------------------------------------------------------------

class TestCombineFanout:
    def test_combines_results_with_names(self):
        result = ray.get(_combine_fanout.remote(
            "original input",
            ["agent_a", "agent_b"],
            "result A",
            "result B",
        ))
        assert "[original]:" in result
        assert "original input" in result
        assert "[agent_a]:" in result
        assert "result A" in result
        assert "[agent_b]:" in result
        assert "result B" in result

    def test_separator_between_results(self):
        result = ray.get(_combine_fanout.remote("x", ["a", "b"], "r1", "r2"))
        assert "---" in result


# ---------------------------------------------------------------------------
# _build_chain
# ---------------------------------------------------------------------------

class TestBuildChain:
    def test_returns_object_ref(self):
        @tool
        def upper(x): return x.upper()
        ref = ray.put("hello")
        from quark import _wrap
        result_ref = _build_chain([_wrap(lambda x: x.upper())], ref)
        assert isinstance(result_ref, ray.ObjectRef)

    def test_sequential_chain_passes_output_forward(self):
        @tool
        def add_bang(x): return x + "!"
        @tool
        def add_q(x): return x + "?"

        ref = ray.put("hello")
        result_ref = _build_chain([add_bang, add_q], ref)
        result = ray.get(result_ref)
        assert result == "hello!?"

    def test_fanout_produces_combined_output(self):
        @tool
        def shout(x): return x.upper()
        @tool
        def whisper(x): return x.lower()

        ref = ray.put("Hello")
        result_ref = _build_chain([[shout, whisper]], ref)
        result = ray.get(result_ref)
        assert "[original]:" in result
        assert "HELLO" in result
        assert "hello" in result


# ---------------------------------------------------------------------------
# ray_run — end-to-end with mocked LLM
# ---------------------------------------------------------------------------

class TestRayRun:
    def test_tool_only_pipeline(self):
        @tool
        def shout(x): return x.upper()
        pipeline = Workflow([shout])
        result = ray_run(pipeline, "hello world")
        assert result == "HELLO WORLD"

    def test_fanout_pipeline_with_tools(self):
        @tool
        def shout(x): return x.upper()
        @tool
        def reverse(x): return x[::-1]
        @tool
        def exclaim(x): return x + "!"

        pipeline = Workflow([[shout, reverse, exclaim]])
        result = ray_run(pipeline, "hello")
        assert "[original]:" in result
        assert "HELLO" in result
        assert "olleh" in result
        assert "hello!" in result

    def test_sequential_tool_pipeline(self):
        @tool
        def add_bang(x): return x + "!"
        @tool
        def add_q(x): return x + "?"

        pipeline = add_bang >> add_q
        result = ray_run(pipeline, "hello")
        assert result == "hello!?"

    def test_batch_tool_pipeline_returns_list(self):
        @tool
        def shout(x): return x.upper()
        pipeline = Workflow([shout])
        results = ray_run(pipeline, ["a", "b", "c"])
        assert results == ["A", "B", "C"]

    def test_batch_preserves_order(self):
        @tool
        def identity(x): return x
        pipeline = Workflow([identity])
        inputs = [f"input_{i}" for i in range(5)]
        results = ray_run(pipeline, inputs)
        assert results == inputs


# ---------------------------------------------------------------------------
# Integration tests (real LLM)
# ---------------------------------------------------------------------------

MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

@pytest.mark.integration
def test_ray_integration_sequential_pipeline():
    summarizer = Agent(system="Repeat the input in exactly 3 words.", model=MODEL, name="summarizer")
    critic = Agent(system="Say 'looks good' if the input is short.", model=MODEL, name="critic")
    pipeline = summarizer >> critic
    result = ray_run(pipeline, "The sky is blue and the sun is shining brightly today.")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.integration
def test_ray_integration_fanout_pipeline():
    summarizer = Agent(system="Summarize in 5 words.", model=MODEL, name="summarizer")
    critic = Agent(system="List one weakness.", model=MODEL, name="critic")
    fact_checker = Agent(system="Say 'facts ok' if nothing is wrong.", model=MODEL, name="fact_checker")
    editor = Agent(system="Write a one sentence final version.", model=MODEL, name="editor")

    pipeline = summarizer >> [critic, fact_checker] >> editor
    result = ray_run(pipeline, "Photosynthesis converts sunlight into energy in plants.")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.integration
def test_ray_integration_batch():
    a = Agent(system="Reply with exactly: pong", model=MODEL, name="pong")
    pipeline = Workflow([a])
    results = ray_run(pipeline, ["ping", "ping", "ping"])
    assert len(results) == 3
    assert all("pong" in r.lower() for r in results)
