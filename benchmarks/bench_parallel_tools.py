"""
Benchmark 1: Parallel tool execution

Tests how frameworks handle multiple tool calls returned in a single LLM response.
Each tool sleeps for TOOL_LATENCY_MS to simulate real I/O (API call, DB query, etc).

If a framework parallelizes: total time ≈ TOOL_LATENCY_MS (flat, regardless of N tools)
If a framework serializes:  total time ≈ N × TOOL_LATENCY_MS

LLM calls are mocked — this benchmark isolates framework tool dispatch only.

Usage:
    python benchmarks/bench_parallel_tools.py
"""

import time, json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock

TOOL_LATENCY_MS = 100
TOOL_COUNTS = [1, 2, 5, 10]
RUNS = 3  # average over this many runs per data point


# ---------------------------------------------------------------------------
# Quark
# ---------------------------------------------------------------------------

def _quark_side_effect(n_tools):
    calls = [0]
    def mock_completion(*a, **kw):
        calls[0] += 1
        msg = MagicMock()
        if calls[0] == 1:
            msg.content = None
            tcs = []
            for i in range(n_tools):
                tc = MagicMock()
                tc.id = f"call_{i}"
                tc.function.name = "slow_tool"
                tc.function.arguments = json.dumps({"x": str(i)})
                tcs.append(tc)
            msg.tool_calls = tcs
        else:
            msg.content = "done"
            msg.tool_calls = None
        r = MagicMock()
        r.choices[0].message = msg
        r.usage = None
        return r
    return mock_completion


def bench_quark(n_tools: int) -> float:
    from quark import Agent

    def slow_tool(x: str) -> str:
        """A slow tool."""
        time.sleep(TOOL_LATENCY_MS / 1000)
        return f"result:{x}"

    times = []
    for _ in range(RUNS):
        with patch("quark.litellm.completion", side_effect=_quark_side_effect(n_tools)):
            agent = Agent(tools={"slow_tool": slow_tool})
            start = time.perf_counter()
            agent.run("go")
            times.append(time.perf_counter() - start)
    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# LangGraph
# ---------------------------------------------------------------------------

def bench_langgraph(n_tools: int) -> float:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, ToolCall, HumanMessage
    from langchain_core.tools import tool as lc_tool
    from langgraph.prebuilt import create_react_agent

    @lc_tool
    def slow_tool(x: str) -> str:
        """A slow tool."""
        time.sleep(TOOL_LATENCY_MS / 1000)
        return f"result:{x}"

    class BindableFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    times = []
    for _ in range(RUNS):
        responses = [
            AIMessage(content="", tool_calls=[
                ToolCall(name="slow_tool", args={"x": str(i)}, id=f"call_{i}")
                for i in range(n_tools)
            ]),
            AIMessage(content="done"),
        ]
        llm = BindableFakeModel(responses=responses)
        agent = create_react_agent(llm, [slow_tool])
        start = time.perf_counter()
        agent.invoke({"messages": [HumanMessage(content="go")]})
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# Strands
# ---------------------------------------------------------------------------

def bench_strands(n_tools: int) -> float:
    from strands import Agent
    from strands.models.bedrock import BedrockModel
    from strands.tools import tool as strands_tool

    @strands_tool
    def slow_tool(x: str) -> str:
        """A slow tool."""
        time.sleep(TOOL_LATENCY_MS / 1000)
        return f"result:{x}"

    # Strands requires a real model client — mock at the boto3 level
    import boto3
    from unittest.mock import patch

    def make_bedrock_response(tool_calls=None, text=None):
        if tool_calls:
            content = [
                {"toolUse": {"toolUseId": f"call_{i}", "name": "slow_tool", "input": {"x": str(i)}}}
                for i in range(tool_calls)
            ]
            stop = "tool_use"
        else:
            content = [{"text": text or "done"}]
            stop = "end_turn"
        return {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": stop,
            "usage": {"inputTokens": 10, "outputTokens": 10, "totalTokens": 20},
            "metrics": {"latencyMs": 100},
        }

    call_counts = [0]
    def mock_converse(*a, **kw):
        call_counts[0] += 1
        if call_counts[0] == 1:
            return make_bedrock_response(tool_calls=n_tools)
        return make_bedrock_response(text="done")

    times = []
    for _ in range(RUNS):
        call_counts[0] = 0
        with patch("boto3.Session.client") as mock_client:
            mock_client.return_value.converse = mock_converse
            try:
                agent = Agent(tools=[slow_tool])
                start = time.perf_counter()
                agent("go")
                times.append(time.perf_counter() - start)
            except Exception:
                return float("nan")
    return sum(times) / len(times) if times else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BENCHMARKS = [
    ("Quark Agents", bench_quark),
    ("LangGraph",    bench_langgraph),
    ("Strands",      bench_strands),
]

if __name__ == "__main__":
    print(f"Tool latency: {TOOL_LATENCY_MS}ms each, averaged over {RUNS} runs")
    print(f"Serial baseline (N × {TOOL_LATENCY_MS}ms): ", end="")
    print("  ".join(f"{n} tools={n*TOOL_LATENCY_MS}ms" for n in TOOL_COUNTS))
    print()

    results = {}
    for name, bench_fn in BENCHMARKS:
        row = {}
        print(f"  {name}...", end=" ", flush=True)
        for n in TOOL_COUNTS:
            try:
                elapsed_ms = bench_fn(n) * 1000
                row[n] = elapsed_ms
                print(f"{n}t={elapsed_ms:.0f}ms", end="  ", flush=True)
            except Exception as e:
                row[n] = float("nan")
                print(f"{n}t=err({e})", end="  ", flush=True)
        results[name] = row
        print()

    # table
    col_w = 12
    print()
    header = f"{'Framework':<20}" + "".join(f"{f'{n} tools':>{col_w}}" for n in TOOL_COUNTS)
    print(header)
    print("-" * len(header))
    for name, row in results.items():
        line = f"{name:<20}"
        for n in TOOL_COUNTS:
            v = row.get(n, float("nan"))
            line += f"{f'{v:.0f}ms':>{col_w}}"
        print(line)

    print()
    print(f"Serial baseline      ", end="")
    for n in TOOL_COUNTS:
        print(f"{f'{n*TOOL_LATENCY_MS}ms':>{col_w}}", end="")
    print()
    print()
    print("Parallel = flat ~100ms regardless of N tools")
    print("Serial   = grows linearly with N tools")
