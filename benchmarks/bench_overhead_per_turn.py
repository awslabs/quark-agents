"""
Per-turn framework overhead benchmark.

Measures how much wall-clock time each framework adds per LLM turn,
excluding the LLM call itself. This is the framework tax that determines
how close you can get to your API rate limit ceiling.

Methodology:
- Mock the LLM at the HTTP level with a fixed MOCK_LATENCY_MS sleep
- Run N_TURNS sequential turns through each framework
- Total time = (N_TURNS × MOCK_LATENCY_MS) + framework_overhead
- Framework overhead = total_time - (N_TURNS × MOCK_LATENCY_MS)
- Overhead per turn = framework_overhead / N_TURNS

If overhead per turn > 1ms, the framework cannot saturate a 1000 RPM quota.
If overhead per turn > 10ms, the framework cannot saturate a 100 RPM quota.

Usage:
    python benchmarks/bench_overhead_per_turn.py
"""

import asyncio, time, json, sys, warnings, statistics
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock

MOCK_LATENCY_MS = 500   # realistic LLM latency
N_TURNS = 50            # turns per run
RUNS = 3                # runs to average over

def make_mock(n_turns_done=[0]):
    """Returns a fresh mock side_effect that always returns plain text."""
    def side_effect(*a, **kw):
        time.sleep(MOCK_LATENCY_MS / 1000)
        msg = MagicMock()
        msg.content = "The answer is 42."
        msg.tool_calls = None
        r = MagicMock()
        r.choices[0].message = msg
        r.usage = None
        return r
    return side_effect

async def make_async_mock():
    async def side_effect(*a, **kw):
        await asyncio.sleep(MOCK_LATENCY_MS / 1000)
        msg = MagicMock()
        msg.content = "The answer is 42."
        msg.tool_calls = None
        r = MagicMock()
        r.choices[0].message = msg
        r.usage = None
        return r
    return side_effect


# ---------------------------------------------------------------------------
# Quark sync
# ---------------------------------------------------------------------------

def bench_quark_sync():
    from quark import Agent
    times = []
    for _ in range(RUNS):
        with patch("quark.litellm.completion", side_effect=make_mock()):
            agent = Agent()
            agent.reset()
            start = time.perf_counter()
            for i in range(N_TURNS):
                agent.run(f"question {i}")
            elapsed = time.perf_counter() - start
        times.append(elapsed)
    return times


# ---------------------------------------------------------------------------
# Quark async
# ---------------------------------------------------------------------------

def bench_quark_async():
    from quark import Agent

    async def _run():
        times = []
        for _ in range(RUNS):
            agent = Agent()
            async def async_mock(*a, **kw):
                await asyncio.sleep(MOCK_LATENCY_MS / 1000)
                msg = MagicMock()
                msg.content = "The answer is 42."
                msg.tool_calls = None
                r = MagicMock()
                r.choices[0].message = msg
                r.usage = None
                return r
            with patch("quark.litellm.acompletion", side_effect=async_mock):
                agent.reset()
                start = time.perf_counter()
                for i in range(N_TURNS):
                    await agent.arun(f"question {i}")
                elapsed = time.perf_counter() - start
            times.append(elapsed)
        return times

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# LangGraph async
# ---------------------------------------------------------------------------

def bench_langgraph_async():
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.prebuilt import create_react_agent

    class SlowFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs): return self
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            time.sleep(MOCK_LATENCY_MS / 1000)
            return super()._generate(messages, stop, run_manager, **kwargs)

    async def _run():
        times = []
        for _ in range(RUNS):
            responses = [AIMessage(content="The answer is 42.")] * (N_TURNS + 5)
            llm = SlowFakeModel(responses=responses)
            agent = create_react_agent(llm, [])
            history = []
            start = time.perf_counter()
            for i in range(N_TURNS):
                result = agent.invoke({"messages": history + [HumanMessage(content=f"question {i}")]})
                history = result["messages"]
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        return times

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Strands
# ---------------------------------------------------------------------------

def bench_strands():
    from strands import Agent

    call_counts = [0]

    def make_bedrock_response():
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "The answer is 42."}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 10, "totalTokens": 20},
            "metrics": {"latencyMs": MOCK_LATENCY_MS},
        }

    def mock_converse(*a, **kw):
        time.sleep(MOCK_LATENCY_MS / 1000)
        call_counts[0] += 1
        return make_bedrock_response()

    times = []
    for _ in range(RUNS):
        call_counts[0] = 0
        try:
            with patch("boto3.Session.client") as mock_client:
                mock_client.return_value.converse = mock_converse
                agent = Agent()
                start = time.perf_counter()
                for i in range(N_TURNS):
                    agent(f"question {i}")
                elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            return None, str(e)
    return times


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(name, times, n_turns=N_TURNS, mock_ms=MOCK_LATENCY_MS):
    if times is None:
        print(f"{name:<25}  error")
        return

    expected_s = (n_turns * mock_ms) / 1000
    overheads = [(t - expected_s) * 1000 for t in times]  # ms total overhead
    per_turn = [o / n_turns for o in overheads]  # ms per turn

    avg_per_turn = statistics.mean(per_turn)
    total_time = statistics.mean(times)

    # at what RPM does this overhead become the bottleneck?
    # overhead_per_turn_s = avg_per_turn / 1000
    # bottleneck_rpm = 60 / overhead_per_turn_s  (if overhead > llm latency)
    overhead_s = avg_per_turn / 1000
    if overhead_s > 0:
        bottleneck_rpm = int(60 / overhead_s)
    else:
        bottleneck_rpm = float("inf")

    print(f"{name:<25}  {total_time:.2f}s total  {avg_per_turn:>8.2f}ms/turn overhead  "
          f"bottleneck>{bottleneck_rpm:>8,} RPM")


BENCHMARKS = [
    ("Quark (sync)",    bench_quark_sync),
    ("Quark (async)",   bench_quark_async),
    ("LangGraph",       bench_langgraph_async),
    ("Strands",         bench_strands),
]

if __name__ == "__main__":
    print(f"Mock LLM latency: {MOCK_LATENCY_MS}ms  |  Turns per run: {N_TURNS}  |  Runs: {RUNS}")
    print(f"Expected baseline (no overhead): {N_TURNS * MOCK_LATENCY_MS / 1000:.1f}s")
    print()

    results = {}
    for name, fn in BENCHMARKS:
        print(f"  running {name}...", flush=True)
        result = fn()
        if isinstance(result, tuple):
            result, err = result
            if result is None:
                print(f"    error: {err}")
                results[name] = None
                continue
        results[name] = result

    print()
    print(f"{'Framework':<25}  {'Total time':>12}  {'Overhead/turn':>15}  {'API bottleneck'}")
    print("-" * 80)
    for name, times in results.items():
        analyze(name, times)

    print()
    print("Overhead/turn = time added by framework per LLM call, excluding LLM latency.")
    print("API bottleneck = RPM at which framework overhead equals one full LLM call.")
    print("Above that RPM, the framework — not the API — becomes the bottleneck.")
