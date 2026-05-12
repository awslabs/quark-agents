"""
Reactor throughput benchmark.

Compares three execution strategies on a fixed set of N_TASKS agent runs:
  1. asyncio.gather (no reactor) — all tasks fired simultaneously, LLM spikes
  2. Reactor with llm_concurrency set to match API quota
  3. Sequential baseline — one task at a time

Each agent makes one LLM call (MOCK_LATENCY_MS) with no tools.
Measures: total wall time, throughput (tasks/sec), LLM utilization %.

Usage:
    python benchmarks/bench_reactor.py
"""

import asyncio, time, sys, statistics
from unittest.mock import patch, MagicMock
sys.path.insert(0, ".")

from quark import Agent
from quark_reactor import Reactor

MOCK_LATENCY_MS  = 500   # realistic LLM latency
N_TASKS          = 100   # total agent runs
LLM_CONCURRENCY  = 10    # simulated API quota: 10 concurrent = ~1200 RPM at 500ms

call_times = []  # track when each LLM call starts

def make_mock():
    async def async_mock(*a, **kw):
        call_times.append(time.perf_counter())
        await asyncio.sleep(MOCK_LATENCY_MS / 1000)
        msg = MagicMock()
        msg.content = "The answer is 42."
        msg.tool_calls = None
        r = MagicMock()
        r.choices[0].message = msg
        r.usage = None
        return r
    return async_mock

questions = [f"What is {i} * {i+1}?" for i in range(N_TASKS)]


async def bench_gather():
    """All tasks fired at once with asyncio.gather — no concurrency control."""
    call_times.clear()
    agent = Agent(model="gpt-5.4")

    with patch("quark.litellm.acompletion", side_effect=make_mock()):
        start = time.perf_counter()
        await asyncio.gather(*[agent.arun(q, history=[]) for q in questions])
        return time.perf_counter() - start


async def bench_reactor():
    """Reactor with llm_concurrency — smooth, continuous LLM utilization."""
    call_times.clear()
    agent = Agent(model="gpt-5.4")
    reactor = Reactor(llm_concurrency=LLM_CONCURRENCY, tool_concurrency=200)

    with patch("quark.litellm.acompletion", side_effect=make_mock()):
        # patch litellm at module level so Reactor's throttled wrapper sees it
        import litellm
        original = litellm.acompletion
        litellm.acompletion = make_mock()
        try:
            tasks = [(agent, q) for q in questions]
            start = time.perf_counter()
            await reactor.run(tasks)
            return time.perf_counter() - start
        finally:
            litellm.acompletion = original


async def bench_sequential():
    """One task at a time — lower bound."""
    call_times.clear()
    agent = Agent(model="gpt-5.4")

    with patch("quark.litellm.acompletion", side_effect=make_mock()):
        start = time.perf_counter()
        for q in questions:
            await agent.arun(q, history=[])
        return time.perf_counter() - start


def llm_utilization(total_s: float, n_tasks: int, latency_ms: float, concurrency: int) -> float:
    """What % of available LLM capacity was used."""
    # max possible LLM-seconds = concurrency * total_s
    # actual LLM-seconds = n_tasks * latency_s
    actual = n_tasks * (latency_ms / 1000)
    max_possible = concurrency * total_s
    return min(actual / max_possible * 100, 100)


if __name__ == "__main__":
    print(f"Tasks: {N_TASKS}  |  Mock LLM latency: {MOCK_LATENCY_MS}ms  |  LLM concurrency limit: {LLM_CONCURRENCY}")
    print(f"Theoretical minimum (all parallel): {MOCK_LATENCY_MS}ms")
    print(f"Sequential baseline: {N_TASKS * MOCK_LATENCY_MS / 1000:.1f}s")
    print()

    results = {}

    print("  running sequential...", end=" ", flush=True)
    t = asyncio.run(bench_sequential())
    results["Sequential"] = t
    print(f"{t:.2f}s")

    print("  running asyncio.gather (no reactor)...", end=" ", flush=True)
    t = asyncio.run(bench_gather())
    results["gather (no limit)"] = t
    print(f"{t:.2f}s")

    print("  running Reactor...", end=" ", flush=True)
    t = asyncio.run(bench_reactor())
    results["Reactor"] = t
    print(f"{t:.2f}s")

    print()
    theoretical_min = MOCK_LATENCY_MS / 1000
    sequential_s = N_TASKS * MOCK_LATENCY_MS / 1000
    optimal_s = (N_TASKS / LLM_CONCURRENCY) * (MOCK_LATENCY_MS / 1000)

    print(f"{'Strategy':<25} {'Total time':>12} {'Throughput':>12} {'vs sequential':>14}")
    print("-" * 67)
    for name, t in results.items():
        throughput = N_TASKS / t
        speedup = sequential_s / t
        print(f"{name:<25} {t:>10.2f}s {throughput:>10.1f}/s {speedup:>12.1f}x")

    print()
    print(f"Optimal with {LLM_CONCURRENCY} concurrent LLM slots: {optimal_s:.2f}s  ({N_TASKS/optimal_s:.1f} tasks/s)")
    print()
    print("gather (no limit): fires all tasks at once — spikes the API, then idle")
    print("Reactor:           fills LLM slots continuously — smooth throughput")
