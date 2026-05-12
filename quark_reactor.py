"""
quark_reactor — Throughput scheduler for Quark Agents.

Install: pip install "quark-agents[reactor]"

The Reactor maximizes end-to-end agent throughput by interleaving turns across
many concurrent agent runs. Rather than running agents in isolated tasks that
spike the LLM simultaneously, the Reactor gates all LLM calls and tool
executions through shared semaphores — keeping API capacity saturated while
preventing thundering herd bursts.

    Without Reactor: agents bunch up, LLM idle between waves, throughput < quota
    With Reactor:    LLM slots filled continuously, throughput approaches quota

Usage:
    from quark import Agent
    from quark_reactor import Reactor

    researcher = Agent(system="You research topics.", tools=[search], model="gpt-5.4")
    writer     = Agent(system="You write summaries.", model="gpt-5.4")

    reactor = Reactor(llm_concurrency=50, tool_concurrency=200)

    results = await reactor.run([
        (researcher, "What is quantum entanglement?"),
        (writer,     "Summarize: black holes"),
        (researcher, "Latest on LLM scaling laws"),
        # ... thousands of tasks
    ])

llm_concurrency:  set to floor(RPM_quota / 60 * avg_latency_s)
                  e.g. 1000 RPM quota, 500ms latency → 1000/60*0.5 ≈ 8
tool_concurrency: set generously — tools are cheap, keep them saturated
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import litellm

try:
    from quark import Agent
except ImportError:
    raise ImportError("quark-agents is required. Install with: pip install quark-agents")


@dataclass
class ReactorResult:
    response: str
    history: list
    duration_s: float


class Reactor:
    """Throughput scheduler — maximizes agent completions per unit time.

    All (agent, input) tasks run concurrently. LLM calls and tool executions
    are gated through shared semaphores so capacity is filled continuously
    rather than in waves.

    Args:
        llm_concurrency:  Max simultaneous LLM calls. Rule of thumb:
                          floor(RPM_quota / 60 * avg_latency_s).
        tool_concurrency: Max simultaneous tool executions.
    """

    def __init__(self, llm_concurrency: int = 50, tool_concurrency: int = 200):
        self.llm_concurrency = llm_concurrency
        self.tool_concurrency = tool_concurrency

    async def run(self, tasks: list[tuple], return_history: bool = False) -> list:
        """Run all tasks concurrently under shared capacity constraints.

        Args:
            tasks: list of (agent, input) or (agent, input, history) tuples.
            return_history: if True, returns list of ReactorResult instead of list of str.

        Returns results in the same order as tasks.
        """
        llm_sem  = asyncio.Semaphore(self.llm_concurrency)
        tool_sem = asyncio.Semaphore(self.tool_concurrency)

        async def throttled_acompletion(*args, **kwargs):
            async with llm_sem:
                return await _original_acompletion(*args, **kwargs)

        _original_acompletion = litellm.acompletion
        litellm.acompletion = throttled_acompletion

        async def throttled_arun_tools(agent_self, tool_calls, h):
            async def _exec(tc):
                async with tool_sem:
                    return await agent_self._exec_tool(tc)
            for m in await asyncio.gather(*[_exec(tc) for tc in tool_calls]):
                h.append(m)

        try:
            coros = []
            for item in tasks:
                agent, inp = item[0], item[1]
                history = list(item[2]) if len(item) > 2 else []
                agent._arun_tools = lambda tcs, h, a=agent: throttled_arun_tools(a, tcs, h)
                coros.append(self._run_one(agent, inp, history, return_history))

            return list(await asyncio.gather(*coros))
        finally:
            litellm.acompletion = _original_acompletion
            for item in tasks:
                agent = item[0]
                agent.__dict__.pop('_arun_tools', None)

    async def _run_one(self, agent: Agent, user: str, history: list, return_history: bool):
        t0 = time.perf_counter()
        result = await agent.arun(user, history=history)
        duration = time.perf_counter() - t0
        if isinstance(result, tuple):
            response, history = result
        else:
            response, history = result, []
        if return_history:
            return ReactorResult(response=response, history=history, duration_s=duration)
        return response
