"""
quark_reactor — Throughput scheduler for Quark Agents.

Install: pip install "quark-agents[reactor]"

The Reactor maximizes end-to-end agent throughput by continuously filling
available LLM and tool capacity across many concurrent agent runs. Rather
than running agents in isolated async tasks that all spike the LLM at once,
the Reactor interleaves turns across agents so LLM slots and tool slots stay
saturated at all times.

Usage:
    from quark import Agent
    from quark_reactor import Reactor

    researcher = Agent(system="You research topics.", tools=[search], model="gpt-5.4")
    writer = Agent(system="You write summaries.", model="gpt-5.4")

    reactor = Reactor(llm_concurrency=50, tool_concurrency=200)

    tasks = [
        (researcher, "What is quantum entanglement?"),
        (writer, "Summarize: black holes"),
        (researcher, "Latest on LLM scaling laws"),
        # ... thousands of tasks
    ]

    results = await reactor.run(tasks)

Design:
    Each task is an independent (agent, input, history) triple. The Reactor
    maintains two semaphores — one for LLM calls, one for tool execution —
    and dispatches turns as capacity becomes available. Agents at different
    stages interleave naturally: while agent A waits on an LLM response,
    agent B's tool result is processed and agent C starts its next turn.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

try:
    from quark import Agent
except ImportError:
    raise ImportError("quark-agents is required. Install with: pip install quark-agents")


@dataclass
class _Task:
    agent: Agent
    input: str
    history: list = field(default_factory=list)
    result: str = None
    done: bool = False


class Reactor:
    """Throughput scheduler — maximizes agent completions per unit time.

    Args:
        llm_concurrency:  Max simultaneous LLM calls in flight. Set to your
                          API rate limit headroom (e.g. RPM / 60 * avg_latency_s).
        tool_concurrency: Max simultaneous tool executions in flight.
    """

    def __init__(self, llm_concurrency: int = 50, tool_concurrency: int = 200):
        self.llm_concurrency = llm_concurrency
        self.tool_concurrency = tool_concurrency

    async def run(self, tasks: list[tuple]) -> list[str]:
        """Run all (agent, input) or (agent, input, history) tasks concurrently.

        Returns results in the same order as tasks.
        """
        llm_sem = asyncio.Semaphore(self.llm_concurrency)
        tool_sem = asyncio.Semaphore(self.tool_concurrency)

        work = []
        for item in tasks:
            if len(item) == 2:
                agent, inp = item
                history = []
            else:
                agent, inp, history = item
            work.append(_Task(agent=agent, input=inp, history=list(history)))

        async def run_task(task: _Task):
            result = await self._run_with_semaphores(task.agent, task.input, task.history, llm_sem, tool_sem)
            if isinstance(result, tuple):
                task.result, task.history = result
            else:
                task.result = result
            task.done = True

        await asyncio.gather(*[run_task(t) for t in work])
        return [t.result for t in work]

    async def _run_with_semaphores(self, agent: Agent, user: str, history: list, llm_sem, tool_sem):
        """Run one agent turn, acquiring semaphores around LLM and tool calls."""
        import json, inspect
        from unittest.mock import patch

        original_acompletion = None

        async def throttled_acompletion(*args, **kwargs):
            async with llm_sem:
                import litellm
                return await litellm.acompletion(*args, **kwargs)

        async def throttled_arun_tools(self_agent, tool_calls, h):
            async def exec_one(tc):
                async with tool_sem:
                    return await self_agent._exec_tool(tc)
            results = await asyncio.gather(*[exec_one(tc) for tc in tool_calls])
            for m in results:
                h.append(m)

        with patch("quark.litellm.acompletion", side_effect=throttled_acompletion):
            agent._arun_tools = lambda tcs, h: throttled_arun_tools(agent, tcs, h)
            return await agent.arun(user, history=history)
