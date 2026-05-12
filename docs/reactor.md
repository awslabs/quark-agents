# Reactor

Reactor is Quark's built-in quota manager for running agents at scale. It gates LLM calls through a semaphore so your API quota becomes a floor — not a ceiling you crash into.

## The problem

`asyncio.gather` fires all tasks simultaneously. At scale, this saturates your API rate limit and triggers 429 throttling errors and exponential backoff retries — turning a 22-second job into a 49-second job with failures.

```
gather — 150 stocks, no limit:
  t=0s   → 150 LLM calls fire simultaneously
  t=1s   → Bedrock returns 429s for ~half the calls
  t=1-9s → litellm retries with backoff
  t=9s   → 75 tasks completed, 75 failed
```

## The solution

Reactor gates calls with `asyncio.Semaphore(llm_concurrency)`. Tasks queue cheaply in memory. As slots free up, the next call dispatches immediately — no wasted time.

```
Reactor — 150 stocks, llm_concurrency=35:
  t=0s  → 35 LLM calls fire, 115 queue
  t=Ns  → as each call finishes, next queues dispatch
  t=22s → 150/150 completed, zero failures
```

## Install

Reactor is included with `quark-agents` — no extra install needed.

```bash
pip install quark-agents
```

## Usage

```python
from quark import Agent
from quark_reactor import Reactor

analyst = Agent(
    system="You are a financial analyst. Give a one-sentence buy/hold/sell.",
    model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
)

# Build (agent, prompt) pairs
tasks = [(analyst, f"Analyze {ticker}") for ticker in stocks]

reactor = Reactor(llm_concurrency=35)
results = await reactor.run(tasks)
```

`reactor.run()` returns a list of response strings in the same order as `tasks`.

### Return history

```python
results, histories = await reactor.run(tasks, return_history=True)
```

## Tuning `llm_concurrency`

Set it to roughly `floor(RPM_quota / 60 * avg_latency_s)`.

For example, with 10,000 RPM and ~3s average LLM latency:
```
floor(10000 / 60 * 3) = 500
```

Start conservative and increase until you see occasional 429s, then back off by 10-20%.

| Provider | Default on-demand quota | Suggested starting point |
|----------|------------------------|--------------------------|
| AWS Bedrock (Haiku 4.5) | varies by region | 35 |
| Anthropic API | 1,000 RPM (Tier 1) | 50 |
| OpenAI (gpt-4o-mini) | 500 RPM (Tier 1) | 25 |

## Benchmark results

Real Bedrock calls (Claude Haiku 4.5), 150 stocks, `llm_concurrency=35`.

| Framework | Mode | Completed | Wall-clock |
|-----------|------|-----------|------------|
| **Quark** | **Reactor (built-in)** | **150/150** | **22s** |
| LangGraph | gather (default) | 76/150 | 9s |
| Strands | gather (default) | 75/150 | 4s |
| CrewAI | gather (default) | 150/150 | 49s |

LangGraph and Strands complete faster in wall-clock — but only because they abandon half the work. CrewAI avoids failures due to high per-task overhead that accidentally self-throttles, but at 2× the time.

> The semaphore pattern can be applied manually to any framework. Quark ships it as a first-class primitive.

## How it works

Reactor monkey-patches `litellm.acompletion` for the duration of the run, wrapping it with a semaphore acquire/release. This means it works regardless of which framework or agent abstraction is calling the LLM underneath.

```python
sem = asyncio.Semaphore(llm_concurrency)

async def throttled(*args, **kwargs):
    async with sem:
        return await original_acompletion(*args, **kwargs)

litellm.acompletion = throttled
results = await asyncio.gather(*tasks)
litellm.acompletion = original_acompletion  # always restored
```

The semaphore is created fresh per `reactor.run()` call, so multiple reactors can run sequentially without interfering.
