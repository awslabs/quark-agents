# Quark Agents

> **Experimental.** An ongoing exploration into the simplest possible agentic framework — use it to learn, hack, and break agentic things.

A ~350-line Python agentic framework, named after the smallest known fundamental particles. Define agents with a system prompt and tools, then compose them into pipelines using the `>>` operator.

Despite being a single file, you get:

- 100+ model providers via litellm
- Multi-agent workflows with `>>`
- Parallel fan-out and tool execution
- Async-first (`arun`, `astream`) — thousands of concurrent agents on one event loop
- Stateless mode — pass history in, get it back, deploy anywhere
- **Reactor** — built-in quota management for running agents at scale
- OpenTelemetry tracing
- Streaming and conversation memory

```python
from quark import Agent

agent = Agent(system="You are a helpful assistant.", model="gpt-5.4")
print(agent.run("What is the capital of France?"))
```

## Why Quark?

Every major agentic framework — LangChain, CrewAI, AutoGen, LlamaIndex — solves the same core problem: call an LLM, execute tools if requested, loop until done. When you strip them down to their source code, the core loop is identical. The rest is abstraction on top of abstraction.

Quark is the irreducible core of what makes an agent useful. A single file you can read, understand, and own in an afternoon. It is not a toy — it supports streaming, parallel tool execution, multi-agent pipelines, async concurrency, and production-grade OpenTelemetry tracing. But it never does more than you asked for.

The thing most frameworks don't ship: **quota management**. When you fire hundreds of LLM calls simultaneously, your API rate limit becomes the bottleneck. Quark ships [Reactor](reactor.md) — a semaphore-gated scheduler that keeps your throughput at your quota ceiling instead of crashing into it.

Quark also ships **Ray integration** (`quark_ray`) for distributing pipelines across an EC2 cluster. The same `>>` pipeline runs locally or across 40 worker slots with one flag change. At 1000 tasks with a 5-LLM-call fanout pipeline, Quark on Ray outperforms LangGraph, Strands, and CrewAI — the parallel fan-out step runs 3 agents concurrently instead of sequentially.

See the [Framework Comparison](comparison.md) for a source-level analysis of 15 frameworks and benchmark results across local, local Ray, and EC2 cluster execution modes.

## Install

```bash
# From PyPI
pip install quark-agents

# From source
git clone https://github.com/awslabs/quark-agents
cd quark-agents
pip install .

# With OpenTelemetry support
pip install "quark-agents[otel]"
```

## At a glance

```python
from quark import Agent

# A plain function — no LLM needed
def fetch_article(url: str) -> str:
    return open("article.txt").read()

# Three agents
summarizer   = Agent(system="Summarize in 3 bullet points.", name="summarizer")
critic       = Agent(system="List 2 weaknesses in this summary.", name="critic")
fact_checker = Agent(system="List 2 facts to verify.", name="fact_checker")
editor       = Agent(system="Write a final improved summary given all feedback.", name="editor")

# Compose into a pipeline
pipeline = fetch_article >> summarizer >> [critic, fact_checker] >> editor

# Run it
result = pipeline.run("https://example.com/article")
```

The `[critic, fact_checker]` step runs both agents in parallel and combines their output before passing it to `editor`.
