# Quark

A ~200-line Python agentic framework. Define agents with a system prompt and tools, then compose them into pipelines using the `>>` operator — just like Airflow, but for LLMs.

```python
from quark import Agent

agent = Agent(system="You are a helpful assistant.", model="gpt-4o")
print(agent.run("What is the capital of France?"))
```

## Why Quark?

Every major agentic framework — LangChain, CrewAI, AutoGen, LlamaIndex — solves the same core problem: call an LLM, execute tools if requested, loop until done. When you strip them down to their source code, the core loop is identical. The rest is abstraction on top of abstraction.

Quark is the irreducible core. A single file you can read, understand, and own in an afternoon. It is not a toy — it supports streaming, parallel tool execution, multi-agent pipelines, and production-grade OpenTelemetry tracing. But it never does more than you asked for.

## Why not just use LangChain?

Every major agentic framework reduces to the same ~20-line loop. Quark exposes that loop directly. See the [Framework Comparison](comparison.md) for a source-level analysis of 15 frameworks and why Quark makes the choices it does.

## The name

In particle physics, quarks are the smallest known fundamental constituents of matter. They cannot exist in isolation — they are always bound together by gluons. The name captures two things: the size (minimal, irreducible) and the purpose (binding agents together into something greater).

## Install

```bash
# From PyPI
pip install quark

# From source
git clone https://gitlab.aws.dev/subshrey/quark
cd quark
pip install .

# With OpenTelemetry support
pip install "quark[otel]"
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
