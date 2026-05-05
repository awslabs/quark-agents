# Quark Agents

> **Experimental.** An ongoing exploration into the simplest possible agentic framework — use it to learn, hack, and break agentic things.

A <300-line Python agentic framework, named after the smallest known fundamental particles. Define agents with a system prompt and tools, then compose them into pipelines using the `>>` operator — just like Airflow, but for LLMs.

```python
from quark import Agent

agent = Agent(system="You are a helpful assistant.", model="gpt-5.4")
print(agent.run("What is the capital of France?"))
```

## Why Quark?

Every major agentic framework — LangChain, CrewAI, AutoGen, LlamaIndex — solves the same core problem: call an LLM, execute tools if requested, loop until done. When you strip them down to their source code, the core loop is identical. The rest is abstraction on top of abstraction.

Quark is the irreducible core parts of what makes an agent useful. A single file you can read, understand, and own in an afternoon. It is not a toy — it supports streaming, parallel tool execution, multi-agent pipelines, and production-grade OpenTelemetry tracing. But it never does more than you asked for.

See the [Framework Comparison](comparison.md) for a source-level analysis of 15 frameworks and why Quark makes the choices it does.

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
