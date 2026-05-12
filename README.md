# Quark Agents

![source lines](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/awslabs/quark-agents/main/.github/badges/lines.json) [![docs](https://img.shields.io/badge/docs-awslabs.github.io-blue)](https://awslabs.github.io/quark-agents/)

> **Experimental.** An ongoing exploration into the simplest possible agentic framework — use it to learn, hack, and break agentic things.

Minimal Python framework for composing agents, tools, and multi-agent workflows. Define agents with a system prompt and tools, then compose them using the `>>` operator. Provider-agnostic via [litellm](https://github.com/BerriAI/litellm).

Despite being a single ~350-line file, you get:
- OpenTelemetry tracing
- 100+ model providers via litellm
- Multi-agent workflows with `>>`
- Parallel fan-out and tool execution
- Async-first (`arun`, `astream`) — thousands of concurrent agents on one event loop
- Stateless mode — pass history in, get it back, deploy anywhere
- Streaming
- Conversation memory

## Install

```bash
pip install quark-agents

# From source
git clone https://github.com/awslabs/quark-agents
cd quark-agents
pip install .

# With OpenTelemetry support
pip install "quark-agents[otel]"
```

### Install with uv

```bash
git clone https://github.com/awslabs/quark-agents
cd quark-agents
uv venv
source .venv/bin/activate

# Core + dev dependencies (pytest, mkdocs)
uv pip install ".[dev]"

# With OpenTelemetry
uv pip install ".[dev,otel]"

# With AWS Bedrock support
uv pip install ".[dev,bedrock]"

# All extras
uv pip install ".[dev,otel,bedrock]"
```

> **Note:** Editable installs (`-e`) require `setuptools>=75`. If you see
> `ModuleNotFoundError: No module named 'setuptools.backends'`, make sure
> `pyproject.toml` has `requires = ["setuptools>=75"]` under `[build-system]`,
> or use a non-editable install (`uv pip install ".[dev]"` without `-e`).

## Usage

### Single agent

```python
from quark import Agent

agent = Agent(
    system="You are a helpful assistant.",
    model="gpt-5.4",  # or any litellm-supported model
    name="assistant",
)

print(agent.run("What is the capital of France?"))
```

### Agent with tools

```python
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22°C in {city}"

agent = Agent(
    system="You are a weather assistant.",
    model="gpt-5.4",
    tools={"get_weather": get_weather},
)

print(agent.run("What's the weather in Paris?"))
```

### Pipelines with `>>`

Chain agents and plain functions using `>>`. Output of each step becomes input to the next.

```python
from quark import Agent

def fetch_article(url: str) -> str:
    """Fetch article content from a URL."""
    return "..."  # your fetch logic

summarizer = Agent(system="Summarize the article in 3 bullet points.", name="summarizer")
critic     = Agent(system="List 2 weaknesses in this summary.", name="critic")
editor     = Agent(system="Write a final improved summary given the feedback.", name="editor")

pipeline = fetch_article >> summarizer >> critic >> editor
result = pipeline.run("https://example.com/article")
```

### Parallel fan-out with lists

Wrap steps in a list to run them in parallel. Their outputs are combined and passed to the next step.

```python
pipeline = fetch_article >> summarizer >> [critic, fact_checker] >> editor
result = pipeline.run("https://example.com/article")
```

### Composing workflows

```python
research = fetch_article >> summarizer
review   = [critic, fact_checker] >> editor

pipeline = research >> review
result = pipeline.run("https://example.com/article")
```

### Streaming

```python
for chunk in agent.stream("Tell me a story."):
    print(chunk, end="", flush=True)
```

### Provider-agnostic

```python
# OpenAI
agent = Agent(model="gpt-5.4")

# Anthropic
agent = Agent(model="claude-opus-4-6")

# AWS Bedrock
agent = Agent(model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0")

# Gemini
agent = Agent(model="gemini/gemini-2.0-flash")

# Ollama (local)
agent = Agent(model="ollama/llama3")
```

### Async and stateless mode

```python
import asyncio

# Stateless — pass history in, get it back, deploy anywhere (Lambda, Ray, etc.)
response, history = await agent.arun("What is 2+2?", history=[])

# Run thousands concurrently
results = await asyncio.gather(*[agent.arun(prompt, history=[]) for prompt in prompts])
```

### Reactor — quota management at scale

`asyncio.gather` fires all LLM calls simultaneously and hits rate limits. Reactor gates calls through a semaphore so your quota becomes a throughput floor, not a ceiling.

```python
from quark import Agent
from quark_reactor import Reactor

analyst = Agent(system="Give a one-sentence buy/hold/sell.", model="bedrock/...")
tasks   = [(analyst, f"Analyze {ticker}") for ticker in stocks]

reactor = Reactor(llm_concurrency=35)
results = await reactor.run(tasks)
```

At 150 stocks with `llm_concurrency=35` against AWS Bedrock: **150/150 completed, zero failures, 22s**.
Plain `asyncio.gather` on the same workload: 75/150 completed, 75 throttled.

See [benchmarks/](benchmarks/) for the full multi-framework comparison.

### Observability (OpenTelemetry)

Set environment variables — tracing is enabled automatically.

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=my-app
```

Every `Agent.run()`, `Workflow.run()`, and tool call emits OTel spans. Compatible with Jaeger, Honeycomb, Grafana Tempo, Datadog, and any OTLP-compatible backend.

## API

### `Agent(*, system, tools, model, max_turns, name)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `system` | `"You are a helpful assistant."` | System prompt |
| `tools` | `{}` | Dict of `{name: callable}` |
| `model` | `"gpt-5.4"` | Any litellm model string |
| `max_turns` | `10` | Max LLM iterations per `run()` call |
| `name` | `"agent"` | Name used in traces and pipeline display |

**Methods:**
- `agent.run(user, history=None)` — blocking; pass `history=[]` for stateless mode → returns `(response, history)`
- `agent.arun(user, history=None)` — async; run thousands concurrently with `asyncio.gather`
- `agent.stream(user)` — yields tokens as they arrive
- `agent.astream(user)` — async streaming
- `agent.reset()` — clears conversation history, keeps system prompt

### `Workflow`

Created automatically by `>>`. Call `.run(input: str) -> str` to execute.

```python
workflow = agent_a >> agent_b >> agent_c
result = workflow.run("input")
```

## Tests

```bash
# Unit tests (no API calls)
pytest tests/

# Integration tests (requires API credentials)
pytest tests/ -m integration
```

If using uv, prefix with `uv run` to ensure the venv's Python is used (avoids conflicts with conda or system Python):

```bash
uv run pytest tests/
uv run pytest tests/ -m "not integration"
uv run pytest tests/ -m integration
```

## Why Quark?

Named after the smallest known fundamental particles — quarks need gluons to bind them together. Quark is the minimal binding layer for AI agents.
