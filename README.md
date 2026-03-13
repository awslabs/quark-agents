# Quark

A ~200-line Python agentic framework. Define agents with a system prompt and tools, then compose them into pipelines using the `>>` operator — just like Airflow, but for LLMs. Provider-agnostic via [litellm](https://github.com/BerriAI/litellm).

## Install

```bash
# From PyPI (once available)
pip install quark

# From source
git clone https://gitlab.aws.dev/subshrey/quark
cd quark
pip install .

# With OpenTelemetry support
pip install "quark[otel]"
```

### Install with uv

```bash
git clone https://gitlab.aws.dev/subshrey/quark
cd quark
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
    model="gpt-4o",  # or any litellm-supported model
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
    model="gpt-4o",
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
agent = Agent(model="gpt-4o")

# Anthropic
agent = Agent(model="claude-opus-4-6")

# AWS Bedrock
agent = Agent(model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0")

# Gemini
agent = Agent(model="gemini/gemini-2.0-flash")

# Ollama (local)
agent = Agent(model="ollama/llama3")
```

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
| `model` | `"gpt-4o"` | Any litellm model string |
| `max_turns` | `10` | Max LLM iterations per `run()` call |
| `name` | `"agent"` | Name used in traces and pipeline display |

**Methods:**
- `agent.run(user: str) -> str` — blocking, returns final answer
- `agent.stream(user: str) -> Generator` — yields tokens as they arrive
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
