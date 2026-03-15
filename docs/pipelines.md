# Pipelines

Quark's pipeline DSL lets you compose agents and plain functions using the `>>` operator. The output of each step becomes the input to the next.

## Basic chaining

```python
from quark import Agent

def fetch(url: str) -> str:
    return open("article.txt").read()

summarizer = Agent(system="Summarize in 3 bullet points.", name="summarizer")
editor     = Agent(system="Polish the summary.", name="editor")

pipeline = fetch >> summarizer >> editor
result   = pipeline.run("https://example.com/article")
```

Steps can be:

- **Plain functions** — `def fn(x: str) -> str`
- **Async functions** — `async def fn(x: str) -> str` (handled transparently)
- **Agents** — `Agent(...)`
- **Workflows** — compose sub-workflows together

## Parallel fan-out

Wrap steps in a list to run them in parallel. All nodes receive the same input. Their outputs — plus the original input — are combined and passed to the next step.

```python
critic       = Agent(system="List 2 weaknesses.", name="critic")
fact_checker = Agent(system="List 2 facts to verify.", name="fact_checker")
editor       = Agent(system="Given all feedback, write a final summary.", name="editor")

pipeline = summarizer >> [critic, fact_checker] >> editor
```

The combined output fed into `editor` looks like:

```
[original]:
<summarizer output>

---

[critic]:
<critic output>

---

[fact_checker]:
<fact_checker output>
```

This gives the next step full context — the original input and all parallel feedback.

## Full example

```python
from quark import Agent

def fetch_article(url: str) -> str:
    """Fetch article content."""
    return "..."  # your fetch logic

summarizer   = Agent(system="Summarize in 3 bullet points.", name="summarizer")
critic       = Agent(system="List 2 weaknesses in this summary.", name="critic")
fact_checker = Agent(system="List 2 facts that need citations.", name="fact_checker")
editor       = Agent(system="Write a final polished summary given all feedback.", name="editor")

pipeline = fetch_article >> summarizer >> [critic, fact_checker] >> editor
result   = pipeline.run("https://example.com/article")
```

## Composing workflows

Workflows can be composed together. Each sub-workflow runs as a unit.

```python
research_flow = fetch_article >> summarizer         # Workflow
review_flow   = [critic, fact_checker] >> editor    # Workflow

full_pipeline = research_flow >> review_flow
result = full_pipeline.run("https://example.com")
```

Sub-workflows preserve their identity — they appear as a single step in the outer workflow, and their internal structure is visible in traces.

## Plain functions

Plain functions connect to agents with no decoration needed:

```python
def clean(text: str) -> str:
    return text.strip().lower()

def truncate(text: str) -> str:
    return text[:500]

# function >> agent — works
pipeline = clean >> agent

# agent >> function >> function — works
pipeline = agent >> clean >> truncate

# function >> function — just compose normally
result = clean(truncate(raw_text))
```

!!! note
    `fn >> fn` (two plain functions, no agent) is not supported via `>>` — just call them normally: `fn2(fn1(x))`. Use `@tool` (see below) if you need a plain function at the start of a chain before a list.

## @tool decorator

Plain functions work in pipelines without any decoration. But wrapping with `@tool` unlocks three additional features:

| Feature | Plain function | `@tool` |
|---------|---------------|---------|
| `>>` with agents | ✓ | ✓ |
| `fn >> [a, b]` (list anchor) | ✗ | ✓ |
| OTel span in traces | ✗ | ✓ |
| Retries on failure | ✗ | ✓ |
| Timeout | ✗ | ✓ |

```python
from quark import tool

# Basic — just adds >> support and OTel tracing
@tool
def fetch(url: str) -> str:
    return requests.get(url).text

# With retries and timeout
@tool(retries=3, timeout=30)
def call_api(query: str) -> str:
    return requests.get(f"https://api.example.com?q={query}").text

# Now fn >> [list] works
pipeline = fetch >> [summarizer, critic] >> editor
pipeline.run("https://example.com/article")
```

`@tool` functions are still callable as regular Python functions:

```python
text = fetch("https://example.com")   # works normally outside a pipeline
```

## Async steps

Any step can be an async function — Quark bridges sync and async transparently:

```python
async def enrich(text: str) -> str:
    async with aiohttp.ClientSession() as s:
        # fetch additional context
        ...
    return enriched_text

pipeline = fetch >> enrich >> agent  # mixed sync/async — just works
```
