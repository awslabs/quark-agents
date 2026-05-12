# API Reference

## Agent

```python
Agent(
    *,
    system: str = "You are a helpful assistant.",
    tools: list[Callable] | dict[str, Callable] | None = None,
    model: str = "gpt-5.4",
    max_turns: int = 10,
    name: str = "agent",
)
```

LLM-backed agent with tool use, conversation memory, and `>>` chaining support.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system` | `str` | `"You are a helpful assistant."` | System prompt sent on every call |
| `tools` | `list[callable]` or `dict[str, callable]` | `{}` | List (name from `__name__`) or dict (explicit names) |
| `model` | `str` | `"gpt-5.4"` | Any [litellm model string](https://docs.litellm.ai/docs/providers) |
| `max_turns` | `int` | `10` | Max LLM iterations before returning |
| `name` | `str` | `"agent"` | Identifier used in traces and pipeline display |

### Methods

#### `run(user: str, history: list | None = None) -> str | tuple[str, list]`

Blocking agentic loop. Returns `str` by default. Pass `history=[]` for stateless mode — returns `(response, history)`.

```python
result = agent.run("What is 42 * 17?")

# stateless
response, history = agent.run("What is 42 * 17?", history=[])
```

#### `arun(user: str, history: list | None = None) -> str | tuple[str, list]`

Async agentic loop. Same signature as `run()`. Use with `await` or `asyncio.gather` for concurrent execution.

```python
result = await agent.arun("What is 42 * 17?")

# fan-out
results = await asyncio.gather(*[agent.arun(q, history=[]) for q in questions])
```

#### `stream(user: str) -> Generator[str, None, None]`

Stream the response token by token. Executes tool calls mid-stream.

```python
for chunk in agent.stream("Tell me a story."):
    print(chunk, end="", flush=True)
```

#### `astream(user: str) -> AsyncGenerator[str, None]`

Async streaming. Yields tokens live, executes tool calls mid-stream.

```python
async for chunk in agent.astream("Tell me a story."):
    print(chunk, end="", flush=True)
```

#### `reset() -> None`

Clear conversation history, keeping the system prompt.

```python
agent.reset()
```

### Operators

#### `agent >> other`

Create a `Workflow` with `agent` followed by `other`. `other` can be an `Agent`, `Workflow`, or any callable.

#### `fn >> agent`

Create a `Workflow` with a plain function `fn` followed by `agent`.

---

## Workflow

Created automatically by `>>`. Represents a sequential pipeline of steps.

```python
workflow = agent_a >> agent_b >> agent_c
```

A list within a workflow runs those steps in parallel:

```python
workflow = agent_a >> [agent_b, agent_c] >> agent_d
```

### Methods

#### `run(x: str) -> str`

Execute all steps in order, passing the output of each as input to the next.

```python
result = workflow.run("initial input")
```

### Operators

#### `workflow >> other`

Extend the workflow with an additional step.

#### `workflow >> [a, b]`

Add a parallel step — `a` and `b` run concurrently, their outputs are combined.

---

## Tool schemas

Tools are plain Python functions. Quark builds the JSON schema automatically from:

- **Name** — the key in the `tools` dict
- **Docstring** — used as the tool description
- **Type hints** — mapped to JSON types (`str→string`, `int→integer`, `float→number`, `bool→boolean`)
- **Default values** — parameters with defaults are optional; those without are required

```python
def search(query: str, max_results: int = 5) -> str:
    """Search the web and return results."""
    ...

# Schema produced:
# {
#   "name": "search",
#   "description": "Search the web and return results.",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "query": {"type": "string"},
#       "max_results": {"type": "integer"}
#     },
#     "required": ["query"]
#   }
# }
```
