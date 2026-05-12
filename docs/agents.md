# Agents

An `Agent` is an LLM with a system prompt, an optional set of tools, and conversation memory. It runs a loop: call the model, execute any tool calls, loop back until the model produces a final answer.

## Creating an agent

```python
from quark import Agent

agent = Agent(
    system="You are a research assistant.",
    model="gpt-5.4",
    name="researcher",
    max_turns=10,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system` | `str` | `"You are a helpful assistant."` | System prompt |
| `tools` | `list[callable]` or `dict[str, callable]` | `{}` | Tools the agent can call |
| `model` | `str` | `"gpt-5.4"` | Any [litellm model string](https://docs.litellm.ai/docs/providers) |
| `max_turns` | `int` | `10` | Max LLM iterations per `run()` call |
| `name` | `str` | `"agent"` | Used in traces and pipeline display |

## Running an agent

```python
result = agent.run("Summarize the latest research on black holes.")
print(result)
```

`run()` is blocking. It returns the final answer as a string.

## Async

Run thousands of agents concurrently on a single event loop — no threads, no overhead:

```python
import asyncio

result = await agent.arun("Summarize the latest research on black holes.")

# fan out across many questions at once
results = await asyncio.gather(*[
    agent.arun(q, history=[]) for q in questions
])
```

`arun()` is the async equivalent of `run()`. Use it on Ray, Lambda, or any async framework.

## Stateless mode

By default agents keep conversation history on the object. Pass `history=[]` to opt into stateless mode — the agent returns `(response, history)` and carries no state itself:

```python
# first turn
response, history = agent.run("My name is Alice.", history=[])

# second turn — pass history back in
response, history = agent.run("What is my name?", history=history)
print(response)  # "Your name is Alice."

# same works with arun
response, history = await agent.arun("My name is Alice.", history=[])
```

Stateless mode makes agents trivially deployable on Lambda, Ray, or any stateless infra — store `history` in Redis or DynamoDB between turns, pass it back in on the next call.

## Streaming

```python
for chunk in agent.stream("Explain quantum entanglement."):
    print(chunk, end="", flush=True)
print()
```

Async streaming:

```python
async for chunk in agent.astream("Explain quantum entanglement."):
    print(chunk, end="", flush=True)
```

`stream()` / `astream()` yield tokens as they arrive from the model.

## Tools

Tools are plain Python functions. The function name, docstring, and type hints are used to build the tool schema automatically.

Pass tools as a list (name inferred from function) or a dict (name explicit):

```python
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

def search_web(query: str) -> str:
    """Search the web."""
    return "..."

# List — simpler, name inferred from function.__name__
agent = Agent(tools=[get_weather, search_web])

# Dict — explicit names, useful when you want to rename
agent = Agent(tools={"weather": get_weather, "search": search_web})
```

```python
def calculate_compound_interest(
    principal: float,
    rate: float,
    years: int,
) -> float:
    """Calculate compound interest."""
    return principal * (1 + rate) ** years

agent = Agent(
    system="You are a financial assistant.",
    tools=[calculate_compound_interest],
)

print(agent.run("If I invest $10,000 at 7% for 20 years, how much will I have?"))
```

When the model decides to call a tool, Quark executes it and feeds the result back automatically. If the model calls multiple tools in one turn, they run in parallel.

### Tool errors

If a tool raises an exception, Quark catches it, formats the error as a string, and sends it back to the model. The model can then decide what to do — retry, ask for clarification, or produce a final answer acknowledging the failure.

```python
def unreliable_tool(query: str) -> str:
    """A tool that might fail."""
    raise ValueError("Service unavailable")

# The model receives: "Error: Service unavailable"
# and handles it gracefully
```

## Conversation memory

Agents remember the full conversation within a session. Each `run()` call appends to the history.

```python
agent = Agent(system="You are a helpful assistant.")

agent.run("My name is Alice.")
response = agent.run("What is my name?")
print(response)  # "Your name is Alice."
```

### Resetting

```python
agent.reset()  # clears history, keeps system prompt
```

## Async steps

If you pass an async function as a tool or pipeline step, Quark handles it transparently — no changes to your code needed.

```python
import asyncio

async def fetch_data(url: str) -> str:
    """Fetch data from a URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

agent = Agent(
    system="Summarize the content.",
    tools={"fetch_data": fetch_data},
)
```
