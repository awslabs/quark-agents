# Quickstart

## Install

```bash
pip install quark-agents
```

## Your first agent

```python
from quark import Agent

agent = Agent(
    system="You are a helpful assistant.",
    model="gpt-5.4",
)

print(agent.run("What is the speed of light?"))
```

## Add tools

Tools are plain Python functions. Type hints become the JSON schema the model uses to call them.

```python
from quark import Agent

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

def search_web(query: str) -> str:
    """Search the web for information."""
    # your search logic here
    return "..."

agent = Agent(
    system="You are a helpful assistant with access to weather and search.",
    model="gpt-5.4",
    tools=[get_weather, search_web],   # or dict: {"get_weather": get_weather, ...}
)

print(agent.run("What's the weather in Tokyo?"))
```

The agent will automatically call `get_weather("Tokyo")`, get the result, and incorporate it into its response. If it needs multiple tools it will call them in parallel.

## Stream the response

```python
for chunk in agent.stream("Write me a short poem about black holes."):
    print(chunk, end="", flush=True)
print()
```

## Your first pipeline

```python
from quark import Agent

def fetch(url: str) -> str:
    """Simulates fetching an article."""
    return "Article content here..."

summarizer = Agent(system="Summarize in 2 sentences.", name="summarizer")
editor     = Agent(system="Polish the summary for a general audience.", name="editor")

pipeline = fetch >> summarizer >> editor
result   = pipeline.run("https://example.com")
print(result)
```

## Switch providers

Change the `model` string to switch providers — no other code changes needed.

```python
# AWS Bedrock
agent = Agent(model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0")

# Anthropic
agent = Agent(model="claude-opus-4-6")

# Gemini
agent = Agent(model="gemini/gemini-2.0-flash")

# Local via Ollama
agent = Agent(model="ollama/llama3")
```

See [Providers](providers.md) for the full list and setup instructions.

## Use MCP tools

Quark works with any [MCP](https://modelcontextprotocol.io) server. The pattern: run the MCP session in a background thread, then inject its tools directly into an agent.

```bash
pip install mcp
```

```python
import asyncio, threading
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from quark import Agent

class MCPClient:
    """Runs an MCP server in a background thread and exposes its tools to Quark."""

    def __init__(self, command: str, args: list[str]):
        self._server = StdioServerParameters(command=command, args=args)
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._session = None
        self._mcp_tools = []
        threading.Thread(target=lambda: self._loop.run_until_complete(self._start()), daemon=True).start()
        self._ready.wait(timeout=10)

    async def _start(self):
        async with stdio_client(self._server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._mcp_tools = (await session.list_tools()).tools
                self._ready.set()
                await asyncio.Event().wait()

    def _call(self, name: str, **kwargs) -> str:
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, kwargs), self._loop
        )
        result = future.result(timeout=30)
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    def inject(self, agent: Agent) -> Agent:
        """Inject MCP tools into a Quark agent using the MCP server's own schemas."""
        for t in self._mcp_tools:
            name = t.name
            def make_fn(n):
                def fn(**kwargs): return self._call(n, **kwargs)
                fn.__name__ = n
                return fn
            agent.tools[name] = make_fn(name)
            agent.schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                }
            })
        return agent
```

Then use it with any agent:

```python
# start the MCP server (here: mcp-server-fetch via uvx)
mcp = MCPClient("uvx", ["mcp-server-fetch"])

agent = Agent(system="You are a helpful assistant.", model="gpt-5.4")
mcp.inject(agent)

print(agent.run("Fetch https://example.com and tell me the title."))
```

`inject()` can be called on multiple agents and mixed with regular Quark tools — they all coexist in the same agent.
