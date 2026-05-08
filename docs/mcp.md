# Use with MCP

[MCP (Model Context Protocol)](https://modelcontextprotocol.io) is a standard for exposing tools to LLMs — file systems, databases, APIs, browsers, and hundreds of community-built servers. Quark works with any MCP server without any special integration, and understanding why reveals something fundamental about how Quark thinks about tools.

## Quark's mental model: everything is a function call

Quark imagines every tool as two separate things:

**1. A callable** — a Python function that gets executed when the model decides to use it.

**2. A schema** — a description sent to the model so it knows the tool exists and how to call it.

These live separately on every agent:

```python
agent.tools    # dict[str, callable] — what actually runs
agent.schemas  # list[dict]          — what the model sees
```

Quark's `_schema()` helper normally generates schemas by introspecting Python type hints. But `agent.tools` and `agent.schemas` are plain public attributes — you can populate them any way you like.

## Why MCP maps perfectly onto this model

When you connect to an MCP server, it hands you exactly these two things for each tool:

- **A callable** — `session.call_tool(name, args)`, an async coroutine you can wrap as a sync Python function
- **A schema** — `tool.inputSchema`, which is standard JSON Schema

That JSON Schema is the exact format that the `parameters` field in an OpenAI tool definition expects. Quark passes tool schemas to [litellm](https://github.com/BerriAI/litellm), which translates them into whatever format the underlying model provider needs (Bedrock, Anthropic, Gemini, etc). So MCP's schemas slot in with zero modification.

The result: **MCP tools require no translation layer**. You wrap the async call as a sync Python callable, pass the schema directly, and the agent works as if you'd written the tool yourself.

This same pattern works for anything that provides an OpenAI-compatible tool schema: LangChain tools, hand-written JSON schemas, Bedrock agent action groups, and so on.

## Setup

```bash
pip install quark-agents mcp
```

## MCPClient helper

The one wrinkle: MCP sessions are async, but Quark's tool executor runs in threads. Calling `asyncio.run()` inside a thread that's already under an event loop deadlocks. The fix is a dedicated background thread with its own event loop, using `run_coroutine_threadsafe` to bridge across:

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
                await asyncio.Event().wait()  # keep session alive

    def _call(self, name: str, **kwargs) -> str:
        # Bridge from Quark's sync thread into the MCP event loop
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, kwargs), self._loop
        )
        result = future.result(timeout=30)
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    def inject(self, agent: Agent) -> Agent:
        """Populate agent.tools and agent.schemas directly from the MCP server."""
        for t in self._mcp_tools:
            name = t.name
            def make_fn(n):
                def fn(**kwargs): return self._call(n, **kwargs)
                fn.__name__ = n
                return fn
            agent.tools[name] = make_fn(name)   # the callable
            agent.schemas.append({              # the schema — passed straight from MCP
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                }
            })
        return agent
```

## Example: fetch tool

```python
# spin up mcp-server-fetch (install separately: uvx mcp-server-fetch)
mcp = MCPClient("uvx", ["mcp-server-fetch"])

agent = Agent(system="You are a helpful assistant.", model="gpt-5.4")
mcp.inject(agent)

print(agent.run("Fetch https://example.com and tell me the title."))
```

## Inject into multiple agents

`inject()` can be called on multiple agents and mixed freely with regular Quark tools:

```python
mcp = MCPClient("uvx", ["mcp-server-fetch"])

researcher = Agent(system="You research topics by fetching pages.", model="gpt-5.4", name="researcher")
editor     = Agent(system="You polish text.", model="gpt-5.4", name="editor")

mcp.inject(researcher)  # only the researcher gets fetch access

pipeline = researcher >> editor
result = pipeline.run("Summarize https://example.com")
```

## Combining MCP tools with regular tools

MCP tools and plain Python functions coexist on the same agent:

```python
def word_count(text: str) -> str:
    """Count the number of words in a text."""
    return str(len(text.split()))

mcp = MCPClient("uvx", ["mcp-server-fetch"])
agent = Agent(
    system="You can fetch pages and count words.",
    model="gpt-5.4",
    tools={"word_count": word_count},  # regular tool
)
mcp.inject(agent)  # MCP tools added on top

print(agent.run("Fetch https://example.com and count the words on the page."))
```
