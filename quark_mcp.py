"""
quark_mcp — MCP integration for Quark Agents.

Install: pip install "quark-agents[mcp]"

Usage:
    from quark import Agent
    from quark_mcp import MCPClient

    mcp = MCPClient("uvx", ["mcp-server-fetch"])
    agent = Agent(system="You are a helpful assistant.", model="gpt-5.4")
    mcp.inject(agent)

    print(agent.run("Fetch https://example.com and tell me the title."))
"""

import asyncio
import threading

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    raise ImportError(
        "mcp is required for quark_mcp. Install it with: pip install 'quark-agents[mcp]'"
    )


class MCPClient:
    """Runs an MCP server in a background thread and exposes its tools to Quark agents.

    The MCP session runs on a dedicated event loop in a daemon thread. Tool calls
    are bridged from Quark's sync/async executor via run_coroutine_threadsafe,
    avoiding event loop conflicts with Quark's own async machinery.
    """

    def __init__(self, command: str, args: list[str], timeout: int = 10):
        self._server = StdioServerParameters(command=command, args=args)
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._session = None
        self._mcp_tools = []
        self._thread = threading.Thread(
            target=lambda: self._loop.run_until_complete(self._start()),
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(f"MCP server did not start within {timeout}s")

    async def _start(self):
        async with stdio_client(self._server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._mcp_tools = (await session.list_tools()).tools
                self._ready.set()
                await asyncio.Event().wait()  # keep session alive

    def _call(self, name: str, **kwargs) -> str:
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, kwargs), self._loop
        )
        result = future.result(timeout=30)
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    @property
    def tools(self) -> list:
        """Raw MCP tool definitions."""
        return self._mcp_tools

    def inject(self, agent) -> "agent":
        """Inject MCP tools into a Quark agent using the MCP server's own schemas.

        Can be called on multiple agents. MCP tools coexist with regular Quark tools.
        """
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
