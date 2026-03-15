"""Quark — a <300-line Python agentic framework. Provider-agnostic via litellm."""

import asyncio, json, inspect, os, typing
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Generator
import litellm

# ---------------------------------------------------------------------------
# Observability — auto-enabled when OTEL_EXPORTER_OTLP_ENDPOINT is set
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource

    _endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if _endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        _resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "quark")})
        _provider = TracerProvider(resource=_resource)
        _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=_endpoint)))
        trace.set_tracer_provider(_provider)
        litellm.callbacks = ["otel"]

    _tracer = trace.get_tracer("quark")
except ImportError:
    class _NoOp:
        @contextmanager
        def start_as_current_span(self, *a, **kw): yield None
    _tracer = _NoOp()
    SpanKind = None


def _span(name):
    """Start an OTel span."""
    return _tracer.start_as_current_span(name, kind=getattr(SpanKind, "INTERNAL", None))


def _attr(span, key, val):
    """Set a span attribute safely."""
    if span and val is not None:
        try: span.set_attribute(key, val)
        except Exception: pass


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """LLM-backed agent with tool use, conversation memory, and >> chaining support."""

    def __init__(self, *, system="You are a helpful assistant.", tools=None,
                 model="gpt-5.4", max_turns=10, name="agent"):
        self.name = name
        self.model = model
        self.max_turns = max_turns
        tools = tools or {}
        self.tools = {fn.__name__: fn for fn in tools} if isinstance(tools, list) else tools
        self.schemas = [_schema(n, fn) for n, fn in self.tools.items()]
        self.history = [{"role": "system", "content": system}]

    def __rshift__(self, other):
        return Workflow([self, _wrap(other)])

    def __rrshift__(self, other):
        return Workflow([_wrap(other), self])

    def run(self, user: str) -> str:
        """Send a message and run the agentic loop until a final answer or max_turns."""
        self.history.append({"role": "user", "content": user})
        with self._agent_span(user) as span:
            for _ in range(self.max_turns):
                content, tool_calls = self._completion()
                if not tool_calls:
                    _attr(span, "output.value", content)
                    return content or ""
                self._run_tools(tool_calls)
            _attr(span, "output.value", content)
            return content or "max turns reached"

    def stream(self, user: str) -> Generator:
        """Stream the response token by token, executing any tool calls mid-stream."""
        self.history.append({"role": "user", "content": user})
        with self._agent_span(user) as span:
            for _ in range(self.max_turns):
                content, tool_calls = yield from self._stream_completion()
                if not tool_calls:
                    _attr(span, "output.value", content)
                    return
                self._run_tools(tool_calls)
            fallback = content or "max turns reached"
            _attr(span, "output.value", fallback)
            yield fallback

    @contextmanager
    def _agent_span(self, user):
        """Open an OTel span for an agent invocation."""
        with _span(f"invoke_agent {self.name}") as s:
            _attr(s, "gen_ai.operation.name", "invoke_agent")
            _attr(s, "gen_ai.agent.name", self.name)
            _attr(s, "input.value", user)
            yield s

    def _completion(self):
        """Single non-streaming LLM call. Returns (content, list[dict] tool_calls)."""
        msg = litellm.completion(model=self.model, messages=self.history,
                                 tools=self.schemas or None, num_retries=3).choices[0].message
        self.history.append(msg)
        tool_calls = [{"id": tc.id, "type": "function",
                       "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                      for tc in (msg.tool_calls or [])]
        return msg.content, tool_calls or None

    def _stream_completion(self):
        """Single streaming LLM call. Yields chunks live, returns (content, tool_calls)."""
        response = litellm.completion(model=self.model, messages=self.history,
                                      tools=self.schemas or None, stream=True, num_retries=3)
        content, tool_acc = "", {}
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
                yield delta.content
            for tc in delta.tool_calls or []:
                i = tc.index
                if i not in tool_acc:
                    tool_acc[i] = {"id": "", "name": "", "arguments": ""}
                if tc.id: tool_acc[i]["id"] += tc.id
                if tc.function:
                    if tc.function.name: tool_acc[i]["name"] += tc.function.name
                    if tc.function.arguments: tool_acc[i]["arguments"] += tc.function.arguments

        if not tool_acc:
            self.history.append({"role": "assistant", "content": content})
            return content, None

        tool_calls = [{"id": v["id"], "type": "function",
                       "function": {"name": v["name"], "arguments": v["arguments"]}}
                      for _, v in sorted(tool_acc.items())]
        self.history.append({"role": "assistant", "content": content or None,
                              "tool_calls": tool_calls})
        return content, tool_calls

    def _run_tools(self, tool_calls):
        """Run tool calls in parallel and append results to history."""
        def _call(tc):
            name, cid, args = tc["function"]["name"], tc["id"], tc["function"]["arguments"]
            with _span(f"execute_tool {name}") as ts:
                _attr(ts, "gen_ai.tool.name", name)
                _attr(ts, "gen_ai.tool.call.id", cid)
                _attr(ts, "gen_ai.tool.call.arguments", args)
                try:
                    result = self.tools[name](**json.loads(args))
                except Exception as e:
                    _attr(ts, "error.type", type(e).__name__)
                    result = f"Error: {e}"
                _attr(ts, "gen_ai.tool.call.result", str(result))
                return {"role": "tool", "tool_call_id": cid, "content": str(result)}

        with ThreadPoolExecutor() as ex:
            for m in ex.map(_call, tool_calls):
                self.history.append(m)

    def reset(self):
        """Clear conversation history, keeping the system prompt."""
        self.history = [self.history[0]]


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class Workflow:
    """Sequential pipeline of steps built by >>; a list-within runs those nodes in parallel."""

    def __init__(self, steps, name="workflow"):
        self.steps = [s if isinstance(s, list) else _wrap(s) for s in steps]
        self.name = name

    def __rshift__(self, other):
        return Workflow(self.steps + [_wrap(other)])

    def __rrshift__(self, other):
        return Workflow([_wrap(other)] + self.steps)

    def run(self, x):
        """Execute all steps in order, passing output of each as input to the next."""
        with _span(f"workflow {self.name}") as span:
            _attr(span, "gen_ai.operation.name", "workflow")
            _attr(span, "input.value", x)
            for step in self.steps:
                if isinstance(step, list):
                    with ThreadPoolExecutor() as ex:
                        results = list(ex.map(lambda s: _run(s, x), step))
                    feedback = "\n\n---\n\n".join(
                        f"[{getattr(s, 'name', getattr(s, '__name__', str(s)))}]:\n{r}"
                        for s, r in zip(step, results)
                    )
                    x = f"[original]:\n{x}\n\n---\n\n{feedback}"
                else:
                    x = _run(step, x)
            _attr(span, "output.value", x)
            return x


# ---------------------------------------------------------------------------
# Tool decorator
# ---------------------------------------------------------------------------

def tool(fn=None, *, retries=0, timeout=None):
    """Decorator making a function pipeline-compatible with >>, OTel tracing, and retries.

    Usage:
        @tool
        def fetch(url): ...

        @tool(retries=3, timeout=30)
        def call_api(query): ...
    """
    if fn is None:
        return lambda f: tool(f, retries=retries, timeout=timeout)

    class _ToolNode:
        name = fn.__name__
        __name__ = fn.__name__
        __doc__ = fn.__doc__

        def __call__(self, *args, **kwargs):
            return fn(*args, **kwargs)

        def run(self, x):
            with _span(f"tool {fn.__name__}") as s:
                _attr(s, "gen_ai.tool.name", fn.__name__)
                _attr(s, "input.value", str(x))
                last_exc = None
                for attempt in range(retries + 1):
                    try:
                        if timeout:
                            with ThreadPoolExecutor(max_workers=1) as ex:
                                result = ex.submit(fn, x).result(timeout=timeout)
                        else:
                            result = fn(x)
                        if inspect.isawaitable(result):
                            result = asyncio.run(result)
                        _attr(s, "gen_ai.tool.call.result", str(result))
                        return str(result)
                    except Exception as e:
                        last_exc = e
                _attr(s, "error.type", type(last_exc).__name__)
                raise last_exc

        def __rshift__(self, other): return Workflow([self, _wrap(other)])
        def __rrshift__(self, other): return Workflow([_wrap(other), self])

    return _ToolNode()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(fn):
    """Wrap a plain callable as a pipeline-compatible node with >> support."""
    if isinstance(fn, list) or hasattr(fn, "run") or hasattr(fn, "__rshift__"):
        return fn
    class _FnNode:
        name = getattr(fn, "__name__", str(fn))
        def run(self, x): return fn(x)
        def __rshift__(self, other): return Workflow([self, _wrap(other)])
        def __rrshift__(self, other): return Workflow([_wrap(other), self])
    return _FnNode()


def _run(node, x):
    """Dispatch a single step; handles sync, async, and plain callables transparently."""
    result = node.run(x) if hasattr(node, "run") else node(x)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    return str(result)


def _schema(name: str, fn: Callable) -> dict:
    """Build an OpenAI-compatible tool schema from a function's type hints and docstring."""
    sig = inspect.signature(fn)
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean",
                list: "array", dict: "object"}

    def _resolve_type(annotation):
        origin = getattr(annotation, "__origin__", None)
        if origin is typing.Union:
            args = [a for a in annotation.__args__ if a is not type(None)]
            if args: return _resolve_type(args[0])
        return type_map.get(annotation, "string")

    properties = {k: {"type": _resolve_type(v.annotation)} for k, v in sig.parameters.items()}
    return {"type": "function", "function": {"name": name, "description": fn.__doc__ or "",
        "parameters": {"type": "object", "properties": properties,
            "required": [k for k, v in sig.parameters.items()
                         if v.default is inspect.Parameter.empty]}}}
