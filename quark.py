"""Quark — a ~200-line Python agentic framework. Provider-agnostic via litellm."""

import json, inspect, os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Generator
import litellm

# ---------------------------------------------------------------------------
# Observability — auto-enabled when OTEL_EXPORTER_OTLP_ENDPOINT is set
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, StatusCode
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
    from contextlib import contextmanager
    class _NoOp:
        @contextmanager
        def start_as_current_span(self, *a, **kw): yield None
    _tracer = _NoOp()
    SpanKind = StatusCode = None


def _span(name):
    """Start an OTel span."""
    return _tracer.start_as_current_span(name, kind=getattr(SpanKind, "INTERNAL", None))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """LLM-backed agent with tool use, conversation memory, and >> chaining support."""

    def __init__(self, *, system="You are a helpful assistant.", tools=None,
                 model="gpt-4o", max_turns=10, name="agent"):
        self.name = name
        self.model = model
        self.max_turns = max_turns
        self.tools = tools or {}
        self.schemas = [_schema(n, fn) for n, fn in self.tools.items()]
        self.history = [{"role": "system", "content": system}]

    def __rshift__(self, other):
        return Workflow([self, _wrap(other)])

    def __rrshift__(self, other):       # called when left side has no __rshift__
        return Workflow([_wrap(other), self])

    def run(self, user: str) -> str:
        """Send a message and run the agentic loop until a final answer or max_turns."""
        self.history.append({"role": "user", "content": user})
        with _span(f"invoke_agent {self.name}") as span:
            _attr(span, "gen_ai.operation.name", "invoke_agent")
            _attr(span, "gen_ai.agent.name", self.name)
            _attr(span, "input.value", user)
            for _ in range(self.max_turns):
                msg = litellm.completion(model=self.model, messages=self.history,
                                         tools=self.schemas or None,
                                         num_retries=3).choices[0].message
                self.history.append(msg)
                if not msg.tool_calls:
                    _attr(span, "output.value", msg.content)
                    return msg.content
                def _call(tc):
                    with _span(f"execute_tool {tc.function.name}") as ts:
                        _attr(ts, "gen_ai.tool.name", tc.function.name)
                        _attr(ts, "gen_ai.tool.call.id", tc.id)
                        _attr(ts, "gen_ai.tool.call.arguments", tc.function.arguments)
                        try:
                            result = self.tools[tc.function.name](**json.loads(tc.function.arguments))
                        except Exception as e:
                            _attr(ts, "error.type", type(e).__name__)
                            result = f"Error: {e}"
                        _attr(ts, "gen_ai.tool.call.result", str(result))
                        return {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
                with ThreadPoolExecutor() as ex:
                    for m in ex.map(_call, msg.tool_calls): self.history.append(m)
        return msg.content or "max turns reached"

    def stream(self, user: str) -> Generator:
        """Stream the response token by token, executing any tool calls mid-stream."""
        self.history.append({"role": "user", "content": user})
        yield from self._stream_turn()

    def _stream_turn(self) -> Generator:
        for _ in range(self.max_turns):
            response = litellm.completion(model=self.model, messages=self.history,
                                          tools=self.schemas or None,
                                          stream=True, num_retries=3)
            content, tool_acc = "", {}
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    content += delta.content; yield delta.content
                for tc in delta.tool_calls or []:
                    i = tc.index
                    if i not in tool_acc: tool_acc[i] = {"id": "", "name": "", "arguments": ""}
                    if tc.id: tool_acc[i]["id"] += tc.id
                    if tc.function:
                        if tc.function.name: tool_acc[i]["name"] += tc.function.name
                        if tc.function.arguments: tool_acc[i]["arguments"] += tc.function.arguments
            if not tool_acc:
                self.history.append({"role": "assistant", "content": content}); return
            tool_calls = [{"id": v["id"], "type": "function",
                           "function": {"name": v["name"], "arguments": v["arguments"]}}
                          for _, v in sorted(tool_acc.items())]
            self.history.append({"role": "assistant", "content": content or None,
                                  "tool_calls": tool_calls})
            def _call(tc):
                try:    result = self.tools[tc["function"]["name"]](**json.loads(tc["function"]["arguments"]))
                except Exception as e: result = f"Error: {e}"
                return {"role": "tool", "tool_call_id": tc["id"], "content": str(result)}
            with ThreadPoolExecutor() as ex:
                for m in ex.map(_call, tool_calls): self.history.append(m)

    def reset(self):
        """Clear conversation history, keeping the system prompt."""
        self.history = [self.history[0]]


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class Workflow:
    """Sequential pipeline of steps built by >>; a list-within runs those nodes in parallel."""

    def __init__(self, steps, name="workflow"):
        self.steps = [_wrap(s) if callable(s) and not hasattr(s, "run") else s for s in steps]
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
# Helpers
# ---------------------------------------------------------------------------

def _wrap(fn):
    """Wrap a plain callable as a pipeline-compatible node with >> support."""
    if hasattr(fn, "run") or hasattr(fn, "__rshift__"):
        return fn
    class _FnNode:
        name = getattr(fn, "__name__", str(fn))
        def run(self, x): return fn(x)
        def __rshift__(self, other): return Workflow([self, _wrap(other)])
        def __rrshift__(self, other): return Workflow([_wrap(other), self])
    return _FnNode()


def _run(node, x):
    """Dispatch a single step; handles sync, async, and plain callables transparently."""
    import asyncio, inspect
    result = node.run(x) if hasattr(node, "run") else node(x)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    return str(result)


def _attr(span, key, val):
    """Set a span attribute safely."""
    if span and val is not None:
        try: span.set_attribute(key, val)
        except Exception: pass


def _schema(name: str, fn: Callable) -> dict:
    """Build an OpenAI-compatible tool schema from a function's type hints and docstring."""
    sig = inspect.signature(fn)
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    properties = {k: {"type": type_map.get(v.annotation, "string")} for k, v in sig.parameters.items()}
    return {"type": "function", "function": {"name": name, "description": fn.__doc__ or "",
        "parameters": {"type": "object", "properties": properties,
            "required": [k for k, v in sig.parameters.items()
                         if v.default is inspect.Parameter.empty]}}}
